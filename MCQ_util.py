# Utils
import time
from typing import List

import langchain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter 

from pydantic import BaseModel

#print(f"LangChain version: {langchain.__version__}")

# Vertex AI
from google.cloud import aiplatform
from langchain.chat_models import ChatVertexAI
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.document_loaders import PyPDFLoader
import os
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser


import streamlit as st


def get_llm_model() :
    # LLM model
    llm = VertexAI(
    model_name="text-bison@001",
    max_output_tokens=256,
    temperature=0.1,
    top_p=0.8,
    top_k=40,
    verbose=True,
)
    return llm
    
    
def get_chat_model() : 
    # Chat
    chat = ChatVertexAI()
    return chat


def split_docs(docs) : 

    chunk_size = 1000
    chunk_overlap = 25

    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    c_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    #Create a split of the document using the text splitter
    text_splitter = r_splitter
    splits = text_splitter.split_documents(docs)
    print(len(splits))
    return splits


# Utility functions for Embeddings API with rate limiting
def rate_limit(max_per_minute):
    period = 60 / max_per_minute
    print("Waiting")
    while True:
        before = time.time()
        yield
        after = time.time()
        elapsed = after - before
        sleep_time = max(0, period - elapsed)
        if sleep_time > 0:
            print(".", end="")
            time.sleep(sleep_time)



class CustomVertexAIEmbeddings(VertexAIEmbeddings, BaseModel):
    requests_per_minute: int
    num_instances_per_batch: int

    # Overriding embed_documents method
    def embed_documents(self, texts: List[str]):
        limiter = rate_limit(self.requests_per_minute)
        results = []
        docs = list(texts)

        while docs:
            # Working in batches because the API accepts maximum 5
            # documents per request to get embeddings
            head, docs = (
                docs[: self.num_instances_per_batch],
                docs[self.num_instances_per_batch :],
            )
            chunk = self.client.get_embeddings(head)
            results.extend(chunk)
            next(limiter)

        return [r.values for r in results]



class DataBase : 
    def __init__(self, docs) : 
        self.database = self.convert_to_vectordb(docs)
        

        

    def convert_to_vectordb(self, docs) :
        splits = split_docs(docs)
        # Embedding
        EMBEDDING_QPM = 100
        EMBEDDING_NUM_BATCH = 5
        embeddings = CustomVertexAIEmbeddings(
            requests_per_minute=EMBEDDING_QPM,
            num_instances_per_batch=EMBEDDING_NUM_BATCH)
        
        persist_directory = os.getcwd() + '/docs/chroma/'
        print(persist_directory)

        # Create the vector store
        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_directory
        )

        return vectordb       


    def as_retriever(self) :
        return self.database.as_retriever()


class PromptFactory : 
    
    @staticmethod
    def retrival_chain_small_db(db, llm ) :   
        # Build prompt
        template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.  . 
        {context}
        Question: {question}
        Helpful Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)# Run chain
        qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        return qa_chain

    @staticmethod
    def format_topics_prompt(text, llm):
        topic_schema = ResponseSchema(name="topics", description="topics generated as a list of strings")
        response_schemas = [topic_schema]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        print(format_instructions)

        pp_template = """ 
        For the following text , extract the topics.
        text: {text}

        {format_instructions}
        """
        pp_prompt = ChatPromptTemplate.from_template(template=pp_template)
        query = text
        #post_process_chain = LLMChain(llm=chat, prompt=post_process_prompt)
        messages = pp_prompt.format_messages(text=text, 
                                format_instructions=format_instructions)
        response = llm(messages)
        print(response.content)
        output_dict = output_parser.parse(response.content)
        return output_dict
    


    @staticmethod
    def retrival_prompt_simple(llm, db) : 
        qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever()
        )
        return qa_chain

    


class PromptApp : 
    def __init__(self, llm_model, chat_model) : 
        self.llm = llm_model
        self.chat = chat_model   


    def get_output(self, query, db) :
        retrival_chain = PromptFactory.retrival_chain_small_db(db, self.llm)
        result = retrival_chain({"query": query})
        # Check the result of the query
        topics_str = result["result"]
        print('topics_str : ' , topics_str)
        output_dict = PromptFactory.format_prompt(topics_str, self.chat)
        print('ouptut_dict : ', output_dict)
        
    



def load_pdf(doc):
    loader = PyPDFLoader(doc)
    #Load the document by calling loader.load()
    pages = loader.load()
    print(len(pages))
    return pages


class QAInterface :
    def __init__(self, doc) : 
        pages = load_pdf(doc)
        self.db = DataBase(pages)
        self.prompt_app = PromptApp(get_llm_model(), get_chat_model())

    def get_topics_from_document(self) :
        '''
        pass the doc to Langchain to extract topics
        '''
        query = 'Who are the important topics here?'
        topics_list = prompt_app.get_output(query, db)
        return topics_list

    def generate_question_from_topic(self, topic , in_context=True) :
        question = "Can you please create a Multiple Choice Question for this topic :" + topic + " ?"
        #print(question)
        PromptFactory.retrival_chain_small_db(db, llm ) 
        PromptApp.retrival_chain_small_db()
        result = mcq_chain({"query": question})
        # Check the result of the query
        print(result["result"])



    
    


