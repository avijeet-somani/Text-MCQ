import streamlit as st
import sys

sys.path.append(".")
import MCQ_util as mcq_util
#import MCQ-util as mcq_util

def main():
    st.title("PDF MCQ")
    
    # Create a file uploader widget
    uploaded_file = st.file_uploader("Upload a file", type=["pdf"])
    
    if uploaded_file is not None:
        st.write("You've uploaded a file!")
        #topics = mcq_util.get_topics_from_document(uploaded_file)    
        topics = [ 'Upanishadic thinkers', 'Jaina teacher Mahavira', 'Buddha', 'Chinese pilgrims' ]
        for topic in topics : 
            if st.button(topic) :
                st.write('topic to be explored: ', topic ) 
                mcq_util.
        
        
        
if __name__ == "__main__":
    main()