import streamlit as st
import os
from typing import List, Dict
import PyPDF2
import openai
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

class LinkedInPostGenerator:
    def __init__(self):
        # Configuration and setup
        self.setup_environment()
        self.vector_store = None
        self.retrieval_chain = None

    def setup_environment(self):
        """
        Set up environment variables and API configurations
        """
        # Ensure these are set in your .env or through Streamlit secrets
        openai.api_key = st.secrets.get("OPENAI_API_KEY")
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo", 
            temperature=0.7
        )

    def extract_text_from_pdf(self, uploaded_file) -> str:
        """
        Extract text from uploaded PDF file
        """
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text

    def create_vector_store(self, text: str):
        """
        Create vector store from extracted text
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        texts = text_splitter.split_text(text)
        
        self.vector_store = FAISS.from_texts(
            texts, 
            self.embeddings
        )

    def generate_linkedin_posts(self, context: str, num_posts: int = 3) -> List[str]:
        """
        Generate LinkedIn post suggestions based on context
        """
        post_prompt = PromptTemplate(
            input_variables=["context"],
            template="""
            Given the following context about a professional topic, 
            generate {num_posts} unique and engaging LinkedIn post suggestions 
            that highlight key insights, add professional value, 
            and encourage audience interaction:

            Context: {context}

            For each post, provide:
            1. A compelling hook
            2. 3-4 key points
            3. A call-to-action or thought-provoking question
            4. Relevant hashtags
            """
        )

        # Create retrieval chain
        retrieval_qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(),
            chain_type_kwargs={
                "prompt": post_prompt
            }
        )

        # Generate posts
        posts = []
        for i in range(num_posts):
            post = retrieval_qa.run(f"Generate a unique LinkedIn post about the context, focusing on professional insights")
            posts.append(post)

        return posts

def main():
    st.title("🚀 LinkedIn Post Generator with RAG")
    
    # Sidebar for configuration
    st.sidebar.header("Document Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload PDF", 
        type=['pdf'], 
        help="Upload a PDF to extract professional context"
    )
    
    # Number of posts slider
    num_posts = st.sidebar.slider(
        "Number of Post Suggestions", 
        min_value=1, 
        max_value=5, 
        value=3
    )

    # Main application logic
    post_generator = LinkedInPostGenerator()

    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            # Extract text from PDF
            pdf_text = post_generator.extract_text_from_pdf(uploaded_file)
            
            # Create vector store
            post_generator.create_vector_store(pdf_text)
            
            # Generate posts
            posts = post_generator.generate_linkedin_posts(
                context=pdf_text, 
                num_posts=num_posts
            )
            
            # Display posts
            st.header("🔥 Generated LinkedIn Post Suggestions")
            for i, post in enumerate(posts, 1):
                with st.expander(f"Post {i}"):
                    st.write(post)
                    st.button(f"Copy Post {i}", key=f"copy_{i}")

    else:
        st.info("Upload a PDF to generate LinkedIn post suggestions")

if __name__ == "__main__":
    main()