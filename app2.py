import streamlit as st
import PyPDF2
from langchain_community.embeddings import OpenAIEmbeddings  # Updated import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI    # Updated import
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS    # Updated import

class SimplePostGenerator:
    def __init__(self):
        # Initialize OpenAI
        self.embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
        self.llm = ChatOpenAI(
            openai_api_key=st.secrets["OPENAI_API_KEY"],
            model="gpt-3.5-turbo",
            temperature=0.7
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.vector_store = None

    def process_pdf(self, pdf_file):
        """Process PDF and create vector store"""
        # Read PDF
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_text(text)

        # Create vector store
        self.vector_store = FAISS.from_texts(texts, self.embeddings)
        return text

    def generate_post(self, query, context=None):
        """Generate LinkedIn post"""
        prompt = f"""
        Create an engaging LinkedIn post about: {query}
        
        Guidelines:
        1. Start with an attention-grabbing hook
        2. Include 2-3 key points
        3. End with a call-to-action
        4. Add relevant hashtags
        Keep it professional and under 3 paragraphs.
        """
        
        response = self.llm.predict(prompt)
        return response

    def chat_with_pdf(self, query):
        """Chat with PDF content"""
        if not self.vector_store:
            return "Please upload and process a PDF first."

        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(),
            memory=self.memory
        )
        
        response = chain({"question": query})
        return response['answer']

def main():
    st.title("📱 LinkedIn Post Generator")
    st.write("Upload a PDF and generate engaging LinkedIn posts!")

    # Initialize generator
    if 'generator' not in st.session_state:
        st.session_state.generator = SimplePostGenerator()

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar for PDF upload
    with st.sidebar:
        st.header("📄 Document Upload")
        uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
        
        if uploaded_file:
            if st.button("Process PDF"):
                with st.spinner("Processing PDF..."):
                    try:
                        st.session_state.generator.process_pdf(uploaded_file)
                        st.success("PDF processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing PDF: {str(e)}")

    # Main area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("🎯 Generate Post")
        post_topic = st.text_area("What would you like to post about?")
        
        if post_topic and st.button("Generate Post"):
            with st.spinner("Generating post..."):
                try:
                    post = st.session_state.generator.generate_post(post_topic)
                    st.text_area("Generated Post", post, height=300)
                except Exception as e:
                    st.error(f"Error generating post: {str(e)}")

    with col2:
        st.header("💬 Chat with PDF")
        question = st.text_input("Ask about your document:")
        
        if question and st.button("Ask"):
            try:
                response = st.session_state.generator.chat_with_pdf(question)
                st.session_state.chat_history.append(("You", question))
                st.session_state.chat_history.append(("Assistant", response))
            except Exception as e:
                st.error(f"Error: {str(e)}")

        # Display chat history
        for role, message in st.session_state.chat_history:
            with st.container():
                if role == "You":
                    st.write(f"👤 **You:** {message}")
                else:
                    st.write(f"🤖 **Assistant:** {message}")

if __name__ == "__main__":
    main()