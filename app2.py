import streamlit as st
import PyPDF2
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

class SimplePostGenerator:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
        self.llm = ChatOpenAI(
            openai_api_key=st.secrets["OPENAI_API_KEY"],
            model="gpt-3.5-turbo",
            temperature=0.7
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True
        )
        self.vector_store = None
        self.raw_text = ""

    def process_pdf(self, pdf_file):
        """Process PDF with improved text extraction and chunking"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            self.raw_text = ""
            
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    self.raw_text += text + "\n\n"
            
            st.write("Preview of extracted text:")
            st.write(self.raw_text[:500] + "...")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            
            texts = text_splitter.split_text(self.raw_text)
            st.write(f"Created {len(texts)} text chunks")
            
            self.vector_store = FAISS.from_texts(
                texts,
                self.embeddings,
                metadatas=[{"source": f"chunk_{i}"} for i in range(len(texts))]
            )
            
            return self.raw_text
            
        except Exception as e:
            st.error(f"Error in PDF processing: {str(e)}")
            raise

    def generate_post(self, query, context=None):
        """Generate LinkedIn post with neutral perspective"""
        try:
            if self.vector_store:
                relevant_docs = self.vector_store.similarity_search(query, k=3)
                context = "\n".join(doc.page_content for doc in relevant_docs)
                
                prompt = f"""
                Create an engaging LinkedIn post about: {query}

                Use these insights from the provided context:
                {context}

                Important guidelines:
                1. Present insights directly
                2. Do NOT mention or reference that these insights come from a paper, report, research, or any other document type
                3. Write as if you are sharing your professional knowledge while using a LinkedIn friendly layout
                4. Focus on the facts and insights themselves
                5. Start with an attention-grabbing statement about the topic that always includes a fitting emoji
                6. Include 2-3 concrete insights or findings
                7. End with a thought-provoking question or call-to-action
                8. Add relevant hashtags
                9. Keep it under 3 paragraphs
                
                Example format:
                [Attention-grabbing statement about the topic]
                
                [Key insight 1 presented as a direct observation]
                [Key insight 2 presented as industry knowledge]
                
                [Engaging question for the audience] #relevanthashtag1 #relevanthashtag2

                Remember: Maintain a professional tone but write as if sharing personal industry knowledge, not summarizing a document.
                """
            else:
                prompt = f"""
                Create an engaging LinkedIn post about: {query}
                
                Guidelines:
                1. Start with an attention-grabbing statement
                2. Share 2-3 key insights as direct professional knowledge
                3. End with an engaging question or call-to-action
                4. Add relevant hashtags
                5. Keep it professional and under 3 paragraphs
                6. Present information as industry expertise
                """
            
            response = self.llm.predict(prompt)
            
            # Additional check to remove any reference to papers or research
            forbidden_terms = ['paper', 'research', 'study', 'report', 'analysis', 'findings', 'document']
            response_lower = response.lower()
            
            if any(term in response_lower for term in forbidden_terms):
                # If forbidden terms are found, regenerate with stricter prompt
                strict_prompt = f"""
                Rewrite this LinkedIn post removing any reference to papers, research, studies, or documents:

                {response}

                Requirements:
                1. Present all insights as direct professional knowledge
                2. Remove any academic or research-related language
                3. Maintain the same key points but express them as industry observations
                4. Keep the engaging and professional tone
                """
                response = self.llm.predict(strict_prompt)
            
            return response
        except Exception as e:
            st.error(f"Error in post generation: {str(e)}")
            raise

    def chat_with_pdf(self, query):
        """Chat with PDF content"""
        if not self.vector_store:
            return "Please upload and process a PDF first."

        try:
            custom_template = """
            You are a helpful assistant answering questions about a specific document.
            Use the following pieces of context to answer the question at the end.
            If you don't know the answer based on the context, say you don't know.
            Don't try to make up an answer.
            
            Context: {context}
            
            Chat History: {chat_history}
            Human: {question}
            Assistant:"""

            CUSTOM_PROMPT = PromptTemplate(
                input_variables=["context", "chat_history", "question"],
                template=custom_template
            )

            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": 3},
                    search_type="mmr"
                ),
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT},
                return_source_documents=True,
                verbose=True
            )
            
            result = chain({"question": query})
            
            # Display relevant chunks in expandable section
            with st.expander("View relevant document sections"):
                for i, doc in enumerate(result['source_documents']):
                    st.write(f"Relevant section {i+1}:")
                    st.write(doc.page_content)
                    st.write("---")
            
            return result['answer']

        except Exception as e:
            st.error(f"Error in chat: {str(e)}")
            raise

def main():
    st.title("📱 LinkedIn Post Generator & PDF Chat")
    
    if 'generator' not in st.session_state:
        st.session_state.generator = SimplePostGenerator()

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.header("📄 Document Upload")
        uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
        
        if uploaded_file:
            if st.button("Process PDF"):
                with st.spinner("Processing PDF..."):
                    try:
                        text = st.session_state.generator.process_pdf(uploaded_file)
                        st.success("PDF processed successfully!")
                        st.session_state.chat_history = []
                    except Exception as e:
                        st.error(f"Error processing PDF: {str(e)}")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("🎯 Generate Post")
        post_topic = st.text_area("What would you like to post about?")
        
        if post_topic and st.button("Generate Post"):
            with st.spinner("Generating post..."):
                try:
                    post = st.session_state.generator.generate_post(post_topic)
                    st.text_area("Generated Post", value=post, height=300)
                    
                    if st.button("📋 Copy to Clipboard"):
                        st.write("Post copied to clipboard!")
                        
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
        if st.session_state.chat_history:
            st.write("Chat History:")
            for role, message in st.session_state.chat_history:
                with st.container():
                    if role == "You":
                        st.write(f"👤 **You:** {message}")
                    else:
                        st.write(f"🤖 **Assistant:** {message}")

if __name__ == "__main__":
    main()