import streamlit as st
import PyPDF2
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import torch
from diffusers import StableDiffusionPipeline
import gc
import os
import psutil

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
                1. Present insights directly and objectively
                2. Do NOT mention or reference that these insights come from a paper, report, research, or any other document type
                3. Write as if you are sharing your professional knowledge
                4. Focus on the facts and insights themselves
                5. Start with an attention-grabbing statement about the topic
                6. Include 2-3 concrete insights or findings
                7. End with a thought-provoking question or call-to-action
                8. Add relevant hashtags
                9. Keep it under 3 paragraphs
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
            return result['answer']

        except Exception as e:
            st.error(f"Error in chat: {str(e)}")
            raise

class M1ResourceMonitor:
    """Monitor system resources for M1 Mac"""
    @staticmethod
    def check_memory():
        try:
            memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024  # GB
            return memory < 14  # Leave 2GB for system
        except Exception:
            return True  # Default to True if unable to check memory

    @staticmethod
    def cleanup():
        try:
            # Check if MPS is available using hasattr
            if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            gc.collect()
        except Exception:
            gc.collect()

class M1OptimizedGenerator:
    def __init__(self):
        """Initialize Stable Diffusion optimized for M1 MacBook Air"""
        self.model_id = "runwayml/stable-diffusion-v1-5"
        self.device = self._get_device()
        self.pipe = None
        self.resource_monitor = M1ResourceMonitor()
        self.setup_pipeline()

    def _get_device(self):
        """Determine the best available device"""
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def setup_pipeline(self):
        """Set up pipeline with M1-specific optimizations"""
        try:
            st.write("Initializing Stable Diffusion...")
            
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            if self.device == "mps":
                st.success("✅ Using M1 Metal acceleration!")
                self.pipe = self.pipe.to(self.device)
            elif self.device == "cuda":
                st.success("✅ Using CUDA acceleration!")
                self.pipe = self.pipe.to(self.device)
            else:
                st.warning("⚠️ Running on CPU - performance will be limited")
            
            self.pipe.enable_attention_slicing(slice_size="max")
            
            st.success("✅ Pipeline ready!")
            
        except Exception as e:
            st.error(f"Setup error: {str(e)}")
            raise

    def generate_image(self, prompt: str, height: int = 512, width: int = 512, 
                      num_inference_steps: int = 20, guidance_scale: float = 7.5):
        """Generate image with M1 optimization and memory management"""
        try:
            if not self.resource_monitor.check_memory():
                self.resource_monitor.cleanup()
                
            with st.spinner("🎨 Generating image (1-2 minutes)..."):
                try:
                    image = self.pipe(
                        prompt=prompt,
                        height=min(height, 512),
                        width=min(width, 512),
                        num_inference_steps=min(num_inference_steps, 25),
                        guidance_scale=guidance_scale
                    ).images[0]
                    
                    self.resource_monitor.cleanup()
                    return image
                    
                except RuntimeError as e:
                    st.warning("⚠️ Attempting memory recovery...")
                    self.resource_monitor.cleanup()
                    
                    return self.pipe(
                        prompt=prompt,
                        height=384,
                        width=384,
                        num_inference_steps=15,
                        guidance_scale=guidance_scale
                    ).images[0]
                
        except Exception as e:
            st.error(f"Generation error: {str(e)}")
            return None

def main():
    st.set_page_config(
        page_title="RAG LinkedIn App",
        page_icon="📱",
        layout="wide"
    )

    st.title("📱 LinkedIn Post Generator & PDF Chat")

    # Initialize text generation components first
    if 'generator' not in st.session_state:
        with st.spinner("Initializing text generation components..."):
            st.session_state.generator = SimplePostGenerator()
            st.success("✅ Text generation ready!")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Defer image generation initialization
    if 'image_generator_initialized' not in st.session_state:
        st.session_state.image_generator_initialized = False

    with st.sidebar:
        st.header("📄 Document Upload")
        uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
        
        if uploaded_file:
            file_size = uploaded_file.size / (1024 * 1024)
            if file_size > 50:
                st.warning("⚠️ Large PDF files might impact performance")
            
            if st.button("Process PDF"):
                with st.spinner("Processing PDF..."):
                    try:
                        text = st.session_state.generator.process_pdf(uploaded_file)
                        st.success("PDF processed successfully!")
                        st.session_state.chat_history = []
                    except Exception as e:
                        st.error(f"Error processing PDF: {str(e)}")

        # Image generation initialization
        if not st.session_state.image_generator_initialized:
            st.header("🎨 Image Generation")
            if st.button("Initialize Image Generator"):
                with st.spinner("Downloading and initializing Stable Diffusion... (This may take 3-5 minutes on first run)"):
                    try:
                        st.session_state.image_generator = M1OptimizedGenerator()
                        st.session_state.image_generator_initialized = True
                        st.success("✅ Image generation ready!")
                    except Exception as e:
                        st.error(f"Error initializing image generator: {str(e)}")
        else:
            st.header("🎨 Image Settings")
            image_settings = {
                'height': st.slider("Image Height", 256, 512, 384, 128),
                'width': st.slider("Image Width", 256, 512, 384, 128),
                'num_inference_steps': st.slider("Quality (Steps)", 15, 25, 20, 5),
                'guidance_scale': st.slider("Creativity", 1.0, 20.0, 7.5, 0.5)
            }

    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("🎯 Generate Post & Image")
        post_topic = st.text_area("What would you like to post about?")
        
        generate_image = st.checkbox(
            "Generate matching image", 
            value=False,
            disabled=not st.session_state.image_generator_initialized,
            help="Initialize image generator first to enable this option"
        )
        
        if post_topic and st.button("Generate Content"):
            with st.spinner("Generating post..."):
                try:
                    post = st.session_state.generator.generate_post(post_topic)
                    st.text_area("Generated Post", value=post, height=200)
                    
                    if generate_image and st.session_state.image_generator_initialized:
                        image = st.session_state.image_generator.generate_image(
                            prompt=post,
                            **image_settings
                        )
                        
                        if image:
                            st.image(image, caption="Generated Image for Post")
                            os.makedirs("generated_images", exist_ok=True)
                            timestamp = torch.rand(1).item()
                            filename = f"linkedin_post_{timestamp:.3f}.png"
                            image.save(os.path.join("generated_images", filename))
                            st.success(f"Image saved as {filename}")
                    
                    if st.button("📋 Copy Post"):
                        st.write("Post copied to clipboard!")
                        
                except Exception as e:
                    st.error(f"Error generating content: {str(e)}")

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