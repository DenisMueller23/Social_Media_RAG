import streamlit as st
from langchain_openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import pinecone
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector
from dotenv import load_dotenv
import os
import PyPDF2

load_dotenv()

# Fetch the OpenAI API key from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
pinecone_api_key = os.getenv("PINCONE_API_KEY")

# initialize pinecone
pinecone.init(openai_api_key=pinecone_api_key, environment=pinecone_env)

# Create or connect to Pinecone index
index_name = "pdf-embedding-index"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536)  # Using OpenAI embeddings dimension
index = pinecone.Index(index_name)

# Create OpenAI embeddings object
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

if not openai_api_key:
    raise ValueError("OpenAI API key not found in the environment variables")
if not pinecone_api_key:
    raise ValueError("Pinecone key not found in the environment variables")

def getLLMResponse(query,job_type,tasktype_option, pdf_content=None):
    # 'text-davinci-003' model is depreciated now, so we are using the openai's recommended model
    llm = OpenAI(temperature=.9, model="gpt-3.5-turbo-instruct")
    
    # Few Shot Prompting approach
    if job_type=="Data Scientist": 

        examples = [
        {
            "query": "What does it take to be a successful data scientist?",
            "answer": "Success in data science requires curiosity, problem-solving skills, and the ability to communicate insights effectively. It's not just about coding or algorithms, but about understanding the context of the data and how to use it to make impactful decisions."
        }, {
            "query": "How do you stay updated with the latest trends in data science?",
            "answer": "I stay updated by constantly learning—reading research papers, attending conferences, participating in online courses, and collaborating with other data scientists. The field evolves quickly, so being proactive in learning is key to staying relevant."
        }, {
            "query": "What’s the most rewarding part of working in data science?",
            "answer": "The most rewarding part is transforming raw data into actionable insights that drive real-world impact. Whether it’s improving customer experiences or optimizing business processes, seeing how data can influence decisions is incredibly fulfilling."
        }, {
            "query": "How do you approach ethical dilemmas in data science?",
            "answer": "Ethics in data science is crucial. I always prioritize transparency, fairness, and privacy when working with data. It's important to build models that are not only accurate but also unbiased and responsible, ensuring that they don’t perpetuate harm."
        }
        ]
    
    elif job_type=="Business Person":  
        examples = [
        {
            "query": "What do you prioritize when making business decisions?",
            "answer": "I prioritize long-term value over short-term gains. Every decision should align with the company’s vision and mission, while also being flexible enough to adapt to changing market dynamics."
        }, {
            "query": "How do you handle failure in business?",
            "answer": "Failure is inevitable, but it’s how you respond to it that matters. I view failure as an opportunity to learn and grow. By analyzing what went wrong and adjusting strategies, I turn setbacks into stepping stones for future success."
        }, {
            "query": "What motivates you to keep pushing forward in business?",
            "answer": "I’m motivated by the desire to create lasting value and to lead teams toward achieving something meaningful. Every new challenge is an opportunity to innovate and to leave a positive impact on the industry and the people involved."
        }, {
            "query": "How do you balance innovation with risk management?",
            "answer": "Balancing innovation and risk management is about careful planning. I encourage creative thinking and experimentation but always within a framework that assesses potential risks. It’s about taking calculated risks, not reckless ones."
        }
        ]
    elif job_type=="C-Level Strategist": 
        examples = [
        {
            "query": "What do you consider when shaping a company's legacy?",
            "answer": "A company’s legacy is built on its values, impact, and the culture it fosters. I focus on creating a sustainable and ethical business that not only delivers profits but also contributes to society and supports future generations of leaders."
        }, {
            "query": "How do you approach succession planning at the executive level?",
            "answer": "Succession planning is about identifying and nurturing future leaders early. I believe in mentoring high-potential individuals and ensuring that they’re equipped with the right skills, values, and vision to take the company forward."
        }, {
            "query": "What’s the key to long-term business growth?",
            "answer": "The key to long-term growth is adaptability. Markets change, technologies evolve, and customer expectations shift. A successful company is one that stays true to its core values while continuously innovating and adapting to these changes."
        }, {
            "query": "How do you ensure that the company's strategy remains relevant?",
            "answer": "I ensure relevance by constantly revisiting the strategy, incorporating market trends, customer feedback, and industry shifts. Staying connected to both the internal culture and external environment allows me to keep the company’s strategy sharp and future-ready."
        }
        ]


    example_template = """
    Question: {query}
    Response: {answer}
    """

    example_prompt = PromptTemplate(
        input_variables=["query", "answer"],
        template=example_template
    )

    # defines the initial part of the prompt to provide contecxt
    # remember: prompt is the input to the LLM exclusively
    prefix = """You are a {template_ageoption}, and {template_tasktype_option}: 
    Here are some examples: 
    """

    suffix = """
    Question: {template_userInput}
    Response: """

    example_selector = LengthBasedExampleSelector(
        examples=examples,
        example_prompt=example_prompt,
        max_length=200
    )


    new_prompt_template = FewShotPromptTemplate(
        example_selector=example_selector,  # use example_selector instead of examples
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["template_userInput","template_ageoption","template_tasktype_option"],
        example_separator="\n"
    )

    # Include the PDF content in the prompt (if available)
    if pdf_content:
        # Combine the PDF content with the user's query, making it clear the query is about the document
        combined_query = f"Here is the content of the document:\n\n{pdf_content}\n\nUser's question: {query}"
    else:
        combined_query = query
    
    # Format the few shot prompts so it contains all the selected parameters as well as PDF input context
    prompt = new_prompt_template.format(
        template_userInput=combined_query,
        template_ageoption=job_type,
        template_tasktype_option=tasktype_option
    )
  
    # print(new_prompt_template.format(template_userInput=query,template_ageoption=age_option,template_tasktype_option=tasktype_option))

    #Recently langchain has recommended to use invoke function for the below please :)
    response=llm.invoke(prompt)

    return response



# configure general settings
st.set_page_config(page_title="Social Media RAG App",
                    page_icon='✅',
                    layout='centered',
                    initial_sidebar_state='auto')


def extract_text_from_uploaded_file(uploaded_file):
    # Check if the file is a PDF
    if uploaded_file.type == "application/pdf":
        # Read PDF file using PyPDF2
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text
    else:
        # For non-PDF files, decode as UTF-8 (assuming it's a text-based file)
        return uploaded_file.read().decode("utf-8")
st.header("Hey, let's move your personal brand forward?")

# UI Starts here

# change color coding / style
## Background, Slider and font color
st.markdown(
    """
    <style>
    /* Change background color to light brown */
    .stApp {
        background-color: #D2B48C;  /* Light brown (tan) */
    }

    /* Change all text (labels, input boxes, dropdowns) to dark grey */
    .stApp h1, .stApp p, .stApp label, .stApp div, .stApp input, .stApp textarea {
        color: #2F2F2F;  /* Dark grey font */
    }

    /* Style for the slider bar (range input) */
    .stSlider > div > div > div > input[type="range"] {
        accent-color: lightblue;  /* Light blue slider thumb */
    }

    /* Change the track background (the bar) */
    .stSlider > div > div > div > div {
        background: black !important;  /* black bar */
    }

    /* Change font color in select box options */
    .stSelectbox div[data-testid="stMarkdownContainer"] * {
        color: #2F2F2F !important;  /* Dark grey font for selection options */
    }

    /* Change font color in the box where the selected value is shown */
    .stSelectbox>div>div>div>div {
        color: #2F2F2F !important;  /* Dark grey font for selected value in dropdown */
    }

    /* Change font color in file uploader box (drag-and-drop area) */
    .stFileUploader label, .stFileUploader div {
        color: #2F2F2F !important;  /* Dark grey font in file uploader */
    }

    </style>
    """,
    unsafe_allow_html=True
)

# File Uploader for Drag-and-Drop
uploaded_file = st.file_uploader("Upload your document for querying", type=["txt", "pdf"])

if uploaded_file is not None:
    file_content = extract_text_from_uploaded_file(uploaded_file)
    
    if file_content:
        st.text_area("Document Content Preview", value=file_content[:500], height=150)  # Show first 500 characters
    else:
        st.write("Could not extract content from the file.")

form_input = st.text_area('Enter text', height=275)

tasktype_option = st.selectbox(
    'Please select the action to be performed?',
    ('Create a LinkedIn post', 'Explain easily to learn', 'Create a summary'),key=1)

age_option= st.selectbox(
    'For which age group?',
    ('Data Scientist', 'Business Person', 'C-Level Strategist'),key=2)

numberOfWords= st.slider('Words limit', 1, 200, 25)

submit = st.button("Generate")

if submit:
    st.write(getLLMResponse(form_input,age_option,tasktype_option))


