import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import base64
import io
from PIL import Image
import pdf2image
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

#Google API

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_t_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_conversational_chain():
    prompt_template = """
    When answering a question, provide as much detail as feasible based on the context submitted. 
    If the answer is not accessible, simply state "answer is not available in the context" instead of providing an incorrect answer.
\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question}
        , return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def get_gemini_response(input_text, pdf_content, prompt):
    try:
        model = genai.GenerativeModel('gemini-pro-vision')
        response = model.generate_content([input_text, pdf_content[0], prompt])
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

def pdf_input_setup(uploaded_file):
    if uploaded_file is not None:
        try:
            images = pdf2image.convert_from_bytes(uploaded_file.read())
            first_page = images[0]
            
            img_byte_arr = io.BytesIO()
            first_page.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            
            pdf_parts = [
                {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(img_byte_arr).decode()
                }
            ]
            return pdf_parts
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            return None
    else:
        st.error("No file uploaded")
        return None
    

def set_footer():
    footer_html = """
    <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: rgba(0, 0, 0, 0.5);
            color: white;
            text-align: center;
            padding: 10px;
            font-size: 14px;
        }
    </style>
    <div class="footer">
        <p>Copyright © rahman_etri@copyright.com</p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

# CSS for transparent background 
def set_response_section_style():
    style = """
    <style>
    .response-section {
        background: rgba(10, 10, 0, 0.7); 
        padding: 20px;
        border-radius: 10px;
        
    }
    </style>
    """
    st.markdown(style, unsafe_allow_html=True)

def get_gemini_vision_response(input,image):
    model = genai.GenerativeModel('gemini-pro-vision')
    if input!="":
       response = model.generate_content([input,image])
    else:
       response = model.generate_content(image)
    return response.text



def extract_yv_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("=")[1]
        
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)

        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]

        return transcript

    except Exception as e:
        raise e
    

def generate_gemini_content(transcript_text, prompt):

    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt + transcript_text)
    return response.text


def main():
    st.set_page_config(page_title="MultiFetch Assistant")
    set_background('bg3.jpg')
    st.title("MultiFetch Assistant")
    st.markdown("<h3 style='color: orange;'>A multi-purpose searching application</h3>", unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        .stButton>button {
            background-color: #007bff;
            color: white;
        }fseeor, 
        </style>
        """,
        unsafe_allow_html=True
    ) 
    
    # Sidebar
    with st.sidebar:
        st.title("Menu:")
        # app_mode = st.radio("Choose the app mode", ["Talk with PDFs", "ATS Resume Expert", "Image Content Extractor"])
        app_mode = st.radio("Choose the app mode", ["Chat with PDFs", "ATS Resume Expert", "Image Content Extractor", "YouTube Transcript to Notes"])


    if app_mode == "Chat with PDFs":
        st.header("Chat with PDFs")

       
        if 'pdfs_processed' not in st.session_state:
            st.session_state.pdfs_processed = False

        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Process Button", accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_t_chunks(raw_text)
                get_vector_store(text_chunks)
                st.session_state.pdfs_processed = True
                st.success("Done")

        
        if st.session_state.pdfs_processed:
            user_question = st.text_input("Ask a Question from the PDF Files")
            if user_question:
                user_input(user_question)
            
            if st.button("Reset"):
                st.session_state.pdfs_processed = False
                st.experimental_rerun()
                

    elif app_mode == "ATS Resume Expert":
        st.header("ATS Resume Expert")

        input_text = st.text_area("Job Description: ", key="input")
        uploaded_file = st.file_uploader("Upload your resume (PDF)...", type=["pdf"])

        if uploaded_file is not None:
            st.write("PDF Uploaded Successfully")

        col1, col2 = st.columns(2)
        submit_1 = col1.button("Tell Me about the Resume")
        submit_2 = col2.button("Tell me about the Percentage match")

        input_prompt1 = """
        Being a Technical Human Resource Manager, you have experience. It is your responsibility to compare the included resume with the job description. 
        Could you please provide your expert assessment of how well the candidate's profile fits the position? 
        Draw attention to the applicant's advantages and disadvantages in light of the job criteria.

        """

        input_prompt3 = """
        You are a skilled applicant tracking system (ATS) scanner who is well-versed in data science and ATS features. 
        It is your responsibility to match the job description provided with the resume. If the resume matches the job description, please let me know what percentage of matches there are. 
        The output should be shown as a percentage first, then any missing keywords, and lastly, the conclusions.

        """

        if submit_1 or submit_2:
            if uploaded_file is not None:
                pdf_content = pdf_input_setup(uploaded_file)
                if pdf_content:
                    if submit_1:
                        response = get_gemini_response(input_text, pdf_content, input_prompt1)
                    else:
                        response = get_gemini_response(input_text, pdf_content, input_prompt3)
                    st.subheader("Response:")
                    set_response_section_style()
                    st.markdown(f'<div class="response-section">{response}</div>', unsafe_allow_html=True)
            else:
                st.write("Please upload the resume")

    elif app_mode == "Image Content Extractor":
        st.header("Image Content Extractor")      
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        image = ""
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image.", use_column_width=True)

        input = st.text_input("Input Prompt: ", key="image_input")
        submit = st.button("Extract Content")

        if submit:
            response = get_gemini_vision_response(input, image)
            st.subheader("Response:")
            set_response_section_style()
            st.markdown(f'<div class="response-section">{response}</div>', unsafe_allow_html=True)
    
    elif app_mode == "YouTube Transcript to Notes":
        st.header("Make Summarized Notes of a Youtube Video")
        youtube_link = st.text_input("Enter YouTube Video Link:")
        prompt="""YouTube video summarizer. The task is to summarize the video in 250 words using the transcript text 
        and key points. Please offer a summary of the text provided here.  """

        if youtube_link:
            video_id = youtube_link.split("=")[1]
            st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

        if st.button("Get Summarized Notes"):
            transcript_text = extract_yv_transcript_details(youtube_link)

            if transcript_text:
                summary = generate_gemini_content(transcript_text, prompt)
                st.markdown("## Detailed Notes:")
                set_response_section_style()
                st.markdown(f'<div class="response-section">{summary}</div>', unsafe_allow_html=True)

    set_footer()

if __name__ == "__main__":
    main()


# to run the application -> streamlit run multiFetch_Assistant.py