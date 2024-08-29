from PyPDF2 import PdfReader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import gradio as gr

# Tải các biến môi trường từ file .env
load_dotenv()

# Lấy giá trị của biến OPENAI_API_KEY từ môi trường
api_key = os.getenv('OPENAI_API_KEY')

if not api_key:
    raise ValueError("API key not found. Please ensure it is set correctly in your environment or .env file.")

def storeDocEmbeds(file_path, filename):
    try:
        with open(file_path, "rb") as file:
            reader = PdfReader(file)
            corpus = ''.join([p.extract_text() for p in reader.pages if p.extract_text()])

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_text(corpus)

        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectors = FAISS.from_texts(chunks, embeddings)

        # Lưu trữ FAISS index
        vectors.save_local(filename + "_faiss")
    except Exception as e:
        return str(e)

def getDocEmbeds(file_path, filename):
    try:
        if not os.path.exists(filename + "_faiss"):
            error = storeDocEmbeds(file_path, filename)
            if error:
                return None, error

        # Tải lại FAISS index với `allow_dangerous_deserialization`
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectors = FAISS.load_local(filename + "_faiss", embeddings, allow_dangerous_deserialization=True)

        return vectors, None
    except Exception as e:
        return None, str(e)

def conversational_chat(query, history, vectors):
    try:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo")  # Thay đổi model nhẹ hơn
        qa = ConversationalRetrievalChain.from_llm(llm, retriever=vectors.as_retriever(), return_source_documents=True)
        result = qa({"question": query, "chat_history": history})
        history.append((query, result["answer"]))
        return result["answer"], history, None
    except Exception as e:
        return None, history, str(e)

def upload_and_process_pdf(uploaded_file_path):
    if uploaded_file_path is not None:
        try:
            vectors, error = getDocEmbeds(uploaded_file_path, "uploaded_file")
            if error:
                return "Error: " + error, None
            return "Upload successful. You can now ask questions.", vectors
        except Exception as e:
            return "Error: " + str(e), None
    return "Please upload a PDF file.", None

def chat_interface(query, vectors, history):
    if vectors is not None:
        response, history, error = conversational_chat(query, history, vectors)
        if error:
            return "Error: " + error, history
        return response, history
    return "Please upload a PDF file and try again.", history

with gr.Blocks() as demo:
    history = gr.State([])
    vectors = gr.State(None)
    
    gr.Markdown("# PDFChat : Ask questions about your PDF")

    with gr.Row():
        with gr.Column():
            upload = gr.File(label="Upload PDF", type="filepath")
            upload_button = gr.Button("Process PDF")
        with gr.Column():
            status = gr.Textbox(label="Status", interactive=False)

    upload_button.click(upload_and_process_pdf, inputs=[upload], outputs=[status, vectors])

    query = gr.Textbox(label="Ask a question")
    response = gr.Textbox(label="Response", interactive=False)
    
    query.submit(chat_interface, inputs=[query, vectors, history], outputs=[response, history])

demo.launch(share=True)
