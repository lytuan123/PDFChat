from PyPDF2 import PdfReader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import json
import re
import gradio as gr
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# Tải các biến môi trường từ file .env
load_dotenv()

# Lấy giá trị của biến OPENAI_API_KEY từ môi trường
api_key = os.getenv('OPENAI_API_KEY')

if not api_key:
    raise ValueError("API key không được tìm thấy. Vui lòng kiểm tra file .env.")

# Tạo FAISS index từ tài liệu PDF
def create_faiss_index(file_path):
    with open(file_path, "rb") as file:
        reader = PdfReader(file)
        corpus = ''.join([p.extract_text() for p in reader.pages if p.extract_text()])

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(corpus)

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectors = FAISS.from_texts(chunks, embeddings)
    return vectors
# Tìm các đoạn văn bản liên quan từ FAISS
def retrieve_relevant_chunks(query, vectors):
    return vectors.similarity_search(query, k=5)  # Trả về 5 đoạn văn bản liên quan nhất

# Tạo câu trả lời bằng cách sử dụng RAG
def generate_answer_with_rag(query, vectors):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")  # Sử dụng GPT-3.5 Turbo hoặc GPT-4
    relevant_chunks = retrieve_relevant_chunks(query, vectors)
    
    # Kết hợp các đoạn văn bản liên quan với truy vấn
    context = " ".join([chunk.page_content for chunk in relevant_chunks])

    chain = load_qa_chain(llm, chain_type="stuff")
    return chain.run(input_documents=relevant_chunks, question=query)
# Load feedback từ file JSON
def load_feedback():
    if os.path.exists("feedback.json"):
        with open("feedback.json", "r") as f:
            return json.load(f)
    return {}

# Lưu feedback vào file JSON
def save_feedback(query, correct_answer):
    feedback = load_feedback()
    feedback[query] = correct_answer
    with open("feedback.json", "w") as f:
        json.dump(feedback, f)

# Chuẩn hóa câu hỏi
def normalize_query(query):
    query = query.lower().strip()
    query = re.sub(r'\W+', ' ', query)
    return query

# Trả lời câu hỏi và xử lý phản hồi người dùng
def answer_query_with_rag(query, vectors):
    try:
        query = normalize_query(query)
        feedback = load_feedback()
        if query in feedback:
            return feedback[query]  # Trả lời từ phản hồi đã lưu

        return generate_answer_with_rag(query, vectors)
    except Exception as e:
        return f"Lỗi khi xử lý truy vấn: {str(e)}"
# Xử lý upload và xử lý PDF
def upload_and_process_pdf(uploaded_file_path):
    if uploaded_file_path is not None:
        try:
            vectors = create_faiss_index(uploaded_file_path)
            return "Tải lên thành công. Bạn có thể bắt đầu đặt câu hỏi.", vectors
        except Exception as e:
            return "Lỗi: " + str(e), None
    return "Vui lòng tải lên một file PDF.", None

# Giao diện Gradio
with gr.Blocks() as demo:
    vectors = gr.State(None)
    user_feedback = gr.Textbox(label="Phản hồi của bạn (nếu có)", placeholder="Nhập câu trả lời chính xác nếu phản hồi không đúng")

    gr.Markdown("# PDFChat : Đặt câu hỏi về tài liệu PDF của bạn")

    with gr.Row():
        with gr.Column():
            upload = gr.File(label="Tải lên PDF", type="filepath")
            upload_button = gr.Button("Xử lý PDF")
        with gr.Column():
            status = gr.Textbox(label="Trạng thái", interactive=False)

    upload_button.click(upload_and_process_pdf, inputs=[upload], outputs=[status, vectors])

    query = gr.Textbox(label="Đặt câu hỏi")
    response = gr.Textbox(label="Câu trả lời", interactive=False)
    feedback_button = gr.Button("Gửi phản hồi")

    query.submit(answer_query_with_rag, inputs=[query, vectors], outputs=response)
    feedback_button.click(save_feedback, inputs=[query, user_feedback], outputs=None)

demo.launch(share=True)
