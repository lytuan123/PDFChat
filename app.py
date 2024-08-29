# Các thư viện cần thiết
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
from langchain.chains import RetrievalQA  # Thay vì ConversationalRetrievalChain

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

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    chunks = splitter.split_text(corpus)

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectors = FAISS.from_texts(chunks, embeddings)
    return vectors

# Hàm trả lời câu hỏi với ngữ cảnh và lưu ngữ cảnh
previous_queries = ""

def generate_answer_with_rag(query, vectors, previous_queries):
    # Tạo mô hình ngôn ngữ
    llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=api_key)
    
    # Kết hợp các câu hỏi trước đó với câu hỏi hiện tại để tạo ngữ cảnh
    contextual_query = f"{previous_queries} {query}"

    # Tạo chuỗi truy xuất với khả năng truy vấn từ chỉ mục FAISS
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectors.as_retriever(), return_source_documents=True)

    # Thực hiện truy vấn với ngữ cảnh và lấy kết quả
    result = qa_chain({"query": contextual_query})

    # Trả về kết quả từ trường 'result'
    return result['result']

def answer_query_with_rag(query, vectors):
    global previous_queries  # Sử dụng biến toàn cục để duy trì ngữ cảnh
    try:
        query = normalize_query(query)
        feedback = load_feedback()
        if query in feedback:
            return feedback[query]  # Trả lời từ phản hồi đã lưu

        # Lấy câu trả lời từ hàm generate_answer_with_rag
        answer = generate_answer_with_rag(query, vectors, previous_queries)
        
        # Cập nhật previous_queries với câu hỏi và câu trả lời hiện tại
        previous_queries += f"Q: {query} A: {answer} "
        
        return answer
    except Exception as e:
        return f"Lỗi khi xử lý truy vấn: {str(e)}"


# Chuẩn hóa câu hỏi
def normalize_query(query):
    query = query.lower().strip()
    query = re.sub(r'\W+', ' ', query)
    return query

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

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8000)
