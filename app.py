# app.py
import os

import streamlit as st
from dotenv import load_dotenv
from google import genai

from rag_engine import RAGEngine

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL = "gemini-2.5-flash"

# ⚙️ ตั้งค่าหน้า
st.set_page_config(
    page_title="Luna - Workhair AI Assistant",
    page_icon="💇",
    
    initial_sidebar_state="expanded",
)

st.set_option("client.showErrorDetails", False)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');

    :root {
        --ink: #1c1a18;
        --ink-soft: #4a423c;
        --accent: #d48f6b;
        --accent-dark: #a96445;
        --cream: #f8f3ee;
        --card: rgba(255, 255, 255, 0.78);
        --border: rgba(89, 73, 63, 0.12);
    }

    html, body, [class*="stApp"] {
        font-family: 'Manrope', sans-serif;
        color: var(--ink);
        background: radial-gradient(circle at top left, #fff9f2 0%, #f3e9de 45%, #efe3d7 100%);
    }

    h1, h2, h3, h4 {
        font-family: 'Playfair Display', serif;
        letter-spacing: 0.2px;
        color: var(--ink);
    }

    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 1.8rem 2.2rem;
        border-radius: 24px;
        background: linear-gradient(120deg, rgba(255, 255, 255, 0.92), rgba(249, 242, 234, 0.82));
        border: 1px solid var(--border);
        box-shadow: 0 22px 60px rgba(72, 50, 33, 0.12);
        margin-bottom: 1.6rem;
    }

    .hero-title {
        font-size: 2.4rem;
        margin-bottom: 0.35rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: var(--ink-soft);
        margin-bottom: 1rem;
    }

    .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        background: rgba(212, 143, 107, 0.14);
        color: var(--accent-dark);
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 0.5rem;
    }

    .chat-shell {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1.2rem 1.4rem 0.6rem;
        box-shadow: 0 16px 50px rgba(77, 52, 34, 0.08);
    }

    .stChatMessage {
        padding: 0.25rem 0.2rem;
    }

    .stChatMessage [data-testid="stMarkdownContainer"] p {
        line-height: 1.6;
        font-size: 1rem;
    }

    [data-testid="stChatInput"] textarea {
        border-radius: 18px;
        border: 1px solid rgba(100, 78, 61, 0.22);
        padding: 0.85rem 1rem;
        font-size: 1rem;
    }

    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.6);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] h2 {
        font-family: 'Playfair Display', serif;
    }

    @media (max-width: 760px) {
        .hero {
            padding: 1.4rem 1.4rem;
        }
        .hero-title {
            font-size: 2rem;
        }
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_rag():
    return RAGEngine("knowledge/workhair_rag.txt")


rag = load_rag()

st.markdown(
    """
    <div class="hero">
        <div class="hero-pill">💇 Workhair Salon AI</div>
        <div class="hero-title">Luna ผู้ช่วย AI ของ Workhair</div>
        <div class="hero-subtitle">ถามเรื่องทรงผม เวลาเปิดร้าน โปรโมชั่น หรือข้อมูลร้านได้เลย</div>
        <div>
            <span class="hero-pill">ตอบจากข้อมูลร้านเท่านั้น</span>
            <span class="hero-pill">สุภาพ ใช้งานง่าย</span>
            <span class="hero-pill">เหมาะกับมือถือ</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="chat-shell">', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("ถามอะไรเกี่ยวกับร้านได้เลย..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # RAG: Search
    context_chunks = rag.search(prompt, top_k=3)
    context = "\n---\n".join(context_chunks)

    # Generate
    full_prompt = f"""คุณคือ Luna ผู้ช่วย AI ของร้าน Workhair ตอบเฉพาะจากข้อมูลด้านล่าง
ถ้าไม่พบข้อมูล ให้บอกว่าไม่ทราบ อย่าแต่งข้อมูลเอง

ข้อมูลร้าน:
{context}

คำถาม: {prompt}
"""
    response = client.models.generate_content(model=MODEL, contents=full_prompt)
    answer = response.text

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)

st.markdown("</div>", unsafe_allow_html=True)
        
        
        
        
        
        
