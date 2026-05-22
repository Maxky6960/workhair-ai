# app.py
import os

import streamlit as st
from dotenv import load_dotenv
import requests

from rag_engine import RAGEngine

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL","nvidia/nemotron-3-super-120b-a12b:free")
API_URL = "https://openrouter.ai/api/v1/chat/completions"


@st.cache_resource
def load_rag():
    return RAGEngine("knowledge/workhair_rag.txt")


rag = load_rag()

st.title("Luna ผู้ช่วย AI ของร้าน Workhair")
st.caption("ถามเรื่องทรงผม เวลาเปิด หรือข้อมูลร้านได้เลย")

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
    system_prompt = f"""คุณคือ Luna ผู้ช่วย AI ของร้าน Workhair ตอบเฉพาะจากข้อมูลด้านล่าง
ถ้าไม่พบข้อมูล ให้บอกว่าไม่ทราบค่ะ อย่าแต่งข้อมูลเอง

ข้อมูลร้าน:
{context}
"""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    answer = response.json()["choices"][0]["message"]["content"]

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
