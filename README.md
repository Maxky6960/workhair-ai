# Workhair AI (Luna)

Workhair AI คือผู้ช่วยแชทสำหรับร้านทำผม Workhair ที่ตอบคำถามลูกค้าจากฐานความรู้ของร้าน เช่น ราคา เวลาเปิด-ปิด บริการที่มี ระยะเวลาทำผม วิธีจองคิว และข้อมูลช่างประจำร้าน

โปรเจกต์นี้เป็นการ pivot จาก MilkLab template มาเป็นธุรกิจร้านทำผมของครอบครัว โดยใช้ RAG chatbot เพื่อช่วยลดภาระการตอบแชทซ้ำ ๆ ของแอดมิน

## Live Demo

- Demo URL: ใส่ลิงก์ HuggingFace Spaces หรือ Streamlit deployment ที่นี่
- Pivot Worksheet: [PIVOT.md](./PIVOT.md)

## Features

- ตอบคำถามลูกค้าจากข้อมูลจริงใน `knowledge/workhair_rag.txt`
- ใช้ RAG pipeline: load, chunk, embed, search, generate
- ใช้ text embedding และ FAISS สำหรับค้นหาข้อมูลที่เกี่ยวข้อง
- ใช้ Gemini API เพื่อสร้างคำตอบภาษาไทยที่เหมาะกับลูกค้าร้านทำผม
- มี retry/backoff เมื่อเจอ rate limit หรือ quota error จาก Gemini API
- มี Streamlit chat UI ที่ปรับ branding เป็น Luna ผู้ช่วยของ Workhair
- รองรับการรันผ่าน Docker Compose

## Tech Stack

- Python
- Streamlit
- Google GenAI SDK
- Sentence Transformers
- FAISS
- Docker / Docker Compose

## Project Structure

```text
.
├── app.py                    # Streamlit app และ Gemini response flow
├── rag_engine.py             # RAG engine: chunk, embed, cache, search
├── knowledge/workhair_rag.txt # ฐานความรู้ของร้าน Workhair
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker image config
├── docker-compose.yml        # Local Docker Compose runner
└── PIVOT.md                  # Pivot worksheet ของโปรเจกต์
```

## Environment Variables

สร้างไฟล์ `.env` จาก `.env.example` แล้วตั้งค่าหลัก ๆ ดังนี้

```env
APITOKEN=your_gemini_api_key
API_MODEL=gemini-2.5-flash
RAG_TOP_K=4
RAG_CACHE_DIR=.rag_cache
LOG_LEVEL=INFO
```

ห้าม commit ไฟล์ `.env` ขึ้น GitHub เพราะมี API key

## Run Locally

ติดตั้ง dependencies แล้วรันด้วย Streamlit

```bash
pip install -r requirements.txt
streamlit run app.py
```

เปิดแอปที่

```text
http://localhost:8501
```

## Run With Docker

```bash
docker compose up -d --build
```

เปิดแอปที่

```text
http://localhost:8501
```

## Example Questions

- ร้านเปิดกี่โมง
- ตัดผมชายราคาเท่าไหร่
- ทำสีผมใช้เวลานานไหม
- มีช่างคนไหนถนัดงานเฟดบ้าง
- ถ้าจะเลื่อนคิวต้องแจ้งล่วงหน้ากี่ชั่วโมง

## Demo Day Self-Check

- [X] Deploy URL ใช้งานได้
- [X] ไม่มี `.env` หรือไฟล์ secret อยู่ใน repository
- [x] `PIVOT.md` ครบ 3 ข้อ
- [x] README อธิบายระบบของ Workhair ไม่ใช่ MilkLab
- [x] Knowledge base, prompt, และ UI ปรับเป็น Workhair แล้ว
- [x] มี rate limit retry สำหรับ Gemini API

## Reflection

สิ่งที่ทำได้ดีที่สุดคือการปรับ RAG chatbot ให้ตอบจากข้อมูลเฉพาะของร้าน Workhair ได้จริง และปรับ UI ให้เหมาะกับธุรกิจร้านทำผมมากกว่า template เดิม

ส่วนที่ยากที่สุดของการ pivot คือการทำให้ differentiation ไม่ใช่แค่เปลี่ยนชื่อร้าน แต่ต้องคิดว่าร้านทำผมมีปัญหาเฉพาะ เช่น การถามราคา การจองคิว การเลือกช่าง และการดูแลผมหลังใช้บริการ
