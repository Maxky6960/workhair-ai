
Workhair AI - Salon Chatbot (Streamlit)

Quick start (local)
1) Create .env in gemini-chatbot with values:
   GOOGLE_API_KEY=your_key
   LINE_CHANNEL_ACCESS_TOKEN=optional
   LINE_GROUP_ID=optional
   SUPABASE_URL=optional
   SUPABASE_SERVICE_ROLE_KEY=optional
2) Install deps
   pip install -r requirements.txt
3) Run
   streamlit run gemini-chatbot/app.py

Hugging Face Spaces (Streamlit)
1) Set Space to Streamlit.
2) Add files from this repo.
3) Configure Secrets in the Space:
   GOOGLE_API_KEY
   LINE_CHANNEL_ACCESS_TOKEN (optional)
   LINE_GROUP_ID (optional)
   SUPABASE_URL (optional)
   SUPABASE_SERVICE_ROLE_KEY (optional)
4) App entry: gemini-chatbot/app.py

Note
- If GOOGLE_API_KEY is missing, the chatbot falls back to knowledge-base answers only.
