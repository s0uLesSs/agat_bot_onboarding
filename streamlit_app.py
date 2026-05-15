import streamlit as st
import requests
from pathlib import Path
import os
from streamlit_pdf_viewer import pdf_viewer

st.set_page_config(page_title="Онбординг Агат", layout="wide")

RAG_API_URL = "http://localhost:8001/ask"

# ------------- начало (если ничего не определено, начальная позиция - обучение) -------------------
if "mode" not in st.session_state:
    st.session_state.mode = "learning"
if "selected_pdf" not in st.session_state:
    st.session_state.selected_pdf = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------- добавление сайдбара (навигации) -------------
with st.sidebar:
    st.markdown("## 🧭 Навигация")
    if st.button("📚 Обучение", use_container_width=True):
        st.session_state.mode = "learning"
        st.rerun()
    if st.button("🤖 RAG-помощник", use_container_width=True):
        st.session_state.mode = "rag"
        st.rerun()
    if st.button("🎓 Тесты", use_container_width=True):
        st.session_state.mode = "tests"
        st.rerun()    

if st.session_state.mode == "learning":
    st.title("📚 Обучение сотрудников")
    
    pdf_files = list(Path("instructions").glob("*.pdf"))
    
    if not pdf_files:
        st.warning("Нет PDF-файлов в папке instructions")
    else:
        cols = st.columns(3)
        for idx, pdf in enumerate(pdf_files):
            with cols[idx % 3]:
                if st.button(f"📘 {pdf.stem[:30]}", use_container_width=True):
                    st.session_state.selected_pdf = str(pdf)
                    st.rerun()
    
    if st.session_state.selected_pdf and os.path.exists(st.session_state.selected_pdf):
        st.divider()
        with open(st.session_state.selected_pdf, "rb") as f:
            st.download_button("📥 Скачать PDF", f, file_name=Path(st.session_state.selected_pdf).name)
        st.info("PDF найден. Нажмите 'Скачать PDF' для просмотра.")

# ---------- предпросмотр пдф ----------
if st.session_state.selected_pdf and os.path.exists(st.session_state.selected_pdf):
    st.divider()
    st.subheader(f"📄 Предпросмотр: {Path(st.session_state.selected_pdf).name}")
    
    # Создаём колонки с отступами по бокам
    col1, col2, col3 = st.columns([0.5, 8, 0.5])  # Центральная колонка шире
    
    with col2:
        with open(st.session_state.selected_pdf, "rb") as f:
            pdf_bytes = f.read()
        
        pdf_viewer(
            input=pdf_bytes,
            width=None,  # Автоматическая ширина
            height=700,
            viewer_align="center"
        )

elif st.session_state.mode == "rag":
    st.title("🤖 RAG-помощник")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    if prompt := st.chat_input("Задайте вопрос..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            try:
                response = requests.post(RAG_API_URL, json={"question": prompt}, timeout=60)
                if response.status_code == 200:
                    answer = response.json()["answer"]
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error("Ошибка API")
            except Exception as e:
                st.error(f"Ошибка: {e}")
    
    if st.button("🗑️ Очистить диалог"):
        st.session_state.messages = []
        st.rerun()