import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
import ollama
import fitz  # Это pymupdf для чтения PDF

app = FastAPI()

# --- 1. НАСТРОЙКА ---
# Папка с вашими PDF файлами
PDF_FOLDER = "instructions"
os.makedirs(PDF_FOLDER, exist_ok=True)

# Настройка ChromaDB (наша локальная база знаний)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
# Создаем "полку" для инструкций, если её нет
collection = chroma_client.get_or_create_collection(
    name="company_instructions",
    # Указываем, какую модель использовать для превращения текста в числа
    embedding_function=embedding_functions.OllamaEmbeddingFunction(
        model_name="nomic-embed-text"
    )
)

# --- 2. ФУНКЦИЯ ДЛЯ ЗАГРУЗКИ PDF В БАЗУ ЗНАНИЙ ---
def index_all_pdfs():
    """Проходим по всем PDF в папке и добавляем их в ChromaDB"""
    print("Начинаю индексацию PDF файлов...")
    # Проверяем, есть ли уже документы в базе, чтобы не дублировать
    if collection.count() > 0:
        print("База знаний уже содержит документы. Пропускаем индексацию.")
        return

    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            file_path = os.path.join(PDF_FOLDER, filename)
            print(f"Обрабатываю: {filename}")
            
            # 1. Открываем PDF и извлекаем текст
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            
            # 2. Разбиваем текст на небольшие кусочки (chunks)
            # Это важно! Мы разбиваем текст на части по 1500 символов.
            chunk_size = 1500
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            
            # 3. Добавляем каждый кусочек в базу ChromaDB
            # Для каждого кусочка генерируем уникальный ID
            for i, chunk in enumerate(chunks):
                chunk_id = f"{filename}_chunk_{i}"
                collection.add(
                    documents=[chunk],
                    metadatas=[{"source": filename}], # Сохраняем имя файла-источника
                    ids=[chunk_id]
                )
            print(f"Добавлено {len(chunks)} кусочков из файла {filename}")
    print("Индексация завершена!")

# --- 3. СОЗДАЕМ API ЭНДПОИНТ ---
class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: QueryRequest):
    """Получает вопрос, ищет ответ в инструкциях и возвращает его"""
    user_question = request.question
    
    # 1. Ищем в базе 30 самых похожих кусочков
    results = collection.query(
        query_texts=[user_question],
        n_results=5
    )

    # 2. Если ничего не нашли
    if not results['documents'] or not results['documents'][0]:
        raise HTTPException(status_code=404, detail="В инструкциях не найдено информации по вашему вопросу.")

    # 3. Готовим контекст — берём ВСЕ найденные чанки (без группировки)
    context_parts = []
    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        context_parts.append(f"[{i+1}] Из файла: {meta['source']}\n{doc}")
    context = "\n\n---\n\n".join(context_parts)
    sources = list(set([meta['source'] for meta in results['metadatas'][0]]))
    
    # 4. Формируем промпт
    prompt = f"""Ты — помощник по документации. Отвечай ТОЛЬКО на основе контекста ниже.

ПРАВИЛА:
1. Если в контексте есть список — перечисли ВСЕ пункты списка.
2. Не сокращай и не обобщай. Приведи дословно или максимально близко к тексту.
3. Если информации нет — напиши "В инструкциях нет информации".

КОНТЕКСТ:
{context}

ВОПРОС: {user_question}

ОТВЕТ (перечисли ВСЁ из контекста):"""
    
    # 5. Задаем вопрос модели через Ollama
    response = ollama.chat(model='llama3.1:8b', messages=[{'role': 'user', 'content': prompt}])
    answer = response['message']['content']
    
    # 6. Возвращаем ответ и список источников
    return {
        "answer": answer,
        "source_documents": sources
    }

@app.get("/peek")
def peek_file(filename: str = None):
    """Показывает содержимое первых 3 чанков из указанного файла"""
    if not filename:
        return {"error": "Укажите параметр filename, например ?filename=дежурство.pdf"}
    
    # Ищем чанки из этого файла
    results = collection.get(where={"source": filename})
    
    if not results['documents']:
        return {"error": f"Файл '{filename}' не найден в базе"}
    
    preview = []
    for i, doc in enumerate(results['documents'][:3]):  # первые 3 чанка
        preview.append({
            "chunk_index": i + 1,
            "text_preview": doc[:500] + "..." if len(doc) > 500 else doc
        })
    
    return {
        "filename": filename,
        "total_chunks": len(results['documents']),
        "preview": preview
    }

# --- 4. ЗАПУСК ---
if __name__ == "__main__":
    # 1. Запускаем процесс индексации всех PDF из папки 'instructions'
    index_all_pdfs()
    # 2. Запускаем веб-сервер
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
    print("Сервер запущен! Отправляйте POST-запросы на http://127.0.0.1:8000/ask")