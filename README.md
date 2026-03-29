# SaaS Support RAG System

A production-ready Retrieval-Augmented Generation (RAG) system built in Python for a SaaS technical support use case. This system ingests your internal Markdown/PDF documentation and accurately answers user queries while strictly preventing hallucinations, with a modern Web UI.

## 🚀 Architecture

*   **API Framework**: [FastAPI](https://fastapi.tiangolo.com/) - High performance, async, built-in validation.
*   **Orchestration**: [LangChain](https://python.langchain.com/) - Chains the LLM, prompts, and vector database.
*   **Vector Database**: [FAISS](https://faiss.ai/) - Local, blazing-fast vector similarity search capable of running on CPU.
*   **LLM & Embeddings**: OpenAI's `gpt-3.5-turbo` and `text-embedding-3-small`.
*   **Frontend**: Vanilla HTML/JS/CSS with a modern dynamic UI (Glassmorphism).

## 📁 Project Structure

```text
rag-saas-support/
├── api/                # FastAPI application and routes
├── data/
│   ├── raw/            # Pon tus archivos Markdown (.md) o .pdf aquí
│   └── vectorstore/    # Base de datos vectorial FAISS generada
├── public/             # Interfaz web del usuario (HTML/CSS/JS)
├── scripts/
│   └── ingest.py       # Script para procesar tus documentos
├── src/                # Core Business Logic (Ingestión, RAG, Retreival)
├── tests/              # Pruebas automatizadas (Pytest)
├── requirements.txt    # Dependencias de Python
└── .env                # Variables de entorno (API Keys)
```

---

## 🛠️ Guía Paso a Paso para Ejecutar el Proyecto

Sigue estos pasos en tu terminal (diseñado para Windows PowerShell):

### Paso 1: Entorno Virtual e Instalación
Inicia un entorno aislado para no ensuciar tu sistema e instala las dependencias:
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Paso 2: Configurar las Variables de Entorno
Copia el archivo de ejemplo y añade tu **OpenAI API Key** real:
```powershell
cp .env.example .env
```
*(Abre el archivo `.env` resultante y pega tu clave donde dice `OPENAI_API_KEY=...`. ¡Asegúrate de que tu cuenta tenga saldo!)*

### Paso 3: Ingestión de Conocimiento (RAG)
Coloca documentos `.md` o `.pdf` con tu conocimiento técnico dentro de la carpeta `data/raw/` (ya dejé un `sample_docs.md` de prueba). Luego, ejecuta el motor para crear la memoria vectorial:
```powershell
.\.venv\Scripts\python.exe scripts/ingest.py
```
*(Esto consumirá tu API de OpenAI para crear los Embeddings matemáticos).*

### Paso 4: Arrancar el Servidor Backend (FastAPI)
Una vez guardado el conocimiento, levanta el "cerebro" (la API):
```powershell
.\.venv\Scripts\uvicorn.exe api.main:app --reload
```
*(No cierres esta terminal. Tu API vivirá en `http://localhost:8000`).*

### Paso 5: Arrancar la Interfaz Web (Frontend)
Abre **una nueva pestaña** de terminal, asegúrate de estar en la carpeta del proyecto, y lanza el servidor visual:
```powershell
python -m http.server 8080 -d public/
```

🚀 **¡Todo listo!** Abre tu navegador en [http://localhost:8080](http://localhost:8080) y disfruta de tu propio Asistente basado en IA.

---

## 🐳 Alternativa: Docker
Si prefieres usar contenedores, puedes empaquetar la aplicación:
```bash
docker-compose up --build
```

## 🧪 Pruebas
Si quieres verificar el código internamente:
```powershell
.\.venv\Scripts\pytest.exe tests/ -v
```
