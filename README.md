# 🤖 3D Avatar Chatbot with RAG

A production-ready AI-powered chatbot with voice interaction, document Q&A capabilities, and memory-optimized RAG (Retrieval-Augmented Generation) system. Designed to handle large PDFs (200-500 pages) on consumer laptops without freezing.

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Python](https://img.shields.io/badge/python-3.11+-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688)
![License](https://img.shields.io/badge/license-MIT-orange)

---

## ✨ Features

### 🎤 Voice Interaction
- **Speech-to-Text**: Whisper model for accurate voice transcription
- **Text-to-Speech**: Natural voice responses using Edge-TTS (Microsoft Azure)
- **3D Avatar**: Ready Player Me character with lip-sync animations
- Support for real-time voice conversations

### 📚 Advanced RAG System
- **Memory-Efficient**: Handles 200-500 page PDFs on 8GB RAM laptops
- **Page-by-Page Processing**: Streaming PDF extraction prevents RAM overflow
- **Background Indexing**: Non-blocking API during document processing
- **Real-Time Progress Tracking**: Monitor PDF indexing status
- **OCR Support**: Fallback for scanned PDFs (pytesseract)

### 🛡️ Hallucination Prevention
- **Strict Relevance Filtering**: Cosine similarity threshold (0.8)
- **Context-Only Answering**: LLM restricted to provided documents
- **Mandatory Source Citations**: Every answer includes document references
- **Safe Refusal**: Returns "not found" for irrelevant queries

### 🚀 Performance Optimized
- **Lightweight Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
- **Batched Processing**: 20 chunks/batch to prevent memory spikes
- **Explicit Garbage Collection**: Memory freed after each batch
- **CPU-Only**: No GPU required

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (HTML/JS)                        │
│  Voice Input │ Text Chat │ PDF Upload │ Progress Tracking   │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Backend (Python)                   │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Whisper    │  │    Gemini    │  │   Edge-TTS   │     │
│  │ (Speech-to-  │  │    1.5 Flash │  │  (Text-to-   │     │
│  │    Text)     │  │     (LLM)    │  │   Speech)    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              RAG Pipeline                             │  │
│  │  PyMuPDF → Chunking → Embeddings → FAISS → LLM     │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│               Vector Database (FAISS)                        │
│  Cosine Similarity Search │ Incremental Indexing            │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend Framework** | FastAPI | Async web framework |
| **Speech Recognition** | Whisper (base) | Voice → Text |
| **LLM** | Google Gemini 1.5 Flash | AI responses |
| **Text-to-Speech** | Edge-TTS | Text → Voice |
| **Embeddings** | all-MiniLM-L6-v2 | Document vectorization |
| **Vector Database** | FAISS | Similarity search |
| **PDF Processing** | PyMuPDF (fitz) | Fast extraction |
| **OCR** | Pytesseract | Scanned PDF support |
| **Text Chunking** | LangChain | Recursive splitting |
| **3D Rendering** | Three.js | Avatar display |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- 8GB RAM (minimum)
- Internet connection (for Gemini API)

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/3d-avatar-chatbot.git
cd 3d-avatar-chatbot
```

### 2. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
pip install PyMuPDF  # Fast PDF processing
```

### 3. Optional: Install OCR Support
Only needed for scanned PDFs:

**Windows:**
```bash
pip install pytesseract Pillow
# Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr
pip install pytesseract Pillow
```

**macOS:**
```bash
brew install tesseract
pip install pytesseract Pillow
```

### 4. Configure Environment
Create `.env` file in `backend/` folder:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Get your API key: https://makersuite.google.com/app/apikey

### 5. Start Server
```bash
python main.py
```

Server will start at: **http://localhost:8000**

API docs: **http://localhost:8000/docs**

---

## 📖 Usage

### Upload PDFs
**PowerShell:**
```powershell
$file = "C:\path\to\document.pdf"
$form = @{ file = Get-Item -Path $file }
$response = Invoke-RestMethod -Uri "http://localhost:8000/upload-pdf" -Method Post -Form $form
Write-Host $response.message
```

**cURL:**
```bash
curl -X POST "http://localhost:8000/upload-pdf" -F "file=@document.pdf"
```

### Track Progress
```powershell
# Replace 'filename.pdf' with your file
Invoke-RestMethod -Uri "http://localhost:8000/index-status/filename.pdf"
```

### Ask Questions
**PowerShell:**
```powershell
$body = @{ message = "What is this document about?" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/chat" -Method Post -Body $body -ContentType "application/json"
```

**cURL:**
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is this document about?"}'
```

---

## 🔌 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Frontend interface |
| `GET` | `/api/status` | Server health check |
| `POST` | `/chat` | Text-based chat |
| `POST` | `/process-audio` | Voice input (WebM) |
| `GET` | `/get-audio/{filename}` | Download audio response |

### Document Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload-pdf` | Upload & index PDF (background) |
| `GET` | `/index-status/{filename}` | Check indexing progress |
| `GET` | `/index-status` | All active indexing jobs |
| `GET` | `/documents` | List uploaded PDFs |
| `DELETE` | `/documents/{filename}` | Delete PDF & rebuild index |

### Utilities

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/new-chat` | Reset conversation history |

---

## 📊 Performance Metrics

Tested on: **Intel i5, 8GB RAM, No GPU**

| Metric | Value |
|--------|-------|
| **PDF Size** | 300 pages (45 MB) |
| **Extraction Time** | ~3 minutes |
| **Embedding Time** | ~4 minutes |
| **Total Indexing** | ~7 minutes |
| **Peak RAM Usage** | 1.8 GB |
| **Query Response** | < 2 seconds |
| **API During Indexing** | ✅ Responsive |

---

## 🛡️ Anti-Hallucination Features

### 5-Layer Defense System

1. **Relevance Threshold** (Pre-LLM)
   - FAISS similarity score must be ≤ 0.8
   - Irrelevant queries blocked before LLM call

2. **Strict Prompt Engineering**
   - Explicit instructions: "Answer ONLY from context"
   - Requires "I don't know" for missing info

3. **Context Isolation**
   - LLM sees only retrieved chunks
   - No access to general knowledge

4. **Mandatory Citations**
   - Every answer includes source filenames
   - Users can verify claims

5. **Two-Stage Validation**
   - Stage 1: Relevance check
   - Stage 2: Context-only generation

### Example Behavior
```
Query: "What's the capital of France?"
Response: "I could not find this information in the provided knowledge base."
```
✅ No hallucination - Query outside knowledge base

```
Query: "What does the author say about assets?"
Response: "The author defines assets as things that put money in your pocket...

📚 Source(s): Rich Dad Poor Dad book.pdf"
```
✅ Grounded answer with citation

---

## 🧮 How It Works

### Cosine Similarity Search

The system uses **cosine similarity** via normalized embeddings:

```python
# Embeddings are normalized (vector length = 1)
encode_kwargs={'normalize_embeddings': True}

# For normalized vectors:
# cosine_similarity = 1 - (L2_distance² / 2)

# FAISS returns L2 distance:
results = faiss_index.similarity_search_with_score(query, k=3)
# Lower score = higher similarity
```

**Threshold:** 0.8 (≈ 68% cosine similarity minimum)

### RAG Pipeline Flow

```
User Query
    ↓
Query → Embedding (384-dim vector)
    ↓
FAISS Cosine Similarity Search (k=3)
    ↓
Score Check: best_score > 0.8?
    ↓
YES → Return "Not found" ❌
    ↓
NO → Context found ✅
    ↓
Build Strict Context-Only Prompt
    ↓
Gemini LLM generates answer
    ↓
Append Source Citations
    ↓
Convert to Speech (Edge-TTS)
    ↓
Return JSON + Audio URL
```

---

## 📁 Project Structure

```
3d-avatar-chatbot/
├── backend/
│   ├── main.py                    # Main FastAPI application
│   ├── requirements.txt           # Python dependencies
│   ├── .env                       # API keys (create this)
│   ├── documents/                 # Uploaded PDFs
│   ├── faiss_store/              # Vector database
│   │   ├── index.faiss           # FAISS index
│   │   └── index.pkl             # Metadata
│   └── audio_temp/               # Generated audio files
├── frontend/
│   ├── index.html                # Main UI with 3D avatar
│   └── test.html                 # Test interface
└── README.md                      # This file
```

---

## 🧪 Testing

Test the chat endpoint:

```powershell
$body = @{ message = "Hello, how are you?" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/chat" -Method Post -Body $body -ContentType "application/json"
```

Check server status:
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/status"
```

---

## 🐛 Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'fitz'`
```bash
pip install PyMuPDF
```

### Issue: Server won't start
```bash
# Kill existing processes
Get-Process python | Stop-Process -Force

# Check if port 8000 is in use
Get-NetTCPConnection -LocalPort 8000
```

### Issue: Out of memory during indexing
Reduce batch sizes in `main.py`:
```python
batch_size = 5           # Default: 10
EMBED_BATCH_SIZE = 10    # Default: 20
```

### Issue: Slow indexing
- Disable OCR if using text-based PDFs
- Increase batch sizes if you have more RAM
- Process smaller PDFs first to verify system works

### Issue: FAISS dimension mismatch
```bash
# Delete old index and restart
Remove-Item "backend\faiss_store\*" -Force
python main.py
```

### Issue: Avatar not loading
- Check console for CORS errors
- Try alternative avatar URLs in code
- Fallback placeholder will show if all fail

### Issue: Microphone access denied
- Browser must have microphone permissions
- Use HTTPS or localhost (required by browser)

---

## 🔧 Configuration

### Embedding Model
Change in `main.py`:
```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # Lightweight
    # Alternative: "BAAI/bge-m3"  # More accurate, slower
)
```

### Chunk Size
Adjust in `extract_pdf_text_streaming()`:
```python
RecursiveCharacterTextSplitter(
    chunk_size=600,      # Characters per chunk
    chunk_overlap=100,   # Overlap for continuity
)
```

### Relevance Threshold
Modify in `retrieve_context()`:
```python
relevance_threshold=0.8  # Lower = stricter (0.0-1.0)
```

### Change Voice
In `backend/main.py`, change the voice parameter:
```python
communicate = edge_tts.Communicate(ai_response, "en-US-JennyNeural")
```

Available voices: https://github.com/rany2/edge-tts#voices

---

## 📈 Roadmap

- [ ] Support additional file formats (DOCX, TXT, Markdown)
- [ ] Add user authentication
- [ ] Implement conversation history persistence
- [ ] Multi-language support
- [ ] Docker containerization
- [ ] Redis for distributed progress tracking
- [ ] Celery for advanced task queue
- [ ] Chunk caching for frequent queries

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **OpenAI Whisper** - Speech recognition
- **Google Gemini** - LLM capabilities
- **Microsoft Edge-TTS** - Natural voice synthesis
- **FAISS** - Efficient similarity search
- **Sentence Transformers** - Quality embeddings
- **PyMuPDF** - Fast PDF processing
- **LangChain** - RAG utilities
- **Three.js** - 3D rendering
- **Ready Player Me** - Avatar models

---

## 📫 Contact

Project Link: [https://github.com/yourusername/3d-avatar-chatbot](https://github.com/yourusername/3d-avatar-chatbot)

---

## ⚡ Quick Commands Reference

```bash
# Start server
python main.py

# Check server status
curl http://localhost:8000/api/status

# Upload PDF
curl -X POST "http://localhost:8000/upload-pdf" -F "file=@document.pdf"

# Ask question
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is this about?"}'
```

---

<div align="center">

**Made with ❤️ for efficient AI document Q&A**

⭐ Star this repo if you find it useful!

</div>
