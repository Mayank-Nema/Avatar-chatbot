from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import whisper
import tempfile
import shutil
import google.generativeai as genai
import edge_tts
import asyncio
import traceback
import uuid
import gc
import threading
from datetime import datetime
from typing import Optional, List, Dict

# RAG imports - Memory Efficient
import fitz  # PyMuPDF for fast PDF extraction
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️ OCR not available. Install pytesseract and Pillow for scanned PDF support.")

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# ==================== PATHS ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
FAISS_DIR = os.path.join(BASE_DIR, "faiss_store")

os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)

# Create FastAPI app
app = FastAPI(title="3D Avatar Chatbot ")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Audio temp directory — use a dedicated folder instead of system temp
AUDIO_DIR = os.path.join(BASE_DIR, "audio_temp")
os.makedirs(AUDIO_DIR, exist_ok=True)

# ==================== PYDANTIC MODELS ====================
class ChatMessage(BaseModel):
    message: str

# ==================== LOAD WHISPER MODEL ====================
print("🔄 Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("✅ Whisper model loaded!")

# ==================== CONFIGURE GEMINI API ====================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # Use valid Gemini model name (try gemini-1.5-flash, gemini-1.5-pro, or gemini-pro)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    chat_session = gemini_model.start_chat(history=[])
    print("✅ Gemini API configured!")
else:
    print("⚠️ Warning: GEMINI_API_KEY not found in .env file")

# ==================== RAG PIPELINE - MEMORY OPTIMIZED ====================
print("🔄 Initializing lightweight embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # Lightweight & fast
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print("✅ Embedding model ready!")

faiss_index = None

# Progress tracking for background indexing
indexing_progress: Dict[str, Dict] = {}
indexing_lock = threading.Lock()

def load_faiss_index():
    """Load existing FAISS index from disk if available."""
    global faiss_index
    index_path = os.path.join(FAISS_DIR, "index.faiss")
    if os.path.exists(index_path):
        faiss_index = FAISS.load_local(
            folder_path=FAISS_DIR, 
            embeddings=embeddings, 
            allow_dangerous_deserialization=True
        )
        print(f"✅ FAISS index loaded from disk!")
    else:
        print("ℹ️ No existing FAISS index found.")

def extract_pdf_text_streaming(pdf_path: str, filename: str) -> list:
    """
    Memory-efficient PDF text extraction using PyMuPDF.
    Processes page-by-page with OCR fallback for scanned PDFs.
    
    Args:
        pdf_path: Path to PDF file
        filename: Name of file for progress tracking
        
    Returns:
        List of text chunks
    """
    chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        print(f"📄 Processing {total_pages} pages from {filename}")
        
        # Process in small batches to manage memory
        batch_text = ""
        batch_size = 10  # Pages per batch
        
        for page_num in range(total_pages):
            # Update progress
            progress_pct = int((page_num / total_pages) * 100)
            with indexing_lock:
                if filename in indexing_progress:
                    indexing_progress[filename]['progress'] = progress_pct
                    indexing_progress[filename]['current_page'] = page_num + 1
            
            page = doc[page_num]
            page_text = page.get_text()
            
            # OCR fallback if no text found
            if not page_text.strip() and OCR_AVAILABLE:
                try:
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    page_text = pytesseract.image_to_string(img)
                    print(f"  📷 OCR applied to page {page_num + 1}")
                except Exception as ocr_err:
                    print(f"  ⚠️ OCR failed on page {page_num + 1}: {ocr_err}")
            
            if page_text.strip():
                batch_text += f"\n[Page {page_num + 1}]\n{page_text}"
            
            # Process batch when size reached
            if (page_num + 1) % batch_size == 0 or page_num == total_pages - 1:
                if batch_text.strip():
                    batch_chunks = splitter.split_text(batch_text)
                    chunks.extend(batch_chunks)
                    print(f"  ✅ Processed pages {max(0, page_num - batch_size + 1)}-{page_num + 1}: {len(batch_chunks)} chunks")
                    batch_text = ""  # Clear batch
                    gc.collect()  # Force garbage collection
        
        doc.close()
        print(f"✅ Extracted {len(chunks)} chunks from {filename}")
        return chunks
        
    except Exception as e:
        print(f"❌ Error extracting PDF {filename}: {e}")
        traceback.print_exc()
        return []

def add_pdf_to_index_batched(pdf_path: str, filename: str) -> int:
    """
    Add a PDF's content to FAISS index using batched processing.
    Memory-safe for large PDFs.
    
    Args:
        pdf_path: Path to PDF file
        filename: Name of file for progress tracking
        
    Returns:
        Number of chunks indexed
    """
    global faiss_index
    
    # Initialize progress tracking
    with indexing_lock:
        indexing_progress[filename] = {
            'status': 'extracting',
            'progress': 0,
            'total_chunks': 0,
            'current_page': 0,
            'error': None
        }
    
    try:
        # Extract text chunks (memory-efficient streaming)
        chunks = extract_pdf_text_streaming(pdf_path, filename)
        
        if not chunks:
            with indexing_lock:
                indexing_progress[filename]['status'] = 'failed'
                indexing_progress[filename]['error'] = 'No text extracted'
            return 0
        
        with indexing_lock:
            indexing_progress[filename]['total_chunks'] = len(chunks)
            indexing_progress[filename]['status'] = 'embedding'
            indexing_progress[filename]['progress'] = 0
        
        # Process embeddings in batches to avoid memory spikes
        EMBED_BATCH_SIZE = 20
        total_batches = (len(chunks) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * EMBED_BATCH_SIZE
            end_idx = min(start_idx + EMBED_BATCH_SIZE, len(chunks))
            batch_chunks = chunks[start_idx:end_idx]
            batch_metadatas = [
                {"source": filename, "chunk": start_idx + i} 
                for i in range(len(batch_chunks))
            ]
            
            # Add to FAISS
            if faiss_index is None:
                faiss_index = FAISS.from_texts(
                    batch_chunks, 
                    embeddings, 
                    metadatas=batch_metadatas
                )
            else:
                faiss_index.add_texts(batch_chunks, metadatas=batch_metadatas)
            
            # Update progress
            progress_pct = int(((batch_idx + 1) / total_batches) * 100)
            with indexing_lock:
                indexing_progress[filename]['progress'] = progress_pct
            
            print(f"  🔹 Embedded batch {batch_idx + 1}/{total_batches} ({len(batch_chunks)} chunks)")
            
            # Clear batch and collect garbage
            del batch_chunks, batch_metadatas
            gc.collect()
        
        # Save index to disk
        faiss_index.save_local(FAISS_DIR)
        
        with indexing_lock:
            indexing_progress[filename]['status'] = 'completed'
            indexing_progress[filename]['progress'] = 100
        
        print(f"✅ Successfully indexed {len(chunks)} chunks from {filename}")
        return len(chunks)
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error indexing {filename}: {error_msg}")
        traceback.print_exc()
        
        with indexing_lock:
            indexing_progress[filename]['status'] = 'failed'
            indexing_progress[filename]['error'] = error_msg
        
        return 0

def rebuild_index():
    """Rebuild the entire FAISS index from all PDFs in the documents folder."""
    global faiss_index
    faiss_index = None

    pdf_files = [f for f in os.listdir(DOCUMENTS_DIR) if f.lower().endswith(".pdf")]
    total_chunks = 0
    for pdf_file in pdf_files:
        pdf_path = os.path.join(DOCUMENTS_DIR, pdf_file)
        count = add_pdf_to_index_batched(pdf_path, pdf_file)
        total_chunks += count
        print(f"  📄 Indexed {pdf_file}: {count} chunks")

    if total_chunks > 0:
        faiss_index.save_local(FAISS_DIR)
    else:
        # Remove old index files if no documents
        for f in os.listdir(FAISS_DIR):
            os.remove(os.path.join(FAISS_DIR, f))

    return total_chunks

def retrieve_context(query: str, k: int = 3, relevance_threshold: float = 0.8) -> tuple:
    """
    Retrieve relevant document chunks for a query with strict relevance filtering.
    
    Args:
        query: User query text
        k: Number of top results to retrieve
        relevance_threshold: Maximum distance score (lower = more similar)
        
    Returns:
        tuple: (context_text, sources_list, is_relevant_bool)
    """
    if faiss_index is None:
        return "", [], False

    results = faiss_index.similarity_search_with_score(query, k=k)

    if not results:
        return "", [], False

    # Check if best result passes threshold
    best_score = results[0][1]
    if best_score > relevance_threshold:
        # No relevant context found
        return "", [], False

    # Collect relevant chunks and sources
    context_parts = []
    sources = []
    for doc, score in results:
        if score <= relevance_threshold:
            context_parts.append(doc.page_content)
            source = doc.metadata.get("source", "Unknown")
            if source not in sources:
                sources.append(source)

    if not context_parts:
        return "", [], False

    context = "\n\n---\n\n".join(context_parts)
    return context, sources, True

# Load existing index on startup
load_faiss_index()

# Note: Auto-indexing disabled to prevent startup freezing with large PDFs
# Use the /upload-pdf endpoint or manually trigger indexing
print("ℹ️ Auto-indexing disabled. Upload PDFs via API to index them.")

# ==================== SHARED PROCESSING ====================
async def process_message(user_text: str) -> dict:
    """
    Process user message with strict RAG pipeline:
    Retrieval → Validation → Context-only Generation → Citation
    """
    # Step 1: Retrieve context with relevance validation
    context, sources, is_relevant = retrieve_context(user_text, k=3, relevance_threshold=0.8)

    # Step 2: Handle irrelevant queries
    if not is_relevant:
        safe_response = "I could not find this information in the provided knowledge base."
        print(f"⚠️ No relevant context found for query")
        
        # Generate audio for safe response
        audio_filename = f"response_{uuid.uuid4().hex[:12]}.mp3"
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        try:
            communicate = edge_tts.Communicate(safe_response, "en-US-AriaNeural")
            await communicate.save(audio_path)
            print(f"✅ Audio saved: {audio_path}")
        except Exception as tts_err:
            print(f"❌ TTS error: {tts_err}")
            audio_filename = None

        return {
            "success": True,
            "user_message": user_text,
            "ai_response": safe_response,
            "audio_url": f"/get-audio/{audio_filename}" if audio_filename else None,
            "sources": []
        }

    # Step 3: Build strict context-only prompt
    strict_prompt = f"""You are a helpful assistant that answers questions STRICTLY based on the provided document context.

**CRITICAL INSTRUCTIONS:**
1. You MUST answer ONLY using information explicitly stated in the context below.
2. If the context does not contain the answer, you MUST respond: "I don't have enough information in the provided documents to answer that question."
3. DO NOT use any external knowledge or general information.
4. DO NOT make assumptions or inferences beyond what is directly stated.
5. Always cite which document(s) you used by mentioning the source name.

**DOCUMENT CONTEXT:**
{context}

**USER QUESTION:**
{user_text}

**YOUR ANSWER (with source citations):**"""

    # Step 4: Get LLM response
    print(f"📚 RAG context from: {', '.join(sources)}")
    print(f"🤖 Querying Gemini with strict context-only prompt...")
    
    try:
        response = chat_session.send_message(strict_prompt)
        ai_response = response.text.strip()
    except Exception as e:
        print(f"❌ LLM error: {e}")
        ai_response = "I encountered an error processing your request."
        sources = []

    # Step 5: Add citation footer if not already present
    if sources and not any(source in ai_response for source in sources):
        citation_text = f"\n\n📚 Source(s): {', '.join(sources)}"
        ai_response += citation_text
    elif sources:
        # Sources mentioned in response, just add footer
        citation_text = f"\n\n📚 Source(s): {', '.join(sources)}"
        if citation_text not in ai_response:
            ai_response += citation_text

    print(f"💬 AI response with citations: {ai_response[:150]}...")

    # Step 6: Generate audio
    audio_filename = f"response_{uuid.uuid4().hex[:12]}.mp3"
    audio_path = os.path.join(AUDIO_DIR, audio_filename)
    print(f"🔊 Converting text to speech → {audio_filename}")
    try:
        communicate = edge_tts.Communicate(ai_response, "en-US-AriaNeural")
        await communicate.save(audio_path)
        print(f"✅ Audio saved: {audio_path} ({os.path.getsize(audio_path)} bytes)")
    except Exception as tts_err:
        print(f"❌ TTS error: {tts_err}")
        traceback.print_exc()
        audio_filename = None

    # Step 7: Return structured response
    return {
        "success": True,
        "user_message": user_text,
        "ai_response": ai_response,
        "audio_url": f"/get-audio/{audio_filename}" if audio_filename else None,
        "sources": sources
    }

# ==================== ROUTES ====================

@app.get("/")
async def root():
    """Serve the main frontend HTML page."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {"error": "Frontend not found", "path": index_path}

@app.get("/test.html")
async def test_page():
    """Serve the test HTML page."""
    test_path = os.path.join(FRONTEND_DIR, "test.html")
    if os.path.exists(test_path):
        return FileResponse(test_path, media_type="text/html")
    return {"error": "Test page not found"}

@app.get("/api/status")
async def api_status():
    """API status endpoint (for testing backend connectivity)."""
    return {
        "status": "online",
        "message": "3D Avatar Chatbot API is running!",
        "version": "2.0.0",
        "gemini_key_loaded": bool(os.getenv("GEMINI_API_KEY")),
        "documents_count": len([f for f in os.listdir(DOCUMENTS_DIR) if f.lower().endswith(".pdf")]),
        "rag_ready": faiss_index is not None
    }

# ==================== AUDIO PROCESSING ====================

@app.post("/process-audio")
async def process_audio(audio: UploadFile = File(...)):
    """Voice input: transcribe with Whisper then process through shared pipeline."""
    temp_audio_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as f:
            shutil.copyfileobj(audio.file, f)
            temp_audio_path = f.name

        print(f"📥 Received audio: {audio.filename}")
        print(f"🧠 Transcribing with Whisper...")

        result = whisper_model.transcribe(temp_audio_path, fp16=False)
        transcribed_text = result["text"].strip()
        print(f"🗣️ User said: {transcribed_text}")

        os.unlink(temp_audio_path)
        temp_audio_path = None

        if not transcribed_text:
            return {"success": False, "message": "No speech detected in audio"}

        response = await process_message(transcribed_text)
        response["transcribed_text"] = transcribed_text
        return response

    except Exception as e:
        print(f"❌ Error processing audio: {e}")
        traceback.print_exc()
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        return {"success": False, "message": f"Error: {str(e)}"}

# ==================== TEXT CHAT ====================

@app.post("/chat")
async def chat(msg: ChatMessage):
    """Text input: process through shared pipeline (skip Whisper)."""
    try:
        if not msg.message.strip():
            return {"success": False, "message": "Empty message"}

        print(f"💬 Text message: {msg.message}")
        response = await process_message(msg.message.strip())
        return response

    except Exception as e:
        print(f"❌ Error processing chat: {e}")
        traceback.print_exc()
        return {"success": False, "message": f"Error: {str(e)}"}

# ==================== AUDIO SERVING ====================

@app.get("/get-audio/{filename}")
async def get_audio(filename: str):
    try:
        file_path = os.path.join(AUDIO_DIR, filename)
        print(f"🔊 Audio requested: {filename}, exists={os.path.exists(file_path)}")

        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✅ Serving audio: {file_path} ({file_size} bytes)")
            return FileResponse(
                file_path,
                media_type="audio/mpeg",
                filename="response.mp3",
                headers={
                    "Accept-Ranges": "bytes",
                    "Cache-Control": "no-cache",
                    "Access-Control-Allow-Origin": "*",
                }
            )
        else:
            print(f"❌ Audio file NOT FOUND: {file_path}")
            return {"success": False, "message": "Audio file not found"}
    except Exception as e:
        print(f"❌ get-audio error: {e}")
        traceback.print_exc()
        return {"success": False, "message": str(e)}

# ==================== PDF UPLOAD & MANAGEMENT ====================

def index_pdf_background(pdf_path: str, filename: str):
    """Background task to index a PDF without blocking the API."""
    try:
        add_pdf_to_index_batched(pdf_path, filename)
    except Exception as e:
        print(f"❌ Background indexing error for {filename}: {e}")
        with indexing_lock:
            if filename in indexing_progress:
                indexing_progress[filename]['status'] = 'failed'
                indexing_progress[filename]['error'] = str(e)

@app.post("/upload-pdf")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload a PDF and index it in the background.
    Returns immediately with indexing status endpoint.
    """
    try:
        if not file.filename.lower().endswith(".pdf"):
            return {"success": False, "message": "Only PDF files are accepted"}

        pdf_path = os.path.join(DOCUMENTS_DIR, file.filename)
        
        # Save file
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        print(f"📄 Uploaded PDF: {file.filename}")
        
        # Initialize progress tracking
        with indexing_lock:
            indexing_progress[file.filename] = {
                'status': 'queued',
                'progress': 0,
                'total_chunks': 0,
                'current_page': 0,
                'error': None
            }
        
        # Start background indexing
        background_tasks.add_task(index_pdf_background, pdf_path, file.filename)

        return {
            "success": True,
            "message": f"PDF '{file.filename}' uploaded. Indexing started in background.",
            "filename": file.filename,
            "status_endpoint": f"/index-status/{file.filename}"
        }

    except Exception as e:
        print(f"❌ Error uploading PDF: {e}")
        traceback.print_exc()
        return {"success": False, "message": f"Error: {str(e)}"}

@app.get("/index-status/{filename}")
async def get_index_status(filename: str):
    """
    Get the current indexing status and progress for a PDF.
    
    Returns:
        - status: 'queued', 'extracting', 'embedding', 'completed', 'failed'
        - progress: 0-100 percentage
        - current_page: Current page being processed
        - total_chunks: Total chunks created (if completed)
        - error: Error message if failed
    """
    with indexing_lock:
        if filename in indexing_progress:
            return {
                "success": True,
                "filename": filename,
                **indexing_progress[filename]
            }
        else:
            # Check if file exists but not in progress (already indexed)
            pdf_path = os.path.join(DOCUMENTS_DIR, filename)
            if os.path.exists(pdf_path):
                return {
                    "success": True,
                    "filename": filename,
                    "status": "completed",
                    "progress": 100,
                    "message": "File already indexed"
                }
            else:
                return {
                    "success": False,
                    "message": "File not found"
                }

@app.get("/index-status")
async def get_all_index_status():
    """Get indexing status for all files currently being processed."""
    with indexing_lock:
        return {
            "success": True,
            "files": dict(indexing_progress)
        }

@app.get("/documents")
async def list_documents():
    """List all uploaded PDF documents."""
    files = [f for f in os.listdir(DOCUMENTS_DIR) if f.lower().endswith(".pdf")]
    return {"success": True, "documents": files, "count": len(files)}

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Delete a PDF and rebuild the index."""
    try:
        pdf_path = os.path.join(DOCUMENTS_DIR, filename)
        if not os.path.exists(pdf_path):
            return {"success": False, "message": "Document not found"}

        os.remove(pdf_path)
        total = rebuild_index()
        print(f"🗑️ Deleted {filename}, rebuilt index ({total} chunks remaining)")

        return {"success": True, "message": f"Deleted '{filename}'", "remaining_chunks": total}

    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}

# ==================== NEW CHAT SESSION ====================

@app.post("/new-chat")
async def new_chat():
    """Reset the Gemini chat session for a new conversation thread."""
    global chat_session
    try:
        chat_session = gemini_model.start_chat(history=[])
        print("🔄 New chat session started!")
        return {"success": True, "message": "New chat session started"}
    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}

# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FastAPI server...")
    print("📡 API will be available at: http://localhost:8000")
    print("📚 API docs at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
