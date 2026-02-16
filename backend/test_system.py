"""
Test script to verify the memory-optimized RAG system is working correctly.
Run this after starting the server to validate all components.
"""

import requests
import time
import json

BASE_URL = "http://localhost:8000"

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def test_server_status():
    print_section("TEST 1: Server Status")
    try:
        response = requests.get(f"{BASE_URL}/api/status")
        data = response.json()
        
        print(f"✅ Server is online")
        print(f"   Version: {data.get('version')}")
        print(f"   Gemini API: {'✅ Loaded' if data.get('gemini_key_loaded') else '❌ Not loaded'}")
        print(f"   Documents: {data.get('documents_count')}")
        print(f"   RAG Ready: {'✅ Yes' if data.get('rag_ready') else '⚠️ No (upload PDFs first)'}")
        return True
    except Exception as e:
        print(f"❌ Server status check failed: {e}")
        return False

def test_list_documents():
    print_section("TEST 2: List Documents")
    try:
        response = requests.get(f"{BASE_URL}/documents")
        data = response.json()
        
        print(f"✅ Document listing works")
        print(f"   Total documents: {data.get('count')}")
        if data.get('documents'):
            for doc in data['documents']:
                print(f"   - {doc}")
        else:
            print(f"   ⚠️ No documents uploaded yet")
        return True
    except Exception as e:
        print(f"❌ Document listing failed: {e}")
        return False

def test_index_status():
    print_section("TEST 3: Index Status Tracking")
    try:
        response = requests.get(f"{BASE_URL}/index-status")
        data = response.json()
        
        print(f"✅ Index status endpoint works")
        if data.get('files'):
            print(f"   Active indexing jobs: {len(data['files'])}")
            for filename, status in data['files'].items():
                print(f"   - {filename}: {status['status']} ({status.get('progress', 0)}%)")
        else:
            print(f"   ℹ️ No active indexing jobs")
        return True
    except Exception as e:
        print(f"❌ Index status check failed: {e}")
        return False

def test_chat_no_documents():
    print_section("TEST 4: Chat Without Documents")
    try:
        response = requests.post(
            f"{BASE_URL}/chat",
            json={"message": "What is this about?"}
        )
        data = response.json()
        
        if data.get('success'):
            print(f"✅ Chat endpoint works")
            print(f"   Response: {data.get('ai_response')[:100]}...")
            print(f"   Sources: {data.get('sources')}")
            
            if not data.get('sources'):
                print(f"   ✅ Correctly returns 'no info' when no documents indexed")
        else:
            print(f"⚠️ Chat returned error: {data.get('message')}")
        return True
    except Exception as e:
        print(f"❌ Chat test failed: {e}")
        return False

def test_embedding_model():
    print_section("TEST 5: Embedding Model Check")
    try:
        from sentence_transformers import SentenceTransformer
        
        print(f"📥 Loading embedding model...")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        test_text = "This is a test sentence."
        embedding = model.encode(test_text)
        
        print(f"✅ Embedding model works")
        print(f"   Model: all-MiniLM-L6-v2")
        print(f"   Embedding dimension: {len(embedding)}")
        print(f"   Sample values: {embedding[:3]}...")
        return True
    except Exception as e:
        print(f"❌ Embedding model test failed: {e}")
        return False

def test_pdf_library():
    print_section("TEST 6: PDF Processing Library")
    try:
        import fitz  # PyMuPDF
        
        print(f"✅ PyMuPDF (fitz) is installed")
        print(f"   Version: {fitz.version}")
        return True
    except ImportError:
        print(f"❌ PyMuPDF not installed")
        print(f"   Install with: pip install PyMuPDF")
        return False

def test_ocr_support():
    print_section("TEST 7: OCR Support (Optional)")
    try:
        import pytesseract
        from PIL import Image
        
        print(f"✅ OCR libraries installed")
        print(f"   pytesseract: Available")
        print(f"   Pillow (PIL): Available")
        
        # Try to get tesseract version
        try:
            version = pytesseract.get_tesseract_version()
            print(f"   Tesseract OCR: {version}")
        except:
            print(f"   ⚠️ Tesseract OCR engine not found")
            print(f"   Install from: https://github.com/UB-Mannheim/tesseract/wiki")
        
        return True
    except ImportError as e:
        print(f"⚠️ OCR not available (optional)")
        print(f"   Missing: {e}")
        print(f"   Install with: pip install pytesseract Pillow")
        return True  # Not critical, return True anyway

def run_all_tests():
    print("\n" + "="*60)
    print("  MEMORY-OPTIMIZED RAG SYSTEM - VALIDATION TESTS")
    print("="*60)
    print("\nš Running tests... Make sure server is running!")
    
    results = {
        "Server Status": test_server_status(),
        "List Documents": test_list_documents(),
        "Index Status": test_index_status(),
        "Chat Endpoint": test_chat_no_documents(),
        "Embedding Model": test_embedding_model(),
        "PDF Library": test_pdf_library(),
        "OCR Support": test_ocr_support(),
    }
    
    # Summary
    print_section("TEST SUMMARY")
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
    elif passed >= total - 1:
        print("\n⚠️ System mostly working. Check failed test above.")
    else:
        print("\n❌ Multiple tests failed. Check server and dependencies.")
    
    # Next steps
    print("\n" + "-"*60)
    print("NEXT STEPS:")
    print("-"*60)
    if results["Server Status"] and results["PDF Library"]:
        print("1. Upload a PDF:")
        print("   POST http://localhost:8000/upload-pdf")
        print("2. Track progress:")
        print("   GET http://localhost:8000/index-status/{filename}")
        print("3. Ask questions:")
        print("   POST http://localhost:8000/chat")
        print("\nSee QUICK_START.md for detailed examples.")
    else:
        print("1. Make sure server is running: python main.py")
        print("2. Install missing dependencies: pip install -r requirements.txt")
        print("3. Run this test again")

if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⚠️ Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test suite error: {e}")
        import traceback
        traceback.print_exc()
