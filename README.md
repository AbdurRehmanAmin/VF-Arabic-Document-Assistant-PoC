# المساعد السعودي للمستندات 🇸🇦
*Arabic Document Assistant*

A sophisticated AI-powered document analysis system that processes Arabic and English documents using advanced retrieval-augmented generation (RAG) technology. Built specifically for Arabic speakers with full RTL support and cultural context.

## 🌟 Features

- **📄 Multi-format Support**: PDF, DOCX, and TXT files
- **🔍 Advanced Document Analysis**: Semantic chunking and intelligent text extraction
- **🌐 Bilingual Support**: Native Arabic and English processing
- **🔄 Real-time Translation**: Instant translation between Arabic and English
- **📚 Multi-document Processing**: Handle up to 5 documents per conversation
- **🎯 Smart Question Generation**: AI-generated starter questions based on document content
- **📖 Accurate Citations**: Precise source references with page and line numbers
- **🚀 Modern UI**: Built with Chainlit for responsive chat interface

## 🏗️ Architecture

![Architecture Diagram]("System Architecture.png")

The system follows a modular RAG (Retrieval-Augmented Generation) architecture:

### Core Components

1. **Document Processing Pipeline**
   - Text extraction from multiple formats
   - Arabic text normalization and RTL handling
   - Semantic chunking with metadata preservation

2. **Vector Store System**
   - FAISS-based similarity search
   - Multilingual sentence transformers
   - Efficient embedding storage and retrieval

3. **LLM Integration**
   - Ollama-powered local inference
   - Specialized Arabic expert system prompts
   - Context-aware response generation

4. **Chat Interface**
   - Streamlit-based responsive UI
   - Real-time file processing
   - Interactive question suggestions

## 🛠️ Technology Stack

- **Backend**: Python 3.8+
- **UI Framework**: Chainlit
- **Vector Database**: FAISS
- **Embeddings**: Sentence Transformers (paraphrase-multilingual-mpnet-base-v2)
- **LLM**: Ollama (Llama 3)
- **Document Processing**: PyPDF2, python-docx
- **Text Processing**: LangChain, langdetect

## 📋 Prerequisites

- Python 3.8 or higher
- Ollama installed and running locally
- At least 4GB RAM for embedding models
- CUDA support (optional, for GPU acceleration)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/arabic-document-assistant.git
   cd arabic-document-assistant
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install and setup Ollama**
   ```bash
   # Install Ollama (visit https://ollama.ai for installation instructions)
   
   # Pull the required model
   ollama pull llama3
   ```

5. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env file with your configurations
   ```

## 🏃‍♂️ Running the Application

1. **Start Ollama server** (if not already running)
   ```bash
   ollama serve
   ```

2. **Launch the application**
   ```bash
   chainlit run app.py -w
   ```

3. **Access the interface**
   - Open your browser to `http://localhost:8000`
   - Upload your documents and start asking questions!

## 📁 Project Structure

```
arabic-document-assistant/
├── app.py                 # Main Chainlit application
├── document_processor.py  # Document parsing and text extraction
├── vector_store.py       # FAISS vector database management
├── ollama_service.py     # LLM integration and prompt management
├── chainlit.md          # Application description
├── requirements.txt     # Python dependencies
├── .env                # Environment configuration
├── .gitignore          # Git ignore patterns
├── uploads/            # Temporary file storage
└── README.md           # This file
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Ollama Configuration
OLLAMA_API_URL=http://localhost:11434/api
OLLAMA_MODEL=llama3

# Application Settings
UPLOAD_DIR=uploads
MAX_FILES_PER_CHAT=5
MAX_FILE_SIZE_MB=20

# Logging
LOG_LEVEL=INFO
```

### Model Configuration

The application uses the following models by default:
- **Embeddings**: `paraphrase-multilingual-mpnet-base-v2`
- **LLM**: `llama3` (via Ollama)

You can modify these in the respective service files.

## 📖 Usage

1. **Upload Documents**: Support for PDF, DOCX, and TXT files
2. **Wait for Processing**: The system will extract text, create embeddings, and generate starter questions
3. **Ask Questions**: Use the suggested questions or ask your own
4. **Get Answers**: Receive detailed responses with source citations
5. **Translate**: Use the translation feature for bilingual support

### Supported File Formats

- **PDF**: Text extraction with Arabic RTL support
- **DOCX**: Full document parsing including tables
- **TXT**: Multiple encoding support (UTF-8, UTF-16, CP1256)

## 🌍 Language Support

- **Arabic**: Full RTL support, text normalization, cultural context
- **English**: Standard processing with multilingual embeddings
- **Translation**: Bidirectional Arabic ↔ English translation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ollama** for local LLM inference
- **Chainlit** for the chat interface framework
- **FAISS** for efficient vector similarity search
- **Sentence Transformers** for multilingual embeddings
- **LangChain** for document processing utilities

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Contact the development team
- Check the documentation in `/docs`

## 🚦 Roadmap

- [ ] Support for more document formats (EPUB, RTF)
- [ ] Advanced analytics and document insights
- [ ] Multi-user support with authentication
- [ ] Cloud deployment options
- [ ] Mobile application
- [ ] Voice interaction support

---

**Made with ❤️ for the Arabic-speaking community**
