# 🎯 Automated Resume Relevance Check System

A comprehensive system that automates resume evaluation against job descriptions, providing relevance scores, suitability verdicts, and AI-powered improvement suggestions.

## 🌟 Features

- **Automated Resume Parsing**: Extract structured information from PDF/DOCX resumes
- **Job Description Analysis**: Parse and extract requirements from job postings
- **Multi-layered Relevance Analysis**:
  - Hard matching using TF-IDF, BM25, and fuzzy matching
  - Semantic matching using embeddings and cosine similarity
- **AI-Powered Insights**: LLM-generated improvement suggestions
- **Comprehensive Scoring**: 0-100 relevance score with detailed breakdowns
- **Suitability Verdicts**: High/Medium/Low classifications
- **Missing Elements Detection**: Identify gaps in skills, certifications, and projects
- **Web Dashboard**: Streamlit-based interface for recruiters
- **REST API**: Flask-based backend for integration
- **Vector Storage**: Support for Chroma, FAISS, and Pinecone

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   Flask API     │    │   Database      │
│   Dashboard     │◄──►│   Backend       │◄──►│   (SQLite)      │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Analysis Engine │
                    │                 │
                    │ • Document Parser│
                    │ • Relevance Engine│
                    │ • LLM Integration│
                    │ • Vector Store   │
                    └─────────────────┘
```

## 🔧 Technologies Used

- **Programming Language**: Python 3.8+
- **Resume Parsing**: PyMuPDF, pdfplumber, python-docx, docx2txt
- **NLP/Text Processing**: spaCy, NLTK
- **AI Orchestration**: LangChain, LangGraph, LangSmith
- **Vector Stores**: Chroma, FAISS, Pinecone
- **LLMs**: OpenAI GPT, Google Gemini, Anthropic Claude
- **Keyword Matching**: TF-IDF, BM25, fuzzy matching
- **Semantic Matching**: Embeddings + cosine similarity
- **Backend**: Flask with SQLAlchemy
- **Frontend**: Streamlit
- **Database**: SQLite (default) / PostgreSQL

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd resume-relevance-system

# Setup environment and install dependencies
python run_system.py setup
```

### 2. Configure API Keys

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` file:
```env
# Required for LLM features
OPENAI_API_KEY=your_openai_api_key_here

# Optional - for additional LLM providers
GOOGLE_API_KEY=your_google_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional - for Pinecone vector store
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment_here
```

### 3. Run the System

```bash
# Run both API and dashboard
python run_system.py

# Or run components separately
python run_system.py api        # API only (port 5000)
python run_system.py dashboard  # Dashboard only (port 8501)
```

### 4. Access the System

- **Streamlit Dashboard**: http://localhost:8501
- **Flask API**: http://localhost:5000
- **API Documentation**: http://localhost:5000/api/health

## 📊 Usage Workflow

### For Recruiters (via Streamlit Dashboard):

1. **Upload Job Description**
   - Navigate to "Upload Job Description"
   - Upload PDF/DOCX/TXT file
   - Review parsed requirements

2. **Upload and Evaluate Resumes**
   - Navigate to "Upload Resume"
   - Enter candidate details
   - Upload resume file
   - Get instant evaluation results

3. **View Dashboard Analytics**
   - Navigate to "Dashboard Overview"
   - View evaluation statistics
   - Filter and search results
   - Download reports

4. **Detailed Analysis**
   - Navigate to "Detailed Analysis"
   - Enter evaluation ID
   - View comprehensive breakdown

### For Developers (via API):

```python
import requests

# Upload job description
with open('job_description.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/upload/job-description',
        files={'file': f}
    )
job_id = response.json()['job_id']

# Upload resume
with open('resume.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/upload/resume',
        files={'file': f},
        data={'candidate_name': 'John Doe'}
    )
resume_id = response.json()['resume_id']

# Evaluate resume
response = requests.post(
    'http://localhost:5000/api/evaluate',
    json={'resume_id': resume_id, 'job_id': job_id}
)
evaluation = response.json()
```

## 🔍 Analysis Engine Details

### Hard Matching (40% weight)
- **TF-IDF Similarity**: Document-level similarity using term frequency
- **Fuzzy Matching**: Skills matching with 70% similarity threshold
- **Keyword Frequency**: Overlap analysis of important terms
- **BM25 Scoring**: Ranking function for keyword relevance

### Semantic Matching (60% weight)
- **Document Embeddings**: Using OpenAI text-embedding-ada-002
- **Cosine Similarity**: Vector space similarity calculation
- **Section-wise Analysis**: Skills, experience, education matching
- **Vector Storage**: Efficient similarity search using Chroma/FAISS

### Scoring Algorithm
```python
relevance_score = (
    hard_match_score * 0.4 + 
    semantic_match_score * 0.6
) * 100

suitability_verdict = {
    score >= 75: "High",
    score >= 50: "Medium",
    score < 50: "Low"
}
```

## 📁 Project Structure

```
resume-relevance-system/
├── config/
│   └── settings.py              # Configuration management
├── src/
│   ├── models/
│   │   └── database.py          # Database models and management
│   ├── parsers/
│   │   └── document_parser.py   # PDF/DOCX parsing logic
│   ├── analysis/
│   │   ├── relevance_engine.py  # Core analysis engine
│   │   └── llm_integration.py   # LLM and vector store integration
│   ├── api/
│   │   └── flask_app.py         # Flask REST API
│   └── dashboard/
│       └── streamlit_app.py     # Streamlit web interface
├── uploads/                     # File upload directory
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── run_system.py               # Main runner script
├── .env.example                # Environment variables template
└── README.md                   # This file
```

## 🔧 API Endpoints

### Core Endpoints
- `GET /api/health` - Health check
- `POST /api/upload/job-description` - Upload job description
- `POST /api/upload/resume` - Upload resume
- `POST /api/evaluate` - Evaluate resume against job
- `GET /api/evaluations` - Get all evaluations (with filters)
- `GET /api/evaluation/{id}` - Get detailed evaluation
- `GET /api/jobs` - Get all job descriptions
- `GET /api/resumes` - Get all resumes

### Query Parameters
- `job_id` - Filter by job ID
- `min_score` / `max_score` - Score range filtering
- `verdict` - Filter by suitability verdict
- `limit` / `offset` - Pagination

## 🎛️ Configuration Options

### Vector Store Selection
```python
# In config/settings.py
VECTOR_STORE_TYPE = 'chroma'  # Options: 'chroma', 'faiss', 'pinecone'
```

### Scoring Weights
```python
HARD_MATCH_WEIGHT = 0.4      # Keyword-based matching weight
SEMANTIC_MATCH_WEIGHT = 0.6  # Semantic similarity weight
```

### Suitability Thresholds
```python
HIGH_SUITABILITY_THRESHOLD = 75    # High suitability cutoff
MEDIUM_SUITABILITY_THRESHOLD = 50  # Medium suitability cutoff
```

## 🔍 Troubleshooting

### Common Issues

1. **spaCy Model Not Found**
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **NLTK Data Missing**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

3. **API Connection Error**
   - Ensure Flask API is running on port 5000
   - Check firewall settings
   - Verify API health: `curl http://localhost:5000/api/health`

4. **File Upload Issues**
   - Check file size (max 16MB)
   - Verify file format (PDF, DOCX, DOC, TXT)
   - Ensure uploads directory exists and is writable

### Performance Optimization

1. **Large File Processing**
   - Increase `MAX_CONTENT_LENGTH` in config
   - Use streaming for large files
   - Implement background processing

2. **Vector Store Performance**
   - Use FAISS for large-scale similarity search
   - Consider Pinecone for production deployments
   - Implement caching for frequent queries

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the API documentation

## 🚀 Future Enhancements

- [ ] Multi-language support
- [ ] Advanced ML models for scoring
- [ ] Integration with ATS systems
- [ ] Batch processing capabilities
- [ ] Advanced analytics and reporting
- [ ] Mobile-responsive dashboard
- [ ] Real-time collaboration features