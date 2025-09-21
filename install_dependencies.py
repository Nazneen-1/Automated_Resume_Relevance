#!/usr/bin/env python3
"""
Script to install all required dependencies and setup the environment
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main installation function"""
    print("🎯 Resume Relevance Check System - Dependency Installation")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install pip packages
    print("\n📦 Installing Python packages...")
    packages = [
        "flask==2.3.3",
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "streamlit==1.28.1",
        "PyMuPDF==1.23.8",
        "pdfplumber==0.10.0",
        "python-docx==1.1.0",
        "docx2txt==0.8",
        "spacy==3.7.2",
        "nltk==3.8.1",
        "scikit-learn==1.3.2",
        "fuzzywuzzy==0.18.0",
        "python-Levenshtein==0.23.0",
        "langchain==0.0.340",
        "langchain-openai==0.0.2",
        "langchain-community==0.0.10",
        "openai==1.3.7",
        "google-generativeai==0.3.2",
        "anthropic==0.7.8",
        "chromadb==0.4.18",
        "faiss-cpu==1.7.4",
        "pinecone-client==2.2.4",
        "sqlalchemy==2.0.23",
        "pandas==2.1.4",
        "numpy==1.25.2",
        "python-multipart==0.0.6",
        "python-dotenv==1.0.0",
        "Pillow==10.1.0",
        "plotly==5.17.0",
        "flask-cors==4.0.0",
        "requests==2.31.0"
    ]
    
    # Install packages in batches to avoid memory issues
    batch_size = 5
    for i in range(0, len(packages), batch_size):
        batch = packages[i:i+batch_size]
        package_list = " ".join(batch)
        if not run_command(f"pip install {package_list}", f"Installing batch {i//batch_size + 1}"):
            print("❌ Package installation failed. Please check your internet connection and try again.")
            sys.exit(1)
    
    # Download spaCy model
    print("\n🔤 Downloading spaCy English model...")
    if not run_command("python -m spacy download en_core_web_sm", "Downloading spaCy model"):
        print("⚠️  Warning: spaCy model download failed. Some NLP features may be limited.")
    
    # Download NLTK data
    print("\n📚 Downloading NLTK data...")
    nltk_script = """
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
print('NLTK data downloaded successfully')
"""
    
    if not run_command(f'python -c "{nltk_script}"', "Downloading NLTK data"):
        print("⚠️  Warning: NLTK data download failed. Some text processing features may be limited.")
    
    # Create necessary directories
    print("\n📁 Creating directories...")
    directories = ["uploads", "logs", "data"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Create .env file if it doesn't exist
    if not Path(".env").exists():
        print("\n📝 Creating .env file...")
        if Path(".env.example").exists():
            import shutil
            shutil.copy(".env.example", ".env")
            print("✅ Created .env file from template")
            print("⚠️  Please edit .env file with your API keys before running the system")
        else:
            # Create basic .env file
            env_content = """# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Google Gemini Configuration (optional)
GOOGLE_API_KEY=your_google_api_key_here

# Anthropic Claude Configuration (optional)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Pinecone Configuration (optional)
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment_here

# Database Configuration
DATABASE_URL=sqlite:///resume_system.db

# Application Settings
FLASK_ENV=development
STREAMLIT_SERVER_PORT=8501
API_SERVER_PORT=8000
"""
            with open(".env", "w") as f:
                f.write(env_content)
            print("✅ Created basic .env file")
            print("⚠️  Please add your API keys to .env file before running the system")
    
    # Test imports
    print("\n🧪 Testing critical imports...")
    test_imports = [
        ("flask", "Flask web framework"),
        ("streamlit", "Streamlit dashboard"),
        ("spacy", "spaCy NLP library"),
        ("sklearn", "Scikit-learn ML library"),
        ("openai", "OpenAI API client"),
        ("langchain", "LangChain framework"),
        ("chromadb", "ChromaDB vector store"),
        ("sqlalchemy", "SQLAlchemy ORM")
    ]
    
    failed_imports = []
    for module, description in test_imports:
        try:
            __import__(module)
            print(f"✅ {description}")
        except ImportError as e:
            print(f"❌ {description} - {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠️  Some imports failed: {', '.join(failed_imports)}")
        print("You may need to install these manually or check for version conflicts.")
    
    print("\n🎉 Installation completed!")
    print("\nNext steps:")
    print("1. Edit .env file with your API keys")
    print("2. Run: python run_system.py")
    print("3. Access dashboard at: http://localhost:8501")
    print("4. Access API at: http://localhost:5000")

if __name__ == "__main__":
    main()