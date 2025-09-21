#!/usr/bin/env python3
"""
Main script to run the Resume Relevance Check System
"""

import os
import sys
import subprocess
import threading
import time
import signal
from pathlib import Path

def run_flask_api():
    """Run the Flask API server"""
    print("🚀 Starting Flask API server...")
    try:
        subprocess.run([
            sys.executable, "-m", "src.api.flask_app"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Flask API server stopped")
    except Exception as e:
        print(f"❌ Error running Flask API: {e}")

def run_streamlit_dashboard():
    """Run the Streamlit dashboard"""
    print("🚀 Starting Streamlit dashboard...")
    try:
        subprocess.run([
            "streamlit", "run", "src/dashboard/streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Streamlit dashboard stopped")
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")

def setup_environment():
    """Setup the environment and install dependencies"""
    print("🔧 Setting up environment...")
    
    # Check if virtual environment exists
    venv_path = Path("venv")
    if not venv_path.exists():
        print("📦 Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
    
    # Determine the correct pip path
    if os.name == 'nt':  # Windows
        pip_path = venv_path / "Scripts" / "pip"
        python_path = venv_path / "Scripts" / "python"
    else:  # Unix/Linux/MacOS
        pip_path = venv_path / "bin" / "pip"
        python_path = venv_path / "bin" / "python"
    
    # Install dependencies
    print("📦 Installing dependencies...")
    subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)
    
    # Download spaCy model
    print("📦 Downloading spaCy English model...")
    try:
        subprocess.run([str(python_path), "-m", "spacy", "download", "en_core_web_sm"], check=True)
    except subprocess.CalledProcessError:
        print("⚠️  Warning: Could not download spaCy model. Some features may be limited.")
    
    # Download NLTK data
    print("📦 Downloading NLTK data...")
    subprocess.run([str(python_path), "-c", 
                   "import nltk; nltk.download('punkt'); nltk.download('stopwords')"], 
                   check=True)
    
    print("✅ Environment setup complete!")

def check_environment():
    """Check if the environment is properly set up"""
    print("🔍 Checking environment...")
    
    # Check if .env file exists
    if not Path(".env").exists():
        print("⚠️  .env file not found. Creating from template...")
        if Path(".env.example").exists():
            import shutil
            shutil.copy(".env.example", ".env")
            print("📝 Please edit .env file with your API keys before running the system.")
            return False
        else:
            print("❌ .env.example file not found!")
            return False
    
    # Check if uploads directory exists
    uploads_dir = Path("uploads")
    if not uploads_dir.exists():
        print("📁 Creating uploads directory...")
        uploads_dir.mkdir(exist_ok=True)
    
    return True

def main():
    """Main function to run the system"""
    print("🎯 Resume Relevance Check System")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "setup":
            setup_environment()
            return
        elif command == "api":
            if not check_environment():
                return
            run_flask_api()
            return
        elif command == "dashboard":
            if not check_environment():
                return
            run_streamlit_dashboard()
            return
        elif command == "help":
            print_help()
            return
    
    # Default: run both services
    if not check_environment():
        print("❌ Environment check failed. Run 'python run_system.py setup' first.")
        return
    
    print("🚀 Starting both API server and Streamlit dashboard...")
    print("📡 API will be available at: http://localhost:5000")
    print("🌐 Dashboard will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop both services")
    
    # Start Flask API in a separate thread
    api_thread = threading.Thread(target=run_flask_api, daemon=True)
    api_thread.start()
    
    # Wait a moment for API to start
    time.sleep(3)
    
    # Start Streamlit dashboard (this will block)
    try:
        run_streamlit_dashboard()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down system...")

def print_help():
    """Print help information"""
    print("""
🎯 Resume Relevance Check System - Help

Usage: python run_system.py [command]

Commands:
  setup      - Set up the environment and install dependencies
  api        - Run only the Flask API server (port 5000)
  dashboard  - Run only the Streamlit dashboard (port 8501)
  help       - Show this help message
  (no args)  - Run both API and dashboard

Setup Instructions:
1. Copy .env.example to .env and add your API keys
2. Run: python run_system.py setup
3. Run: python run_system.py

Required API Keys (add to .env file):
- OPENAI_API_KEY (for GPT models and embeddings)
- GOOGLE_API_KEY (optional, for Gemini)
- ANTHROPIC_API_KEY (optional, for Claude)
- PINECONE_API_KEY (optional, for Pinecone vector store)

System Components:
- Flask API: Handles file uploads, parsing, and analysis
- Streamlit Dashboard: Web interface for recruiters
- SQLite Database: Stores resumes, jobs, and evaluations
- Vector Store: For semantic similarity matching
- LLM Integration: For improvement suggestions

For more information, see the README.md file.
    """)

if __name__ == "__main__":
    main()