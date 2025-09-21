import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
    
    # Database
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///resume_system.db')
    
    # Application Settings
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt'}
    
    # Scoring Weights
    HARD_MATCH_WEIGHT = 0.4
    SEMANTIC_MATCH_WEIGHT = 0.6
    
    # Thresholds
    HIGH_SUITABILITY_THRESHOLD = 75
    MEDIUM_SUITABILITY_THRESHOLD = 50
    
    # Vector Store Settings
    VECTOR_STORE_TYPE = 'chroma'  # Options: 'chroma', 'faiss', 'pinecone'
    EMBEDDING_MODEL = 'text-embedding-ada-002'
    
    @staticmethod
    def init_app(app):
        pass

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}