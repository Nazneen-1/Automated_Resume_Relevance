from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from config.settings import Config

Base = declarative_base()

class JobDescription(Base):
    __tablename__ = 'job_descriptions'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    company = Column(String(200))
    description = Column(Text, nullable=False)
    must_have_skills = Column(Text)
    good_to_have_skills = Column(Text)
    qualifications = Column(Text)
    location = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    evaluations = relationship("ResumeEvaluation", back_populates="job_description")

class Resume(Base):
    __tablename__ = 'resumes'
    
    id = Column(Integer, primary_key=True)
    candidate_name = Column(String(200))
    email = Column(String(200))
    phone = Column(String(50))
    filename = Column(String(200), nullable=False)
    file_path = Column(String(500), nullable=False)
    extracted_text = Column(Text)
    skills = Column(Text)
    experience = Column(Text)
    education = Column(Text)
    certifications = Column(Text)
    projects = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    evaluations = relationship("ResumeEvaluation", back_populates="resume")

class ResumeEvaluation(Base):
    __tablename__ = 'resume_evaluations'
    
    id = Column(Integer, primary_key=True)
    resume_id = Column(Integer, ForeignKey('resumes.id'), nullable=False)
    job_description_id = Column(Integer, ForeignKey('job_descriptions.id'), nullable=False)
    
    # Scores
    relevance_score = Column(Float, nullable=False)
    hard_match_score = Column(Float)
    semantic_match_score = Column(Float)
    
    # Analysis Results
    suitability_verdict = Column(String(20), nullable=False)  # High/Medium/Low
    missing_skills = Column(Text)
    missing_certifications = Column(Text)
    missing_projects = Column(Text)
    improvement_suggestions = Column(Text)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    resume = relationship("Resume", back_populates="evaluations")
    job_description = relationship("JobDescription", back_populates="evaluations")

class DatabaseManager:
    def __init__(self, database_url=None):
        self.database_url = database_url or Config.DATABASE_URL
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def create_tables(self):
        Base.metadata.create_all(bind=self.engine)
        
    def get_session(self):
        return self.SessionLocal()
        
    def close_session(self, session):
        session.close()

# Global database instance
db_manager = DatabaseManager()