import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz, process
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import Dict, List, Tuple, Optional
import logging
import re

logger = logging.getLogger(__name__)

class RelevanceEngine:
    """Core engine for analyzing resume relevance against job descriptions"""
    
    def __init__(self):
        self.nlp = None
        self.stop_words = set()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3),
            lowercase=True
        )
        self._initialize_nlp()
    
    def _initialize_nlp(self):
        """Initialize NLP models and resources"""
        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Please install: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        try:
            # Download NLTK resources
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            logger.warning(f"NLTK initialization failed: {e}")
    
    def analyze_relevance(self, resume_data: Dict, job_data: Dict) -> Dict:
        """
        Main method to analyze resume relevance against job description
        
        Args:
            resume_data: Parsed resume information
            job_data: Parsed job description information
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Step 1: Hard matching (keyword-based)
            hard_match_results = self._perform_hard_matching(resume_data, job_data)
            
            # Step 2: Semantic matching (embedding-based)
            semantic_match_results = self._perform_semantic_matching(resume_data, job_data)
            
            # Step 3: Calculate weighted relevance score
            relevance_score = self._calculate_relevance_score(
                hard_match_results, semantic_match_results
            )
            
            # Step 4: Determine suitability verdict
            suitability_verdict = self._determine_suitability(relevance_score)
            
            # Step 5: Identify missing elements
            missing_analysis = self._analyze_missing_elements(resume_data, job_data)
            
            # Compile results
            analysis_results = {
                'relevance_score': relevance_score,
                'hard_match_score': hard_match_results['overall_score'],
                'semantic_match_score': semantic_match_results['overall_score'],
                'suitability_verdict': suitability_verdict,
                'missing_skills': missing_analysis['missing_skills'],
                'missing_certifications': missing_analysis['missing_certifications'],
                'missing_projects': missing_analysis['missing_projects'],
                'detailed_analysis': {
                    'hard_match_details': hard_match_results,
                    'semantic_match_details': semantic_match_results,
                    'missing_analysis_details': missing_analysis
                }
            }
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in relevance analysis: {str(e)}")
            raise
    
    def _perform_hard_matching(self, resume_data: Dict, job_data: Dict) -> Dict:
        """Perform keyword-based hard matching using TF-IDF, BM25, and fuzzy matching"""
        
        # Extract text for analysis
        resume_text = self._combine_resume_text(resume_data)
        job_text = self._combine_job_text(job_data)
        
        # TF-IDF similarity
        tfidf_score = self._calculate_tfidf_similarity(resume_text, job_text)
        
        # Fuzzy matching for skills
        skills_match_score = self._calculate_skills_fuzzy_match(resume_data, job_data)
        
        # Keyword frequency analysis
        keyword_score = self._calculate_keyword_frequency_score(resume_text, job_text)
        
        # BM25 scoring (simplified implementation)
        bm25_score = self._calculate_bm25_score(resume_text, job_text)
        
        # Combine scores
        overall_score = (
            tfidf_score * 0.3 +
            skills_match_score * 0.4 +
            keyword_score * 0.2 +
            bm25_score * 0.1
        ) * 100
        
        return {
            'overall_score': min(overall_score, 100),
            'tfidf_score': tfidf_score * 100,
            'skills_match_score': skills_match_score * 100,
            'keyword_score': keyword_score * 100,
            'bm25_score': bm25_score * 100
        }
    
    def _perform_semantic_matching(self, resume_data: Dict, job_data: Dict) -> Dict:
        """Perform semantic matching using embeddings and cosine similarity"""
        
        if not self.nlp:
            logger.warning("spaCy model not available, using simplified semantic matching")
            return self._simplified_semantic_matching(resume_data, job_data)
        
        # Get document embeddings
        resume_text = self._combine_resume_text(resume_data)
        job_text = self._combine_job_text(job_data)
        
        resume_doc = self.nlp(resume_text)
        job_doc = self.nlp(job_text)
        
        # Calculate semantic similarity
        semantic_similarity = resume_doc.similarity(job_doc)
        
        # Section-wise semantic analysis
        section_similarities = {}
        for section in ['skills', 'experience', 'education']:
            if resume_data.get(section) and job_data.get('must_have_skills'):
                resume_section_doc = self.nlp(resume_data[section])
                job_section_doc = self.nlp(job_data['must_have_skills'])
                section_similarities[section] = resume_section_doc.similarity(job_section_doc)
        
        # Calculate overall semantic score
        overall_score = semantic_similarity
        if section_similarities:
            section_avg = np.mean(list(section_similarities.values()))
            overall_score = (semantic_similarity * 0.6 + section_avg * 0.4)
        
        return {
            'overall_score': overall_score * 100,
            'document_similarity': semantic_similarity * 100,
            'section_similarities': {k: v * 100 for k, v in section_similarities.items()}
        }
    
    def _simplified_semantic_matching(self, resume_data: Dict, job_data: Dict) -> Dict:
        """Simplified semantic matching when spaCy is not available"""
        resume_text = self._combine_resume_text(resume_data)
        job_text = self._combine_job_text(job_data)
        
        # Use TF-IDF as a proxy for semantic similarity
        tfidf_similarity = self._calculate_tfidf_similarity(resume_text, job_text)
        
        return {
            'overall_score': tfidf_similarity * 100,
            'document_similarity': tfidf_similarity * 100,
            'section_similarities': {}
        }
    
    def _calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Calculate TF-IDF cosine similarity between two texts"""
        try:
            corpus = [text1, text2]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            return similarity_matrix[0][1]
        except Exception as e:
            logger.error(f"Error calculating TF-IDF similarity: {e}")
            return 0.0
    
    def _calculate_skills_fuzzy_match(self, resume_data: Dict, job_data: Dict) -> float:
        """Calculate fuzzy matching score for skills"""
        resume_skills = self._extract_skills_list(resume_data.get('skills', ''))
        job_skills = self._extract_skills_list(job_data.get('must_have_skills', ''))
        
        if not resume_skills or not job_skills:
            return 0.0
        
        total_score = 0
        matches = 0
        
        for job_skill in job_skills:
            best_match = process.extractOne(job_skill, resume_skills, scorer=fuzz.token_sort_ratio)
            if best_match and best_match[1] > 70:  # 70% similarity threshold
                total_score += best_match[1] / 100
                matches += 1
        
        return total_score / len(job_skills) if job_skills else 0.0
    
    def _calculate_keyword_frequency_score(self, resume_text: str, job_text: str) -> float:
        """Calculate score based on keyword frequency overlap"""
        resume_keywords = self._extract_keywords(resume_text)
        job_keywords = self._extract_keywords(job_text)
        
        if not job_keywords:
            return 0.0
        
        common_keywords = set(resume_keywords) & set(job_keywords)
        return len(common_keywords) / len(job_keywords)
    
    def _calculate_bm25_score(self, resume_text: str, job_text: str) -> float:
        """Simplified BM25 scoring implementation"""
        # This is a simplified version - in production, use rank_bm25 library
        resume_words = set(resume_text.lower().split())
        job_words = set(job_text.lower().split())
        
        if not job_words:
            return 0.0
        
        common_words = resume_words & job_words
        return len(common_words) / len(job_words)
    
    def _calculate_relevance_score(self, hard_match_results: Dict, semantic_match_results: Dict) -> float:
        """Calculate weighted relevance score"""
        from config.settings import Config
        
        hard_score = hard_match_results['overall_score'] / 100
        semantic_score = semantic_match_results['overall_score'] / 100
        
        weighted_score = (
            hard_score * Config.HARD_MATCH_WEIGHT +
            semantic_score * Config.SEMANTIC_MATCH_WEIGHT
        ) * 100
        
        return min(weighted_score, 100)
    
    def _determine_suitability(self, relevance_score: float) -> str:
        """Determine suitability verdict based on relevance score"""
        from config.settings import Config
        
        if relevance_score >= Config.HIGH_SUITABILITY_THRESHOLD:
            return "High"
        elif relevance_score >= Config.MEDIUM_SUITABILITY_THRESHOLD:
            return "Medium"
        else:
            return "Low"
    
    def _analyze_missing_elements(self, resume_data: Dict, job_data: Dict) -> Dict:
        """Analyze missing skills, certifications, and projects"""
        
        # Extract required elements from job description
        required_skills = self._extract_skills_list(job_data.get('must_have_skills', ''))
        preferred_skills = self._extract_skills_list(job_data.get('good_to_have_skills', ''))
        
        # Extract candidate's elements
        candidate_skills = self._extract_skills_list(resume_data.get('skills', ''))
        candidate_certs = self._extract_certifications_list(resume_data.get('certifications', ''))
        candidate_projects = self._extract_projects_list(resume_data.get('projects', ''))
        
        # Find missing elements
        missing_skills = self._find_missing_skills(candidate_skills, required_skills + preferred_skills)
        missing_certifications = self._suggest_missing_certifications(missing_skills)
        missing_projects = self._suggest_missing_projects(missing_skills)
        
        return {
            'missing_skills': missing_skills,
            'missing_certifications': missing_certifications,
            'missing_projects': missing_projects
        }
    
    def _extract_skills_list(self, skills_text: str) -> List[str]:
        """Extract individual skills from skills text"""
        if not skills_text:
            return []
        
        # Common skill separators
        skills_text = re.sub(r'[,;|•\n\r]', ',', skills_text)
        skills = [skill.strip() for skill in skills_text.split(',')]
        
        # Filter out empty strings and common non-skills
        filtered_skills = []
        for skill in skills:
            if skill and len(skill) > 2 and not skill.lower() in ['and', 'or', 'with', 'using']:
                filtered_skills.append(skill)
        
        return filtered_skills
    
    def _extract_certifications_list(self, cert_text: str) -> List[str]:
        """Extract certifications from text"""
        if not cert_text:
            return []
        
        # Look for common certification patterns
        cert_patterns = [
            r'[A-Z]{2,}\s+[A-Z][a-z]+',  # AWS Certified, etc.
            r'Certified\s+[A-Z][a-z]+',
            r'[A-Z][a-z]+\s+Certification'
        ]
        
        certifications = []
        for pattern in cert_patterns:
            matches = re.findall(pattern, cert_text)
            certifications.extend(matches)
        
        return list(set(certifications))
    
    def _extract_projects_list(self, projects_text: str) -> List[str]:
        """Extract project names from text"""
        if not projects_text:
            return []
        
        # Simple extraction - split by common delimiters
        projects = re.split(r'[•\n\r]', projects_text)
        project_names = []
        
        for project in projects:
            project = project.strip()
            if project and len(project) > 10:  # Reasonable project description length
                # Extract first line as project name
                first_line = project.split('.')[0].strip()
                if first_line:
                    project_names.append(first_line)
        
        return project_names
    
    def _find_missing_skills(self, candidate_skills: List[str], required_skills: List[str]) -> List[str]:
        """Find skills that are required but missing from candidate's profile"""
        missing = []
        
        for req_skill in required_skills:
            # Use fuzzy matching to check if skill exists
            best_match = process.extractOne(req_skill, candidate_skills, scorer=fuzz.token_sort_ratio)
            if not best_match or best_match[1] < 70:  # Less than 70% similarity
                missing.append(req_skill)
        
        return missing
    
    def _suggest_missing_certifications(self, missing_skills: List[str]) -> List[str]:
        """Suggest relevant certifications based on missing skills"""
        cert_mapping = {
            'aws': ['AWS Certified Solutions Architect', 'AWS Certified Developer'],
            'azure': ['Microsoft Azure Fundamentals', 'Azure Solutions Architect'],
            'google cloud': ['Google Cloud Professional Cloud Architect'],
            'kubernetes': ['Certified Kubernetes Administrator (CKA)'],
            'docker': ['Docker Certified Associate'],
            'python': ['Python Institute Certifications'],
            'java': ['Oracle Java Certifications'],
            'project management': ['PMP', 'Scrum Master Certification'],
            'data science': ['Google Data Analytics Certificate', 'IBM Data Science Certificate'],
            'cybersecurity': ['CompTIA Security+', 'CISSP']
        }
        
        suggested_certs = []
        for skill in missing_skills:
            skill_lower = skill.lower()
            for key, certs in cert_mapping.items():
                if key in skill_lower:
                    suggested_certs.extend(certs)
        
        return list(set(suggested_certs))
    
    def _suggest_missing_projects(self, missing_skills: List[str]) -> List[str]:
        """Suggest project types based on missing skills"""
        project_mapping = {
            'web development': ['E-commerce Website', 'Portfolio Website', 'Blog Platform'],
            'mobile development': ['Mobile App', 'Cross-platform App'],
            'data science': ['Data Analysis Project', 'Machine Learning Model', 'Data Visualization Dashboard'],
            'machine learning': ['Predictive Model', 'Classification Project', 'Recommendation System'],
            'cloud': ['Cloud Migration Project', 'Serverless Application', 'Infrastructure as Code'],
            'devops': ['CI/CD Pipeline', 'Container Orchestration', 'Monitoring System'],
            'database': ['Database Design Project', 'Data Migration', 'Performance Optimization']
        }
        
        suggested_projects = []
        for skill in missing_skills:
            skill_lower = skill.lower()
            for key, projects in project_mapping.items():
                if any(keyword in skill_lower for keyword in key.split()):
                    suggested_projects.extend(projects)
        
        return list(set(suggested_projects))
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        if not text:
            return []
        
        # Remove stop words and extract meaningful terms
        words = word_tokenize(text.lower())
        keywords = [word for word in words if word.isalpha() and word not in self.stop_words and len(word) > 2]
        
        return list(set(keywords))
    
    def _combine_resume_text(self, resume_data: Dict) -> str:
        """Combine all resume sections into single text"""
        sections = ['skills', 'experience', 'education', 'projects', 'certifications', 'summary']
        combined_text = []
        
        for section in sections:
            if resume_data.get(section):
                combined_text.append(resume_data[section])
        
        return ' '.join(combined_text)
    
    def _combine_job_text(self, job_data: Dict) -> str:
        """Combine all job description sections into single text"""
        sections = ['must_have_skills', 'good_to_have_skills', 'qualifications', 'full_text']
        combined_text = []
        
        for section in sections:
            if job_data.get(section):
                combined_text.append(job_data[section])
        
        return ' '.join(combined_text)