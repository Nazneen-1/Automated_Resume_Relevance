from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import openai
import google.generativeai as genai
from anthropic import Anthropic
from typing import Dict, List, Optional
import logging
from config.settings import Config

logger = logging.getLogger(__name__)

class LLMIntegration:
    """Integration with various LLM providers for generating improvement suggestions"""
    
    def __init__(self, provider: str = "openai"):
        self.provider = provider.lower()
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate LLM client"""
        try:
            if self.provider == "openai" and Config.OPENAI_API_KEY:
                self.client = ChatOpenAI(
                    openai_api_key=Config.OPENAI_API_KEY,
                    model_name="gpt-3.5-turbo",
                    temperature=0.7
                )
            elif self.provider == "gemini" and Config.GOOGLE_API_KEY:
                genai.configure(api_key=Config.GOOGLE_API_KEY)
                self.client = genai.GenerativeModel('gemini-pro')
            elif self.provider == "claude" and Config.ANTHROPIC_API_KEY:
                self.client = Anthropic(api_key=Config.ANTHROPIC_API_KEY)
            else:
                logger.warning(f"No API key found for {self.provider} or unsupported provider")
                
        except Exception as e:
            logger.error(f"Error initializing {self.provider} client: {e}")
    
    def generate_improvement_suggestions(self, analysis_results: Dict, resume_data: Dict, job_data: Dict) -> str:
        """Generate personalized improvement suggestions using LLM"""
        
        if not self.client:
            return self._generate_fallback_suggestions(analysis_results)
        
        try:
            prompt = self._create_improvement_prompt(analysis_results, resume_data, job_data)
            
            if self.provider == "openai":
                return self._generate_openai_suggestions(prompt)
            elif self.provider == "gemini":
                return self._generate_gemini_suggestions(prompt)
            elif self.provider == "claude":
                return self._generate_claude_suggestions(prompt)
            else:
                return self._generate_fallback_suggestions(analysis_results)
                
        except Exception as e:
            logger.error(f"Error generating LLM suggestions: {e}")
            return self._generate_fallback_suggestions(analysis_results)
    
    def _create_improvement_prompt(self, analysis_results: Dict, resume_data: Dict, job_data: Dict) -> str:
        """Create a comprehensive prompt for improvement suggestions"""
        
        prompt = f"""
        As a professional career advisor and resume expert, analyze the following resume evaluation results and provide specific, actionable improvement suggestions.

        **Job Description Summary:**
        - Title: {job_data.get('title', 'Not specified')}
        - Company: {job_data.get('company', 'Not specified')}
        - Required Skills: {job_data.get('must_have_skills', 'Not specified')[:500]}
        - Preferred Skills: {job_data.get('good_to_have_skills', 'Not specified')[:500]}

        **Resume Analysis Results:**
        - Overall Relevance Score: {analysis_results.get('relevance_score', 0):.1f}/100
        - Hard Match Score: {analysis_results.get('hard_match_score', 0):.1f}/100
        - Semantic Match Score: {analysis_results.get('semantic_match_score', 0):.1f}/100
        - Suitability Verdict: {analysis_results.get('suitability_verdict', 'Unknown')}

        **Missing Elements:**
        - Missing Skills: {', '.join(analysis_results.get('missing_skills', [])[:10])}
        - Suggested Certifications: {', '.join(analysis_results.get('missing_certifications', [])[:5])}
        - Suggested Projects: {', '.join(analysis_results.get('missing_projects', [])[:5])}

        **Current Resume Sections:**
        - Skills: {resume_data.get('skills', 'Not provided')[:300]}
        - Experience: {resume_data.get('experience', 'Not provided')[:300]}
        - Education: {resume_data.get('education', 'Not provided')[:200]}

        Please provide specific, actionable improvement suggestions in the following format:

        ## Immediate Actions (1-2 weeks)
        [List 3-5 specific actions the candidate can take immediately]

        ## Short-term Goals (1-3 months)
        [List 3-5 skills to develop or certifications to pursue]

        ## Long-term Development (3-6 months)
        [List 2-3 major projects or experiences to gain]

        ## Resume Enhancement Tips
        [Specific suggestions for improving resume content and presentation]

        ## Interview Preparation
        [Key areas to focus on for interview preparation]

        Keep suggestions specific, realistic, and directly relevant to the job requirements. Focus on the most impactful improvements first.
        """
        
        return prompt
    
    def _generate_openai_suggestions(self, prompt: str) -> str:
        """Generate suggestions using OpenAI"""
        try:
            messages = [
                SystemMessage(content="You are a professional career advisor and resume expert with 10+ years of experience helping candidates improve their job applications."),
                HumanMessage(content=prompt)
            ]
            
            response = self.client(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def _generate_gemini_suggestions(self, prompt: str) -> str:
        """Generate suggestions using Google Gemini"""
        try:
            response = self.client.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    def _generate_claude_suggestions(self, prompt: str) -> str:
        """Generate suggestions using Anthropic Claude"""
        try:
            message = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1500,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text
            
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise
    
    def _generate_fallback_suggestions(self, analysis_results: Dict) -> str:
        """Generate basic suggestions when LLM is not available"""
        
        score = analysis_results.get('relevance_score', 0)
        verdict = analysis_results.get('suitability_verdict', 'Unknown')
        missing_skills = analysis_results.get('missing_skills', [])
        missing_certs = analysis_results.get('missing_certifications', [])
        
        suggestions = f"""
## Resume Improvement Suggestions

**Overall Assessment:** Your resume scored {score:.1f}/100 with a {verdict} suitability rating.

## Immediate Actions (1-2 weeks)
"""
        
        if score < 50:
            suggestions += """
- Review and update your skills section to better match job requirements
- Quantify your achievements with specific numbers and metrics
- Tailor your professional summary to highlight relevant experience
- Use industry-specific keywords throughout your resume
"""
        elif score < 75:
            suggestions += """
- Add more specific examples of your technical skills in action
- Include relevant projects that demonstrate required competencies
- Update your experience descriptions to emphasize transferable skills
- Consider adding a technical skills matrix or proficiency levels
"""
        else:
            suggestions += """
- Fine-tune your resume formatting for better readability
- Add any recent projects or achievements
- Consider adding links to your portfolio or GitHub
- Ensure all contact information is current and professional
"""
        
        if missing_skills:
            suggestions += f"""
## Short-term Goals (1-3 months)
Focus on developing these missing skills:
{chr(10).join([f"- {skill}" for skill in missing_skills[:5]])}
"""
        
        if missing_certs:
            suggestions += f"""
## Recommended Certifications
Consider pursuing these certifications:
{chr(10).join([f"- {cert}" for cert in missing_certs[:3]])}
"""
        
        suggestions += """
## Resume Enhancement Tips
- Use action verbs to start each bullet point
- Keep descriptions concise but impactful
- Ensure consistent formatting throughout
- Proofread for grammar and spelling errors
- Save and submit as PDF to preserve formatting

## Interview Preparation
- Research the company and role thoroughly
- Prepare specific examples using the STAR method
- Practice explaining technical concepts in simple terms
- Prepare questions about the role and company culture
"""
        
        return suggestions

class VectorStoreManager:
    """Manages vector storage for semantic search and similarity matching"""
    
    def __init__(self, store_type: str = "chroma"):
        self.store_type = store_type.lower()
        self.vector_store = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize the vector store based on configuration"""
        try:
            if self.store_type == "chroma":
                import chromadb
                self.client = chromadb.Client()
                self.collection = self.client.create_collection(
                    name="resume_job_embeddings",
                    get_or_create=True
                )
            elif self.store_type == "faiss":
                import faiss
                # Initialize FAISS index
                self.dimension = 1536  # OpenAI embedding dimension
                self.index = faiss.IndexFlatIP(self.dimension)
            elif self.store_type == "pinecone":
                import pinecone
                if Config.PINECONE_API_KEY:
                    pinecone.init(
                        api_key=Config.PINECONE_API_KEY,
                        environment=Config.PINECONE_ENVIRONMENT
                    )
                    
        except Exception as e:
            logger.error(f"Error initializing vector store {self.store_type}: {e}")
    
    def store_embeddings(self, documents: List[str], metadata: List[Dict], embeddings: List[List[float]]):
        """Store document embeddings in the vector store"""
        try:
            if self.store_type == "chroma":
                self.collection.add(
                    documents=documents,
                    metadatas=metadata,
                    embeddings=embeddings,
                    ids=[f"doc_{i}" for i in range(len(documents))]
                )
            elif self.store_type == "faiss":
                import numpy as np
                embeddings_array = np.array(embeddings).astype('float32')
                self.index.add(embeddings_array)
                
        except Exception as e:
            logger.error(f"Error storing embeddings: {e}")
    
    def search_similar(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        try:
            if self.store_type == "chroma":
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k
                )
                return results
            elif self.store_type == "faiss":
                import numpy as np
                query_array = np.array([query_embedding]).astype('float32')
                distances, indices = self.index.search(query_array, top_k)
                return {"distances": distances[0], "indices": indices[0]}
                
        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            return []

def get_embeddings(text: str, model: str = "text-embedding-ada-002") -> List[float]:
    """Get embeddings for text using OpenAI"""
    try:
        if Config.OPENAI_API_KEY:
            openai.api_key = Config.OPENAI_API_KEY
            response = openai.Embedding.create(
                input=text,
                model=model
            )
            return response['data'][0]['embedding']
        else:
            logger.warning("OpenAI API key not available for embeddings")
            return []
    except Exception as e:
        logger.error(f"Error getting embeddings: {e}")
        return []