import fitz  # PyMuPDF
import pdfplumber
import docx2txt
from docx import Document
import re
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class DocumentParser:
    """Handles parsing of PDF and DOCX documents"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.doc', '.txt']
    
    def parse_document(self, file_path: str) -> Dict[str, str]:
        """
        Parse document and extract structured information
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing extracted text and sections
        """
        try:
            file_extension = file_path.lower().split('.')[-1]
            
            if file_extension == 'pdf':
                return self._parse_pdf(file_path)
            elif file_extension in ['docx', 'doc']:
                return self._parse_docx(file_path)
            elif file_extension == 'txt':
                return self._parse_txt(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            logger.error(f"Error parsing document {file_path}: {str(e)}")
            raise
    
    def _parse_pdf(self, file_path: str) -> Dict[str, str]:
        """Parse PDF using PyMuPDF and pdfplumber"""
        text_content = ""
        
        try:
            # Primary method: PyMuPDF
            doc = fitz.open(file_path)
            for page in doc:
                text_content += page.get_text()
            doc.close()
            
            # Fallback: pdfplumber for better table extraction
            if not text_content.strip():
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_content += page_text + "\n"
                            
        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {str(e)}")
            raise
        
        return self._extract_sections(text_content)
    
    def _parse_docx(self, file_path: str) -> Dict[str, str]:
        """Parse DOCX using python-docx and docx2txt"""
        text_content = ""
        
        try:
            # Primary method: docx2txt
            text_content = docx2txt.process(file_path)
            
            # Fallback: python-docx for structured extraction
            if not text_content.strip():
                doc = Document(file_path)
                for paragraph in doc.paragraphs:
                    text_content += paragraph.text + "\n"
                    
        except Exception as e:
            logger.error(f"Error parsing DOCX {file_path}: {str(e)}")
            raise
        
        return self._extract_sections(text_content)
    
    def _parse_txt(self, file_path: str) -> Dict[str, str]:
        """Parse plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text_content = file.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as file:
                text_content = file.read()
                
        return self._extract_sections(text_content)
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract structured sections from text"""
        # Clean and normalize text
        text = self._clean_text(text)
        
        sections = {
            'full_text': text,
            'contact_info': self._extract_contact_info(text),
            'summary': self._extract_summary(text),
            'experience': self._extract_experience(text),
            'education': self._extract_education(text),
            'skills': self._extract_skills(text),
            'certifications': self._extract_certifications(text),
            'projects': self._extract_projects(text)
        }
        
        return sections
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\-\(\)\@\+\#]', '', text)
        return text.strip()
    
    def _extract_contact_info(self, text: str) -> str:
        """Extract contact information"""
        contact_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone
            r'\b(?:\+\d{1,3}\s?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'  # International phone
        ]
        
        contact_info = []
        for pattern in contact_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            contact_info.extend(matches)
        
        return ' | '.join(contact_info)
    
    def _extract_summary(self, text: str) -> str:
        """Extract professional summary or objective"""
        summary_keywords = [
            'summary', 'objective', 'profile', 'about', 'overview',
            'professional summary', 'career objective'
        ]
        
        lines = text.split('\n')
        summary_section = ""
        capture = False
        
        for line in lines:
            line_lower = line.lower().strip()
            
            if any(keyword in line_lower for keyword in summary_keywords):
                capture = True
                continue
            
            if capture:
                if line.strip() and not any(keyword in line_lower for keyword in 
                    ['experience', 'education', 'skills', 'projects', 'certifications']):
                    summary_section += line + " "
                else:
                    break
        
        return summary_section.strip()
    
    def _extract_experience(self, text: str) -> str:
        """Extract work experience section"""
        return self._extract_section_by_keywords(text, [
            'experience', 'work experience', 'employment', 'career history',
            'professional experience', 'work history'
        ])
    
    def _extract_education(self, text: str) -> str:
        """Extract education section"""
        return self._extract_section_by_keywords(text, [
            'education', 'academic', 'degree', 'university', 'college',
            'school', 'qualification'
        ])
    
    def _extract_skills(self, text: str) -> str:
        """Extract skills section"""
        return self._extract_section_by_keywords(text, [
            'skills', 'technical skills', 'core competencies', 'expertise',
            'technologies', 'programming languages', 'tools'
        ])
    
    def _extract_certifications(self, text: str) -> str:
        """Extract certifications section"""
        return self._extract_section_by_keywords(text, [
            'certifications', 'certificates', 'licenses', 'credentials',
            'professional certifications'
        ])
    
    def _extract_projects(self, text: str) -> str:
        """Extract projects section"""
        return self._extract_section_by_keywords(text, [
            'projects', 'personal projects', 'academic projects',
            'portfolio', 'work samples'
        ])
    
    def _extract_section_by_keywords(self, text: str, keywords: List[str]) -> str:
        """Generic method to extract sections based on keywords"""
        lines = text.split('\n')
        section_content = ""
        capture = False
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Check if this line contains section keywords
            if any(keyword in line_lower for keyword in keywords):
                capture = True
                continue
            
            if capture:
                # Stop if we hit another section
                other_sections = [
                    'experience', 'education', 'skills', 'projects', 
                    'certifications', 'summary', 'objective', 'contact'
                ]
                if any(section in line_lower for section in other_sections) and \
                   not any(keyword in line_lower for keyword in keywords):
                    break
                
                if line.strip():
                    section_content += line + "\n"
        
        return section_content.strip()

class JobDescriptionParser(DocumentParser):
    """Specialized parser for job descriptions"""
    
    def parse_job_description(self, file_path: str) -> Dict[str, str]:
        """Parse job description and extract structured information"""
        sections = self.parse_document(file_path)
        
        jd_info = {
            'full_text': sections['full_text'],
            'title': self._extract_job_title(sections['full_text']),
            'company': self._extract_company_name(sections['full_text']),
            'must_have_skills': self._extract_must_have_skills(sections['full_text']),
            'good_to_have_skills': self._extract_good_to_have_skills(sections['full_text']),
            'qualifications': self._extract_qualifications(sections['full_text']),
            'location': self._extract_location(sections['full_text'])
        }
        
        return jd_info
    
    def _extract_job_title(self, text: str) -> str:
        """Extract job title from job description"""
        lines = text.split('\n')[:10]  # Check first 10 lines
        
        title_keywords = ['position', 'role', 'job title', 'title', 'opening']
        
        for line in lines:
            line_clean = line.strip()
            if line_clean and len(line_clean) < 100:  # Reasonable title length
                # Check if it's likely a title
                if any(keyword in line.lower() for keyword in title_keywords) or \
                   (len(line_clean.split()) <= 5 and line_clean[0].isupper()):
                    return line_clean
        
        return "Not specified"
    
    def _extract_company_name(self, text: str) -> str:
        """Extract company name from job description"""
        company_patterns = [
            r'Company:\s*([^\n]+)',
            r'Organization:\s*([^\n]+)',
            r'Employer:\s*([^\n]+)'
        ]
        
        for pattern in company_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return "Not specified"
    
    def _extract_must_have_skills(self, text: str) -> str:
        """Extract must-have/required skills"""
        must_have_keywords = [
            'required', 'must have', 'essential', 'mandatory',
            'minimum requirements', 'qualifications required'
        ]
        
        return self._extract_skills_by_keywords(text, must_have_keywords)
    
    def _extract_good_to_have_skills(self, text: str) -> str:
        """Extract good-to-have/preferred skills"""
        good_to_have_keywords = [
            'preferred', 'nice to have', 'good to have', 'desirable',
            'plus', 'bonus', 'additional'
        ]
        
        return self._extract_skills_by_keywords(text, good_to_have_keywords)
    
    def _extract_skills_by_keywords(self, text: str, keywords: List[str]) -> str:
        """Extract skills based on section keywords"""
        lines = text.split('\n')
        skills_content = ""
        capture = False
        
        for line in lines:
            line_lower = line.lower().strip()
            
            if any(keyword in line_lower for keyword in keywords):
                capture = True
                skills_content += line + "\n"
                continue
            
            if capture:
                if line.strip() and not line_lower.startswith(('responsibilities', 'duties', 'about')):
                    skills_content += line + "\n"
                elif line_lower.startswith(('responsibilities', 'duties', 'about')):
                    break
        
        return skills_content.strip()
    
    def _extract_qualifications(self, text: str) -> str:
        """Extract qualifications section"""
        return self._extract_section_by_keywords(text, [
            'qualifications', 'requirements', 'education', 'degree',
            'experience required'
        ])
    
    def _extract_location(self, text: str) -> str:
        """Extract job location"""
        location_patterns = [
            r'Location:\s*([^\n]+)',
            r'Based in:\s*([^\n]+)',
            r'Office:\s*([^\n]+)'
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return "Not specified"