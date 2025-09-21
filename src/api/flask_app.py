from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
from werkzeug.utils import secure_filename
import logging
from datetime import datetime
import json

from config.settings import Config
from src.models.database import db_manager, JobDescription, Resume, ResumeEvaluation
from src.parsers.document_parser import DocumentParser, JobDescriptionParser
from src.analysis.relevance_engine import RelevanceEngine
from src.analysis.llm_integration import LLMIntegration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Initialize components
document_parser = DocumentParser()
job_parser = JobDescriptionParser()
relevance_engine = RelevanceEngine()
llm_integration = LLMIntegration()

# Ensure upload directory exists
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/upload/job-description', methods=['POST'])
def upload_job_description():
    """Upload and parse job description"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(Config.UPLOAD_FOLDER, unique_filename)
        file.save(file_path)
        
        # Parse job description
        job_data = job_parser.parse_job_description(file_path)
        
        # Save to database
        session = db_manager.get_session()
        try:
            job_desc = JobDescription(
                title=job_data.get('title', 'Not specified'),
                company=job_data.get('company', 'Not specified'),
                description=job_data.get('full_text', ''),
                must_have_skills=job_data.get('must_have_skills', ''),
                good_to_have_skills=job_data.get('good_to_have_skills', ''),
                qualifications=job_data.get('qualifications', ''),
                location=job_data.get('location', 'Not specified')
            )
            session.add(job_desc)
            session.commit()
            
            response_data = {
                'job_id': job_desc.id,
                'title': job_desc.title,
                'company': job_desc.company,
                'location': job_desc.location,
                'parsed_data': job_data,
                'message': 'Job description uploaded and parsed successfully'
            }
            
            return jsonify(response_data), 201
            
        finally:
            db_manager.close_session(session)
            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
                
    except Exception as e:
        logger.error(f"Error uploading job description: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/upload/resume', methods=['POST'])
def upload_resume():
    """Upload and parse resume"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(Config.UPLOAD_FOLDER, unique_filename)
        file.save(file_path)
        
        # Parse resume
        resume_data = document_parser.parse_document(file_path)
        
        # Save to database
        session = db_manager.get_session()
        try:
            resume = Resume(
                candidate_name=request.form.get('candidate_name', 'Not provided'),
                email=request.form.get('email', ''),
                phone=request.form.get('phone', ''),
                filename=filename,
                file_path=file_path,
                extracted_text=resume_data.get('full_text', ''),
                skills=resume_data.get('skills', ''),
                experience=resume_data.get('experience', ''),
                education=resume_data.get('education', ''),
                certifications=resume_data.get('certifications', ''),
                projects=resume_data.get('projects', '')
            )
            session.add(resume)
            session.commit()
            
            response_data = {
                'resume_id': resume.id,
                'candidate_name': resume.candidate_name,
                'filename': resume.filename,
                'parsed_data': resume_data,
                'message': 'Resume uploaded and parsed successfully'
            }
            
            return jsonify(response_data), 201
            
        finally:
            db_manager.close_session(session)
                
    except Exception as e:
        logger.error(f"Error uploading resume: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/evaluate', methods=['POST'])
def evaluate_resume():
    """Evaluate resume against job description"""
    try:
        data = request.get_json()
        resume_id = data.get('resume_id')
        job_id = data.get('job_id')
        
        if not resume_id or not job_id:
            return jsonify({'error': 'Resume ID and Job ID are required'}), 400
        
        session = db_manager.get_session()
        try:
            # Get resume and job description
            resume = session.query(Resume).filter(Resume.id == resume_id).first()
            job_desc = session.query(JobDescription).filter(JobDescription.id == job_id).first()
            
            if not resume or not job_desc:
                return jsonify({'error': 'Resume or Job Description not found'}), 404
            
            # Prepare data for analysis
            resume_data = {
                'full_text': resume.extracted_text,
                'skills': resume.skills,
                'experience': resume.experience,
                'education': resume.education,
                'certifications': resume.certifications,
                'projects': resume.projects
            }
            
            job_data = {
                'full_text': job_desc.description,
                'title': job_desc.title,
                'company': job_desc.company,
                'must_have_skills': job_desc.must_have_skills,
                'good_to_have_skills': job_desc.good_to_have_skills,
                'qualifications': job_desc.qualifications
            }
            
            # Perform relevance analysis
            analysis_results = relevance_engine.analyze_relevance(resume_data, job_data)
            
            # Generate improvement suggestions
            improvement_suggestions = llm_integration.generate_improvement_suggestions(
                analysis_results, resume_data, job_data
            )
            
            # Save evaluation results
            evaluation = ResumeEvaluation(
                resume_id=resume_id,
                job_description_id=job_id,
                relevance_score=analysis_results['relevance_score'],
                hard_match_score=analysis_results['hard_match_score'],
                semantic_match_score=analysis_results['semantic_match_score'],
                suitability_verdict=analysis_results['suitability_verdict'],
                missing_skills=json.dumps(analysis_results['missing_skills']),
                missing_certifications=json.dumps(analysis_results['missing_certifications']),
                missing_projects=json.dumps(analysis_results['missing_projects']),
                improvement_suggestions=improvement_suggestions
            )
            
            session.add(evaluation)
            session.commit()
            
            response_data = {
                'evaluation_id': evaluation.id,
                'relevance_score': analysis_results['relevance_score'],
                'hard_match_score': analysis_results['hard_match_score'],
                'semantic_match_score': analysis_results['semantic_match_score'],
                'suitability_verdict': analysis_results['suitability_verdict'],
                'missing_skills': analysis_results['missing_skills'],
                'missing_certifications': analysis_results['missing_certifications'],
                'missing_projects': analysis_results['missing_projects'],
                'improvement_suggestions': improvement_suggestions,
                'detailed_analysis': analysis_results.get('detailed_analysis', {}),
                'message': 'Resume evaluation completed successfully'
            }
            
            return jsonify(response_data), 200
            
        finally:
            db_manager.close_session(session)
            
    except Exception as e:
        logger.error(f"Error evaluating resume: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/evaluations', methods=['GET'])
def get_evaluations():
    """Get all evaluations with filtering options"""
    try:
        # Get query parameters
        job_id = request.args.get('job_id', type=int)
        min_score = request.args.get('min_score', type=float)
        max_score = request.args.get('max_score', type=float)
        verdict = request.args.get('verdict')
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        session = db_manager.get_session()
        try:
            # Build query
            query = session.query(ResumeEvaluation).join(Resume).join(JobDescription)
            
            if job_id:
                query = query.filter(ResumeEvaluation.job_description_id == job_id)
            if min_score is not None:
                query = query.filter(ResumeEvaluation.relevance_score >= min_score)
            if max_score is not None:
                query = query.filter(ResumeEvaluation.relevance_score <= max_score)
            if verdict:
                query = query.filter(ResumeEvaluation.suitability_verdict == verdict)
            
            # Get total count
            total_count = query.count()
            
            # Apply pagination
            evaluations = query.order_by(ResumeEvaluation.relevance_score.desc())\
                              .offset(offset).limit(limit).all()
            
            # Format response
            results = []
            for eval in evaluations:
                results.append({
                    'evaluation_id': eval.id,
                    'resume_id': eval.resume_id,
                    'job_id': eval.job_description_id,
                    'candidate_name': eval.resume.candidate_name,
                    'job_title': eval.job_description.title,
                    'company': eval.job_description.company,
                    'relevance_score': eval.relevance_score,
                    'suitability_verdict': eval.suitability_verdict,
                    'created_at': eval.created_at.isoformat(),
                    'missing_skills': json.loads(eval.missing_skills) if eval.missing_skills else [],
                    'missing_certifications': json.loads(eval.missing_certifications) if eval.missing_certifications else []
                })
            
            return jsonify({
                'evaluations': results,
                'total_count': total_count,
                'limit': limit,
                'offset': offset
            }), 200
            
        finally:
            db_manager.close_session(session)
            
    except Exception as e:
        logger.error(f"Error getting evaluations: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/evaluation/<int:evaluation_id>', methods=['GET'])
def get_evaluation_details(evaluation_id):
    """Get detailed evaluation results"""
    try:
        session = db_manager.get_session()
        try:
            evaluation = session.query(ResumeEvaluation)\
                               .join(Resume)\
                               .join(JobDescription)\
                               .filter(ResumeEvaluation.id == evaluation_id)\
                               .first()
            
            if not evaluation:
                return jsonify({'error': 'Evaluation not found'}), 404
            
            response_data = {
                'evaluation_id': evaluation.id,
                'resume': {
                    'id': evaluation.resume.id,
                    'candidate_name': evaluation.resume.candidate_name,
                    'email': evaluation.resume.email,
                    'phone': evaluation.resume.phone,
                    'filename': evaluation.resume.filename
                },
                'job_description': {
                    'id': evaluation.job_description.id,
                    'title': evaluation.job_description.title,
                    'company': evaluation.job_description.company,
                    'location': evaluation.job_description.location
                },
                'scores': {
                    'relevance_score': evaluation.relevance_score,
                    'hard_match_score': evaluation.hard_match_score,
                    'semantic_match_score': evaluation.semantic_match_score
                },
                'suitability_verdict': evaluation.suitability_verdict,
                'missing_elements': {
                    'skills': json.loads(evaluation.missing_skills) if evaluation.missing_skills else [],
                    'certifications': json.loads(evaluation.missing_certifications) if evaluation.missing_certifications else [],
                    'projects': json.loads(evaluation.missing_projects) if evaluation.missing_projects else []
                },
                'improvement_suggestions': evaluation.improvement_suggestions,
                'created_at': evaluation.created_at.isoformat()
            }
            
            return jsonify(response_data), 200
            
        finally:
            db_manager.close_session(session)
            
    except Exception as e:
        logger.error(f"Error getting evaluation details: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/jobs', methods=['GET'])
def get_jobs():
    """Get all job descriptions"""
    try:
        session = db_manager.get_session()
        try:
            jobs = session.query(JobDescription).order_by(JobDescription.created_at.desc()).all()
            
            results = []
            for job in jobs:
                results.append({
                    'id': job.id,
                    'title': job.title,
                    'company': job.company,
                    'location': job.location,
                    'created_at': job.created_at.isoformat()
                })
            
            return jsonify({'jobs': results}), 200
            
        finally:
            db_manager.close_session(session)
            
    except Exception as e:
        logger.error(f"Error getting jobs: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/resumes', methods=['GET'])
def get_resumes():
    """Get all resumes"""
    try:
        session = db_manager.get_session()
        try:
            resumes = session.query(Resume).order_by(Resume.created_at.desc()).all()
            
            results = []
            for resume in resumes:
                results.append({
                    'id': resume.id,
                    'candidate_name': resume.candidate_name,
                    'email': resume.email,
                    'filename': resume.filename,
                    'created_at': resume.created_at.isoformat()
                })
            
            return jsonify({'resumes': results}), 200
            
        finally:
            db_manager.close_session(session)
            
    except Exception as e:
        logger.error(f"Error getting resumes: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Initialize database
    db_manager.create_tables()
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=Config.DEBUG
    )