import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import io
import base64

# Configure Streamlit page
st.set_page_config(
    page_title="Resume Relevance Check System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:5000/api"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-card {
        background-color: #d4edda;
        border-color: #28a745;
    }
    .warning-card {
        background-color: #fff3cd;
        border-color: #ffc107;
    }
    .danger-card {
        background-color: #f8d7da;
        border-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

def make_api_request(endpoint, method="GET", data=None, files=None):
    """Make API request with error handling"""
    try:
        url = f"{API_BASE_URL}/{endpoint}"
        
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            if files:
                response = requests.post(url, data=data, files=files)
            else:
                response = requests.post(url, json=data)
        
        if response.status_code == 200 or response.status_code == 201:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code} - {response.text}"
            
    except requests.exceptions.ConnectionError:
        return None, "Connection Error: Unable to connect to API server. Please ensure the Flask API is running."
    except Exception as e:
        return None, f"Error: {str(e)}"

def upload_job_description():
    """Job Description Upload Section"""
    st.header("📋 Upload Job Description")
    
    uploaded_file = st.file_uploader(
        "Choose a job description file",
        type=['pdf', 'docx', 'doc', 'txt'],
        help="Upload the job description in PDF, DOCX, or TXT format"
    )
    
    if uploaded_file is not None:
        if st.button("Upload and Parse Job Description", type="primary"):
            with st.spinner("Uploading and parsing job description..."):
                files = {"file": uploaded_file}
                result, error = make_api_request("upload/job-description", "POST", files=files)
                
                if error:
                    st.error(f"Upload failed: {error}")
                else:
                    st.success("Job description uploaded successfully!")
                    
                    # Display parsed information
                    st.subheader("Parsed Information")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.info(f"**Title:** {result.get('title', 'N/A')}")
                        st.info(f"**Company:** {result.get('company', 'N/A')}")
                        st.info(f"**Location:** {result.get('location', 'N/A')}")
                    
                    with col2:
                        st.info(f"**Job ID:** {result.get('job_id', 'N/A')}")
                    
                    # Store job ID in session state
                    st.session_state.current_job_id = result.get('job_id')
                    
                    # Display parsed sections
                    parsed_data = result.get('parsed_data', {})
                    if parsed_data.get('must_have_skills'):
                        st.subheader("Required Skills")
                        st.text_area("Must-have skills", parsed_data['must_have_skills'], height=100, disabled=True)
                    
                    if parsed_data.get('good_to_have_skills'):
                        st.subheader("Preferred Skills")
                        st.text_area("Good-to-have skills", parsed_data['good_to_have_skills'], height=100, disabled=True)

def upload_resume():
    """Resume Upload Section"""
    st.header("📄 Upload Resume")
    
    # Check if job is selected
    if 'current_job_id' not in st.session_state:
        st.warning("Please upload a job description first.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        candidate_name = st.text_input("Candidate Name")
        email = st.text_input("Email (optional)")
    
    with col2:
        phone = st.text_input("Phone (optional)")
    
    uploaded_file = st.file_uploader(
        "Choose a resume file",
        type=['pdf', 'docx', 'doc'],
        help="Upload the resume in PDF or DOCX format"
    )
    
    if uploaded_file is not None and candidate_name:
        if st.button("Upload and Evaluate Resume", type="primary"):
            with st.spinner("Uploading resume and performing evaluation..."):
                # Upload resume
                files = {"file": uploaded_file}
                data = {
                    "candidate_name": candidate_name,
                    "email": email,
                    "phone": phone
                }
                
                resume_result, error = make_api_request("upload/resume", "POST", data=data, files=files)
                
                if error:
                    st.error(f"Resume upload failed: {error}")
                    return
                
                st.success("Resume uploaded successfully!")
                resume_id = resume_result.get('resume_id')
                
                # Perform evaluation
                with st.spinner("Analyzing resume relevance..."):
                    eval_data = {
                        "resume_id": resume_id,
                        "job_id": st.session_state.current_job_id
                    }
                    
                    eval_result, error = make_api_request("evaluate", "POST", data=eval_data)
                    
                    if error:
                        st.error(f"Evaluation failed: {error}")
                        return
                    
                    # Display evaluation results
                    display_evaluation_results(eval_result)

def display_evaluation_results(result):
    """Display evaluation results"""
    st.header("📊 Evaluation Results")
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    relevance_score = result.get('relevance_score', 0)
    verdict = result.get('suitability_verdict', 'Unknown')
    
    with col1:
        st.metric("Relevance Score", f"{relevance_score:.1f}/100")
    
    with col2:
        color = "success" if verdict == "High" else "warning" if verdict == "Medium" else "danger"
        st.markdown(f'<div class="metric-card {color}-card"><h3>Suitability</h3><h2>{verdict}</h2></div>', 
                   unsafe_allow_html=True)
    
    with col3:
        st.metric("Hard Match Score", f"{result.get('hard_match_score', 0):.1f}/100")
    
    with col4:
        st.metric("Semantic Match Score", f"{result.get('semantic_match_score', 0):.1f}/100")
    
    # Score breakdown chart
    st.subheader("Score Breakdown")
    
    fig = go.Figure(data=[
        go.Bar(name='Scores', x=['Relevance', 'Hard Match', 'Semantic Match'], 
               y=[relevance_score, result.get('hard_match_score', 0), result.get('semantic_match_score', 0)],
               marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ])
    
    fig.update_layout(
        title="Score Analysis",
        yaxis_title="Score (0-100)",
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Missing elements
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Missing Skills")
        missing_skills = result.get('missing_skills', [])
        if missing_skills:
            for skill in missing_skills[:10]:  # Show top 10
                st.write(f"• {skill}")
        else:
            st.write("No critical skills missing!")
    
    with col2:
        st.subheader("Suggested Certifications")
        missing_certs = result.get('missing_certifications', [])
        if missing_certs:
            for cert in missing_certs[:5]:  # Show top 5
                st.write(f"• {cert}")
        else:
            st.write("No specific certifications suggested")
    
    with col3:
        st.subheader("Suggested Projects")
        missing_projects = result.get('missing_projects', [])
        if missing_projects:
            for project in missing_projects[:5]:  # Show top 5
                st.write(f"• {project}")
        else:
            st.write("No specific projects suggested")
    
    # Improvement suggestions
    st.subheader("💡 Improvement Suggestions")
    suggestions = result.get('improvement_suggestions', '')
    if suggestions:
        st.markdown(suggestions)
    else:
        st.write("No specific suggestions available.")

def dashboard_overview():
    """Dashboard Overview Section"""
    st.header("📈 Dashboard Overview")
    
    # Get evaluations data
    evaluations_data, error = make_api_request("evaluations?limit=100")
    
    if error:
        st.error(f"Failed to load dashboard data: {error}")
        return
    
    evaluations = evaluations_data.get('evaluations', [])
    
    if not evaluations:
        st.info("No evaluations found. Upload job descriptions and resumes to see analytics.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(evaluations)
    df['created_at'] = pd.to_datetime(df['created_at'])
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Evaluations", len(df))
    
    with col2:
        avg_score = df['relevance_score'].mean()
        st.metric("Average Score", f"{avg_score:.1f}")
    
    with col3:
        high_suitability = len(df[df['suitability_verdict'] == 'High'])
        st.metric("High Suitability", high_suitability)
    
    with col4:
        unique_jobs = df['job_id'].nunique()
        st.metric("Active Jobs", unique_jobs)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Score distribution
        fig = px.histogram(df, x='relevance_score', nbins=20, 
                          title="Score Distribution",
                          labels={'relevance_score': 'Relevance Score', 'count': 'Number of Candidates'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Suitability verdict distribution
        verdict_counts = df['suitability_verdict'].value_counts()
        fig = px.pie(values=verdict_counts.values, names=verdict_counts.index,
                    title="Suitability Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent evaluations table
    st.subheader("Recent Evaluations")
    
    # Add filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        job_filter = st.selectbox("Filter by Job", ["All"] + df['job_title'].unique().tolist())
    
    with col2:
        verdict_filter = st.selectbox("Filter by Verdict", ["All", "High", "Medium", "Low"])
    
    with col3:
        min_score = st.slider("Minimum Score", 0, 100, 0)
    
    # Apply filters
    filtered_df = df.copy()
    
    if job_filter != "All":
        filtered_df = filtered_df[filtered_df['job_title'] == job_filter]
    
    if verdict_filter != "All":
        filtered_df = filtered_df[filtered_df['suitability_verdict'] == verdict_filter]
    
    filtered_df = filtered_df[filtered_df['relevance_score'] >= min_score]
    
    # Display filtered results
    display_columns = ['candidate_name', 'job_title', 'company', 'relevance_score', 'suitability_verdict', 'created_at']
    st.dataframe(
        filtered_df[display_columns].sort_values('created_at', ascending=False),
        use_container_width=True
    )
    
    # Download report button
    if st.button("Download Report"):
        csv = filtered_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="evaluation_report.csv">Download CSV Report</a>'
        st.markdown(href, unsafe_allow_html=True)

def detailed_analysis():
    """Detailed Analysis Section"""
    st.header("🔍 Detailed Analysis")
    
    # Get evaluation ID from user
    evaluation_id = st.number_input("Enter Evaluation ID", min_value=1, step=1)
    
    if st.button("Load Detailed Analysis"):
        result, error = make_api_request(f"evaluation/{evaluation_id}")
        
        if error:
            st.error(f"Failed to load evaluation: {error}")
            return
        
        # Display detailed results
        st.subheader("Evaluation Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Candidate Information**")
            resume = result.get('resume', {})
            st.write(f"Name: {resume.get('candidate_name', 'N/A')}")
            st.write(f"Email: {resume.get('email', 'N/A')}")
            st.write(f"Phone: {resume.get('phone', 'N/A')}")
            st.write(f"Resume File: {resume.get('filename', 'N/A')}")
        
        with col2:
            st.write("**Job Information**")
            job = result.get('job_description', {})
            st.write(f"Title: {job.get('title', 'N/A')}")
            st.write(f"Company: {job.get('company', 'N/A')}")
            st.write(f"Location: {job.get('location', 'N/A')}")
        
        # Scores
        st.subheader("Detailed Scores")
        scores = result.get('scores', {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Relevance Score", f"{scores.get('relevance_score', 0):.1f}/100")
        with col2:
            st.metric("Hard Match Score", f"{scores.get('hard_match_score', 0):.1f}/100")
        with col3:
            st.metric("Semantic Match Score", f"{scores.get('semantic_match_score', 0):.1f}/100")
        
        # Missing elements
        st.subheader("Missing Elements Analysis")
        missing = result.get('missing_elements', {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Missing Skills**")
            for skill in missing.get('skills', []):
                st.write(f"• {skill}")
        
        with col2:
            st.write("**Suggested Certifications**")
            for cert in missing.get('certifications', []):
                st.write(f"• {cert}")
        
        with col3:
            st.write("**Suggested Projects**")
            for project in missing.get('projects', []):
                st.write(f"• {project}")
        
        # Improvement suggestions
        st.subheader("Improvement Suggestions")
        suggestions = result.get('improvement_suggestions', '')
        if suggestions:
            st.markdown(suggestions)

def main():
    """Main Streamlit application"""
    st.markdown('<h1 class="main-header">📄 Resume Relevance Check System</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Dashboard Overview", "Upload Job Description", "Upload Resume", "Detailed Analysis"]
    )
    
    # API health check
    health_result, health_error = make_api_request("health")
    if health_error:
        st.error("⚠️ API Server is not responding. Please ensure the Flask API is running on http://localhost:5000")
        st.stop()
    else:
        st.sidebar.success("✅ API Server Connected")
    
    # Route to appropriate page
    if page == "Dashboard Overview":
        dashboard_overview()
    elif page == "Upload Job Description":
        upload_job_description()
    elif page == "Upload Resume":
        upload_resume()
    elif page == "Detailed Analysis":
        detailed_analysis()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**System Status**")
    st.sidebar.info(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()