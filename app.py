"""
Interview Preparation Coach - Modern Premium Streamlit UI Application.

Features a state-of-the-art glassmorphism design system, interactive multi-agent 
processing indicators, animated progress timelines, and rich Q&A report cards.
"""

import sys
import time
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from graph.orchestrator import get_interview_guide
from agents.interviewer_agent import generate_interview_guide, generate_demo_qa_report


# Extensible Job Roles List
JOB_ROLES = [
    "Software Engineering",
    "DevOps",
    "Business Analyst",
    "QA Engineering",
    "Data Science",
    "Cloud Architecture",
    "Cybersecurity",
    "AI/ML Engineering",
    "UI/UX Design"
]

INTERVIEW_TYPES = ["HR", "Technical", "Coding"]


# Page Configuration
st.set_page_config(
    page_title="AI Interview Prep Coach",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Glassmorphism & High-Aesthetic CSS Design System
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap" rel="stylesheet">

<style>
    /* Global Typography & Background */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
    }

    /* Hero Banner Header */
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #311B92 100%);
        border-radius: 20px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.15rem;
        font-weight: 400;
        margin-bottom: 0;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.6rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
    }
    
    /* Interactive Pills */
    .badge-role {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%);
        color: #FFFFFF;
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
        box-shadow: 0 4px 6px -1px rgba(2, 132, 199, 0.3);
        display: inline-block;
    }
    .badge-type {
        background: linear-gradient(135deg, #7C3AED 0%, #6D28D9 100%);
        color: #FFFFFF;
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
        box-shadow: 0 4px 6px -1px rgba(124, 58, 237, 0.3);
        display: inline-block;
    }

    /* Section Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Theme Item Box */
    .theme-item {
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #38BDF8;
        padding: 0.8rem 1rem;
        border-radius: 0 10px 10px 0;
        margin-bottom: 0.6rem;
        color: #E2E8F0;
        font-weight: 500;
    }

    /* Tip Item Box */
    .tip-item {
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #C084FC;
        padding: 0.8rem 1rem;
        border-radius: 0 10px 10px 0;
        margin-bottom: 0.6rem;
        color: #E2E8F0;
        font-weight: 500;
    }

    /* Model Answer Container */
    .model-answer-box {
        background: rgba(6, 78, 59, 0.25);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 1.2rem;
        color: #ECFDF5;
        font-size: 0.98rem;
        line-height: 1.6;
        margin-top: 0.5rem;
    }

    /* Criteria & Advice Containers */
    .criteria-box {
        background: rgba(120, 53, 15, 0.25);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 12px;
        padding: 1.1rem;
        color: #FEF3C7;
    }
    .tips-box {
        background: rgba(12, 74, 110, 0.25);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 12px;
        padding: 1.1rem;
        color: #E0F2FE;
    }

    /* Streamlit UI Tweaks */
    .stSelectbox label, .stTextInput label {
        color: #CBD5E1 !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)


def init_session():
    """Initialize Streamlit session state variables."""
    if "selected_role" not in st.session_state:
        st.session_state.selected_role = JOB_ROLES[0]
    if "selected_type" not in st.session_state:
        st.session_state.selected_type = "Technical"
    if "report_generated" not in st.session_state:
        st.session_state.report_generated = False
    if "guide_data" not in st.session_state:
        st.session_state.guide_data = None
    if "qa_report" not in st.session_state:
        st.session_state.qa_report = None


def reset_session():
    """Reset session state."""
    st.session_state.report_generated = False
    st.session_state.guide_data = None
    st.session_state.qa_report = None


def render_sidebar():
    """Render sidebar with agent routing architecture and actions."""
    with st.sidebar:
        st.markdown("### ⚡ Multi-Agent System")
        st.caption("University Agentic AI Assignment")
        st.markdown("---")

        st.markdown("#### 🤖 Active AI Agents")
        st.markdown("""
        - **🔍 RAG Retriever**: ChromaDB + Sentence Transformers (`all-MiniLM-L6-v2`)
        - **🧠 Interviewer Agent**: Groq (`llama-3.1-8b-instant`)
        - **📊 Evaluator Agent**: Groq (`llama-3.1-8b-instant`)
        - **💎 Coach Agent**: OpenRouter (`llama-3.3-70b-instruct`)
        """)
        st.markdown("---")

        if st.session_state.report_generated:
            if st.button("🔄 Select Different Role / Round", type="primary", use_container_width=True):
                reset_session()
                st.rerun()


def render_role_selection():
    """Render Hero Header, Role & Category Selection, and Interactive Agent Pipeline Trigger."""
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🎓 AI Interview Preparation Coach</div>
        <div class="hero-subtitle">Interactive Multi-Agent System for Domain Guides & Model Q&A Scorecard Generation</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("### 🎯 Select Your Target Role & Interview Round")
        col1, col2 = st.columns(2)

        with col1:
            role = st.selectbox(
                "📌 Step 1: Target Job Position",
                options=JOB_ROLES,
                index=0,
                key="role_select",
                help="Select your career position to customize knowledge retrieval."
            )
            st.session_state.selected_role = role

        with col2:
            if role == "Business Analyst":
                available_types = ["HR", "Technical"]
                st.caption("ℹ️ *Coding round is automatically disabled for Business Analyst roles.*")
            else:
                available_types = ["HR", "Technical", "Coding"]

            itype = st.selectbox(
                "🎯 Step 2: Interview Round Category",
                options=available_types,
                index=0,
                key="type_select",
                help="Select the specific interview stage to prepare for."
            )
            st.session_state.selected_type = itype

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🚀 Generate AI Interview Guide & Q&A Report", type="primary", use_container_width=True):
            # Interactive Multi-Agent Processing Indicator Timeline
            progress_bar = st.progress(0)
            status_box = st.status("🤖 **Multi-Agent Processing Initialized...**", expanded=True)

            try:
                with status_box:
                    # Agent Step 1: RAG Retrieval
                    st.write("🔍 **Step 1/3: RAG Retriever** searching ChromaDB vector store (`all-MiniLM-L6-v2` embeddings)...")
                    progress_bar.progress(33)
                    time.sleep(0.4)

                    # Agent Step 2: Interviewer Agent
                    st.write(f"🧠 **Step 2/3: Interviewer Agent** generating guide overview for `{role}` (`{itype}` round) via Groq API...")
                    guide_data = generate_interview_guide(target_role=role, interview_type=itype)
                    progress_bar.progress(66)
                    time.sleep(0.4)

                    # Agent Step 3: Coach & Evaluator Agents
                    st.write("💎 **Step 3/3: Coach & Evaluator Agents** formulating exemplar model answers & scoring rubrics...")
                    qa_report = generate_demo_qa_report(target_role=role, interview_type=itype)
                    progress_bar.progress(100)
                    time.sleep(0.3)

                    status_box.update(label="✅ **Multi-Agent Report Successfully Synthesized!**", state="complete", expanded=False)

                st.session_state.guide_data = guide_data
                st.session_state.qa_report = qa_report
                st.session_state.report_generated = True
                st.rerun()

            except Exception as e:
                status_box.update(label="❌ **Generation Failed**", state="error", expanded=True)
                st.error(f"⚠️ Error during multi-agent processing: {str(e)}. Please check your API keys.")


def render_guide_and_report():
    """Render High-Aesthetic Glassmorphism Guide Card and Demo Q&A Report Cards."""
    role = st.session_state.selected_role
    itype = st.session_state.selected_type
    guide = st.session_state.guide_data or {}
    qa_report = st.session_state.qa_report or []

    # Hero Header Banner
    st.markdown(f"""
    <div class="hero-container">
        <div class="hero-title">🎓 Interview Master Preparation Report</div>
        <div style="margin-top: 0.8rem;">
            <span class="badge-role">Role: {role}</span> &nbsp;&nbsp;
            <span class="badge-type">Round: {itype} Interview</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # SECTION 1: RAG-Grounded Interview Guide
    st.markdown('<div class="section-header">📚 Step 3: RAG-Grounded Preparation Guide</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="glass-card">
        <h4 style="color: #38BDF8; margin-top: 0;">🔍 Overview & Expectations</h4>
        <p style="color: #CBD5E1; font-size: 1.05rem; line-height: 1.6;">
            {guide.get('overview', '')}
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #38BDF8; margin-top: 0;">🎯 Key Question Themes</h4>
        """, unsafe_allow_html=True)
        themes = guide.get("question_themes", [])
        for theme in themes:
            st.markdown(f'<div class="theme-item">⚡ {theme}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_g2:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #C084FC; margin-top: 0;">💡 Behavioral & Etiquette Tips</h4>
        """, unsafe_allow_html=True)
        tips = guide.get("behavior_tips", [])
        for tip in tips:
            st.markdown(f'<div class="tip-item">✨ {tip}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # SECTION 2: Demo Questions & Model Answers Report
    st.markdown('<div class="section-header">📋 Step 4: Demo Questions & Ideal Model Answers Report</div>', unsafe_allow_html=True)
    st.caption("Exemplar questions, model responses, scoring criteria, and coaching strategies synthesized by the agentic pipeline.")

    if qa_report:
        for idx, item in enumerate(qa_report, start=1):
            with st.expander(f"📌 Question {idx}: {item.get('question')}", expanded=(idx == 1)):
                st.markdown(f"""
                <div class="model-answer-box">
                    <strong style="color: #34D399; font-size: 1.05rem;">✨ Ideal Model Answer:</strong><br><br>
                    {item.get('model_answer')}
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                col_e1, col_e2 = st.columns(2)

                with col_e1:
                    st.markdown(f"""
                    <div class="criteria-box">
                        <strong style="color: #FBBF24;">🔍 Key Evaluation Criteria:</strong><br><br>
                        {item.get('evaluation_criteria', 'N/A')}
                    </div>
                    """, unsafe_allow_html=True)

                with col_e2:
                    st.markdown(f"""
                    <div class="tips-box">
                        <strong style="color: #38BDF8;">💡 Coach Tips & Best Practices:</strong><br><br>
                        {item.get('coaching_tips', 'N/A')}
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.warning("No sample Q&A report items available.")

    st.markdown("<br><hr>", unsafe_allow_html=True)
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("🔄 Select Another Job Role / Category", type="primary", use_container_width=True):
            reset_session()
            st.rerun()


def main():
    """Main Application Controller."""
    init_session()
    render_sidebar()

    if not st.session_state.report_generated:
        render_role_selection()
    else:
        render_guide_and_report()


if __name__ == "__main__":
    main()
