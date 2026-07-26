"""
Interview Preparation Coach - Modern Full-Width Streamlit UI Application.

Features a full-width layout without sidebar, top-left navigation controls, 
interactive multi-tab preparation reports, and animated multi-agent status indicators.
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


# Page Configuration - Collapsed Sidebar & Full-Width Layout
st.set_page_config(
    page_title="AI Interview Prep Coach",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom High-Aesthetic Glassmorphism CSS Design System
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap" rel="stylesheet">

<style>
    /* Completely Hide Left Sidebar */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Expand Main Container Margins */
    .main .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 95% !important;
    }

    /* Global Typography */
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
        margin-bottom: 1.8rem;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
    }
    .hero-title {
        font-size: 2.7rem;
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
        background: rgba(30, 41, 59, 0.75);
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
        padding: 0.4rem 1.1rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.95rem;
        box-shadow: 0 4px 6px -1px rgba(2, 132, 199, 0.3);
        display: inline-block;
    }
    .badge-type {
        background: linear-gradient(135deg, #7C3AED 0%, #6D28D9 100%);
        color: #FFFFFF;
        padding: 0.4rem 1.1rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.95rem;
        box-shadow: 0 4px 6px -1px rgba(124, 58, 237, 0.3);
        display: inline-block;
    }

    /* Theme & Tip Item Boxes */
    .theme-item {
        background: rgba(15, 23, 42, 0.7);
        border-left: 4px solid #38BDF8;
        padding: 0.9rem 1.2rem;
        border-radius: 0 12px 12px 0;
        margin-bottom: 0.7rem;
        color: #E2E8F0;
        font-weight: 500;
        font-size: 1.02rem;
    }
    .tip-item {
        background: rgba(15, 23, 42, 0.7);
        border-left: 4px solid #C084FC;
        padding: 0.9rem 1.2rem;
        border-radius: 0 12px 12px 0;
        margin-bottom: 0.7rem;
        color: #E2E8F0;
        font-weight: 500;
        font-size: 1.02rem;
    }

    /* Model Answer Container */
    .model-answer-box {
        background: rgba(6, 78, 59, 0.28);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 12px;
        padding: 1.3rem;
        color: #ECFDF5;
        font-size: 1.02rem;
        line-height: 1.65;
        margin-top: 0.5rem;
    }

    /* Criteria & Advice Containers */
    .criteria-box {
        background: rgba(120, 53, 15, 0.28);
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-radius: 12px;
        padding: 1.2rem;
        color: #FEF3C7;
        font-size: 0.98rem;
    }
    .tips-box {
        background: rgba(12, 74, 110, 0.28);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 12px;
        padding: 1.2rem;
        color: #E0F2FE;
        font-size: 0.98rem;
    }

    /* Streamlit UI Element Styling */
    .stSelectbox label {
        color: #CBD5E1 !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
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
    """Reset session state to return to role selection."""
    st.session_state.report_generated = False
    st.session_state.guide_data = None
    st.session_state.qa_report = None


def render_role_selection():
    """Render Hero Header, Role & Category Selection, and Interactive Multi-Agent Trigger."""
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🎓 AI Interview Preparation Coach</div>
        <div class="hero-subtitle">Multi-Agent System for Retrieval-Grounded Guides & Exemplar Q&A Scorecard Generation</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("### 🎯 Select Target Role & Interview Stage")
        col1, col2 = st.columns(2)

        with col1:
            role = st.selectbox(
                "📌 Step 1: Select Target Job Position",
                options=JOB_ROLES,
                index=0,
                key="role_select",
                help="Select your career position to customize knowledge base retrieval."
            )
            st.session_state.selected_role = role

        with col2:
            if role == "Business Analyst":
                available_types = ["HR", "Technical"]
                st.caption("ℹ️ *Coding round is automatically disabled for Business Analyst roles.*")
            else:
                available_types = ["HR", "Technical", "Coding"]

            itype = st.selectbox(
                "🎯 Step 2: Select Interview Round Category",
                options=available_types,
                index=0,
                key="type_select",
                help="Select the specific interview stage to prepare for."
            )
            st.session_state.selected_type = itype

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🚀 Generate AI Interview Guide & Q&A Report", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_box = st.status("🤖 **Multi-Agent Processing Initialized...**", expanded=True)

            try:
                with status_box:
                    # Agent Step 1: RAG Retrieval
                    st.write("🔍 **Step 1/3: RAG Retriever** querying ChromaDB vector store (`all-MiniLM-L6-v2` embeddings)...")
                    progress_bar.progress(33)
                    time.sleep(0.4)

                    # Agent Step 2: Interviewer Agent
                    st.write(f"🧠 **Step 2/3: Interviewer Agent** synthesizing domain guidelines for `{role}` (`{itype}` round) via Groq API...")
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
    """Render Top-Left Reset Navigation Button, Hero Header, and Interactive Multi-Tab Report Panels."""
    role = st.session_state.selected_role
    itype = st.session_state.selected_type
    guide = st.session_state.guide_data or {}
    qa_report = st.session_state.qa_report or []

    # TOP-LEFT NAVIGATION CONTROL ROW
    col_nav1, col_nav2 = st.columns([1.5, 4.5])
    with col_nav1:
        if st.button("← 🔄 Select New Job Role / Reset", type="primary", use_container_width=True):
            reset_session()
            st.rerun()

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

    # INTERACTIVE REPORT TABS
    tab_guide, tab_report, tab_architecture = st.tabs([
        "📚 Step 3: Preparation Guide & Strategy",
        "📋 Step 4: Demo Questions & Model Answers Report",
        "⚡ Agentic AI System Architecture"
    ])

    # TAB 1: RAG-Grounded Preparation Guide
    with tab_guide:
        st.markdown(f"""
        <div class="glass-card">
            <h4 style="color: #38BDF8; margin-top: 0; font-size: 1.3rem;">🔍 Overview & What to Expect</h4>
            <p style="color: #CBD5E1; font-size: 1.08rem; line-height: 1.65;">
                {guide.get('overview', '')}
            </p>
        </div>
        """, unsafe_allow_html=True)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("""
            <div class="glass-card">
                <h4 style="color: #38BDF8; margin-top: 0; font-size: 1.25rem;">🎯 Key Question Themes</h4>
            """, unsafe_allow_html=True)
            themes = guide.get("question_themes", [])
            for theme in themes:
                st.markdown(f'<div class="theme-item">⚡ {theme}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_g2:
            st.markdown("""
            <div class="glass-card">
                <h4 style="color: #C084FC; margin-top: 0; font-size: 1.25rem;">💡 Behavioral & Etiquette Tips</h4>
            """, unsafe_allow_html=True)
            tips = guide.get("behavior_tips", [])
            for tip in tips:
                st.markdown(f'<div class="tip-item">✨ {tip}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # TAB 2: Demo Questions & Ideal Model Answers Report
    with tab_report:
        st.markdown("### 📋 Exemplar Questions & Model Answers Scorecard")
        st.caption("Review these sample interview questions, ideal response structures, scoring criteria, and coaching advice.")

        if qa_report:
            for idx, item in enumerate(qa_report, start=1):
                with st.expander(f"📌 Question {idx}: {item.get('question')}", expanded=(idx == 1)):
                    st.markdown(f"""
                    <div class="model-answer-box">
                        <strong style="color: #34D399; font-size: 1.08rem;">✨ Ideal Model Answer:</strong><br><br>
                        {item.get('model_answer')}
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    col_e1, col_e2 = st.columns(2)

                    with col_e1:
                        st.markdown(f"""
                        <div class="criteria-box">
                            <strong style="color: #FBBF24; font-size: 1.02rem;">🔍 Key Evaluation Criteria:</strong><br><br>
                            {item.get('evaluation_criteria', 'N/A')}
                        </div>
                        """, unsafe_allow_html=True)

                    with col_e2:
                        st.markdown(f"""
                        <div class="tips-box">
                            <strong style="color: #38BDF8; font-size: 1.02rem;">💡 Coach Tips & Best Practices:</strong><br><br>
                            {item.get('coaching_tips', 'N/A')}
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.warning("No sample Q&A report items available.")

    # TAB 3: Agent Architecture & System Specs
    with tab_architecture:
        st.markdown("### ⚡ Multi-Agent System Routing & Model Specs")
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #38BDF8; margin-top: 0;">🤖 Agentic Pipeline Specifications</h4>
            <ul>
                <li><strong>RAG Context Retriever</strong>: ChromaDB local vector database with <code>sentence-transformers/all-MiniLM-L6-v2</code> embeddings.</li>
                <li><strong>Interviewer Agent</strong>: Groq API backing <code>llama-3.1-8b-instant</code> for rapid sub-second guide and question synthesis.</li>
                <li><strong>Evaluator Agent</strong>: Groq API for rapid rubric evaluation against retrieved vector chunks.</li>
                <li><strong>Coach Agent</strong>: OpenRouter API backing <code>meta-llama/llama-3.3-70b-instruct</code> for deep reflection and structured coaching advice.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main Application Controller."""
    init_session()

    if not st.session_state.report_generated:
        render_role_selection()
    else:
        render_guide_and_report()


if __name__ == "__main__":
    main()
