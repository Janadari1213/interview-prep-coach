"""
Interview Preparation Coach - Main Streamlit UI Application.

Interactive multi-agent platform delivering RAG-grounded Interview Guides 
and complete Demo Questions & Model Answers Reports using LangGraph orchestration, 
Groq, OpenRouter, and ChromaDB vector retrieval.
"""

import sys
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from graph.orchestrator import get_interview_guide


# Extensible Job Roles Constant
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
    page_title="Interview Preparation Coach",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .guide-box {
        background-color: #F8FAFC;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #CBD5E1;
        margin-bottom: 1.5rem;
    }
    .qa-card {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 1.25rem;
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-role {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-type {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
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
    """Render sidebar info."""
    with st.sidebar:
        st.header("⚙️ Session Config")
        st.write("University Agentic AI Assignment")
        st.markdown("---")
        
        st.subheader("🤖 Agent Architecture")
        st.markdown("""
        - **Interviewer Agent**: Groq (`llama-3.1-8b-instant`)
        - **RAG Context**: ChromaDB + Sentence Transformers (`all-MiniLM-L6-v2`)
        - **Evaluator Agent**: Groq (`llama-3.1-8b-instant`)
        - **Coach Agent**: OpenRouter (`llama-3.3-70b-instruct`)
        """)
        st.markdown("---")

        if st.session_state.report_generated:
            if st.button("🔄 Select New Role / Reset", use_container_width=True):
                reset_session()
                st.rerun()


def render_role_selection():
    """Render Step 1 (Role Selection) and Step 2 (Interview Type Selection)."""
    st.markdown('<div class="main-title">🎓 Interview Preparation Coach</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">AI-Powered Interview Guides & Model Answers Report Generator</div>', unsafe_allow_html=True)

    with st.container():
        st.subheader("Step 1 & 2: Select Job Role and Interview Category")
        col1, col2 = st.columns(2)

        with col1:
            role = st.selectbox(
                "📌 Step 1: Select Target Job Role",
                options=JOB_ROLES,
                index=0,
                key="role_select"
            )
            st.session_state.selected_role = role

        with col2:
            # Business Analyst excludes Coding round
            if role == "Business Analyst":
                available_types = ["HR", "Technical"]
                st.caption("ℹ️ *Note: 'Coding' round is disabled for Business Analyst roles.*")
            else:
                available_types = ["HR", "Technical", "Coding"]

            itype = st.selectbox(
                "🎯 Step 2: Select Interview Round Category",
                options=available_types,
                index=0 if "Technical" in available_types else 0,
                key="type_select"
            )
            st.session_state.selected_type = itype

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📖 Generate Interview Guide & Q&A Report", type="primary", use_container_width=True):
            try:
                with st.spinner(f"Retrieving RAG context & generating guide for {role} ({itype} round)..."):
                    res = get_interview_guide(role=role, interview_type=itype)
                    st.session_state.guide_data = res.get("interview_guide", {})
                    st.session_state.qa_report = res.get("demo_qa_report", [])
                    st.session_state.report_generated = True
                    st.rerun()
            except Exception as e:
                st.error(f"⚠️ Failed to generate guide: {str(e)}. Please check your API keys.")


def render_guide_and_report():
    """Render Step 3 (Interview Guide) and Step 4 (Demo Questions & Model Answers Report)."""
    role = st.session_state.selected_role
    itype = st.session_state.selected_type
    guide = st.session_state.guide_data or {}
    qa_report = st.session_state.qa_report or []

    st.markdown('<div class="main-title">🎓 Interview Preparation Report</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="badge-role">Role: {role}</span> &nbsp; <span class="badge-type">Category: {itype} Round</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Section 1: RAG-Grounded Interview Guide
    st.subheader("📚 Step 3: Interview Preparation Guide")
    with st.container():
        st.markdown('<div class="guide-box">', unsafe_allow_html=True)
        st.markdown(f"#### 🔍 Overview & What to Expect\n{guide.get('overview', '')}")
        st.markdown("---")

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("#### 🎯 Expected Question Themes")
            themes = guide.get("question_themes", [])
            for theme in themes:
                st.markdown(f"- **{theme}**")

        with col_g2:
            st.markdown("#### 💡 Behavioral & Etiquette Tips")
            tips = guide.get("behavior_tips", [])
            for tip in tips:
                st.markdown(f"- {tip}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Section 2: Demo Questions & Model Answers Report
    st.subheader("📋 Step 4: Demo Questions & Ideal Model Answers Report")
    st.caption("Review these sample interview questions, exemplar answers, scoring rubrics, and coaching advice.")

    if qa_report:
        for idx, item in enumerate(qa_report, start=1):
            with st.expander(f"Question {idx}: {item.get('question')}", expanded=(idx == 1)):
                st.info(f"**Ideal Model Answer:**\n\n{item.get('model_answer')}")

                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    st.markdown("##### 🔍 Key Evaluation Criteria")
                    st.write(item.get('evaluation_criteria', 'N/A'))

                with col_e2:
                    st.markdown("##### 💡 Coach Tips & Best Practices")
                    st.write(item.get('coaching_tips', 'N/A'))
    else:
        st.warning("No sample Q&A report items available.")

    st.markdown("<br><hr>", unsafe_allow_html=True)
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("🔄 Select Another Role or Category", use_container_width=True):
            reset_session()
            st.rerun()


def main():
    """Main Streamlit Application Controller."""
    init_session()
    render_sidebar()

    if not st.session_state.report_generated:
        render_role_selection()
    else:
        render_guide_and_report()


if __name__ == "__main__":
    main()
