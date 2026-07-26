"""
Interview Preparation Coach - Main Streamlit UI Application.

Interactive multi-agent mock interview platform using Streamlit session state,
LangGraph orchestration, Groq, OpenRouter, and ChromaDB vector retrieval.
"""

import sys
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


from graph.orchestrator import run_interview_step, InterviewState
from agents.interviewer_agent import generate_interview_question


# Page Configuration
st.set_page_config(
    page_title="Interview Preparation Coach",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .card-box {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
    .strength-item {
        color: #15803D;
        font-weight: 500;
        margin-bottom: 0.3rem;
    }
    .gap-item {
        color: #B91C1C;
        font-weight: 500;
        margin-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session():
    """Initialize Streamlit session state variables."""
    if "session_started" not in st.session_state:
        st.session_state.session_started = False
    if "state" not in st.session_state:
        st.session_state.state = {
            "interview_type": "Technical",
            "target_role": "Software Engineer",
            "current_question": "",
            "user_answer": "",
            "retrieved_context": [],
            "evaluation": {},
            "feedback": {},
            "question_count": 0,
            "is_complete": False
        }
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = None


def reset_session():
    """Reset mock interview session."""
    st.session_state.session_started = False
    st.session_state.state = {
        "interview_type": "Technical",
        "target_role": "Software Engineer",
        "current_question": "",
        "user_answer": "",
        "retrieved_context": [],
        "evaluation": {},
        "feedback": {},
        "question_count": 0,
        "is_complete": False
    }
    st.session_state.history = []
    st.session_state.last_feedback = None


def render_sidebar():
    """Render sidebar configuration options."""
    with st.sidebar:
        st.header("⚙️ Session Config")
        st.write("University Agentic AI Assignment")
        st.markdown("---")
        
        st.subheader("🤖 Agent Routing Architecture")
        st.markdown("""
        - **Interviewer Agent**: Groq (`llama-3.1-8b-instant`)
        - **Evaluator Agent**: Groq (`llama-3.1-8b-instant`) + ChromaDB RAG
        - **Coach Agent**: OpenRouter (`llama-3.3-70b-instruct`)
        """)
        st.markdown("---")

        if st.session_state.session_started:
            if st.button("🔄 Restart Interview", use_container_width=True):
                reset_session()
                st.rerun()


def render_landing_view():
    """Render landing page for candidate interview setup."""
    st.markdown('<div class="main-title">🎓 Interview Preparation Coach</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">AI-Powered Multi-Agent Mock Interview & Real-Time Coaching System</div>', unsafe_allow_html=True)

    with st.container():
        st.subheader("Setup Your Mock Interview")
        col1, col2 = st.columns(2)

        with col1:
            target_role = st.selectbox(
                "Select Target Role",
                ["Software Engineer", "DevOps Engineer", "Data Scientist", "Frontend Developer", "Backend Developer", "System Architect", "Product Manager"]
            )
            custom_role = st.text_input("Or type custom target role:", placeholder="e.g. ML Platform Engineer")
            if custom_role.strip():
                target_role = custom_role.strip()

        with col2:
            interview_type = st.selectbox(
                "Select Interview Category",
                ["Technical", "Behavioral", "Coding", "HR"]
            )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Start Mock Interview", type="primary", use_container_width=True):
            try:
                with st.spinner("Initializing AI Interviewer Agent..."):
                    res = generate_interview_question(
                        user_input=f"Preparing for {target_role} {interview_type} interview.",
                        target_role=target_role,
                        interview_type=interview_type
                    )
                    st.session_state.state["interview_type"] = res.get("interview_type", interview_type)
                    st.session_state.state["target_role"] = res.get("target_role", target_role)
                    st.session_state.state["current_question"] = res.get("current_question", "Explain the difference between an abstract class and an interface.")
                    st.session_state.state["question_count"] = 1
                    st.session_state.session_started = True
                    st.rerun()
            except Exception as e:
                st.error(f"⚠️ Failed to start interview: {str(e)}. Please verify your API keys.")


def render_main_interview_loop():
    """Render active interview question, answer submission, and feedback cards."""
    state = st.session_state.state
    count = state.get("question_count", 1)

    st.markdown('<div class="main-title">🎓 Interview Preparation Coach</div>', unsafe_allow_html=True)
    
    # Progress Indicator
    progress_val = min(count / 5.0, 1.0)
    st.progress(progress_val)
    st.caption(f"**Question {count} of 5** | Role: `{state.get('target_role')}` | Category: `{state.get('interview_type')}`")

    st.markdown("---")

    # Current Question Card
    st.info(f"### ❓ Question {count}:\n{state.get('current_question')}")

    # Answer Input Form
    with st.form(key=f"answer_form_{count}"):
        user_answer = st.text_area(
            "Your Answer:",
            height=160,
            placeholder="Type your detailed answer here... (You can also type 'end interview' to exit early)"
        )
        submit_btn = st.form_submit_button("📤 Submit Answer", type="primary", use_container_width=True)

    if submit_btn:
        if not user_answer.strip():
            st.warning("Please enter an answer before submitting.")
        else:
            try:
                with st.spinner("Agents evaluating response and generating coaching feedback..."):
                    updated_state = run_interview_step(state, user_answer.strip())

                    # Save turn to history
                    st.session_state.history.append({
                        "question": state.get("current_question"),
                        "answer": user_answer,
                        "evaluation": updated_state.get("evaluation", {}),
                        "feedback": updated_state.get("feedback", {})
                    })

                    st.session_state.last_feedback = updated_state.get("feedback", {})
                    st.session_state.state = updated_state

                    if updated_state.get("is_complete"):
                        st.session_state.state["is_complete"] = True

                    st.rerun()

            except Exception as e:
                st.error(f"⚠️ An error occurred during evaluation: {str(e)}. Don't worry, your progress is saved!")

    # Display Feedback for Previous Turn if available
    if st.session_state.last_feedback and st.session_state.history:
        last_turn = st.session_state.history[-1]
        st.markdown("---")
        st.subheader("💡 Latest AI Coach Feedback")

        eval_data = last_turn.get("evaluation", {})
        fb_data = last_turn.get("feedback", {})

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Correctness", eval_data.get("correctness", "N/A"))
        col_b.metric("Clarity", eval_data.get("clarity", "N/A"))
        col_c.metric("Completeness", eval_data.get("completeness", "N/A"))

        tab1, tab2, tab3 = st.tabs(["✓ Strengths", "✗ Areas to Improve", "📌 Coaching Recommendations"])

        with tab1:
            strengths = fb_data.get("strengths", [])
            if strengths:
                for s in strengths:
                    st.markdown(f'<div class="strength-item">{s}</div>', unsafe_allow_html=True)
            else:
                st.write("✓ Good effort in answering the prompt.")

        with tab2:
            gaps = fb_data.get("gaps", [])
            if gaps:
                for g in gaps:
                    st.markdown(f'<div class="gap-item">{g}</div>', unsafe_allow_html=True)
            else:
                st.write("✗ Focus on adding more concrete examples.")

        with tab3:
            suggs = fb_data.get("suggestions", [])
            if suggs:
                for sug in suggs:
                    st.write(f"- {sug}")
            else:
                st.write("- Keep practicing structured responses.")


def render_summary_screen():
    """Render completion scorecard summary after 5 questions."""
    st.markdown('<div class="main-title">🎉 Session Complete!</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Congratulations on completing your mock interview session.</div>', unsafe_allow_html=True)

    state = st.session_state.state
    st.success(f"**Mock Interview Summary** | Role: `{state.get('target_role')}` | Category: `{state.get('interview_type')}`")

    st.markdown("### 📋 Response Scorecard")
    for idx, turn in enumerate(st.session_state.history, start=1):
        with st.expander(f"Question {idx}: {turn.get('question')}", expanded=(idx == 1)):
            st.write(f"**Your Answer:** {turn.get('answer')}")
            eval_data = turn.get("evaluation", {})
            fb_data = turn.get("feedback", {})

            st.write(f"**Scores:** Correctness: `{eval_data.get('correctness')}` | Clarity: `{eval_data.get('clarity')}` | Completeness: `{eval_data.get('completeness')}`")
            st.write(f"**Evaluator Notes:** {eval_data.get('notes')}")

            st.markdown("**Key Feedback:**")
            for str_item in fb_data.get("strengths", []):
                st.write(str_item)
            for gap_item in fb_data.get("gaps", []):
                st.write(gap_item)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Start New Mock Interview", type="primary", use_container_width=True):
        reset_session()
        st.rerun()


def main():
    """Main Streamlit Application Controller."""
    init_session()
    render_sidebar()

    state = st.session_state.state

    if not st.session_state.session_started:
        render_landing_view()
    elif state.get("is_complete") or state.get("question_count", 0) > 5:
        render_summary_screen()
    else:
        render_main_interview_loop()


if __name__ == "__main__":
    main()
