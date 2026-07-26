"""
LangGraph Orchestrator Module for Interview Preparation Coach.

Defines the state schema and StateGraph workflow connecting Interviewer,
Evaluator, and Coach agents. Manages pre-interview guide generation, demo Q&A reports,
and multi-turn interview orchestration.
"""

from typing import Any, Dict, List, TypedDict, Optional
from langgraph.graph import StateGraph, START, END

from agents.interviewer_agent import (
    generate_interview_guide,
    generate_demo_qa_report,
    generate_interview_question
)
from agents.evaluator_agent import evaluate_candidate_answer
from agents.coach_agent import generate_coaching_feedback


class InterviewState(TypedDict, total=False):
    """
    State schema for the LangGraph interview orchestration workflow.
    """
    role: str
    target_role: str
    interview_type: str
    interview_guide: Dict[str, Any]
    demo_qa_report: List[Dict[str, Any]]
    current_question: str
    user_answer: str
    retrieved_context: List[Any]
    evaluation: Dict[str, Any]
    feedback: Dict[str, Any]
    question_count: int
    is_complete: bool


# Node Definitions
def guide_node(state: InterviewState) -> Dict[str, Any]:
    """
    Node 0: Generate retrieval-grounded Interview Guide and Demo Q&A Report.
    """
    role = state.get("target_role") or state.get("role", "Software Engineering")
    itype = state.get("interview_type", "Technical")

    guide = generate_interview_guide(target_role=role, interview_type=itype)
    qa_report = generate_demo_qa_report(target_role=role, interview_type=itype)

    return {
        "target_role": role,
        "role": role,
        "interview_type": itype,
        "interview_guide": guide,
        "demo_qa_report": qa_report
    }


def interviewer_node(state: InterviewState) -> Dict[str, Any]:
    """
    Node 1: Generate next role-tailored interview question.
    """
    role = state.get("target_role") or state.get("role", "Software Engineering")
    itype = state.get("interview_type", "Technical")
    count = state.get("question_count", 0)

    res = generate_interview_question(
        target_role=role,
        interview_type=itype,
        user_input=state.get("user_answer", "")
    )
    return {
        "interview_type": itype,
        "target_role": role,
        "role": role,
        "current_question": res.get("current_question", ""),
        "question_count": count + 1
    }


def evaluator_node(state: InterviewState) -> Dict[str, Any]:
    """
    Node 2: Retrieve RAG context and evaluate candidate's response.
    """
    q = state.get("current_question", "")
    a = state.get("user_answer", "")
    res = evaluate_candidate_answer(current_question=q, user_answer=a)
    return {
        "evaluation": res.get("evaluation", {}),
        "retrieved_context": res.get("retrieved_context", [])
    }


def coach_node(state: InterviewState) -> Dict[str, Any]:
    """
    Node 3: Generate reflection-based coaching feedback and actionable suggestions.
    """
    res = generate_coaching_feedback(
        current_question=state.get("current_question", ""),
        user_answer=state.get("user_answer", ""),
        retrieved_context=state.get("retrieved_context", []),
        evaluation=state.get("evaluation", {})
    )
    return {
        "feedback": res.get("feedback", {})
    }


def should_continue(state: InterviewState) -> str:
    """
    Conditional edge router checking termination criteria.
    """
    user_answer = str(state.get("user_answer", "")).strip().lower()
    count = state.get("question_count", 0)

    if count >= 5 or user_answer in ["end interview", "exit", "quit", "end"]:
        return "end_session"
    return "next_question"


def build_interview_graph():
    """
    Construct and compile the LangGraph state graph workflow.
    """
    builder = StateGraph(InterviewState)

    builder.add_node("guide_node", guide_node)
    builder.add_node("interviewer_node", interviewer_node)
    builder.add_node("evaluator_node", evaluator_node)
    builder.add_node("coach_node", coach_node)

    # Wire Edges: START -> guide_node -> interviewer_node -> evaluator_node -> coach_node
    builder.add_edge(START, "guide_node")
    builder.add_edge("guide_node", "interviewer_node")
    builder.add_edge("interviewer_node", "evaluator_node")
    builder.add_edge("evaluator_node", "coach_node")

    # Conditional edge after coach_node: loop to interviewer_node OR END
    builder.add_conditional_edges(
        "coach_node",
        should_continue,
        {
            "next_question": "interviewer_node",
            "end_session": END
        }
    )

    return builder.compile()


# Global compiled graph instance
interview_graph = build_interview_graph()


def get_interview_guide(role: str, interview_type: str) -> Dict[str, Any]:
    """
    Standalone runner executing guide_node to return the retrieval-grounded 
    Interview Guide and Demo Q&A Report.
    
    :param role: Target job role (e.g. "Software Engineering", "DevOps").
    :param interview_type: Selected interview round (e.g. "Technical", "HR", "Coding").
    :return: Dict containing interview_guide and demo_qa_report.
    """
    init_state: InterviewState = {
        "role": role,
        "target_role": role,
        "interview_type": interview_type
    }
    res = guide_node(init_state)
    return res


def run_interview_step(state: InterviewState, user_answer: str) -> InterviewState:
    """
    Execute a single turn of the mock interview Q&A pipeline.
    """
    updated_state = dict(state)
    updated_state["user_answer"] = user_answer

    eval_res = evaluator_node(updated_state)
    updated_state.update(eval_res)

    coach_res = coach_node(updated_state)
    updated_state.update(coach_res)

    count = updated_state.get("question_count", 1)

    if count >= 5 or user_answer.strip().lower() in ["end interview", "exit", "quit", "end"]:
        updated_state["is_complete"] = True
    else:
        next_q_res = interviewer_node(updated_state)
        updated_state["current_question"] = next_q_res.get("current_question", "")
        updated_state["question_count"] = next_q_res.get("question_count", count + 1)
        updated_state["is_complete"] = False

    return updated_state
