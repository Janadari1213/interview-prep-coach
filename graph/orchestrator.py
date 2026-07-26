"""
LangGraph Orchestrator Module for Interview Preparation Coach.

Defines the state schema and StateGraph workflow connecting Interviewer,
Evaluator, and Coach agents. Manages multi-question session loops, context updates,
and turn-by-turn execution via run_interview_step().
"""

from typing import Any, Dict, List, TypedDict, Optional
from langgraph.graph import StateGraph, START, END

from agents.interviewer_agent import generate_interview_question
from agents.evaluator_agent import evaluate_candidate_answer
from agents.coach_agent import generate_coaching_feedback


class InterviewState(TypedDict, total=False):
    """
    State schema for the LangGraph interview orchestration workflow.
    """
    interview_type: str
    target_role: str
    current_question: str
    user_answer: str
    retrieved_context: List[Any]
    evaluation: Dict[str, Any]
    feedback: Dict[str, Any]
    question_count: int
    is_complete: bool


# Node Definitions
def interviewer_node(state: InterviewState) -> Dict[str, Any]:
    """
    Node 1: Generate next role-tailored interview question.
    """
    count = state.get("question_count", 0)
    res = generate_interview_question(
        user_input=state.get("user_answer", ""),
        target_role=state.get("target_role"),
        interview_type=state.get("interview_type")
    )
    return {
        "interview_type": res.get("interview_type", state.get("interview_type", "Technical")),
        "target_role": res.get("target_role", state.get("target_role", "Software Engineer")),
        "current_question": res.get("current_question", ""),
        "question_count": count + 1
    }


def evaluator_node(state: InterviewState) -> Dict[str, Any]:
    """
    Node 2: Retrieve RAG context and evaluate user's response.
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
    Conditional edge router: Check session termination criteria.
    Terminates if question_count >= 5 or if user typed 'end interview' / 'exit'.
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

    builder.add_node("interviewer_node", interviewer_node)
    builder.add_node("evaluator_node", evaluator_node)
    builder.add_node("coach_node", coach_node)

    # Wire Edges: START -> interviewer -> evaluator -> coach
    builder.add_edge(START, "interviewer_node")
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


def run_interview_step(state: InterviewState, user_answer: str) -> InterviewState:
    """
    Execute a single turn of the interview pipeline.
    
    1. Updates state with candidate's user_answer.
    2. Evaluates response via evaluator_node.
    3. Generates feedback via coach_node.
    4. Advances to next question via interviewer_node if session continues.
    
    :param state: Current InterviewState dictionary.
    :param user_answer: Candidate's latest response text.
    :return: Updated InterviewState dictionary.
    """
    # Copy current state
    updated_state = dict(state)
    updated_state["user_answer"] = user_answer

    # Step A: Evaluate candidate's answer
    eval_res = evaluator_node(updated_state)
    updated_state.update(eval_res)

    # Step B: Generate coaching feedback
    coach_res = coach_node(updated_state)
    updated_state.update(coach_res)

    # Step C: Increment question count & check termination
    count = updated_state.get("question_count", 1)
    
    if count >= 5 or user_answer.strip().lower() in ["end interview", "exit", "quit", "end"]:
        updated_state["is_complete"] = True
    else:
        # Step D: Generate next question for next turn
        next_q_res = interviewer_node(updated_state)
        updated_state["current_question"] = next_q_res.get("current_question", "")
        updated_state["question_count"] = next_q_res.get("question_count", count + 1)
        updated_state["is_complete"] = False

    return updated_state
