"""
Command-Line End-to-End Test Script for Interview Preparation Coach.

Simulates 2-3 turns of the mock interview pipeline using hardcoded candidate answers,
printing intermediate state transformations after each turn to verify graph execution.
"""

import json
import sys
import io
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

load_dotenv()

from agents.interviewer_agent import generate_interview_question
from graph.orchestrator import run_interview_step, InterviewState



def run_e2e_simulation():
    """Execute end-to-end simulation across 3 turns."""
    print("=" * 70, flush=True)
    print("🚀 STARTING E2E INTERVIEW AGENT FLOW SIMULATION", flush=True)
    print("=" * 70, flush=True)

    # Turn 0: Initialization
    init_res = generate_interview_question(
        user_input="I am preparing for a Software Engineer interview, ask me technical questions.",
        target_role="Software Engineer",
        interview_type="Technical"
    )

    state: InterviewState = {
        "interview_type": init_res.get("interview_type", "Technical"),
        "target_role": init_res.get("target_role", "Software Engineer"),
        "current_question": init_res.get("current_question", "What is the difference between an abstract class and an interface?"),
        "user_answer": "",
        "retrieved_context": [],
        "evaluation": {},
        "feedback": {},
        "question_count": 1,
        "is_complete": False
    }

    print(f"\n[Turn 1 Initial Question]: {state['current_question']}", flush=True)

    fake_answers = [
        "An abstract class allows default method implementations and fields, while an interface defines contracts for multiple inheritance.",
        "The STAR method stands for Situation, Task, Action, and Result. You use it to structure behavioral responses cleanly.",
        "For Two Sum, I use a hash map to store seen elements and check for target - current in O(n) time."
    ]

    for turn_idx, answer in enumerate(fake_answers, start=1):
        print(f"\n" + "-" * 60, flush=True)
        print(f"📍 TURN {turn_idx} EXECUTION", flush=True)
        print(f"Candidate Answer: {answer}", flush=True)

        state = run_interview_step(state, user_answer=answer)

        print("\n--- Evaluation Result ---", flush=True)
        print(json.dumps(state.get("evaluation", {}), indent=2), flush=True)

        print("\n--- Coaching Feedback ---", flush=True)
        fb = state.get("feedback", {})
        print("Strengths:", fb.get("strengths", []), flush=True)
        print("Gaps:", fb.get("gaps", []), flush=True)
        print("Suggestions:", fb.get("suggestions", []), flush=True)

        print(f"\n--- Context Snippets Retrieved ({len(state.get('retrieved_context', []))}) ---", flush=True)
        for c in state.get("retrieved_context", []):
            src = c.get("source", "unknown") if isinstance(c, dict) else "unknown"
            txt = c.get("text", str(c))[:100] if isinstance(c, dict) else str(c)[:100]
            print(f"  • Source [{src}]: {txt}...", flush=True)

        print(f"\nNext Question ({state.get('question_count')}/5): {state.get('current_question')}", flush=True)
        if state.get("is_complete"):
            print("Session Complete Flag: True", flush=True)
            break

    print("\n" + "=" * 70, flush=True)
    print("✅ E2E SIMULATION COMPLETED SUCCESSFULLY", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    run_e2e_simulation()
