"""
Interviewer Agent Module for Interview Preparation Coach.

Analyzes candidate preferences, routes interview type (HR/Technical/Coding) and 
target role, and generates dynamic, role-specific interview questions.
Imports call_groq from graph.model_router for fast LLM inference.
"""

import json
from typing import Any, Dict, List, Optional
from graph.model_router import call_groq


def generate_interview_question(
    user_input: str = "I am preparing for a Software Engineer interview, ask me technical questions.",
    target_role: Optional[str] = None,
    interview_type: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Extract interview parameters and generate an appropriate interview question.
    
    :param user_input: Raw candidate request or target preference string.
    :param target_role: Optional explicit target role string.
    :param interview_type: Optional explicit interview category.
    :param conversation_history: List of past Q&A turns.
    :return: Dict containing interview_type, target_role, and current_question.
    """
    history_str = ""
    if conversation_history:
        history_str = "\n".join([f"Q: {q.get('question')}\nA: {q.get('answer')}" for q in conversation_history])

    prompt = f"""You are an expert AI Job Interviewer.

User Input: "{user_input}"
Target Role Hint: "{target_role or 'Not specified'}"
Interview Type Hint: "{interview_type or 'Not specified'}"
Previous Q&A History:
{history_str or 'None'}

Perform two tasks:
1. Identify or refine the 'interview_type' (must be one of: "Technical", "Behavioral", "Coding", or "HR").
2. Identify or refine the 'target_role' (e.g., "Software Engineer", "DevOps Engineer", "Data Scientist").
3. Generate a realistic, challenging, role-tailored interview question.

Output valid JSON strictly with this schema:
{{
  "interview_type": "Technical",
  "target_role": "Software Engineer",
  "current_question": "Can you explain the difference between an abstract class and an interface, and when you would use each?"
}}
"""

    raw_response = call_groq(
        prompt=prompt,
        model_name="llama-3.1-8b-instant",
        system_prompt="You are an AI Interviewer. Output only JSON.",
        json_mode=True
    )

    if raw_response:
        try:
            content = raw_response.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            parsed = json.loads(content.strip())
            return {
                "interview_type": parsed.get("interview_type", interview_type or "Technical"),
                "target_role": parsed.get("target_role", target_role or "Software Engineer"),
                "current_question": parsed.get("current_question", "What is the STAR method and how do you apply it in interviews?")
            }
        except Exception as e:
            print(f"[InterviewerAgent Warning] JSON parse error: {e}")

    # Robust fallback questions tailored to context
    role = target_role or "Software Engineer"
    itype = interview_type or "Technical"
    fallback_q = "Can you explain the difference between an abstract class and an interface in OOP?"
    if "behavioral" in itype.lower() or "hr" in itype.lower():
        fallback_q = "Tell me about a challenging situation you faced in a project and how you resolved it using the STAR method."
    elif "coding" in itype.lower():
        fallback_q = "How would you design an algorithm to solve the Two Sum problem in O(n) time complexity?"

    return {
        "interview_type": itype,
        "target_role": role,
        "current_question": fallback_q
    }


class InterviewerAgent:
    """
    Stateful Interviewer Agent wrapper class.
    """

    def __init__(self, model_name: str = "llama-3.1-8b-instant") -> None:
        self.model_name = model_name

    def generate_question(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        State-driven question generation.
        """
        user_input = state.get("user_answer") or f"Preparing for {state.get('target_role', 'Software Engineer')} {state.get('interview_type', 'Technical')} interview."
        return generate_interview_question(
            user_input=user_input,
            target_role=state.get("target_role"),
            interview_type=state.get("interview_type")
        )
