"""
Interviewer Agent Module for Interview Preparation Coach.

Analyzes candidate role and interview category selections, queries RAG retriever
for domain-specific preparation guidelines, generates pre-interview preparation guides,
and constructs dynamic, role-tailored interview questions.
Imports call_groq from graph.model_router and get_relevant_chunks from rag.retriever.
"""

import json
from typing import Any, Dict, List, Optional
from graph.model_router import call_groq
from rag.retriever import get_relevant_chunks


def generate_interview_guide(target_role: str, interview_type: str) -> Dict[str, Any]:
    """
    Generate a retrieval-grounded Interview Guide prior to starting the Q&A session.
    
    :param target_role: Candidate's selected job role (e.g. "Software Engineering", "DevOps").
    :param interview_type: Candidate's selected interview round (e.g. "Technical", "HR", "Coding").
    :return: Dict containing overview, question_themes, and behavior_tips.
    """
    # Query RAG retriever for role + interview type guidelines
    query = f"{interview_type} interview {target_role} structure expectations questions behavior guidelines"
    retrieved_chunks = get_relevant_chunks(query=query, top_k=4)

    context_str = "\n---\n".join([c.get("text", "") for c in retrieved_chunks]) if retrieved_chunks else "No specific ground truth document retrieved."

    prompt = f"""You are an expert AI Interview Coach. Generate a comprehensive "Interview Preparation Guide" for a candidate applying for the role of '{target_role}' in an '{interview_type}' interview round.

Reference Knowledge Base Context:
{context_str}

Generate a structured guide tailored specifically to {target_role} ({interview_type} round).

Output valid JSON strictly with this schema:
{{
  "overview": "Detailed overview explaining what to expect in a {interview_type} interview for {target_role}, including typical stages, duration (e.g., 45-60 mins), and interviewer expectations.",
  "question_themes": [
    "Theme 1 grounded in reference context (e.g., Core OOP abstractions and class hierarchies)",
    "Theme 2 grounded in reference context (e.g., System performance & complexity analysis)",
    "Theme 3 grounded in reference context (e.g., Practical scenario problem-solving)"
  ],
  "behavior_tips": [
    "Behavioral/Etiquette tip 1 (e.g., Use the STAR method for scenario questions)",
    "Communication tip 2 (e.g., 'Think aloud' and articulate your trade-offs clearly)",
    "Pacing tip 3 (e.g., Ask clarifying questions before diving into code or solutions)"
  ]
}}

Format requirements:
1. 'overview' MUST be 2-3 informative sentences.
2. 'question_themes' MUST contain 3 to 5 specific themes grounded in the knowledge context.
3. 'behavior_tips' MUST contain 3 to 5 actionable etiquette, tone, or structural tips.
4. Output ONLY valid JSON, no markdown code block backticks or conversational text.
"""

    raw_response = call_groq(
        prompt=prompt,
        model_name="llama-3.1-8b-instant",
        system_prompt="You are an expert AI Interviewer. Output only JSON.",
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
            parsed_guide = json.loads(content.strip())
            return {
                "overview": parsed_guide.get("overview", f"In a {interview_type} interview for {target_role}, expect 45-60 minutes evaluating technical competency, problem-solving, and communication."),
                "question_themes": parsed_guide.get("question_themes", [
                    f"Core {target_role} fundamentals and architecture",
                    "Practical scenario-based troubleshooting",
                    "Performance optimization and trade-offs"
                ]),
                "behavior_tips": parsed_guide.get("behavior_tips", [
                    "Listen carefully and ask clarifying questions before answering.",
                    "Structure your answers clearly using the STAR framework when appropriate.",
                    "Think aloud to explain your decision-making process."
                ])
            }
        except Exception as e:
            print(f"[InterviewerAgent Warning] Guide JSON parse error: {e}")

    # Grounded fallback guide
    fallback_guide = {
        "overview": f"In a {interview_type} interview for {target_role}, you will typically face a 45-60 minute session conducted by senior engineering or hiring managers evaluating technical proficiency, analytical thinking, and cultural fit.",
        "question_themes": [
            f"Core {target_role} domain principles and technical concepts",
            "Real-world application trade-offs and complexity analysis",
            "Incident response, debugging, and system design patterns"
        ],
        "behavior_tips": [
            "Use the STAR method (Situation, Task, Action, Result) for behavioral questions.",
            "For technical and coding rounds, 'think aloud' to share your problem-solving logic.",
            "Maintain a confident, collaborative tone and verify assumptions early."
        ]
    }
    return fallback_guide


def generate_interview_question(
    target_role: str = "Software Engineering",
    interview_type: str = "Technical",
    user_input: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Generate a realistic, role-tailored interview question using explicit target_role and interview_type.
    
    :param target_role: Target job position (e.g., "Software Engineering", "DevOps").
    :param interview_type: Category of interview (e.g., "Technical", "HR", "Coding").
    :param user_input: Optional candidate preference or prompt string.
    :param conversation_history: List of past Q&A turns.
    :return: Dict containing interview_type, target_role, and current_question.
    """
    history_str = ""
    if conversation_history:
        history_str = "\n".join([f"Q: {q.get('question')}\nA: {q.get('answer')}" for q in conversation_history])

    prompt = f"""You are an expert AI Job Interviewer.

Target Role: "{target_role}"
Interview Category: "{interview_type}"
Candidate Preferences / Input: "{user_input or 'Standard interview setup'}"
Previous Q&A History in Session:
{history_str or 'None'}

Generate an engaging, challenging, role-appropriate interview question for a {target_role} position in a {interview_type} round.

Output valid JSON strictly with this schema:
{{
  "interview_type": "{interview_type}",
  "target_role": "{target_role}",
  "current_question": "Can you explain the difference between an abstract class and an interface, and when you would choose one over the other?"
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
                "interview_type": parsed.get("interview_type", interview_type),
                "target_role": parsed.get("target_role", target_role),
                "current_question": parsed.get("current_question", f"What are the core technical responsibilities of a {target_role}?")
            }
        except Exception as e:
            print(f"[InterviewerAgent Warning] Question JSON parse error: {e}")

    # Fallback question tailored to role and interview type
    fallback_q = f"What are the key technical principles and design trade-offs involved in {target_role}?"
    itype_lower = interview_type.lower()
    role_lower = target_role.lower()

    if "coding" in itype_lower:
        fallback_q = "Given an array of integers and a target sum, how would you find the indices of the two numbers that add up to the target in O(n) time?"
    elif "hr" in itype_lower or "behavioral" in itype_lower:
        fallback_q = "Tell me about a challenging technical setback you faced in a project and how you resolved it using the STAR method."
    elif "devops" in role_lower:
        fallback_q = "Can you explain how a CI/CD pipeline automates build, test, and deployment stages, and how Docker and Kubernetes fit in?"
    elif "business" in role_lower:
        fallback_q = "How do you bridge the gap between technical developers and non-technical business stakeholders when gathering requirements?"

    return {
        "interview_type": interview_type,
        "target_role": target_role,
        "current_question": fallback_q
    }


class InterviewerAgent:
    """
    Stateful Interviewer Agent wrapper class.
    """

    def __init__(self, model_name: str = "llama-3.1-8b-instant") -> None:
        self.model_name = model_name

    def generate_guide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        State-driven interview guide generation.
        """
        role = state.get("target_role") or state.get("role", "Software Engineering")
        itype = state.get("interview_type", "Technical")
        return generate_interview_guide(target_role=role, interview_type=itype)

    def generate_question(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        State-driven question generation.
        """
        role = state.get("target_role") or state.get("role", "Software Engineering")
        itype = state.get("interview_type", "Technical")
        user_input = state.get("user_answer") or f"Preparing for {role} {itype} interview."
        return generate_interview_question(
            target_role=role,
            interview_type=itype,
            user_input=user_input
        )
