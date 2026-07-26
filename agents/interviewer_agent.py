"""
Interviewer Agent Module for Interview Preparation Coach.

Analyzes candidate role and interview category selections, queries RAG retriever
for domain-specific preparation guidelines, generates pre-interview preparation guides,
and constructs dynamic role-tailored questions along with complete demo Q&A reports.
Imports call_groq from graph.model_router and get_relevant_chunks from rag.retriever.
"""

import json
from typing import Any, Dict, List, Optional
from graph.model_router import call_groq
from rag.retriever import get_relevant_chunks


def generate_interview_guide(target_role: str, interview_type: str) -> Dict[str, Any]:
    """
    Generate a retrieval-grounded Interview Guide for the selected job role and interview round.
    
    :param target_role: Selected job role (e.g. "Software Engineering", "DevOps").
    :param interview_type: Selected interview round (e.g. "Technical", "HR", "Coding").
    :return: Dict containing overview, question_themes, and behavior_tips.
    """
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
                "overview": parsed_guide.get("overview", f"In a {interview_type} interview for {target_role}, expect a 45-60 minute session evaluating domain competency, problem-solving, and professional communication."),
                "question_themes": parsed_guide.get("question_themes", [
                    f"Core {target_role} domain fundamentals and architecture",
                    "Practical scenario-based troubleshooting and design trade-offs",
                    "System reliability, testing, and performance optimization"
                ]),
                "behavior_tips": parsed_guide.get("behavior_tips", [
                    "Listen carefully and clarify scope before answering.",
                    "Structure scenario responses using the STAR method (Situation, Task, Action, Result).",
                    "Think aloud to articulate your thought process and trade-offs."
                ])
            }
        except Exception as e:
            print(f"[InterviewerAgent Warning] Guide JSON parse error: {e}")

    # Fallback guide
    return {
        "overview": f"In a {interview_type} interview for {target_role}, you will typically face a 45-60 minute structured evaluation conducted by senior team members assessing domain knowledge, analytical problem-solving, and collaborative communication.",
        "question_themes": [
            f"Core {target_role} engineering principles and concepts",
            "Real-world application trade-offs and complexity analysis",
            "Debugging, architecture patterns, and operational practices"
        ],
        "behavior_tips": [
            "Use the STAR framework for behavioral or situational prompts.",
            "For technical and coding topics, 'think aloud' to share your reasoning.",
            "State assumptions explicitly and explain architectural decisions."
        ]
    }


def generate_demo_qa_report(target_role: str, interview_type: str) -> List[Dict[str, Any]]:
    """
    Generate a complete Demo Q&A Report featuring 3-4 exemplar questions, 
    ideal model answers, key evaluation criteria, and coaching tips.
    
    :param target_role: Selected job role (e.g., "Software Engineering", "DevOps").
    :param interview_type: Selected interview round (e.g., "Technical", "HR", "Coding").
    :return: List of dicts representing sample questions with ideal answers and coaching advice.
    """
    query = f"{interview_type} interview questions model answers evaluation criteria {target_role}"
    retrieved_chunks = get_relevant_chunks(query=query, top_k=4)

    context_str = "\n---\n".join([c.get("text", "") for c in retrieved_chunks]) if retrieved_chunks else "No specific ground truth document retrieved."

    prompt = f"""You are an expert AI Interview Coach and Hiring Manager. Generate a comprehensive Demo Questions & Model Answers Report for a '{target_role}' candidate in an '{interview_type}' interview round.

Reference Knowledge Base Context:
{context_str}

Generate 3 high-quality, realistic interview questions tailored to {target_role} ({interview_type} round). For EACH question, provide an ideal model answer, key evaluation criteria, and actionable coaching tips.

Output valid JSON strictly with this schema:
{{
  "qa_report": [
    {{
      "question": "Question 1 tailored to {target_role} {interview_type}",
      "model_answer": "Ideal, structured exemplar answer demonstrating best practices...",
      "evaluation_criteria": "What interviewers look for: Key technical points, clarity, completeness...",
      "coaching_tips": "Coaching advice on structure, key concepts to highlight, and pitfalls to avoid."
    }},
    {{
      "question": "Question 2 tailored to {target_role} {interview_type}",
      "model_answer": "Ideal, structured exemplar answer demonstrating best practices...",
      "evaluation_criteria": "What interviewers look for: Key technical points, clarity, completeness...",
      "coaching_tips": "Coaching advice on structure, key concepts to highlight, and pitfalls to avoid."
    }},
    {{
      "question": "Question 3 tailored to {target_role} {interview_type}",
      "model_answer": "Ideal, structured exemplar answer demonstrating best practices...",
      "evaluation_criteria": "What interviewers look for: Key technical points, clarity, completeness...",
      "coaching_tips": "Coaching advice on structure, key concepts to highlight, and pitfalls to avoid."
    }}
  ]
}}

Format requirements:
1. Output MUST contain 3 complete Q&A report items.
2. 'model_answer' MUST be a full, well-structured exemplar response (not a summary).
3. 'evaluation_criteria' MUST highlight specific scoring rubrics.
4. Output ONLY valid JSON.
"""

    raw_response = call_groq(
        prompt=prompt,
        model_name="llama-3.1-8b-instant",
        system_prompt="You are an expert AI Interview Coach. Output only JSON.",
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
            report = parsed.get("qa_report", [])
            if report and len(report) >= 1:
                return report
        except Exception as e:
            print(f"[InterviewerAgent Warning] QA Report JSON parse error: {e}")

    # Grounded fallback Q&A report
    itype_lower = interview_type.lower()
    role_lower = target_role.lower()

    if "coding" in itype_lower:
        return [
            {
                "question": "How do you solve the Two Sum problem efficiently in O(n) time complexity?",
                "model_answer": "To solve Two Sum in O(n) time, use a Hash Map (dictionary) to store each number's complement (target - current_val) as you iterate through the array. For each element `num` at index `i`, compute `complement = target - num`. If `complement` exists in the hash map, return `[map[complement], i]`. Otherwise, store `map[num] = i`. This completes in a single pass with O(n) time and O(n) space complexity.",
                "evaluation_criteria": "Correctness of algorithmic logic, identification of optimal O(n) time vs O(n^2) brute force complexity, and space trade-off awareness.",
                "coaching_tips": "Start by briefly stating the brute force O(n^2) approach, then present the Hash Map O(n) optimization. Explicitly mention both time and space complexities."
            },
            {
                "question": "What is the difference between an Abstract Class and an Interface in OOP?",
                "model_answer": "An Abstract Class represents an 'is-a' relationship and serves as a base class with shared state (instance variables) and concrete method implementations. An Interface represents a 'can-do' contract defining public method signatures without maintaining state. Classes support single inheritance for abstract classes, but can implement multiple interfaces.",
                "evaluation_criteria": "Clear distinction between state vs contract, single vs multiple inheritance rules, and practical design choice justification.",
                "coaching_tips": "Structure your answer by comparing Purpose ('is-a' vs 'can-do'), State (instance variables vs constants), and Inheritance (single vs multiple)."
            }
        ]
    elif "hr" in itype_lower or "behavioral" in itype_lower:
        return [
            {
                "question": "Tell me about a time you faced a major technical failure in a project and how you handled it.",
                "model_answer": "During a database migration, an unexpected index lock caused production API latency to spike. (Situation & Task). I immediately rolled back the deployment, alerted stakeholders via Slack, and initiated a post-mortem investigation. (Action). I identified missing staging stress tests and implemented automated CI migration validation. (Result). This reduced deployment incident rates by 80%.",
                "evaluation_criteria": "Adherence to STAR structure, accountability without blaming others, clear corrective actions, and quantified learning outcomes.",
                "coaching_tips": "Focus 60% of your time on Action and Result. Always quantify the outcome with metrics whenever possible."
            },
            {
                "question": "How do you handle tight deadlines and competing priorities from stakeholders?",
                "model_answer": "I prioritize tasks using the Eisenhower Matrix and MoSCoW framework (Must-have, Should-have, Could-have). I communicate transparently with product managers to clarify core requirements, break work into iterative MVP sprints, and renegotiate deadlines proactively when unforeseen scope complexity arises.",
                "evaluation_criteria": "Structured prioritization methodology, proactive communication, and ability to manage scope without sacrificing quality.",
                "coaching_tips": "Mention specific prioritization frameworks and emphasize proactive communication before deadlines pass."
            }
        ]
    else:
        return [
            {
                "question": f"What are the core technical responsibilities and architectural patterns in {target_role}?",
                "model_answer": f"In {target_role}, core responsibilities include designing scalable system components, ensuring high availability, maintaining automated test/deployment workflows, and performing complexity analysis on architectural trade-offs.",
                "evaluation_criteria": "Breadth of domain knowledge, clarity on architectural patterns, and practical engineering trade-off rationale.",
                "coaching_tips": "Begin with a high-level summary of responsibilities, then dive into 2-3 specific technical pillars with concrete tools/frameworks."
            },
            {
                "question": f"How do you approach debugging and root-cause analysis when an incident occurs in a {target_role} workflow?",
                "model_answer": "I follow a systematic 4-step triage process: 1. Isolate and replicate the issue using centralized metrics/logs. 2. Formulate testable hypotheses. 3. Apply targeted patches in staging. 4. Conduct a post-mortem to update automated regression tests.",
                "evaluation_criteria": "Structured troubleshooting methodology, effective use of observability tools, and focus on long-term prevention.",
                "coaching_tips": "Outline your step-by-step triage sequence clearly and emphasize post-incident preventive measures."
            }
        ]


def generate_interview_question(
    target_role: str = "Software Engineering",
    interview_type: str = "Technical",
    user_input: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Generate a realistic, role-tailored interview question using explicit target_role and interview_type.
    """
    prompt = f"""You are an expert AI Job Interviewer.

Target Role: "{target_role}"
Interview Category: "{interview_type}"

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

    return {
        "interview_type": interview_type,
        "target_role": target_role,
        "current_question": f"What are the key technical principles and design trade-offs involved in {target_role}?"
    }


class InterviewerAgent:
    """
    Stateful Interviewer Agent wrapper class.
    """

    def __init__(self, model_name: str = "llama-3.1-8b-instant") -> None:
        self.model_name = model_name

    def generate_guide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """State-driven interview guide generation."""
        role = state.get("target_role") or state.get("role", "Software Engineering")
        itype = state.get("interview_type", "Technical")
        return generate_interview_guide(target_role=role, interview_type=itype)

    def generate_qa_report(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """State-driven demo Q&A report generation."""
        role = state.get("target_role") or state.get("role", "Software Engineering")
        itype = state.get("interview_type", "Technical")
        return generate_demo_qa_report(target_role=role, interview_type=itype)
