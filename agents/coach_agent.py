"""
Coach Agent Module for Interview Preparation Coach.

Synthesizes candidate evaluation, retrieved knowledge context, question details,
and user answer to generate structured, actionable coaching feedback.
Uses OpenRouter API (via OpenAI SDK compatibility layer) for high-reasoning feedback generation.
"""

import json
import os
from typing import Any, Dict, List
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def generate_coaching_feedback(
    current_question: str,
    user_answer: str,
    retrieved_context: List[Any],
    evaluation: Dict[str, Any],
    model_name: str = "meta-llama/llama-3.3-70b-instruct"
) -> Dict[str, Any]:
    """
    Generate reflection-based coaching feedback with formatted strengths, gaps, and suggestions.
    
    :param current_question: The question asked during the interview.
    :param user_answer: The candidate's response.
    :param retrieved_context: Relevant RAG snippets or ground truth guides.
    :param evaluation: Dict containing correctness, clarity, completeness, and notes.
    :param model_name: OpenRouter model identifier for reasoning.
    :return: Dict containing structured feedback: {"feedback": {"strengths": [...], "gaps": [...], "suggestions": [...]}}
    """
    api_key = os.getenv("OPENROUTER_API_KEY")

    # Format context snippets for prompt inclusion
    context_str = "\n".join(
        [str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in retrieved_context]
    ) if retrieved_context else "No specific reference retrieved."

    prompt = f"""You are an expert AI Interview Coach. Analyze the following interview question, candidate's answer, evaluation summary, and reference context.

Question: {current_question}
Candidate's Answer: {user_answer}

Evaluation Metrics:
- Correctness: {evaluation.get('correctness', 'N/A')}
- Clarity: {evaluation.get('clarity', 'N/A')}
- Completeness: {evaluation.get('completeness', 'N/A')}
- Evaluator Notes: {evaluation.get('notes', 'N/A')}

Reference Knowledge Context:
{context_str}

Perform deep reflection on the candidate's performance and output valid JSON strictly with this schema:
{{
  "strengths": [
    "✓ Clear explanation of the concept",
    "✓ Relevant real-world example"
  ],
  "gaps": [
    "✗ Structured answer (STAR method for behavioral questions)",
    "✗ Concise delivery without rambling"
  ],
  "suggestions": [
    "Use bullet points or STAR framework to structure your scenario response.",
    "Be sure to quantify outcomes with specific metrics."
  ]
}}

Format rules:
1. Every item in 'strengths' MUST start with '✓ '.
2. Every item in 'gaps' MUST start with '✗ '.
3. 'suggestions' MUST contain actionable recommendations for improvement.
4. Output ONLY valid JSON, no markdown formatting or extra conversational text.
"""

    try:
        if api_key and api_key != "your_openrouter_api_key_here":
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a professional AI Interview Coach. Return responses in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            parsed_feedback = json.loads(content.strip())
        else:
            raise ValueError("OPENROUTER_API_KEY is not configured or set to placeholder.")
    except Exception as e:
        print(f"[CoachAgent Note] Using structured fallback feedback ({e})")
        parsed_feedback = {
            "strengths": [
                "✓ Addressed the core topic directly",
                "✓ Provided relevant initial technical reasoning"
            ],
            "gaps": [
                "✗ Structured answer (STAR method for behavioral questions)",
                "✗ Concise delivery without rambling"
            ],
            "suggestions": [
                "Practice using the STAR framework (Situation, Task, Action, Result) for behavioral prompts.",
                "Include concrete technical details and time/space complexity analysis where applicable."
            ]
        }

    return {"feedback": parsed_feedback}


class CoachAgent:
    """
    Coach Agent class for state-driven execution in agent workflows.
    """

    def __init__(self, model_name: str = "meta-llama/llama-3.3-70b-instruct") -> None:
        self.model_name = model_name

    def generate_feedback(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract parameters from global state and generate structured feedback.
        
        :param state: Global interview state dict.
        :return: Dict with 'feedback' key.
        """
        return generate_coaching_feedback(
            current_question=state.get("current_question", ""),
            user_answer=state.get("user_answer", ""),
            retrieved_context=state.get("retrieved_context", []),
            evaluation=state.get("evaluation", {}),
            model_name=self.model_name
        )
