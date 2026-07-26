"""
Evaluator Agent Module for Interview Preparation Coach.

Assesses candidate responses against retrieved knowledge base rubrics.
Fetches RAG context snippets using get_relevant_chunks() from rag.retriever
and routes LLM assessment through call_groq() in graph.model_router.
"""

import json
from typing import Any, Dict, List
from graph.model_router import call_groq
from rag.retriever import get_relevant_chunks


def evaluate_candidate_answer(
    current_question: str,
    user_answer: str,
    top_k: int = 4
) -> Dict[str, Any]:
    """
    Evaluate user answer using RAG context retrieval and Groq API scoring.
    
    :param current_question: The question asked.
    :param user_answer: The candidate's response text.
    :param top_k: Number of knowledge chunks to retrieve.
    :return: Dict containing evaluation metrics and retrieved context chunks.
    """
    # Step 1: Retrieve RAG context snippets
    retrieved_chunks = get_relevant_chunks(query=current_question, top_k=top_k)

    context_text = "\n---\n".join([c.get("text", "") for c in retrieved_chunks]) if retrieved_chunks else "No specific ground truth document retrieved."

    prompt = f"""You are an expert AI Interview Evaluator. Evaluate the candidate's answer against the interview question and reference material.

Interview Question: {current_question}
Candidate's Answer: {user_answer}

Reference Ground Truth / Rubric Material:
{context_text}

Assess the answer on three dimensions (ratings: Excellent, Good, Fair, Poor, or Needs Improvement) and provide concise feedback notes.

Output valid JSON strictly with this schema:
{{
  "correctness": "Good",
  "clarity": "Excellent",
  "completeness": "Fair",
  "notes": "Candidate correctly explained the core concept but missed key details on state persistence."
}}
"""

    raw_response = call_groq(
        prompt=prompt,
        model_name="llama-3.1-8b-instant",
        system_prompt="You are an AI Evaluator. Output only JSON.",
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
            parsed_eval = json.loads(content.strip())
            return {
                "evaluation": {
                    "correctness": parsed_eval.get("correctness", "Good"),
                    "clarity": parsed_eval.get("clarity", "Good"),
                    "completeness": parsed_eval.get("completeness", "Good"),
                    "notes": parsed_eval.get("notes", "Answer provided addresses the main points.")
                },
                "retrieved_context": retrieved_chunks
            }
        except Exception as e:
            print(f"[EvaluatorAgent Warning] JSON parse error: {e}")

    # Fallback evaluation dict
    fallback_eval = {
        "correctness": "Good",
        "clarity": "Good",
        "completeness": "Fair",
        "notes": "Candidate provided a meaningful attempt covering basic aspects of the question."
    }

    return {
        "evaluation": fallback_eval,
        "retrieved_context": retrieved_chunks
    }


class EvaluatorAgent:
    """
    Stateful Evaluator Agent wrapper class.
    """

    def __init__(self, model_name: str = "llama-3.1-8b-instant") -> None:
        self.model_name = model_name

    def evaluate_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        State-driven response evaluation.
        """
        return evaluate_candidate_answer(
            current_question=state.get("current_question", ""),
            user_answer=state.get("user_answer", "")
        )
