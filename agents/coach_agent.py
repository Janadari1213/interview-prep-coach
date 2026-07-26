"""
Coach Agent Module.

Responsible for generating constructive feedback, actionable recommendations, 
sample improved responses, and personalized study tips for the candidate.
"""

class CoachAgent:
    """
    Agent responsible for delivering constructive coaching feedback and recommendations.
    """

    def __init__(self, model_name: str = "llama3-70b-8192") -> None:
        """
        Initialize the Coach Agent with prompt templates and coaching strategies.
        
        :param model_name: Identifier for the LLM backing the agent.
        """
        pass

    def generate_feedback(self, question: str, answer: str, evaluation: dict) -> dict:
        """
        Synthesize detailed feedback and actionable tips based on candidate response evaluation.
        
        :param question: The interview question.
        :param answer: The candidate's original answer.
        :param evaluation: The evaluation dict output from EvaluatorAgent.
        :return: Structured dictionary containing feedback summary, suggested answer, and tips.
        """
        pass
