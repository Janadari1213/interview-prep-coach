# Agent Communication Sequence Diagram

The sequence diagram below illustrates the message flow and shared state transitions across the multi-agent pipeline during a single interview turn.

```mermaid
sequenceDiagram
    autonumber
    actor User as Candidate User
    participant UI as Streamlit UI (app.py)
    participant Graph as LangGraph Orchestrator
    participant Interviewer as Interviewer Agent (Groq)
    participant RAG as RAG Retriever (ChromaDB)
    participant Evaluator as Evaluator Agent (Groq)
    participant Coach as Coach Agent (OpenRouter)

    User->>UI: Select Role & Category -> Click 'Start'
    UI->>Graph: Initialize InterviewState (role, type, count=0)
    Graph->>Interviewer: Request question for target_role
    Interviewer-->>Graph: Return current_question
    Graph-->>UI: Update InterviewState & Display Question

    User->>UI: Input candidate answer -> Click 'Submit'
    UI->>Graph: run_interview_step(state, user_answer)
    
    Graph->>RAG: get_relevant_chunks(current_question, top_k=4)
    RAG-->>Graph: Return retrieved_context snippets

    Graph->>Evaluator: Evaluate (question, user_answer, retrieved_context)
    Evaluator-->>Graph: Return evaluation dict (correctness, clarity, completeness, notes)

    Graph->>Coach: Generate feedback (question, user_answer, evaluation, retrieved_context)
    Coach-->>Graph: Return feedback dict (✓ strengths, ✗ gaps, suggestions)

    alt Question Count < 5 AND session not ended
        Graph->>Interviewer: Generate next_question
        Interviewer-->>Graph: Return next current_question, count++
        Graph-->>UI: Return updated state (feedback + next question)
        UI-->>User: Display feedback card + next question card
    else Question Count >= 5 OR 'end interview'
        Graph-->>UI: Set is_complete = True
        UI-->>User: Display final performance scorecard summary
    end
```

### Shared State Lifecycle

At each step in the sequence, the shared `InterviewState` `TypedDict` object is enriched:

```python
{
    "interview_type": "Technical",
    "target_role": "Software Engineer",
    "current_question": "Explain abstract class vs interface.",
    "user_answer": "Abstract classes support single inheritance...",
    "retrieved_context": [{"text": "...", "source": "technical_prep.txt"}],
    "evaluation": {"correctness": "Good", "clarity": "Excellent", "completeness": "Good"},
    "feedback": {"strengths": ["✓ ..."], "gaps": ["✗ ..."], "suggestions": ["..."]},
    "question_count": 1,
    "is_complete": False
}
```
