# Agent Communication Sequence Diagram

The sequence diagram below illustrates the message flow and retrieval-grounded generation sequence for the **Interview Preparation Guide** and **Demo Q&A Model Answers Report**.

```mermaid
sequenceDiagram
    autonumber
    actor User as Candidate User
    participant UI as Streamlit UI (app.py)
    participant Graph as LangGraph Orchestrator (graph/orchestrator.py)
    participant Interviewer as Interviewer Agent (agents/interviewer_agent.py)
    participant RAG as RAG Retriever (rag/retriever.py)
    participant Groq as Groq API (llama-3.1-8b-instant)

    User->>UI: Select Role (e.g. Software Engineering) & Category (Technical)
    User->>UI: Click 'Generate Interview Guide & Q&A Report'
    
    UI->>Graph: get_interview_guide(role, interview_type)
    Graph->>Interviewer: generate_guide(role, interview_type)
    
    Interviewer->>RAG: get_relevant_chunks(role + interview_type query, top_k=4)
    RAG-->>Interviewer: Return retrieved context snippets

    Interviewer->>Groq: Request structured guide JSON (overview, themes, behavior tips)
    Groq-->>Interviewer: Return JSON guide response

    Graph->>Interviewer: generate_demo_qa_report(role, interview_type)
    Interviewer->>Groq: Request sample questions + ideal model answers + criteria + tips
    Groq-->>Interviewer: Return JSON QA Report response

    Graph-->>UI: Return state dict (interview_guide, demo_qa_report)
    UI-->>User: Display Interview Preparation Guide & Demo Questions/Model Answers Report
```

### Shared State Schema

During guide & report generation, the `InterviewState` dictionary contains:

```python
{
    "role": "Software Engineering",
    "target_role": "Software Engineering",
    "interview_type": "Technical",
    "interview_guide": {
        "overview": "Detailed overview of the 45-60 min technical round...",
        "question_themes": ["Core OOP abstractions", "System performance", "Incident response"],
        "behavior_tips": ["Use STAR method", "Think aloud for technical logic", "Verify assumptions"]
    },
    "demo_qa_report": [
        {
            "question": "What is the difference between an abstract class and an interface?",
            "model_answer": "An abstract class represents an 'is-a' hierarchy with shared state...",
            "evaluation_criteria": "Single vs. multiple inheritance, state vs. contract...",
            "coaching_tips": "Compare Purpose, State, and Inheritance structured into 3 points."
        }
    ]
}
```
