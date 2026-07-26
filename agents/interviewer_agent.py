"""
Interviewer Agent Module for Interview Preparation Coach.

Analyzes candidate role and interview category selections, queries RAG retriever
for domain-specific preparation guidelines across 2,638 knowledge base chunks, 
and generates high-impact, deeply grounded Interview Preparation Guides and 
8-Question Master Q&A Scorecards.
"""

import json
from typing import Any, Dict, List, Optional
from graph.model_router import call_groq, call_openrouter
from rag.retriever import get_relevant_chunks


def generate_interview_guide(target_role: str, interview_type: str) -> Dict[str, Any]:
    """
    Generate an in-depth, RAG-grounded Interview Preparation Guide for a target role & round.
    
    :param target_role: Selected job role (e.g., "Software Engineering", "DevOps", "Data Science").
    :param interview_type: Selected interview round (e.g., "Technical", "HR", "Coding").
    :return: Dictionary containing overview, question_themes, and behavior_tips.
    """
    query = f"{target_role} {interview_type} interview preparation guide expectations key topics skills star method"
    retrieved_chunks = get_relevant_chunks(query=query, top_k=6)

    context_str = "\n---\n".join([c.get("text", "") for c in retrieved_chunks]) if retrieved_chunks else "Standard industry interview guidelines."

    prompt = f"""You are an elite Tech Hiring Manager and Lead Career Coach. Create an authoritative, highly detailed Interview Preparation Guide for a candidate interviewing for a '{target_role}' role in a '{interview_type}' round.

Retrieved Knowledge Base Context (Extracted from domain PDFs and prep guides):
{context_str}

Analyze the role '{target_role}' and round '{interview_type}'. Ground your guide directly in the retrieved context above and industry standards.

Output valid JSON strictly matching this schema:
{{
  "overview": "Comprehensive 3-paragraph executive overview detailing stage duration, core candidate expectations, technical depth evaluated, and overall evaluation goals for {target_role} ({interview_type} round).",
  "question_themes": [
    "Theme 1: Deep core technical skill or domain area with description",
    "Theme 2: System design, architecture, or workflow competence with description",
    "Theme 3: Problem solving, analytical trade-offs, or optimization",
    "Theme 4: Collaboration, communication, or operational best practices"
  ],
  "behavior_tips": [
    "Tip 1: Communication strategy (e.g. STAR method or think-aloud technique)",
    "Tip 2: Technical depth and assumption clarification guidelines",
    "Tip 3: Pacing, structural response framing, and body language",
    "Tip 4: Handling unknown edge cases and asking smart closing questions"
  ]
}}

Format Requirements:
1. Output ONLY valid JSON.
2. Provide actionable, domain-specific insights rather than generic advice.
"""

    raw_response = call_groq(
        prompt=prompt,
        model_name="llama-3.1-8b-instant",
        system_prompt="You are an expert AI Interview Coach. Output valid JSON only.",
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
            if "overview" in parsed and "question_themes" in parsed:
                return parsed
        except Exception as e:
            print(f"[InterviewerAgent Warning] Guide JSON parse error: {e}")

    # Fallback Guide
    return {
        "overview": f"In the {interview_type} interview round for {target_role}, candidates undergo a rigorous 45-60 minute evaluation. Interviewers assess core domain competency, structured problem-solving under pressure, analytical trade-off reasoning, and collaborative communication.",
        "question_themes": [
            f"Core {target_role} fundamentals and architectural principles",
            "Real-world application trade-offs and complexity analysis",
            "Debugging, system design patterns, and operational practices",
            "Cross-functional collaboration and incident post-mortems"
        ],
        "behavior_tips": [
            "Use the STAR framework (Situation, Task, Action, Result) for behavioral prompts.",
            "For technical and coding topics, 'think aloud' to share your problem-solving reasoning.",
            "State your assumptions explicitly before jumping into solution implementation.",
            "Quantify your achievements with concrete metrics whenever possible."
        ]
    }


def generate_demo_qa_report(target_role: str, interview_type: str) -> List[Dict[str, Any]]:
    """
    Generate an authoritative 8-Question Master Q&A Scorecard report featuring 
    realistic questions, difficulty ratings, why interviewers ask, exemplar model answers, 
    scoring criteria vs red flags, and coach tips.
    
    :param target_role: Selected job role.
    :param interview_type: Selected interview category.
    :return: List of 8 detailed dictionary items.
    """
    query = f"{target_role} {interview_type} interview questions model answers code snippets evaluation criteria"
    retrieved_chunks = get_relevant_chunks(query=query, top_k=8)

    context_str = "\n---\n".join([c.get("text", "") for c in retrieved_chunks]) if retrieved_chunks else "No specific ground truth document retrieved."

    prompt = f"""You are an elite Senior Hiring Manager and Master AI Coach. Synthesize a comprehensive 8-Question Demo Q&A Scorecard Report for a candidate applying for '{target_role}' in a '{interview_type}' interview round.

Ground your questions, model answers, and criteria in the retrieved context below:
{context_str}

Generate EXACTLY 8 distinct, realistic, and highly relevant interview questions specifically tailored for {target_role} ({interview_type} round).

Output valid JSON strictly with this schema:
{{
  "qa_report": [
    {{
      "difficulty": "Easy / Medium / Hard",
      "question": "Clear, precise interview question tailored to {target_role} ({interview_type})",
      "why_asked": "1-2 sentences explaining why hiring managers evaluate this specific capability",
      "model_answer": "Complete, highly detailed exemplar model answer demonstrating top-tier response structure...",
      "evaluation_criteria": "Key points required for 5/5 score",
      "red_flags": "Common candidate mistakes or red flags that cause failure",
      "coaching_tips": "Actionable coach advice to stand out from other applicants"
    }}
  ]
}}

Format Rules:
1. 'qa_report' array MUST contain EXACTLY 8 items.
2. Ensure realistic mix of difficulty levels: 2 Easy, 4 Medium, 2 Hard.
3. 'model_answer' MUST be thorough, comprehensive, and well-explained (not abbreviated).
4. Output ONLY valid JSON.
"""

    raw_response = call_groq(
        prompt=prompt,
        model_name="llama-3.1-8b-instant",
        system_prompt="You are a Senior Hiring Manager. Output valid JSON only.",
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
            if report and len(report) >= 4:
                return report
        except Exception as e:
            print(f"[InterviewerAgent Warning] QA Report JSON parse error: {e}")

    # Fallback to role-specific 8-question master bank
    return get_role_fallback_questions(target_role, interview_type)


def get_role_fallback_questions(target_role: str, interview_type: str) -> List[Dict[str, Any]]:
    """Return an extensive 8-question master bank tailored by role and category."""
    role_lower = target_role.lower()
    itype_lower = interview_type.lower()

    if "coding" in itype_lower:
        return [
            {
                "difficulty": "Easy",
                "question": "How do you solve the Two Sum problem efficiently in O(n) time complexity?",
                "why_asked": "Tests foundational hash table data structure usage and optimization from O(n^2) to O(n).",
                "model_answer": "To solve Two Sum in O(n) time, use a Hash Map (dictionary) to store each number's complement (target - num) as you iterate through the array. For each element `num` at index `i`, check if `target - num` exists in the hash map. If found, return `[map[complement], i]`. Otherwise, record `map[num] = i`. Time complexity is O(n), Space complexity is O(n).",
                "evaluation_criteria": "O(1) hash map lookup, O(n) single pass time complexity vs O(n^2) brute force.",
                "red_flags": "Using nested loops resulting in O(n^2) time complexity without identifying the hash map optimization.",
                "coaching_tips": "State the brute force O(n^2) approach first, then present the Hash Map O(n) solution with time/space complexity analysis."
            },
            {
                "difficulty": "Medium",
                "question": "What is the difference between an Abstract Class and an Interface in OOP?",
                "why_asked": "Evaluates object-oriented architecture principles, state management, and contract enforcement.",
                "model_answer": "An Abstract Class represents an 'is-a' relationship with shared instance state and concrete method implementations. An Interface defines a 'can-do' contract specifying method signatures without maintaining state. Subclasses support single inheritance for abstract classes, but can implement multiple interfaces.",
                "evaluation_criteria": "Single vs multiple inheritance rules, state vs stateless contract, design choice justification.",
                "red_flags": "Confusing interface default methods with abstract class instance fields.",
                "coaching_tips": "Compare Purpose ('is-a' vs 'can-do'), State (fields vs constants), and Inheritance."
            },
            {
                "difficulty": "Medium",
                "question": "How do you reverse a Singly Linked List iteratively and recursively?",
                "why_asked": "Tests pointer manipulation accuracy, recursion stack management, and edge-case handling.",
                "model_answer": "Iteratively: Maintain three pointers (`prev`, `curr`, `next`). In a loop, store `next = curr.next`, set `curr.next = prev`, advance `prev = curr` and `curr = next`. Return `prev` as new head. Recursively: Base case `head == None or head.next == None`. Reverse rest, then set `head.next.next = head` and `head.next = None`.",
                "evaluation_criteria": "Pointer updates without breaking references, edge cases (empty list, single node), space complexity analysis.",
                "red_flags": "Losing pointer references resulting in orphaned nodes or infinite memory loops.",
                "coaching_tips": "Draw pointer transitions step-by-step before writing code."
            },
            {
                "difficulty": "Easy",
                "question": "Explain how Binary Search works and analyze its Time Complexity.",
                "why_asked": "Evaluates divide-and-conquer strategy and logarithmic complexity recognition.",
                "model_answer": "Binary Search operates on a sorted array by repeatedly dividing the search space in half. Compare target with middle element `mid`. If target == mid, return index. If target < mid, search left half (`high = mid - 1`). If target > mid, search right half (`low = mid + 1`). Time complexity is O(log n).",
                "evaluation_criteria": "Sorted array prerequisite, integer overflow prevention (`low + (high - low) // 2`), logarithmic complexity.",
                "red_flags": "Applying Binary Search on an unsorted array without sorting first.",
                "coaching_tips": "Always mention verifying sorted order before executing Binary Search."
            },
            {
                "difficulty": "Medium",
                "question": "How do you detect a cycle in a Linked List using Floyd's Cycle Detection Algorithm?",
                "why_asked": "Evaluates two-pointer technique and O(1) space optimization.",
                "model_answer": "Use two pointers (`slow` and `fast`). Move `slow` by 1 step and `fast` by 2 steps. If there is a cycle, `fast` will eventually meet `slow` (`slow == fast`). If `fast` or `fast.next` reaches null, no cycle exists. Time complexity O(n), Space complexity O(1).",
                "evaluation_criteria": "Floyd's Tortoise and Hare algorithm, memory efficiency (O(1) space), edge-case null checks.",
                "red_flags": "Using O(n) space hash set when O(1) two-pointer solution is requested.",
                "coaching_tips": "Contrast O(1) space Floyd's algorithm with O(n) space Hash Set tracking."
            },
            {
                "difficulty": "Medium",
                "question": "What is the difference between Depth-First Search (DFS) and Breadth-First Search (BFS)?",
                "why_asked": "Assesses graph traversal strategies and data structure selection.",
                "model_answer": "DFS explores graph/tree branches as deep as possible before backtracking using a Stack (or call stack). BFS explores level-by-level using a Queue, making it ideal for finding shortest path in unweighted graphs. Both run in O(V + E) time.",
                "evaluation_criteria": "Data structure choice (Stack vs Queue), shortest path applications, space complexity (tree height vs width).",
                "red_flags": "Using DFS for shortest path in unweighted graphs when BFS is optimal.",
                "coaching_tips": "Use BFS for shortest path in unweighted graphs; use DFS for topological sort and cycle detection."
            },
            {
                "difficulty": "Hard",
                "question": "Explain Dynamic Programming and the difference between Memoization (Top-Down) and Tabulation (Bottom-Up).",
                "why_asked": "Evaluates advanced optimization of recursive problems with overlapping subproblems.",
                "model_answer": "Dynamic Programming optimizes recursive problems with overlapping subproblems and optimal substructure. Memoization (Top-Down) uses recursion with a cache dictionary. Tabulation (Bottom-Up) builds an iterative lookup table starting from base cases.",
                "evaluation_criteria": "Identifying overlapping subproblems, subproblem state formulation, recursion stack vs iteration space.",
                "red_flags": "Re-computing overlapping subproblems without caching (pure exponential recursion).",
                "coaching_tips": "Define subproblem state `dp[i]` clearly before writing code."
            },
            {
                "difficulty": "Hard",
                "question": "How do you implement an LRU (Least Recently Used) Cache using Hash Map and Doubly Linked List?",
                "model_answer": "Combine a Hash Map storing `key -> node` for O(1) lookups with a Doubly Linked List keeping nodes in access order. On access/update, move node to head. On insertion exceeding capacity, evict tail node and remove its entry from map. Both `get` and `put` execute in O(1) time.",
                "evaluation_criteria": "Combining Hash Map with Doubly Linked List, O(1) time bounds for get/put, sentinel dummy head/tail nodes.",
                "red_flags": "Using a single array or list resulting in O(n) eviction time.",
                "coaching_tips": "Explain why a single Array or Linked List cannot achieve O(1) for both lookup and deletion."
            }
        ]

    elif "business" in role_lower:
        return [
            {
                "difficulty": "Easy",
                "question": "What is the difference between Functional and Non-Functional Requirements?",
                "why_asked": "Verifies ability to distinguish user-facing features from technical system quality attributes.",
                "model_answer": "Functional Requirements define WHAT the system should do (e.g., process credit card payments). Non-Functional Requirements specify HOW the system should perform (e.g., SLA latency under 200ms, 99.99% availability, ISO security compliance).",
                "evaluation_criteria": "Distinction between business features vs quality attributes (usability, reliability, performance).",
                "red_flags": "Omitting performance, security, and scalability from requirement specifications.",
                "coaching_tips": "Provide concrete examples of each type and explain why non-functional requirements dictate technical architecture."
            },
            {
                "difficulty": "Medium",
                "question": "How do you bridge the gap between technical developers and non-technical business stakeholders?",
                "why_asked": "Evaluates communication adaptation, stakeholder management, and requirement translation skills.",
                "model_answer": "I act as a translator using domain models, user stories, Wireframes, and BPMN. I facilitate alignment workshops, establish shared vocabulary, and validate acceptance criteria to ensure business intent matches technical execution.",
                "evaluation_criteria": "Stakeholder management, communication adaptation, requirements traceability, active listening.",
                "red_flags": "Using overly technical jargon with business users or ambiguous requirements with developers.",
                "coaching_tips": "Emphasize using visual diagrams (use case diagrams, flowcharts) rather than technical jargon."
            },
            {
                "difficulty": "Medium",
                "question": "What is the difference between a Business Requirement Document (BRD) and System Requirement Specification (SRS)?",
                "why_asked": "Assesses knowledge of project documentation lifecycle and stakeholder audience targeting.",
                "model_answer": "A BRD focuses on business goals, problem statements, and ROI objectives from the business perspective. An SRS translates BRD goals into detailed technical specs, data flow diagrams, API interactions, and system behavior for developers.",
                "evaluation_criteria": "Target audience differentiation, level of abstraction, requirements lifecycle mapping.",
                "red_flags": "Putting developer API specs in a high-level BRD intended for executive sponsors.",
                "coaching_tips": "Highlight that BRD answers 'WHY' and 'WHAT', while SRS details 'HOW' systems behave."
            },
            {
                "difficulty": "Medium",
                "question": "How do you perform Gap Analysis in a Business Process?",
                "model_answer": "Gap Analysis evaluates the difference between current state ('As-Is') and desired future state ('To-Be'). Steps: 1. Mapping As-Is workflows. 2. Defining target To-Be capabilities. 3. Identifying missing features, bottlenecks, and data gaps. 4. Formulating a prioritized remediation roadmap.",
                "evaluation_criteria": "As-Is vs To-Be process mapping, root cause identification, actionable roadmap creation.",
                "red_flags": "Designing To-Be solutions without thoroughly analyzing existing As-Is process bottlenecks.",
                "coaching_tips": "Walk through a practical example like transitioning from manual paper approvals to automated digital workflows."
            },
            {
                "difficulty": "Easy",
                "question": "Explain the INVEST criteria for writing effective User Stories.",
                "why_asked": "Tests Agile backlog grooming skills and user story quality standards.",
                "model_answer": "INVEST stands for Independent, Negotiable, Valuable, Estimable, Small (fits within a sprint), and Testable (has explicit acceptance criteria).",
                "evaluation_criteria": "Agile methodology mastery, story sizing, acceptance criteria formatting (Given-When-Then).",
                "red_flags": "Writing giant monolithic user stories that cannot be completed within a single sprint.",
                "coaching_tips": "Write a sample user story: 'As a customer, I want to filter products by price, so that I can find affordable items.'"
            },
            {
                "difficulty": "Hard",
                "question": "How do you conduct Requirement Elicitation with conflicting stakeholders?",
                "why_asked": "Evaluates conflict resolution, negotiation, and objective prioritization.",
                "model_answer": "I conduct 1-on-1 interviews, facilitated alignment workshops, and MoSCoW prioritization sessions. I ground decisions in objective user data and business ROI metrics to resolve conflicting opinions.",
                "evaluation_criteria": "Conflict resolution, active listening, objective data usage, consensus building.",
                "red_flags": "Taking sides subjectively instead of facilitating data-driven decision making.",
                "coaching_tips": "Focus stakeholders on shared business goals rather than subjective feature choices."
            },
            {
                "difficulty": "Medium",
                "question": "What is Root Cause Analysis and how do you use the 5 Whys technique?",
                "why_asked": "Evaluates analytical depth in uncovering systemic process defects.",
                "model_answer": "Root Cause Analysis identifies underlying causes of operational problems. The 5 Whys iteratively asks 'Why did this happen?' five times until the foundational process or systemic flaw is revealed.",
                "evaluation_criteria": "Problem-solving depth, 5 Whys methodology, Fishbone diagram usage.",
                "red_flags": "Stopping analysis at surface symptoms instead of uncovering the root cause.",
                "coaching_tips": "Walk through a real supply chain delay or bug outbreak using 5 Whys."
            },
            {
                "difficulty": "Hard",
                "question": "How do you manage Scope Creep during project execution?",
                "why_asked": "Assesses project governance, change control, and trade-off negotiations.",
                "model_answer": "I enforce formal Change Control procedures. When new requirements are requested: 1. Analyze impact on budget and schedule. 2. Present impact to Change Control Board. 3. Swap lower-priority backlog items before accepting scope changes.",
                "evaluation_criteria": "Change control process, impact analysis, backlog trade-offs, stakeholder management.",
                "red_flags": "Accepting informal scope additions without updating project schedule or budget.",
                "coaching_tips": "Reframe saying 'no' into offering trade-off choices with transparent impact analysis."
            }
        ]

    elif "data" in role_lower or "ai" in role_lower or "ml" in role_lower:
        return [
            {
                "difficulty": "Easy",
                "question": "What is the difference between Supervised, Unsupervised, and Reinforcement Learning?",
                "why_asked": "Evaluates foundational understanding of machine learning paradigms and objective functions.",
                "model_answer": "Supervised Learning trains models on labeled input-output data to perform classification or regression. Unsupervised Learning discovers hidden patterns or clusters in unlabeled data (K-Means, PCA). Reinforcement Learning trains agents to maximize cumulative rewards through trial-and-error environment interaction.",
                "evaluation_criteria": "Data labeling distinction, objective functions, practical algorithm examples per category.",
                "red_flags": "Confusing clustering (unsupervised) with classification (supervised).",
                "coaching_tips": "Give clear real-world examples for each: spam filtering (Supervised), customer segmentation (Unsupervised), game AI (Reinforcement)."
            },
            {
                "difficulty": "Medium",
                "question": "Explain Bias-Variance Tradeoff and how to prevent Overfitting.",
                "why_asked": "Tests ability to diagnose model generalization errors and apply regularization techniques.",
                "model_answer": "High Bias leads to Underfitting (model is too simple). High Variance leads to Overfitting (model memorizes training noise and fails on unseen test data). Regularization (L1/L2), Cross-Validation, Dropout, Early Stopping, and Ensembling balance the tradeoff to minimize total generalization error.",
                "evaluation_criteria": "Underfitting vs overfitting identification, mathematical MSE error decomposition, regularization methods.",
                "red_flags": "Increasing model complexity when a model is already severely overfitting.",
                "coaching_tips": "Explain how L1 (Lasso) performs feature selection while L2 (Ridge) shrinks parameter weights."
            },
            {
                "difficulty": "Medium",
                "question": "What is the difference between Precision, Recall, F1-Score, and ROC-AUC?",
                "why_asked": "Evaluates classification metric selection under class imbalance.",
                "model_answer": "Precision measures true positives out of all predicted positives (reduces false alarms). Recall measures true positives out of all actual positives (reduces missed cases). F1-Score is the harmonic mean of Precision and Recall. ROC-AUC measures true positive rate vs false positive rate across classification thresholds.",
                "evaluation_criteria": "Confusion matrix metrics, class imbalance metrics selection, trade-off scenarios.",
                "red_flags": "Relying on raw accuracy for severely imbalanced datasets (e.g. 99% majority class).",
                "coaching_tips": "Use medical diagnosis (high Recall preferred) vs spam detection (high Precision preferred) to illustrate choices."
            },
            {
                "difficulty": "Hard",
                "question": "What is the Transformer Architecture and how does Self-Attention work?",
                "model_answer": "Transformers replace recurrent neural networks (RNNs) with parallelizable Self-Attention mechanisms. Scaled Dot-Product Attention calculates relevance scores across all token pairs using Query (Q), Key (K), and Value (V) matrices: `Softmax(Q K^T / sqrt(d_k)) V`, enabling long-range context modeling without sequential bottlenecks.",
                "evaluation_criteria": "Attention matrix math (Q, K, V), Multi-Head Attention, positional encodings, parallelization benefits.",
                "red_flags": "Failing to explain why Query, Key, and Value matrices are used in Self-Attention.",
                "coaching_tips": "Contrast RNN O(N) sequential processing time with Transformer parallel O(1) step processing."
            },
            {
                "difficulty": "Medium",
                "question": "What is Retrieval-Augmented Generation (RAG) in LLM applications?",
                "why_asked": "Tests modern Generative AI architecture skills for reducing LLM hallucinations.",
                "model_answer": "RAG connects Large Language Models to external vector databases (ChromaDB). Documents are chunked, embedded using vector models, and indexed. At query time, top-K relevant chunks are retrieved via vector similarity search and injected into the LLM prompt context to eliminate hallucinations and supply up-to-date domain knowledge.",
                "evaluation_criteria": "Vector embeddings, cosine similarity search, chunking strategies, hallucination reduction.",
                "red_flags": "Suggesting fine-tuning LLMs for fast dynamic data updates instead of RAG.",
                "coaching_tips": "Explain the 3 core RAG phases: Ingestion (chunking/embedding), Retrieval (vector search), and Generation (prompt context injection)."
            },
            {
                "difficulty": "Medium",
                "question": "How do you detect and handle Data Leakage in ML pipelines?",
                "why_asked": "Assesses rigor in building production-ready ML validation pipelines.",
                "model_answer": "Data Leakage occurs when information from the target variable or validation/test set leaks into the training pipeline. Common causes: scaling/imputing dataset BEFORE splitting, incorporating future temporal features, or duplicate records across train/test splits. Prevention: fit transformers ONLY on training folds within cross-validation pipelines.",
                "evaluation_criteria": "Identifying train-test contamination, time-series data splitting, pipeline leakage prevention.",
                "red_flags": "Fitting StandardScaler on the entire dataset prior to train/test split.",
                "coaching_tips": "Emphasize wrapping preprocessors and models inside `sklearn.pipeline.Pipeline`."
            },
            {
                "difficulty": "Easy",
                "question": "What is the Central Limit Theorem (CLT) and why is it important in Statistics?",
                "why_asked": "Tests statistical foundations for hypothesis testing and confidence intervals.",
                "model_answer": "CLT states that the sample mean distribution of N independent, identically distributed random variables approaches a normal (Gaussian) distribution as sample size N increases, regardless of the underlying population distribution. This enables hypothesis testing (Z-tests, T-tests) and confidence interval construction.",
                "evaluation_criteria": "Normal distribution convergence, sample size assumptions (N >= 30), hypothesis testing applications.",
                "red_flags": "Thinking CLT applies to individual data points rather than sample means.",
                "coaching_tips": "Explain why CLT allows us to make statistical inferences about population parameters from sample data."
            },
            {
                "difficulty": "Hard",
                "question": "Explain Model Deployment, MLOps, and Concept Drift Monitoring.",
                "why_asked": "Evaluates end-to-end MLOps lifecycle management in production.",
                "model_answer": "MLOps automates model training, CI/CD deployment (Docker/FastAPI), and monitoring. Concept Drift occurs when statistical properties of target variables change over time; Data Drift occurs when input feature distributions change. Monitoring calculates Population Stability Index (PSI) or KS tests to trigger automated retraining pipelines.",
                "evaluation_criteria": "MLOps architecture, Model Drift vs Data Drift detection, automated retraining triggers.",
                "red_flags": "Deploying models as static artifacts without monitoring prediction distribution shifts over time.",
                "coaching_tips": "Explain how monitoring feature distribution shifts prevents silent production model degradation."
            }
        ]

    else:
        # Default Software Engineering / Technical Master Bank
        return [
            {
                "difficulty": "Easy",
                "question": "What is the difference between an Abstract Class and an Interface in OOP?",
                "why_asked": "Evaluates core object-oriented design principles and inheritance rules.",
                "model_answer": "An Abstract Class represents an 'is-a' relationship and serves as a base class with shared state (instance variables) and concrete method implementations. An Interface defines a 'can-do' contract defining public method signatures without maintaining state. Subclasses support single inheritance for abstract classes, but can implement multiple interfaces.",
                "evaluation_criteria": "Single vs. multiple inheritance rules, state vs. stateless contract, design choice justification.",
                "red_flags": "Confusing interface default methods with abstract class instance fields.",
                "coaching_tips": "Structure your answer by comparing Purpose ('is-a' vs 'can-do'), State (instance variables vs constants), and Inheritance."
            },
            {
                "difficulty": "Easy",
                "question": "How do you solve the Two Sum problem efficiently in O(n) time complexity?",
                "why_asked": "Tests fundamental data structure selection and time complexity optimization.",
                "model_answer": "Use a Hash Map (dictionary) to store each number's complement (target - num) as you iterate through the array. For each element `num` at index `i`, check if `target - num` exists in the hash map. If found, return `[map[complement], i]`. Otherwise, store `map[num] = i`.",
                "evaluation_criteria": "Hash table lookup O(1) efficiency, O(n) single pass time complexity vs O(n^2) brute force.",
                "red_flags": "Using nested loops resulting in O(n^2) time complexity without identifying the hash map optimization.",
                "coaching_tips": "Briefly state brute force O(n^2), then present the Hash Map O(n) optimization with time/space complexity analysis."
            },
            {
                "difficulty": "Medium",
                "question": "Explain SOLID principles in Object-Oriented Software Design.",
                "why_asked": "Assesses clean code architecture and maintainable software design standards.",
                "model_answer": "SOLID stands for Single Responsibility (one reason to change), Open/Closed (open for extension, closed for modification), Liskov Substitution (subtypes must be substitutable for base types), Interface Segregation (fine-grained interfaces), and Dependency Inversion (depend on abstractions, not concretions).",
                "evaluation_criteria": "Explaining all 5 acronym letters accurately with real-world refactoring examples.",
                "red_flags": "Inability to explain Dependency Inversion or confusing it with Dependency Injection.",
                "coaching_tips": "Use concise single-sentence definitions for each letter, then provide a code refactoring example for Dependency Inversion."
            },
            {
                "difficulty": "Medium",
                "question": "What is Database Normalization and why is 3NF important?",
                "why_asked": "Evaluates relational database schema design and anomaly prevention.",
                "model_answer": "Normalization reduces data redundancy and prevents update anomalies by organizing tables into normal forms. 1NF eliminates duplicate columns; 2NF removes partial key dependencies; 3NF removes transitive dependencies where non-key attributes depend on other non-key attributes.",
                "evaluation_criteria": "Identifying anomalies (insertion, deletion, update) and defining functional dependencies.",
                "red_flags": "Failing to identify transitive dependencies in 3NF decomposition.",
                "coaching_tips": "Use a simple Employee-Department table example to demonstrate 3NF decomposition."
            },
            {
                "difficulty": "Medium",
                "question": "Compare SQL (Relational) vs NoSQL (Document/Key-Value) Databases.",
                "why_asked": "Assesses data storage trade-offs, scaling strategies, and CAP theorem application.",
                "model_answer": "SQL databases (PostgreSQL, MySQL) enforce ACID transactions, structured schemas, and relational joins, ideal for complex transactions. NoSQL databases (MongoDB, DynamoDB) offer dynamic schemas, horizontal scaling (sharding), and eventual consistency (BASE), ideal for high throughput unstructured data.",
                "evaluation_criteria": "CAP theorem trade-offs, ACID vs BASE consistency, and scaling strategies.",
                "red_flags": "Claiming NoSQL is always faster than SQL regardless of data access patterns.",
                "coaching_tips": "Frame the choice around data access patterns and scaling needs rather than popularity."
            },
            {
                "difficulty": "Hard",
                "question": "What is Microservices Architecture and how does it compare to Monolithic?",
                "why_asked": "Evaluates distributed systems architecture, service boundaries, and operational overhead.",
                "model_answer": "A Monolith packages all business domains into a single deployment unit. Microservices decompose functionality into independently deployable, loosely coupled services communicating via APIs (gRPC/REST) or message queues (Kafka).",
                "evaluation_criteria": "Domain-driven design, service boundaries, Saga pattern, operational overhead.",
                "red_flags": "Recommending microservices for small early-stage startups without considering operational CI/CD complexity.",
                "coaching_tips": "Discuss operational complexity alongside microservice benefits."
            },
            {
                "difficulty": "Hard",
                "question": "Explain the CAP Theorem in Distributed Systems.",
                "why_asked": "Tests foundational distributed database trade-off analysis under network partitions.",
                "model_answer": "CAP theorem states that a distributed data store can simultaneously provide at most two of three guarantees: Consistency, Availability, and Partition Tolerance.",
                "evaluation_criteria": "Understanding that Partition Tolerance is mandatory, leaving choices between CP and AP.",
                "red_flags": "Claiming a distributed system can achieve all three (C, A, and P) simultaneously.",
                "coaching_tips": "Give concrete examples: HBase/MongoDB (CP) vs Cassandra/DynamoDB (AP)."
            },
            {
                "difficulty": "Medium",
                "question": "What is Dependency Injection and how does it promote testability?",
                "why_asked": "Evaluates software decoupling techniques and automated unit testing practices.",
                "model_answer": "Dependency Injection (DI) is an implementation of Dependency Inversion where dependencies are injected from the outside rather than created internally, allowing injection of mock objects during unit testing.",
                "evaluation_criteria": "Decoupling components, constructor injection vs field injection, test mock creation.",
                "red_flags": "Hardcoding `new ConcreteService()` inside domain logic classes.",
                "coaching_tips": "Contrast tight coupling with loose coupling constructor injection."
            }
        ]


def generate_interview_question(
    target_role: str,
    interview_type: str,
    previous_questions: List[str]
) -> str:
    """Legacy helper for single question generation."""
    prev_str = ", ".join(previous_questions) if previous_questions else "None"
    prompt = f"Generate 1 high-quality {interview_type} question for {target_role}. Previous: {prev_str}."
    return call_groq(prompt, "llama-3.1-8b-instant")
