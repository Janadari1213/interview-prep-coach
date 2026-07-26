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
    Generate a complete Demo Q&A Report featuring 8 exemplar questions, 
    ideal model answers, key evaluation criteria, and coaching tips grounded 
    in the 2,638 RAG chunks from uploaded PDFs and TXT files.
    
    :param target_role: Selected job role (e.g., "Software Engineering", "DevOps", "Business Analyst", "QA Engineering", "UI/UX Design").
    :param interview_type: Selected interview round (e.g., "Technical", "HR", "Coding").
    :return: List of dicts representing sample questions with ideal answers and coaching advice.
    """
    query = f"{interview_type} interview questions model answers evaluation criteria {target_role} architecture principles design patterns"
    retrieved_chunks = get_relevant_chunks(query=query, top_k=8)

    context_str = "\n---\n".join([c.get("text", "") for c in retrieved_chunks]) if retrieved_chunks else "No specific ground truth document retrieved."

    prompt = f"""You are an expert AI Interview Coach and Senior Hiring Manager. Generate a comprehensive Demo Questions & Model Answers Report for a '{target_role}' candidate in an '{interview_type}' interview round.

Reference RAG Knowledge Base Context (Retrieved from uploaded PDFs and prep documents):
{context_str}

Generate 8 high-quality, realistic, and diverse interview questions tailored specifically to {target_role} ({interview_type} round). For EACH question, provide an ideal model answer, key evaluation criteria, and actionable coaching tips grounded in the context above.

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
    }},
    {{
      "question": "Question 4 tailored to {target_role} {interview_type}",
      "model_answer": "Ideal, structured exemplar answer demonstrating best practices...",
      "evaluation_criteria": "What interviewers look for: Key technical points, clarity, completeness...",
      "coaching_tips": "Coaching advice on structure, key concepts to highlight, and pitfalls to avoid."
    }},
    {{
      "question": "Question 5 tailored to {target_role} {interview_type}",
      "model_answer": "Ideal, structured exemplar answer demonstrating best practices...",
      "evaluation_criteria": "What interviewers look for: Key technical points, clarity, completeness...",
      "coaching_tips": "Coaching advice on structure, key concepts to highlight, and pitfalls to avoid."
    }},
    {{
      "question": "Question 6 tailored to {target_role} {interview_type}",
      "model_answer": "Ideal, structured exemplar answer demonstrating best practices...",
      "evaluation_criteria": "What interviewers look for: Key technical points, clarity, completeness...",
      "coaching_tips": "Coaching advice on structure, key concepts to highlight, and pitfalls to avoid."
    }},
    {{
      "question": "Question 7 tailored to {target_role} {interview_type}",
      "model_answer": "Ideal, structured exemplar answer demonstrating best practices...",
      "evaluation_criteria": "What interviewers look for: Key technical points, clarity, completeness...",
      "coaching_tips": "Coaching advice on structure, key concepts to highlight, and pitfalls to avoid."
    }},
    {{
      "question": "Question 8 tailored to {target_role} {interview_type}",
      "model_answer": "Ideal, structured exemplar answer demonstrating best practices...",
      "evaluation_criteria": "What interviewers look for: Key technical points, clarity, completeness...",
      "coaching_tips": "Coaching advice on structure, key concepts to highlight, and pitfalls to avoid."
    }}
  ]
}}

Format requirements:
1. Output MUST contain 8 complete Q&A report items.
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

    # Role-based Q&A Report Fallbacks (8 Questions per Category)
    itype_lower = interview_type.lower()
    role_lower = target_role.lower()

    if "coding" in itype_lower:
        return [
            {
                "question": "How do you solve the Two Sum problem efficiently in O(n) time complexity?",
                "model_answer": "Use a Hash Map to store each number's complement (target - num) as you iterate. For each element `num` at index `i`, check if `target - num` exists in the map. If found, return `[map[complement], i]`. Otherwise, store `map[num] = i`. Time complexity is O(n), Space complexity is O(n).",
                "evaluation_criteria": "O(1) hash map lookup, O(n) single pass complexity vs O(n^2) brute force, space trade-off awareness.",
                "coaching_tips": "Briefly state brute force O(n^2), then present Hash Map O(n) optimization with explicit complexity analysis."
            },
            {
                "question": "What is the difference between an Abstract Class and an Interface in OOP?",
                "model_answer": "An Abstract Class represents an 'is-a' relationship with shared state (fields) and concrete method implementations. An Interface defines a 'can-do' contract with public method signatures. Subclasses support single inheritance for abstract classes, but can implement multiple interfaces.",
                "evaluation_criteria": "Single vs multiple inheritance rules, state vs stateless contract, and design choice justification.",
                "coaching_tips": "Structure by comparing Purpose ('is-a' vs 'can-do'), State (fields vs constants), and Inheritance."
            },
            {
                "question": "How do you reverse a Singly Linked List iteratively and recursively?",
                "model_answer": "Iteratively: Maintain three pointers (`prev`, `curr`, `next`). In a loop, store `next = curr.next`, set `curr.next = prev`, advance `prev = curr` and `curr = next`. Return `prev` as new head. Recursively: Base case `head == None or head.next == None`. Reverse rest, then `head.next.next = head; head.next = None`.",
                "evaluation_criteria": "Pointer manipulation accuracy, edge cases (empty list, single node), and space complexity (O(1) iterative vs O(n) call stack recursive).",
                "coaching_tips": "Draw pointer transitions step-by-step before writing code."
            },
            {
                "question": "Explain how Binary Search works and analyze its Time Complexity.",
                "model_answer": "Binary Search operates on a sorted array by repeatedly dividing the search space in half. Compare target with middle element `mid`. If target == mid, return index. If target < mid, search left half (`high = mid - 1`). If target > mid, search right half (`low = mid + 1`). Time complexity is O(log n).",
                "evaluation_criteria": "Sorted array prerequisite, integer overflow prevention in mid calculation (`low + (high - low) // 2`), logarithmic complexity.",
                "coaching_tips": "Always mention checking if the array is sorted before applying Binary Search."
            },
            {
                "question": "How do you detect a cycle in a Linked List using Floyd's Cycle Detection Algorithm?",
                "model_answer": "Use two pointers (`slow` and `fast`). Move `slow` by 1 step and `fast` by 2 steps. If there is a cycle, `fast` will eventually meet `slow` (`slow == fast`). If `fast` or `fast.next` reaches null, no cycle exists. Time complexity O(n), Space complexity O(1).",
                "evaluation_criteria": "Floyd's Tortoise and Hare algorithm, memory efficiency (O(1) space), edge-case checks.",
                "coaching_tips": "Contrast O(1) space Floyd's algorithm with O(n) space Hash Set tracking."
            },
            {
                "question": "What is the difference between Depth-First Search (DFS) and Breadth-First Search (BFS)?",
                "model_answer": "DFS explores graph/tree branches as deep as possible before backtracking using a Stack (or recursion call stack). BFS explores level-by-level using a Queue, making it ideal for finding shortest path in unweighted graphs. Both run in O(V + E) time.",
                "evaluation_criteria": "Data structure choice (Stack vs Queue), shortest path applications, space complexity (tree height vs width).",
                "coaching_tips": "Use BFS for shortest path in unweighted graphs; use DFS for topological sort and cycle detection."
            },
            {
                "question": "Explain Dynamic Programming and the difference between Memoization (Top-Down) and Tabulation (Bottom-Up).",
                "model_answer": "Dynamic Programming optimizes recursive problems with overlapping subproblems and optimal substructure. Memoization (Top-Down) uses recursion with a cache dictionary. Tabulation (Bottom-Up) builds an iterative lookup table starting from base cases.",
                "evaluation_criteria": "Identifying overlapping subproblems, subproblem state formulation, recursion stack vs iteration space.",
                "coaching_tips": "Define subproblem state `dp[i]` clearly before writing code."
            },
            {
                "question": "How do you implement LRU (Least Recently Used) Cache using Hash Map and Doubly Linked List?",
                "model_answer": "Combine a Hash Map storing `key -> node` for O(1) lookups with a Doubly Linked List keeping nodes in access order. On access/update, move node to head. On insertion exceeding capacity, evict tail node and remove its entry from map. Both `get` and `put` execute in O(1) time.",
                "evaluation_criteria": "Combining Hash Map with Doubly Linked List, O(1) time bounds for get/put, sentinel dummy head/tail nodes.",
                "coaching_tips": "Explain why a single Array or Linked List cannot achieve O(1) for both lookup and deletion."
            }
        ]
    elif "hr" in itype_lower or "behavioral" in itype_lower:
        return [
            {
                "question": "Tell me about a time you faced a major technical failure in a project and how you handled it.",
                "model_answer": "During a database migration, an index lock caused production API latency to spike. (Situation & Task). I immediately rolled back deployment, alerted stakeholders via Slack, and led post-mortem investigation. (Action). I added staging stress tests and automated CI migration validation. (Result). Reduced deployment incident rates by 80%.",
                "evaluation_criteria": "STAR method adherence, ownership without blaming, actionable remediation, and metric-driven results.",
                "coaching_tips": "Focus 60% of time on Action and Result steps; quantify impact with metrics."
            },
            {
                "question": "How do you handle tight deadlines and competing priorities from multiple stakeholders?",
                "model_answer": "I prioritize tasks using the Eisenhower Matrix and MoSCoW framework. I meet with product managers to clarify core MVP requirements, break work into iterative sprints, and communicate trade-offs proactively when scope changes arise.",
                "evaluation_criteria": "Prioritization methodology, transparent communication, MVP scope control, stakeholder alignment.",
                "coaching_tips": "Mention specific prioritization frameworks and proactive risk escalation."
            },
            {
                "question": "Describe a scenario where you disagreed with a senior engineer or manager on a technical decision.",
                "model_answer": "A senior engineer advocated building a custom message queue, while I advocated AWS SQS. (Situation). I created a benchmark comparison analyzing maintenance overhead, cost, and reliability. (Action). We reviewed data together and adopted SQS, saving 3 weeks of dev effort. (Result).",
                "evaluation_criteria": "Data-driven communication, constructive disagreement ('Disagree and Commit'), professional collaboration.",
                "coaching_tips": "Highlight backing opinions with objective benchmarks rather than emotion."
            },
            {
                "question": "Tell me about a time you mentored a junior team member or onboarding colleague.",
                "model_answer": "I paired with a newly hired developer struggles with microservices architecture. I established weekly 1-on-1 code reviews, created architectural wiki guides, and broke down complex tickets into incremental milestones. The colleague delivered independent features within 4 weeks.",
                "evaluation_criteria": "Empathy, leadership initiative, knowledge sharing, and measurable growth outcomes.",
                "coaching_tips": "Demonstrate patience and structured pedagogical mentoring."
            },
            {
                "question": "Why do you want to join our company and what draws you to this role?",
                "model_answer": "I admire your platform's scale and commitment to robust agentic AI architecture. My background in distributed systems and continuous delivery aligns directly with your mission to empower developers through intelligent automation.",
                "evaluation_criteria": "Company research, alignment with core mission, genuine enthusiasm, and skill match.",
                "coaching_tips": "Connect personal accomplishments to 2-3 specific engineering achievements of the company."
            },
            {
                "question": "How do you handle constructive criticism and feedback on your pull requests?",
                "model_answer": "I view PR code reviews as learning opportunities. I review comments objectively, ask clarifying questions on suggestions, update code promptly with test coverage, and thank reviewers for identifying edge cases.",
                "evaluation_criteria": "Ego-free growth mindset, collaboration, code quality standards.",
                "coaching_tips": "Emphasize separating personal identity from code execution."
            },
            {
                "question": "Describe a project where you had to quickly learn a new technology or framework.",
                "model_answer": "Our team needed real-time WebSockets, but I only had REST experience. Within 3 days, I built a prototype using Socket.io, completed documentation, and load-tested concurrent connections, successfully shipping the live chat feature on schedule.",
                "evaluation_criteria": "Self-directed learning speed, adaptability, practical application, and execution under pressure.",
                "coaching_tips": "Outline your systematic learning process: Docs -> Prototype -> Load Test -> Production."
            },
            {
                "question": "How do you manage work-life balance and prevent burnout during high-intensity delivery cycles?",
                "model_answer": "I set clear boundary expectations, practice time-blocking for deep focus work, automate repetitive tasks, and communicate capacity constraints to team leads early during sprint planning.",
                "evaluation_criteria": "Self-awareness, time management, sustainable productivity, proactive communication.",
                "coaching_tips": "Highlight automation and sprint capacity management as burnout preventatives."
            }
        ]
    elif "business" in role_lower:
        return [
            {
                "question": "What is the difference between Functional and Non-Functional Requirements?",
                "model_answer": "Functional Requirements define WHAT the system should do (e.g., process credit card payments). Non-Functional Requirements specify HOW the system should perform (e.g., SLA latency under 200ms, 99.99% availability, security compliance).",
                "evaluation_criteria": "Distinction between business features vs quality attributes (usability, reliability, performance).",
                "coaching_tips": "Provide concrete examples of each type and explain why non-functional requirements dictate technical architecture."
            },
            {
                "question": "How do you bridge the gap between technical developers and non-technical business stakeholders?",
                "model_answer": "I act as a translator using domain models, user stories, Wireframes, and BPMN. I facilitate alignment workshops, establish shared vocabulary, and validate acceptance criteria to ensure business intent matches technical execution.",
                "evaluation_criteria": "Stakeholder management, communication adaptation, requirements traceability, and active listening.",
                "coaching_tips": "Emphasize using visual diagrams (use case diagrams, flowcharts) rather than technical jargon."
            },
            {
                "question": "What is the difference between a Business Requirement Document (BRD) and System Requirement Specification (SRS)?",
                "model_answer": "A BRD focuses on business goals, problem statements, and ROI objectives from the business perspective. An SRS translates BRD goals into detailed technical specs, data flow diagrams, API interactions, and system behavior for developers.",
                "evaluation_criteria": "Target audience differentiation, level of abstraction, and requirements lifecycle mapping.",
                "coaching_tips": "Highlight that BRD answers 'WHY' and 'WHAT', while SRS details 'HOW' systems behave."
            },
            {
                "question": "How do you perform Gap Analysis in a Business Process?",
                "model_answer": "Gap Analysis evaluates the difference between current state ('As-Is') and desired future state ('To-Be'). Steps: 1. Mapping As-Is workflows. 2. Defining target To-Be capabilities. 3. Identifying missing features, bottlenecks, and data gaps. 4. Formulating a prioritized remediation roadmap.",
                "evaluation_criteria": "As-Is vs To-Be process mapping, root cause identification, and actionable roadmap creation.",
                "coaching_tips": "Walk through a practical example like transitioning from manual paper approvals to automated digital workflows."
            },
            {
                "question": "Explain the INVEST criteria for writing effective User Stories.",
                "model_answer": "INVEST stands for Independent, Negotiable, Valuable, Estimable, Small (fits within a sprint), and Testable (has explicit acceptance criteria).",
                "evaluation_criteria": "Agile methodology mastery, story sizing, and acceptance criteria formatting (Given-When-Then).",
                "coaching_tips": "Write a sample user story: 'As a customer, I want to filter products by price, so that I can find affordable items.'"
            },
            {
                "question": "How do you conduct Requirement Elicitation with conflicting stakeholders?",
                "model_answer": "I conduct 1-on-1 interviews, facilitated alignment workshops, and MoSCoW prioritization sessions. I ground decisions in objective user data and business ROI metrics to resolve conflicting opinions.",
                "evaluation_criteria": "Conflict resolution, active listening, objective data usage, consensus building.",
                "coaching_tips": "Focus stakeholders on shared business goals rather than subjective feature choices."
            },
            {
                "question": "What is Root Cause Analysis and how do you use the 5 Whys technique?",
                "model_answer": "Root Cause Analysis identifies underlying causes of operational problems. The 5 Whys iteratively asks 'Why did this happen?' five times until the foundational process or systemic flaw is revealed.",
                "evaluation_criteria": "Problem-solving depth, 5 Whys methodology, Fishbone diagram usage.",
                "coaching_tips": "Walk through a real supply chain delay or bug outbreak using 5 Whys."
            },
            {
                "question": "How do you manage Scope Creep during project execution?",
                "model_answer": "I enforce formal Change Control procedures. When new requirements are requested: 1. Analyze impact on budget and schedule. 2. Present impact to Change Control Board. 3. Swap lower-priority backlog items before accepting scope changes.",
                "evaluation_criteria": "Change control process, impact analysis, backlog trade-offs, stakeholder management.",
                "coaching_tips": "Reframe saying 'no' into offering trade-off choices with transparent impact analysis."
            }
        ]
    elif "qa" in role_lower:
        return [
            {
                "question": "What is the difference between Manual Testing and Automated Testing?",
                "model_answer": "Manual Testing involves human testers executing test cases step-by-step without scripts, ideal for exploratory and UI/UX testing. Automated Testing uses test scripts and execution frameworks (Selenium, Playwright) to execute repetitive regression and performance tests rapidly.",
                "evaluation_criteria": "Test coverage trade-offs, ROI of automation, exploratory vs regression selection.",
                "coaching_tips": "Explain that automation supplements manual testing by freeing testers for high-value exploratory testing."
            },
            {
                "question": "How do you design an Automation Framework architecture using Page Object Model (POM)?",
                "model_answer": "Page Object Model (POM) creates an abstraction layer where each web page has a corresponding class storing element locators and action methods. Test scripts interact with Page Objects rather than hardcoded DOM locators, reducing code maintenance when UI changes occur.",
                "evaluation_criteria": "POM design pattern, locator encapsulation, code reusability, maintenance reduction.",
                "coaching_tips": "Explain how POM isolates UI locator changes to a single page object file."
            },
            {
                "question": "What is Boundary Value Analysis (BVA) and Equivalence Partitioning (EP)?",
                "model_answer": "Equivalence Partitioning divides input data into valid/invalid partitions. Boundary Value Analysis tests values at partition boundaries (e.g. for age 18-65: test 17, 18, 19, 64, 65, 66) as edge cases frequently hide defects.",
                "evaluation_criteria": "Black box test design, boundary identification, test case minimization.",
                "coaching_tips": "Demonstrate with a numerical boundary example (e.g., password length 8-20 characters)."
            },
            {
                "question": "How do you write a complete Bug / Defect Report?",
                "model_answer": "A professional Defect Report contains: 1. Unique Defect ID & Title. 2. Environment Details. 3. Severity & Priority. 4. Detailed Steps to Reproduce. 5. Expected vs Actual Result. 6. Attachments (screenshots, logs).",
                "evaluation_criteria": "Clarity of reproduction steps, Severity vs Priority distinction, diagnostic evidence attached.",
                "coaching_tips": "Highlight the difference between Severity (System Crash) and Priority (Logo typo on landing page)."
            },
            {
                "question": "What is the difference between Regression Testing and Smoke Testing?",
                "model_answer": "Smoke Testing is a quick build-verification test running a small subset of critical cases to verify build stability. Regression Testing is a comprehensive suite verifying recent code changes haven't broken existing features.",
                "evaluation_criteria": "Scope, execution frequency, CI/CD pipeline integration, execution speed.",
                "coaching_tips": "Frame Smoke testing as an automated pipeline gatekeeper before running full Regression suites."
            },
            {
                "question": "What is API Testing and how do you validate endpoints using Postman or REST Assured?",
                "model_answer": "API testing validates server endpoints directly. I verify: HTTP Status Codes (200, 201, 400), Response Payload JSON schema, Headers, Latency, Auth tokens (Bearer/JWT), and Data Integrity against database records.",
                "evaluation_criteria": "API payload validation, status code verification, negative scenario edge-case testing.",
                "coaching_tips": "Mention testing negative scenarios: missing mandatory keys, invalid auth tokens, SQL injection payloads."
            },
            {
                "question": "How do you handle Flaky Tests in Automated Test Suites?",
                "model_answer": "Flaky tests pass/fail inconsistently. I resolve flakiness by: 1. Replacing hardcoded `sleep()` with explicit dynamic waits. 2. Isolating test data using unique dynamic test entities. 3. Ensuring independent test execution without global state dependencies.",
                "evaluation_criteria": "Root cause analysis, dynamic waits usage, test isolation.",
                "coaching_tips": "Strongly advise against `Thread.sleep()` and recommend explicit element visibility waiters."
            },
            {
                "question": "What is the Requirement Traceability Matrix (RTM)?",
                "model_answer": "RTM is a grid document linking business requirements directly to corresponding test cases and defect IDs, ensuring 100% test coverage and tracking execution status throughout STLC.",
                "evaluation_criteria": "Requirements coverage, backward/forward traceability, test gap analysis.",
                "coaching_tips": "Explain how RTM proves to stakeholders that all business requirements were thoroughly tested."
            }
        ]
    elif "devops" in role_lower:
        return [
            {
                "question": "Can you explain how a CI/CD pipeline automates build, test, and deployment stages, and how Docker and Kubernetes fit in?",
                "model_answer": "CI automatically compiles code, runs unit tests, and builds Docker container images upon git push. CD automates deploying container images into staging/production Kubernetes clusters, executing rolling updates with zero downtime.",
                "evaluation_criteria": "Pipeline automation, containerization concepts, Kubernetes deployment strategies, rolling updates.",
                "coaching_tips": "Step through a complete git push -> CI build -> Docker image -> Kubernetes deployment pipeline."
            },
            {
                "question": "What is Infrastructure as Code (IaC) and how do Terraform and CloudFormation work?",
                "model_answer": "IaC manages infrastructure provisioning via declarative configuration files. Terraform uses HCL and cloud provider plugins to manage multi-cloud resources, tracking state in `.tfstate` files to perform idempotent deployments.",
                "evaluation_criteria": "Declarative vs imperative syntax, state file management, idempotency, drift detection.",
                "coaching_tips": "Explain remote state storage with state locking (S3 + DynamoDB) to prevent concurrent deployment collisions."
            },
            {
                "question": "What is the difference between Docker Containerization and Virtual Machines?",
                "model_answer": "VMs virtualize hardware stacks running complete guest OS instances on hypervisors. Docker containers virtualize the OS kernel, sharing the host OS kernel to run lightweight isolated application user-spaces with sub-second startup times.",
                "evaluation_criteria": "Resource overhead, isolation guarantees, kernel sharing, image layer caching.",
                "coaching_tips": "Highlight portability, startup speed, and resource efficiency."
            },
            {
                "question": "How do Kubernetes Pods, Deployments, Services, and Ingress Work?",
                "model_answer": "Pods are the smallest deployable units containing 1+ containers. Deployments manage Pod replica counts, rolling updates, and self-healing. Services provide stable internal IP load balancing. Ingress manages external HTTP/HTTPS routing into cluster Services.",
                "evaluation_criteria": "K8s architecture primitives, rolling updates, service discovery, external traffic routing.",
                "coaching_tips": "Walk through a full deployment flow: Ingress -> Service -> Deployment -> Pods."
            },
            {
                "question": "How do you implement Monitoring, Logging, and Observability in DevOps (Prometheus & Grafana)?",
                "model_answer": "Observability relies on Metrics (Prometheus), Logs (ELK/Loki), and Traces (Jaeger). Prometheus scrapes time-series metrics, Grafana visualizes dashboards and alerts, and centralized logs enable rapid root-cause analysis.",
                "evaluation_criteria": "Three pillars of observability (Metrics, Logs, Traces), Prometheus scraping, alert thresholding.",
                "coaching_tips": "Explain how proactive metrics alerts detect memory leaks before outages occur."
            },
            {
                "question": "What is the Shared Responsibility Model in Cloud Computing (AWS/Azure/GCP)?",
                "model_answer": "Cloud Providers manage security OF the cloud (physical data centers, host hardware, hypervisors, global networking). Customers manage security IN the cloud (guest OS patching, IAM policies, firewall rules, data encryption, app code).",
                "evaluation_criteria": "IaaS/PaaS/SaaS responsibility shifts, customer vs cloud provider boundaries.",
                "coaching_tips": "Emphasize that customer data security and access management always remain customer responsibilities."
            },
            {
                "question": "What is Zero Trust Architecture and its core principles?",
                "model_answer": "Zero Trust operates on 'Never Trust, Always Verify'. It assumes network perimeters are compromised and requires explicit authentication, continuous authorization, least privilege access controls, and end-to-end encryption.",
                "evaluation_criteria": "Micro-segmentation, identity-centric security, least privilege, continuous session verification.",
                "coaching_tips": "Contrast traditional perimeter-based security (castle-and-moat) with Zero Trust."
            },
            {
                "question": "Explain Cloud High Availability, Fault Tolerance, and Disaster Recovery (RPO / RTO).",
                "model_answer": "High Availability ensures continuous operation using load balancers and multi-AZ auto-scaling. Disaster Recovery handles region failures. Recovery Point Objective (RPO) is maximum acceptable data loss. Recovery Time Objective (RTO) is maximum acceptable downtime.",
                "evaluation_criteria": "Multi-AZ vs Multi-Region design, RPO vs RTO calculation, data replication strategies.",
                "coaching_tips": "Illustrate with active-active vs active-passive database replication across cloud regions."
            }
        ]
    else:
        return [
            {
                "question": "What is the difference between an Abstract Class and an Interface in OOP?",
                "model_answer": "An Abstract Class represents an 'is-a' relationship and serves as a base class with shared state (instance variables) and concrete method implementations. An Interface defines a 'can-do' contract defining public method signatures without maintaining state. Subclasses support single inheritance for abstract classes, but can implement multiple interfaces.",
                "evaluation_criteria": "Single vs. multiple inheritance rules, state vs. stateless contract, and design choice justification.",
                "coaching_tips": "Structure your answer by comparing Purpose ('is-a' vs 'can-do'), State (instance variables vs constants), and Inheritance."
            },
            {
                "question": "How do you solve the Two Sum problem efficiently in O(n) time complexity?",
                "model_answer": "Use a Hash Map (dictionary) to store each number's complement (target - num) as you iterate through the array. For each element `num` at index `i`, check if `target - num` exists in the hash map. If found, return `[map[complement], i]`. Otherwise, store `map[num] = i`.",
                "evaluation_criteria": "Hash table lookup O(1) efficiency, O(n) single pass time complexity vs O(n^2) brute force.",
                "coaching_tips": "Briefly state brute force O(n^2), then present the Hash Map O(n) optimization with time/space complexity analysis."
            },
            {
                "question": "Explain SOLID principles in Object-Oriented Software Design.",
                "model_answer": "SOLID stands for Single Responsibility (one reason to change), Open/Closed (open for extension, closed for modification), Liskov Substitution (subtypes must be substitutable for base types), Interface Segregation (fine-grained interfaces), and Dependency Inversion (depend on abstractions, not concretions).",
                "evaluation_criteria": "Explaining all 5 acronym letters accurately with real-world refactoring examples.",
                "coaching_tips": "Use concise single-sentence definitions for each letter, then provide a code refactoring example for Dependency Inversion."
            },
            {
                "question": "What is Database Normalization and why is 3NF important?",
                "model_answer": "Normalization reduces data redundancy and prevents update anomalies by organizing tables into normal forms. 1NF eliminates duplicate columns; 2NF removes partial key dependencies; 3NF removes transitive dependencies where non-key attributes depend on other non-key attributes.",
                "evaluation_criteria": "Identifying anomalies (insertion, deletion, update) and defining functional dependencies.",
                "coaching_tips": "Use a simple Employee-Department table example to demonstrate 3NF decomposition."
            },
            {
                "question": "Compare SQL (Relational) vs NoSQL (Document/Key-Value) Databases.",
                "model_answer": "SQL databases (PostgreSQL, MySQL) enforce ACID transactions, structured schemas, and relational joins, ideal for complex transactions. NoSQL databases (MongoDB, DynamoDB) offer dynamic schemas, horizontal scaling (sharding), and eventual consistency (BASE), ideal for high throughput unstructured data.",
                "evaluation_criteria": "CAP theorem trade-offs, ACID vs BASE consistency, and scaling strategies.",
                "coaching_tips": "Frame the choice around data access patterns and scaling needs rather than popularity."
            },
            {
                "question": "What is Microservices Architecture and how does it compare to Monolithic?",
                "model_answer": "A Monolith packages all business domains into a single deployment unit. Microservices decompose functionality into independently deployable, loosely coupled services communicating via APIs (gRPC/REST) or message queues (Kafka).",
                "evaluation_criteria": "Domain-driven design, service boundaries, Saga pattern, operational overhead.",
                "coaching_tips": "Discuss operational complexity alongside microservice benefits."
            },
            {
                "question": "Explain the CAP Theorem in Distributed Systems.",
                "model_answer": "CAP theorem states that a distributed data store can simultaneously provide at most two of three guarantees: Consistency, Availability, and Partition Tolerance.",
                "evaluation_criteria": "Understanding that Partition Tolerance is mandatory, leaving choices between CP and AP.",
                "coaching_tips": "Give concrete examples: HBase/MongoDB (CP) vs Cassandra/DynamoDB (AP)."
            },
            {
                "question": "What is Dependency Injection and how does it promote testability?",
                "model_answer": "Dependency Injection (DI) is an implementation of Dependency Inversion where dependencies are injected from the outside rather than created internally, allowing injection of mock objects during unit testing.",
                "evaluation_criteria": "Decoupling components, constructor injection vs field injection, test mock creation.",
                "coaching_tips": "Contrast tight coupling with loose coupling constructor injection."
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
