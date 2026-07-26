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
    Generate a complete Demo Q&A Report featuring 5 exemplar questions, 
    ideal model answers, key evaluation criteria, and coaching tips.
    
    :param target_role: Selected job role (e.g., "Software Engineering", "DevOps", "Business Analyst", "QA Engineering").
    :param interview_type: Selected interview round (e.g., "Technical", "HR", "Coding").
    :return: List of dicts representing sample questions with ideal answers and coaching advice.
    """
    query = f"{interview_type} interview questions model answers evaluation criteria {target_role} architecture principles"
    retrieved_chunks = get_relevant_chunks(query=query, top_k=6)

    context_str = "\n---\n".join([c.get("text", "") for c in retrieved_chunks]) if retrieved_chunks else "No specific ground truth document retrieved."

    prompt = f"""You are an expert AI Interview Coach and Hiring Manager. Generate a comprehensive Demo Questions & Model Answers Report for a '{target_role}' candidate in an '{interview_type}' interview round.

Reference Knowledge Base Context:
{context_str}

Generate 5 high-quality, realistic, and diverse interview questions tailored specifically to {target_role} ({interview_type} round). For EACH question, provide an ideal model answer, key evaluation criteria, and actionable coaching tips.

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
    }}
  ]
}}

Format requirements:
1. Output MUST contain 5 complete Q&A report items.
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

    # Role-based Q&A Report Fallbacks
    itype_lower = interview_type.lower()
    role_lower = target_role.lower()

    if "business" in role_lower:
        return [
            {
                "question": "What is the difference between Functional and Non-Functional Requirements?",
                "model_answer": "Functional Requirements define WHAT the system should do (e.g., process credit card payments). Non-Functional Requirements specify HOW the system should perform (e.g., SLA latency under 200ms, 99.99% availability, security compliance).",
                "evaluation_criteria": "Distinction between business features vs quality attributes (usability, reliability, performance).",
                "coaching_tips": "Provide concrete examples of each type and explain why non-functional requirements dictate technical architecture."
            },
            {
                "question": "How do you bridge the gap between technical developers and non-technical business stakeholders?",
                "model_answer": "I act as a translator using clear domain models, user stories, Wireframes, and BPMN. I facilitate alignment workshops, establish shared vocabulary, and validate acceptance criteria to ensure business intent matches technical execution.",
                "evaluation_criteria": "Stakeholder management, communication adaptation, requirements traceability, and active listening.",
                "coaching_tips": "Emphasize using visual diagrams (use case diagrams, flowcharts) rather than technical jargon."
            },
            {
                "question": "What is the difference between a Business Requirement Document (BRD) and System Requirement Specification (SRS)?",
                "model_answer": "A BRD focuses on business goals, problem statements, and high-level ROI objectives from the business perspective. An SRS translates BRD goals into detailed technical specs, data flow diagrams, API interactions, and system behavior for developers.",
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
            }
        ]
    elif "qa" in role_lower:
        return [
            {
                "question": "What is the difference between Manual Testing and Automated Testing?",
                "model_answer": "Manual Testing involves human testers executing test cases step-by-step without scripts, ideal for exploratory, UI/UX, and initial ad-hoc testing. Automated Testing uses test scripts and execution frameworks (Selenium, Playwright, PyTest) to execute repetitive regression and performance tests rapidly.",
                "evaluation_criteria": "Test coverage trade-offs, ROI of automation, and exploratory vs regression test selection.",
                "coaching_tips": "Explain that automation supplements manual testing by freeing testers for high-value exploratory testing."
            },
            {
                "question": "How do you design an Automation Framework architecture using Page Object Model (POM)?",
                "model_answer": "Page Object Model (POM) creates an abstraction layer where each web page has a corresponding class storing web element locators and action methods. Test scripts interact with Page Objects rather than hardcoded DOM locators, reducing code duplication and maintenance when UI changes occur.",
                "evaluation_criteria": "POM design pattern, locator encapsulation, code reusability, and maintenance reduction.",
                "coaching_tips": "Explain how POM isolates UI locator changes to a single page object file."
            },
            {
                "question": "What is Boundary Value Analysis (BVA) and Equivalence Partitioning (EP)?",
                "model_answer": "Equivalence Partitioning divides input data into valid and invalid partitions where the system behaves similarly. Boundary Value Analysis tests values at the boundaries of these partitions (e.g. for age input 18-65: test 17, 18, 19, 64, 65, 66) as edge cases frequently hide defects.",
                "evaluation_criteria": "Black box test design techniques, boundary identification, and test case minimization.",
                "coaching_tips": "Demonstrate with a numerical boundary example (e.g., password length 8-20 characters)."
            },
            {
                "question": "How do you write a complete Bug / Defect Report?",
                "model_answer": "A professional Defect Report contains: 1. Unique Defect ID & Title. 2. Environment Details (OS, Browser). 3. Severity & Priority. 4. Detailed Steps to Reproduce. 5. Expected vs Actual Result. 6. Diagnostic Attachments (screenshots, console logs).",
                "evaluation_criteria": "Clarity of reproduction steps, Severity vs Priority distinction, and diagnostic evidence attached.",
                "coaching_tips": "Highlight the difference between Severity (System Crashes) and Priority (Logo typo on landing page)."
            },
            {
                "question": "What is the difference between Regression Testing and Smoke Testing?",
                "model_answer": "Smoke Testing is a quick build-verification test running a small subset of critical test cases to verify build stability. Regression Testing is a comprehensive suite verifying that recent code changes haven't broken existing, unchanged features.",
                "evaluation_criteria": "Scope, execution frequency, build pipeline integration, and execution speed.",
                "coaching_tips": "Frame Smoke testing as an automated pipeline gatekeeper before running full Regression suites."
            }
        ]
    elif "devops" in role_lower:
        return [
            {
                "question": "Can you explain how a CI/CD pipeline automates build, test, and deployment stages, and how Docker and Kubernetes fit in?",
                "model_answer": "Continuous Integration (CI) automatically compiles code, runs unit tests, and builds lightweight Docker container images upon git push. Continuous Deployment (CD) automates deploying container images into staging and production Kubernetes clusters, executing rolling updates with zero downtime.",
                "evaluation_criteria": "Pipeline automation, containerization concepts, Kubernetes deployment strategies, and rolling updates.",
                "coaching_tips": "Step through a complete git push -> CI build -> Docker image -> Kubernetes deployment pipeline."
            },
            {
                "question": "What is Infrastructure as Code (IaC) and how do Terraform and CloudFormation work?",
                "model_answer": "IaC manages infrastructure provisioning via declarative configuration files rather than manual dashboard clicks. Terraform uses HCL (HashiCorp Configuration Language) and cloud provider plugins to manage multi-cloud resources, tracking infrastructure state in a `.tfstate` file to perform idempotent deployments.",
                "evaluation_criteria": "Declarative vs imperative syntax, state file management, idempotency, and drift detection.",
                "coaching_tips": "Explain remote state storage with state locking (S3 + DynamoDB) to prevent concurrent deployment collisions."
            },
            {
                "question": "What is the difference between Docker Containerization and Virtual Machines?",
                "model_answer": "VMs virtualize full hardware hardware stacks and run complete guest OS instances on a hypervisor. Docker containers virtualize the operating system kernel, sharing the host OS kernel to run lightweight, isolated application user-spaces with sub-second startup times.",
                "evaluation_criteria": "Resource overhead, isolation guarantees, kernel sharing, and image layer caching.",
                "coaching_tips": "Highlight portability, startup speed, and resource efficiency."
            },
            {
                "question": "How do Kubernetes Pods, Deployments, Services, and Ingress Work?",
                "model_answer": "Pods are the smallest deployable units containing 1+ containers sharing storage/network. Deployments manage Pod replica counts, rolling updates, and self-healing. Services provide stable internal IP networking and load balancing across Pod replicas. Ingress manages external HTTP/HTTPS routing into cluster Services.",
                "evaluation_criteria": "K8s architecture primitives, rolling updates, service discovery, and external traffic routing.",
                "coaching_tips": "Walk through a full deployment flow: Ingress -> Service -> Deployment -> Pods."
            },
            {
                "question": "How do you implement Monitoring, Logging, and Observability in DevOps (Prometheus & Grafana)?",
                "model_answer": "Observability relies on Metrics (Prometheus), Logs (ELK Stack/Loki), and Traces (Jaeger). Prometheus scrapes time-series metrics from application endpoints, Grafana visualizes dashboards and metrics alerts, and centralized logs enable rapid incident root-cause analysis.",
                "evaluation_criteria": "Three pillars of observability (Metrics, Logs, Traces), Prometheus scraping, and alert thresholding.",
                "coaching_tips": "Explain how proactive metrics alerts detect memory leaks before outages occur."
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
