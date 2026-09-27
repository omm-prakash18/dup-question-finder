---
name: agency-swarm
description: Role-based multi-agent orchestration and specialist swarm delegation framework. Use when breaking down complex goals into specialized role perspectives (Architect, Core Engineer, Security Auditor, QA/Test Engineer, DevOps), resolving multi-disciplinary tasks, or executing rigorous multi-axis verification before shipping. Automatically activates during complex feature builds, refactors, and end-to-end debugging loops.
---

# Agency Swarm: Role-Based Multi-Agent Orchestrator

Agency Swarm organizes agent execution into focused specialist personas with strict handoffs, responsibility domains, and quality gates.

## Specialist Agent Personas

```
                     ┌──────────────────────────────┐
                     │     Lead Architect Agent     │
                     │  (Planning, Contracts, Specs)│
                     └──────────────┬───────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
          ▼                         ▼                         ▼
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│   Core Engineer   │     │ Security Auditor  │     │   QA Engineer     │
│(Code, Math, Logic)│     │(Auth, Sanitization│     │(Test Suites, Edge │
│                   │     │ Input Boundaries) │     │ Cases, Regression)│
└─────────┬─────────┘     └─────────┬─────────┘     └─────────┬─────────┘
          │                         │                         │
          └─────────────────────────┼─────────────────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │    DevOps & Release Agent    │
                     │  (Docker, CI/CD, Git, Prod)  │
                     └──────────────────────────────┘
```

### 1. 🏛️ Lead Architect Agent
- **Domain**: High-level system design, schema definitions, API contracts, modularity, and data structures.
- **Rules**: Defines clear interfaces before coding; prevents unnecessary dependencies (YAGNI).

### 2. ⚡ Core Engineer Agent
- **Domain**: Algorithm implementation, ML/NLP pipelines, vector search, database queries, and async performance.
- **Rules**: Writes production-ready, clean, well-typed code without stubs or placeholders.

### 3. 🛡️ Security & Hardening Agent
- **Domain**: Input validation, sanitization, boundary checks, OWASP top 10, error disclosure, and secrets safety.
- **Rules**: Ensures no unchecked user input crashes services or leaks internal system state.

### 4. 🧪 QA & Verification Agent
- **Domain**: Comprehensive test suite creation, unit tests, property/adversarial tests, load testing, and regression suites.
- **Rules**: Enforces *Evidence Over Assertions* — nothing is marked done until tests pass 100%.

### 5. 🚢 DevOps & Release Agent
- **Domain**: Packaging, Docker builds, environment configuration, automated commits, and repository synchronization.
- **Rules**: Keeps working tree clean and commits atomic, well-documented changes.

## Automatic Swarm Protocol

Whenever a task is received:
1. **Decomposition**: Architect identifies required specialist domains.
2. **Sequential / Parallel Execution**: Core engineer writes logic; security checks boundaries; QA writes tests.
3. **Verification Gate**: QA executes tests and confirms output logs before completion.
4. **Release**: DevOps packages and commits clean code.
