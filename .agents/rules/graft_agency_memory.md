# Autonomous Architecture: Graft, Agency Swarm & Codebase Memory Protocol

This rule establishes **Graft (Code Graph Navigation)**, **Agency Agents (Specialist Swarm Orchestration)**, and **Codebase Memory (Persistent Cross-Session Knowledge)** across projects.

---

## 1. Graft: Static Graph & Precision Slicing
- **Self-Activation Trigger**: Whenever exploring unfamiliar code, tracing cross-file dependencies, locating symbol definitions/callers, or planning refactors.
- **Rule**: Prioritize static symbol mapping and targeted slice viewing (`view_file` with `StartLine`/`EndLine`) over reading whole files or running redundant broad searches.

---

## 2. Agency Agents: Multi-Specialist Role Swarm
- **Self-Activation Trigger**: Whenever handling multi-step tasks, new feature development, debugging, or releases.
- **Swarm Execution**:
  - **🏛️ Architect**: Establish interface contracts, schemas, and clean abstractions before writing code.
  - **⚡ Core Engineer**: Write complete, robust, production-grade implementations without placeholders.
  - **🛡️ Security Auditor**: Sanitize inputs, enforce boundary validations, and secure error disclosure.
  - **🧪 QA Engineer**: Write and execute comprehensive test suites (unit, integration, regression) to prove correctness.
  - **🚢 DevOps**: Ensure atomic commits, clean git working trees, and continuous deployment sync.

---

## 3. Codebase Memory: Persistent Knowledge & Decisions
- **Self-Activation Trigger**: At session start and whenever encountering non-obvious project constraints or bugs.
- **Protocol**:
  - **Recall First**: Check `.agents/memory/` before performing redundant analysis.
  - **Record Decisions**: Document non-trivial architectural decisions, gotchas, and environment nuances in `.agents/memory/` so future sessions build upon them automatically.
  - **Zero Regressions**: Uphold recorded constraints, SLAs, and performance thresholds across iterations.
