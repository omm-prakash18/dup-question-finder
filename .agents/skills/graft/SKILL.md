---
name: graft
description: Static code graph, AST symbol navigator, and dependency tree mapper. Use when navigating unfamiliar code, locating function/class definitions and references across modules, understanding cross-file call hierarchies, or minimizing token usage by exploring structural code maps before opening large files. Automatically activates when exploring multi-module architecture or tracing data flows.
---

# Graft: Codebase Symbol & Dependency Graph Navigator

Graft gives the AI agent a structural, token-efficient map of the codebase. Instead of blindly reading or grepping through large files, Graft navigates functions, classes, imports, call hierarchies, and cross-module dependencies using static symbol mapping.

## Core Capabilities

1. **Symbol & Interface Mapping**: Instantly locate where classes, functions, routes, and schemas are defined and imported.
2. **Dependency & Call Hierarchy Graphing**: Trace data and control flow across pipelines (e.g. `API -> Router -> Pipeline -> Model -> Database`).
3. **Targeted Slice Reading**: Pinpoint exact line ranges to view or edit, preserving context budget.
4. **Impact Analysis**: Determine what downstream modules, tests, or APIs will be affected before refactoring a function or schema.

## When Graft Self-Activates

Graft activates automatically when:
- Tracing where a specific function, class, or variable is defined or called across files.
- Refactoring public interfaces, schemas, or models to prevent breaking downstream consumers.
- Onboarding to a new subsystem or module without scanning entire folders.
- Investigating import cycles, missing dependencies, or module boundaries.

## Graft Navigation Protocol

```
[1. Target Symbol / Path] ──► [2. Query AST / Symbol Index] ──► [3. Map Dependencies & Callers]
                                                                        │
[5. Execute Precise Action] ◄── [4. Extract Precise Slice / Range] ◄───┘
```

### Step 1: Symbol Identification
Identify the target symbol (e.g. `SimilarQuestionRetriever`, `find_similar`, `clean_classical`).

### Step 2: Cross-File Trace
Trace exports and imports across module boundaries:
- Definitions (`def`, `class`, type aliases)
- Instantiations & call sites
- Schema dependencies & Pydantic models

### Step 3: Module Boundary Graphing
Graph how data flows between layers:
```
[Data Ingestion (src/data)] ──► [Feature Pipeline (src/features)] ──► [Model/FAISS (src/models)]
                                                                               │
                                                                               ▼
[Test Suites (tests/)] ◄────────── [FastAPI Microservice (api/)] ◄─────────────┘
```

### Step 4: Line Range Slicing
Use exact line ranges (`StartLine`, `EndLine`) when reading or modifying code rather than replacing whole files.
