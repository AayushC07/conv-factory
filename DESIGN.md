# DESIGN.md — Conversation Factory

## Table of Contents

1. [Architecture & Decisions](#1-architecture--decisions)
2. [Data Model & Tool Registry](#2-data-model--tool-registry)
3. [Tool Graph & Sampling](#3-tool-graph--sampling)
4. [Multi-Agent System Design](#4-multi-agent-system-design) — [4.1 Roles & protocol](#41-roles-and-communication-protocol) · [4.2 Structured output](#42-structured-output-json-mode) · [4.3 Multi-step / multi-tool split](#43-multi-step-and-multi-tool-enforcement-generate) · [4.4 Parallel label vs execution](#44-parallel-chain-label-vs-runtime-execution) · [4.5 Disambiguation](#45-multi-turn-disambiguation)
5. [Offline Execution Model](#5-offline-execution-model)
6. [Context Management Design](#6-context-management-design)
7. [Quality Evaluation Pipeline](#7-quality-evaluation-pipeline)
8. [Prompt Design](#8-prompt-design)
9. [Diversity & Quality Analysis](#9-diversity--quality-analysis)
10. [What I Would Do Next](#10-what-i-would-do-next)
11. [Changelog](#11-changelog)

---

## 1. Architecture & Decisions

### System Overview

The system is a pipeline with five stages, each implemented as a composable module:

```
ToolBench Data → [Registry] → [Tool Graph] → [Sampler] → [Generator] → [Evaluator] → Dataset
                                                  ↑                          ↓
                                          [Steering] ←──────────────── [Quality Scores]
```

**Key architectural principle: separation of "what to generate" from "how to generate."** The sampler decides *which* tools to chain (structural planning), while the multi-agent generator decides *how* to weave those tools into a natural conversation (content generation). This separation means we can improve sampling diversity independently of conversation quality, and vice versa.

### Why a Pipeline, Not an End-to-End LLM

An alternative design would be to prompt a single LLM to generate entire conversations including tool selections. I rejected this for three reasons:

1. **Controllability**: With a pipeline, we can enforce structural constraints (multi-step ratio, category coverage) at the sampling stage rather than hoping the LLM complies.
2. **Diversity**: Graph-based sampling with diversity steering produces far more varied tool combinations than an LLM, which gravitates toward common patterns in its training data.
3. **Debuggability**: When a conversation is poor, we can isolate whether the issue was the tool chain (sampler), the scenario (planner), the conversation flow (agents), or the mock data (executor).

The tradeoff is complexity — five modules to maintain instead of one prompt. For a production data generation system, I believe this is the right call.

### Component Communication

Components communicate via well-defined data structures (Pydantic models, typed dicts):

- **Registry → Graph**: `ToolRegistry` provides normalized `Tool`/`Endpoint` objects
- **Graph → Sampler**: `ToolGraph` exposes a NetworkX graph; `ToolChainSampler` walks it
- **Sampler → Generator**: `ToolChain` (list of endpoint FQNs + pattern + metadata)
- **Generator → Evaluator**: Conversation dicts with messages, metadata, tool calls
- **Evaluator → Steering**: `JudgeScores` + conversation metadata feed back to `DiversitySteering`

All artifacts are serializable to JSON for reproducibility and inspection.

---

## 2. Data Model & Tool Registry

### Why Pydantic

ToolBench JSON is inconsistent: missing fields, varying parameter formats, inconsistent naming. I chose Pydantic v2 for the data model because:

1. **Validation with coercion**: `field_validator` normalizes type strings ("str" → STRING, "int" → INTEGER) rather than rejecting them
2. **Serialization round-trips**: `model_dump`/`model_validate` for saving/loading the registry
3. **Self-documenting schema**: The model classes serve as documentation of the canonical format

### Model Hierarchy

```
Tool
├── tool_name: str (unique identifier)
├── tool_id: str (MD5-derived, stable)
├── description: str
├── category: ToolCategory
└── endpoints: list[Endpoint]
    ├── name: str
    ├── description: str
    ├── method: str (GET/POST/...)
    ├── parameters: list[Parameter]
    │   ├── name, description, param_type
    │   ├── required: bool
    │   └── default, enum_values, example
    └── response_schema: list[ResponseField]
```

### Design Decision: Flat FQNs

I use fully-qualified names (`tool_name/endpoint_name`) as the universal identifier throughout the system. This is simpler than maintaining separate tool/endpoint ID systems, and makes graph nodes, sampled chains, and conversation tool calls all use the same format.

### Loader Resilience

The loader handles several ToolBench inconsistencies:
- **Missing parameters**: Creates `Parameter(name=..., required=False)` with string type
- **Alternative field names**: Checks `api_list`/`apis`/`endpoints`/`functions`, `tool_name`/`name`/`title`
- **Malformed files**: Logs warnings and skips, rather than failing the entire load
- **Two directory layouts**: Single JSON file or category-directory tree

### Response Schema Inference

Most ToolBench endpoints have **empty `response_schema`**, which starves the tool graph of data-flow edges. `ToolBenchLoader._infer_response_schema` uses endpoint name keywords and HTTP method to synthesize plausible `ResponseField` lists:

| Endpoint pattern | Inferred fields |
|------------------|----------------|
| `search`, `list`, `find`, `query` | `results` (array), `id`, `name`, `total_count` |
| `get`, `detail`, `fetch`, `retrieve` | `id`, `name`, `description`, plus mirrored `*_id` params |
| `book`, `create`, `reserve`, `submit` | `id`, `status`, `confirmation_code` |
| `update`, `modify`, `edit` | `id`, `status`, `updated` |
| `delete`, `cancel`, `remove` | `status`, `success` |
| Fallback POST / GET | `id` + `status` or `id` + `name` |

**Tradeoff**: These are heuristics — they increase data-flow edge count (and thus graph connectivity and sampler chain quality) at the cost of possible false edges. The executor ignores inferred schemas and always uses its richer keyword-based response templates, so mock data quality is unaffected.

---

## 3. Tool Graph & Sampling

### Graph Schema

Nodes are endpoints. Edges represent these relationship types:

| Edge Type | Typical weight | Meaning |
|-----------|----------------|---------|
| `same_tool` | 0.05–0.8 | Intra-tool steps; **forward** discovery→action edges are strong, backward edges are near-zero |
| `same_category` | 0.3 | Cross-tool, same domain (e.g., Hotels + Flights in Travel) |
| `data_flow` | 1.0 | Output of A can feed input of B (param name + type match) |
| `shared_params` | ≤0.3 | Cross-tool when both endpoints take meaningful shared inputs (`city`, `date`, …); merged onto an existing edge if one already exists |

**Same-tool ordering:** Endpoint names are assigned a coarse **operation rank** from REST-style keywords (e.g. `search`/`list` → rank 1, `get`/`detail` → rank 2, `book`/`purchase` → rank 3). Forward rank transitions get weight **0.8**, backward **0.05**. When ranks **tie**, **ToolBench declaration order** (`endpoint_index`) breaks the tie (0.5 forward along index, 0.2 backward). This is explainable in a write-up: naming conventions encode workflow phase more reliably than arbitrary JSON ordering alone.

**Why these weights?** Data-flow edges (1.0) remain the strongest signal for real I/O chaining. Same-tool forward edges use **rank gap**: adjacent ranks (e.g. search→get, gap 1) get weight **0.8**; skipping a phase (e.g. search→purchase, gap ≥ 2) gets **0.3** so typical flows are preferred. Same-tool edges bias the walk toward discovery→action without requiring a separate planner. Category and shared-parameter edges are weaker bridges for diversity.

**Summaries:** `ToolGraph.summary()` counts edges by `relation` and also reports **`data_flow_edges`** — the number of edges with `data_flow: true` (including upgrades where `relation` stayed `same_tool`).

**Data-flow weights:** Parameter-derived data-flow edges use the same coarse ranks: **adjacent** forward step (`gap == 1`) gets weight **1.0**, **skipping** a rank (`gap ≥ 2`) gets **0.3**, so `search → get_details` strongly dominates `search → purchase` when both exist.

**Category cross-tool weights:** For `same_category` edges between tools, the **target** endpoint’s rank sets the weight (rank **1** → **0.4**, rank **2** → **0.2**, else **0.1**) so cross-domain hops prefer landing on search/list APIs before book/detail.

### Parameter Compatibility Heuristic

Two parameters are considered compatible if types match and:

1. Names are an exact match, suffix-compatible, or both ID-like (non-generic), **or**
2. The source field is the synthetic **`id`** from a schema-less GET: then it may only match a target `entity_id` if **`entity` appears in the source tool name** (e.g. `hotel_id` ↔ `Hotels_API/...`, not `Weather_API/...`).

Exact/suffix/ID-like matching still has false positives for unrelated `*_id` fields; the synthetic-`id` rule specifically reduces cross-domain bogus data-flow edges. The executor’s grounding layer still ensures values are consistent at runtime.

### Constrained Sampling

The `SamplingConstraints` dataclass supports:
- `min_steps` / `max_steps`: Chain length bounds
- `required_categories` / `required_tools`: Inclusion constraints
- `excluded_endpoints`: Exclusion set (from diversity steering)
- `require_multi_tool` / `min_distinct_tools`: Multi-tool enforcement
- `pattern`: Sequential vs. parallel vs. mixed

The sampler performs weighted random walks on the graph, boosting edge weights based on constraint satisfaction. It makes up to 50 attempts to find a valid chain before falling back to random sampling.

**Start-node bias:** The first endpoint is preferentially chosen from **search/list/find/query**-style names and similar discovery patterns (`get_forecast`, `get_current`), then **GET** methods, then any remaining candidate. That strongly biases each chain to begin with discovery rather than booking. **Only the first step is guaranteed** to follow this rule; later steps still follow graph edges, so cross-tool segments can occasionally look suboptimal—global operation order across heterogeneous tools is **not** enforced (future work: post-ordering or template constraints).

**Minimum chain length when starting “deep”:** If the chosen start node has **two or more** same-tool `data_flow` successors, the sampled target length is bumped to **at least 3** (capped by `max_steps`), so two-step chains rarely end after a single hop when a longer intra-tool path exists.

**Strict walk mode:** `SamplingConstraints.strict_graph_walk` (on by default in `generate`) disables the sampler’s fallback that **randomly jumps** when the current node has no out-neighbors. When strict, the walk stops early if stuck (more retries/fallback chains instead).

**Tradeoff**: The graph **biases** chains toward realistic ordering; it does **not** guarantee perfect narratives on every sample—cross-tool hops trade some ordering for corpus diversity. Measured qualitative checks on sampled chains are more honest than claiming 100% correctness.

**Tradeoff (retries)**: 50 attempts balance constraint satisfaction and speed; fallback sampling may still emit weak chains if constraints are tight.

---

## 4. Multi-Agent System Design

This section maps the **assignment requirements** to code: three runtime roles (user, assistant, executor), **JSON mode** for decision-making agents, **corpus-level** multi-step/multi-tool quotas via `generate`, and **clarifying turns** when intent or fields are missing.

### 4.1 Roles and communication protocol

Three components drive each synthetic conversation:

| Role | Implementation | Responsibility |
|------|----------------|----------------|
| **User** | `UserAgent` | Opening message and follow-ups (free text); answers clarification questions using persona + `missing_info`. |
| **Assistant** | `AssistantAgent` | Chooses **call_tool**, **ask_clarification**, or **respond** via structured JSON; receives tool descriptions and session context. |
| **Tool executor** | `MockExecutor` | Runs registry-backed mock calls, updates `SessionState`; optional LLM-enriched responses. |

Other LLM-backed pieces sit **around** the turn loop: **Scenario Planner** (one shot per conversation) and **Judge** (quality pass). They are not alternate “turn” agents but part of the same pipeline.

**Protocol** (`ConversationOrchestrator.generate_conversation`):

1. **Plan** — Scenario planner → `scenario`, `user_persona`, `initial_request`, `missing_info`, `expected_steps`.
2. **Build plan IR** — `ConversationPlan.from_scenario` maps each expected step to its endpoint FQN, records step dependencies, and creates a typed progress tracker.
3. **Configure** — User agent gets persona + missing info; assistant gets the sampled endpoint list (`ToolChain.endpoints`).
4. **Loop** (max 15 turns) — User message → assistant structured decision → if `call_tool`, executor + grounding + `plan.mark_executed()`; if `ask_clarification`, user reply; if `respond`, optional continuation or stop when `_should_end` fires.

```
Orchestrator
├── Scenario Planner → scenario dict (JSON)
├── ConversationPlan.from_scenario(scenario, endpoints) → typed plan IR
├── UserAgent.configure(persona, scenario, initial_request, missing_info)
├── AssistantAgent.set_available_tools(chain endpoints)
└── for turn in 1..max_turns:
      UserAgent → text
      AssistantAgent.decide(...) → JSON action
      ├── call_tool → Grounding.ground_arguments → MockExecutor.execute → plan.mark_executed → tool message
      ├── ask_clarification → UserAgent.respond
      └── respond → maybe UserAgent.respond or end
```

### Conversation Plan IR

`ConversationPlan` (`agents/conversation_plan.py`) is an intermediate representation between scenario planning and turn-by-turn execution:

- **`PlannedStep`**: endpoint FQN, description, dependencies, execution status, actual args/response
- **`ConversationPlan`**: scenario metadata + ordered steps with progress tracking
- **`from_scenario()`**: factory that zips LLM scenario output with sampled endpoint FQNs
- **`mark_executed()`**: records actual arguments and response per step
- **`progress_summary()`**: human-readable status for assistant context injection
- **`to_dict()`**: serialized into `metadata.plan` on each conversation record

**Why a typed IR**: The scenario planner's raw JSON is unstructured (`expected_steps` is a string list). The plan IR gives the orchestrator a typed contract: it can check `plan.is_complete`, get `plan.next_step`, and include `plan.progress_summary()` in assistant prompts. It also makes the conversation's execution trace inspectable in the output JSONL.

### 4.2 Structured output (JSON mode)

**Requirement:** at least one agent must use structured output, not free-text-only for decisions.

**Implementation:** `LLMClient.complete_structured` sets OpenAI **`response_format: {"type": "json_object"}`** (JSON mode). These agents use it:

| Agent | Module | Schema highlights |
|-------|--------|-------------------|
| Scenario Planner | `orchestrator._plan_scenario` | `scenario`, `user_persona`, `initial_request`, `expected_steps`, `missing_info`, `disambiguation_needed` |
| Assistant | `AssistantAgent.decide` | `action`, `tool_calls`, `message`, `reasoning` |
| Judge | `LLMJudge` | `scores`, `explanation`, `issues` |

The **User Agent** uses **`complete` (plain text)** only — by design, to simulate natural user language.

So: **multiple** agents use JSON mode; the hard requirement is exceeded while keeping the user channel unstructured.

### 4.3 Multi-step and multi-tool enforcement (`generate`)

**Requirement:** a large fraction of the corpus (target **50–60%**) should be **multi-step** (≥ **3** endpoint calls in the chain) and **multi-tool** (≥ **2** distinct tools).

**Implementation:** `ConversationPipeline.generate` splits conversations by index. Helper **`sampling_constraints_for_cohort`** builds **`SamplingConstraints`** per slot:

**Three cohorts** (`sampling_constraints_for_cohort`):

1. **Short cohort** (first `floor(n * short_conversation_ratio)`, default **15%**): 1–2 steps, single-tool — genuinely short dialogues.
2. **Multi cohort** (next `floor(n * multi_step_ratio)`, default **65%**): `require_multi_tool=True`, `min_distinct_tools=2`, `min_steps = max(min_multi_step_calls, min_steps)` (defaults **3**), sequential/mixed/parallel pattern.
3. **Remainder cohort**: `min_steps=max(2, min_steps)` — medium-length, single-tool.

This is **enforced at the sampler**, before any LLM turn — structure is guaranteed independently of model behavior. The orchestrator additionally **caps tool calls at chain length**: once `num_tool_calls >= len(tool_chain.endpoints)`, further tool-call actions are converted to final responses, preventing the bimodal distribution (2 or 10 calls) observed in earlier iterations.

Tune **`--multi-step-ratio`** (default **0.65**), **`--short-conversation-ratio`** (default **0.15**), **`--min-multi-step-calls`** (default **3**). Actual **`num_tool_calls ≥ 3`** in generated dialogue is tracked in **`metrics.json`** (`coverage.multi_step_ratio`).

### 4.4 Parallel chain label vs runtime execution

When the sampler sets **`tool_chain.pattern == "parallel"`**, it marks a **planned** multi-endpoint batch (e.g. weather + events in one logical step). At runtime the orchestrator still runs tools **sequentially**: the assistant may emit **one** assistant turn with **multiple `tool_calls`**, and **`MockExecutor`** appends **one `tool` message per call** in order. Export metadata adds **`parallel_chain_note`** so the JSONL record does not imply true concurrent HTTP. This matches offline execution; true parallel I/O would be future work.

### 4.5 Multi-turn disambiguation

**Requirement:** when intent is ambiguous or required fields are missing, the **assistant asks**; the user supplies detail in a later turn.

**Mechanisms:**

1. **Planner** — For tool chains with **≥ 3 endpoints**, the orchestrator sets the planner prompt to favor a **vague** `initial_request` and non-empty **`missing_info`** so the user has not yet stated dates, budget, etc.
2. **Assistant** — Prompts require **`ask_clarification`** when required parameters are unknown; `AssistantAction.is_clarification` routes to **`UserAgent.respond`** with instruction to answer the question.
3. **Grounding** — Injects available IDs into assistant instructions so, after clarification, tool arguments can still align with session state.

Example shape (idealized):

```
User: "I need to plan a trip" (vague)
Assistant: "Where would you like to go, and what dates?" (clarification)
User: "Paris, next weekend"
Assistant: [calls search_hotels …]
```

**Metrics:** Each conversation records **`num_clarifications`** in metadata for corpus analysis.

### Why structured output for the assistant (summary)

Free-text tool calls are fragile. JSON mode yields parseable **`tool_calls`**, explicit **`action`** typing, and a **`reasoning`** field that does not surface to the user.

---

## 5. Offline Execution Model

### Dual Strategy

The executor has two modes:
1. **Schema-derived** (default, no API cost): Infers response structure from the endpoint name and schema
2. **LLM-generated** (higher quality): Asks the LLM to produce a realistic response given the endpoint, arguments, and session history

Schema-derived responses use heuristics: `search_*` endpoints return result lists with IDs, `book_*`/`create_*` return confirmation IDs, etc. This is fast and deterministic but produces generic data.

**Entity vs resource `id` (two layers):** Mock responses intentionally separate (1) **entity / foreign-key fields** that carry data flow across steps—e.g. `hotel_id` is the **same string** from search → details → booking (one hotel entity end-to-end)—and (2) a per-response **`id`** generated by `SessionState.generate_id` for *that* response object (detail record, booking row), analogous to APIs that return a resource id plus referenced entity ids. So `get_hotel_details` might return `id: "item_0003"` (this payload’s synthetic row id) while `hotel_id` stays aligned with the search result; **`book_hotel` arguments and response use that same `hotel_id`** (not a new hotel id); only the booking’s own **`id`** is new (e.g. `book_0004`). **Stable cross-step keys** are the semantically named fields (`hotel_id`, …); **top-level `id`** is for uniqueness within the call. That separation keeps traces easy to audit.

### Session State

`SessionState` is the core mechanism for chain consistency. It:
1. Records every tool call and response
2. Indexes response values by key path (e.g., `results.*.id` → `["htl_001", "htl_002"]`)
3. Provides `get_available_ids()` for the grounding layer to find real values
4. Generates unique mock IDs via a monotonic counter (deterministic for a fixed sequence of calls; the executor seed affects other randomized fields, e.g. scores, not the counter)

**Design decision**: I chose key-path indexing over a flat key-value store because nested responses (arrays of objects with ID fields) are the primary data-flow pattern. The wildcard notation (`results.*.id`) captures "all IDs from a search result" naturally.

---

## 6. Context Management Design

### 6.1 Within-Conversation Grounding

**Problem**: When the LLM generates tool call arguments, it might fabricate IDs (`"hotel_id": "htl_12345"`) instead of using the actual ID returned by a previous search (`"htl_001"`).

**Solution**: Two-layer grounding:

1. **Prompt-level**: The grounding module injects session context into the assistant agent's prompt: "IMPORTANT: Use these real values from previous tool responses: results.*.id: ['htl_001', 'htl_002']". This guides the LLM toward correct values.

2. **Post-hoc correction**: After the LLM generates arguments, the grounding module scans for placeholder patterns (`{hotel_id}`, `<id>`, `placeholder_*`, etc.) and replaces them with matching values from the session state. The matching uses fuzzy key matching — `hotel_id` matches stored keys containing `id`.

**Tradeoff**: Prompt injection works ~80% of the time with GPT-4o-mini. Post-hoc correction catches most remaining cases. However, it can't fix semantic misuse (e.g., using a hotel ID where a flight ID is needed). A more sophisticated approach would type-check values against the parameter's domain, but this requires richer schema information than ToolBench provides.

### 6.2 Cross-Conversation Steering

**Problem**: Without steering, the sampler gravitates toward well-connected graph regions (tools with many edges), producing a corpus dominated by a few popular tools.

**Solution**: Counter-based diversity steering that tracks:
- Endpoint usage frequency
- Category usage frequency
- Tool usage frequency
- Tool-pair co-occurrence

Before each new conversation, the steering module:
1. Identifies **deficit categories** (used < 50% of expected uniform frequency) and adds them as `required_categories` constraints
2. Identifies **overused endpoints** (> 3x fair share or > 50% of total usage) and adds them to `excluded_endpoints`

**Why counters, not vectors**: A vector-backed approach (e.g., via mem0) would capture semantic similarity between conversations — detecting that two conversations doing "hotel booking" and "resort reservation" are conceptually similar even though they use different tools. However:
- Counter-based steering is deterministic (reproducible across runs)
- It avoids the complexity of embedding model selection and ANN search tuning
- For our scale (hundreds of conversations, not millions), exact counting is sufficient

**Where this breaks down**: Counter-based steering can't detect *thematic* repetition. Two conversations might use completely different tools but both involve "booking a trip" — counters won't flag this overlap. At scale (10K+ conversations), I'd add embedding-based deduplication as a post-generation filter rather than a sampling constraint.

**Implementation**: The `--no-cross-conversation-steering` CLI flag disables steering by initializing `DiversitySteering(enabled=False)`, which returns base constraints unchanged.

---

## 7. Quality Evaluation Pipeline

**Assignment alignment (§4.1 / §4.2):**

- **§4.1 — LLM-as-judge:** Four dimensions (≥3 required): **naturalness**, **tool_correctness**, **task_completion**, **coherence** (`LLMJudge`, JSON mode). Each conversation record in JSONL includes **`judge_scores`** (dict) and **`judge_issues`** (list). **`convfactory evaluate`** loads **`dataset.jsonl`**, re-scores rows missing judge output, writes **`evaluation.json`** + corpus metrics, and updates the JSONL when new scores are computed.
- **§4.2 — Retry/repair:** After judging, **`should_attempt_repair`** (see `evaluation/repair_policy.py`) runs **`ConversationRepairer`** when: (1) **structural** failure (e.g. runaway identical tool calls), (2) **mean score** below **`quality_threshold`**, or (3) **non-empty judge issues** and **tool_correctness** or **coherence** below **`per_dimension_floor` (default 3.5)** — so minor judge nitpicks alone do not trigger repair; only clearer factual/coherence gaps do, together with reported issues. The repairer may **regenerate** (very low mean) or **targeted repair**; the result is kept with **`metadata.repair_action`**. Conversations are **not** silently discarded.

### Judge Dimensions

| Dimension | What it measures | Why it matters for training |
|-----------|-----------------|---------------------------|
| **Naturalness** (1-5) | Conversational flow, language quality | Models trained on robotic data produce robotic outputs |
| **Tool Correctness** (1-5) | Valid tool calls, correct argument types | The primary skill we're training |
| **Task Completion** (1-5) | Whether the user's goal is achieved | End-to-end task success is the ultimate metric |
| **Coherence** (1-5) | Logical chain consistency, output referencing | Correct chaining is what distinguishes good tool-use from random calls |

These four dimensions were chosen because they decompose "conversation quality" along axes that are independently actionable. If naturalness is low but correctness is high, we know the tool logic is sound but the language needs work — the repair agent can target language specifically.

### Quality Threshold: 3.0

I set the default threshold at 3.0/5.0 (60th percentile). This is the "acceptable" midpoint — conversations below this are structurally flawed or incoherent, while conversations above may have minor issues but are usable for training. The threshold is configurable via CLI.

### Retry/Repair Strategy

When a conversation scores below threshold:

1. **If mean < 2.0 and regenerate_fn exists**: Try full regeneration first (the conversation is too broken to repair)
2. **Targeted repair**: Feed the conversation + identified issues + low-scoring dimensions to a repair agent that rewrites the conversation while preserving the tool chain
3. **Accept best effort**: After `max_repair_attempts` (default 3), accept whichever version scored highest

**Design decision**: I chose to repair rather than discard because:
- Regeneration is expensive (full pipeline re-run)
- Repair can fix specific issues (e.g., naturalness) without disturbing correct tool logic
- Even imperfect conversations add diversity to the training set

**What doesn't work well**: Repair tends to "sanitize" conversations — making them correct but formulaic. The repaired versions often score higher on correctness/coherence but lower on naturalness compared to the original. I'd address this by using a two-pass approach: first repair structural issues, then inject variety in a second pass.

**Repair-rate tuning (honesty):** An earlier **`per_dimension_floor` of 4.5** caused **too many** repairs when the judge listed small issues while numeric means stayed high — inflating repair metadata without always improving transcripts. Lowering the floor to **3.5** keeps the structural and mean-score gates unchanged but requires **clearer** defects (or very low dimension scores) before issue-triggered repair, reducing noisy repair loops on the fixture corpus. If factual issues slip through, tighten prompts/judge or raise the floor slightly for your model.

**Narration vs tool consistency:** The judge rubric (`prompts_judge.py`) explicitly requires **user-facing** assistant prose to be traceable to **tool JSON** (prices, counts, dates, names). This aligns **`tool_correctness`** with “no invented numbers,” not only valid endpoint shapes.

**Registry-aware required parameters:** When the **`ToolRegistry`** is available, **`validate_before_execute`** rejects calls with **missing or empty required parameters** (per OpenAPI-style `required_parameters` on the endpoint) so `get_*` calls are not executed with `{}` unless the schema marks all fields optional.

---

## 8. Prompt Design

### Scenario Planner Prompt (Structured Output)

```
You are a scenario planner for generating realistic tool-use conversations.
Given a set of available tools and their endpoints, create a natural user scenario
that would require using these tools in sequence.

Output JSON with this schema:
{
  "scenario": "Brief description of what the user wants to accomplish",
  "user_persona": "Brief description of the user",
  "initial_request": "The user's opening message — natural, may be vague",
  "expected_steps": ["step1 description", ...],
  "disambiguation_needed": true/false,
  "missing_info": ["list of info the user hasn't provided that tools require"]
}
```

**Key design choices**:
- `missing_info` field explicitly plans what the user *hasn't* said yet, enabling natural disambiguation
- `initial_request` instruction emphasizes naturalness ("may be vague or incomplete")
- Schema is minimal — only fields that downstream agents need
- **Temporal coherence:** The planner receives a deterministic **`reference_date`** derived from the conversation seed (`_reference_date_for_seed`) and is instructed to keep **all concrete dates** in the scenario in the same calendar neighborhood — reducing mixed-year artifacts (e.g. 2023 vs 2026) in one dialogue. **`metadata.simulation_reference_date`** is stored on the exported record.

### Assistant Agent Prompt (Structured Output)

```
Output JSON with this schema:
{
  "reasoning": "Brief internal reasoning about what to do next",
  "action": "call_tool" | "ask_clarification" | "respond",
  "message": "Your message to the user (null if calling a tool)",
  "tool_calls": [{"endpoint": "...", "arguments": {...}}]
}
```

The `reasoning` field serves as chain-of-thought that doesn't leak into the conversation. The action enum constrains the LLM to exactly three behaviors, preventing ambiguous hybrid outputs.

### Prompt Iteration That Didn't Work

**Failed approach**: In an early iteration, I had the assistant agent output tool calls as free text:

```
I'll search for hotels. TOOL_CALL: Hotels_API/search_hotels(city="Paris", max_price=200)
```

**What went wrong**: GPT-4o-mini frequently:
1. Mixed the tool call syntax with natural language ("Let me search... TOOL_CALL...")
2. Used inconsistent argument formatting (sometimes JSON, sometimes kwargs)
3. Omitted required arguments or added fabricated ones

**What I learned**: For any output that needs reliable parsing, structured output (JSON mode) is worth the small quality reduction in naturalness. The assistant's natural language responses are generated in the `message` field, separate from the structured decision — this gives the best of both worlds.

**Second failed approach**: I initially tried a single-agent design where one LLM call generated the entire conversation. This produced syntactically valid but shallow conversations — the model would make a single tool call and immediately resolve the task, skipping disambiguation and multi-step chains. Decomposing into separate agents (planner → user → assistant loop) forces multi-turn interaction.

---

## 9. Diversity & Quality Analysis

### Metrics Chosen

Each metric is tied to a specific steering mechanism it is designed to detect:

| Metric | What It Measures | Steering Mechanism It Captures | Observed Delta (A→B) |
|--------|-----------------|-------------------------------|---------------------|
| **Endpoint Entropy (normalized)** | Shannon entropy of endpoint usage distribution, normalized by log₂(\|graph\|) | **Per-endpoint frequency budget** (`_excluded_by_frequency_budget`): caps how many times any single endpoint can appear, plus **unused tool nudge** (`_unused_tool_nudge`): steers toward endpoints with zero uses. Both flatten the usage distribution → higher entropy. | **83.9% → 86.5% (+2.6pp)** |
| **Category Entropy** | Shannon entropy of category mention counts across the corpus | **Deficit category boost** (`_get_deficit_categories`): requires underrepresented categories in sampling constraints. **Travel hotspot thinning** (`_travel_hotspot_exclusions`): excludes top-2 Travel endpoints when Travel dominates. **Weather floor** (`_weather_boost_categories`): ensures Weather gets minimum representation. All three rebalance category mass → higher entropy. | **1.94 → 2.26 (+0.32)** |
| **Category Coverage** | Fraction of available registry categories represented in the corpus | **Weather floor** and deficit boosting surface thin categories. Unsteered sampling can miss an entire category when the graph is hub-dominated (e.g. Travel); steering raises coverage when it was previously incomplete. | **83.3% → 100% (+16.7pp)** |
| **Mean Pairwise Jaccard Distance** | Average dissimilarity between conversations' tool sets | **Endpoint exclusion + tool nudging** combined: by rotating which endpoints are available, steering forces different conversations to use different tool subsets, increasing pairwise distance. | **97.5% → 97.9% (+0.4pp)** |

**Why these four?** The assignment asks for at least two diversity metrics tied to what steering is designed to change. Our steering combines endpoint frequency caps, category rebalancing, and unused-tool nudging; these four metrics cover endpoint-level spread (normalized entropy, Jaccard), category-level balance (entropy), and whether every domain appears at least once (coverage). Endpoint entropy is the primary signal for “flattening” overuse of hub endpoints. Category coverage and category entropy together separate “all categories appear” from “mentions are balanced across categories.”

### Empirical Run A vs Run B

#### Real LLM run (GPT-4o-mini, `sample_data`, seed 42)

We ran the full pipeline twice: **Run A** — `--no-cross-conversation-steering`; **Run B** — default steering. Both use **`sample_data`** (24 ToolBench tools, **6** categories, **125** graph endpoints), **`--num-conversations 100`**, **`--seed 42`**, and GPT-4o-mini. Artifacts: `output/run_a/` and `output/run_b/`; metrics from `metrics.json` / `evaluation.json` after `convfactory evaluate`.

| Metric | Run A (steering off) | Run B (steering on) | Delta |
|--------|----------------------|----------------------|-------|
| **Endpoint entropy (normalized)** | 83.9% | **86.5%** | **+2.6pp** |
| **Category coverage** | 83.3% (5/6) | **100%** (6/6) | **+16.7pp** |
| **Category entropy** | 1.94 | **2.26** | **+0.32** |
| **Mean pairwise Jaccard distance** | 97.5% | **97.9%** | **+0.4pp** |
| Unique endpoints used (of 125) | 71 | **85** | **+14** |
| Unique tools used (steering stats) | 10 | **22** | **+12** |
| Mean judge score | 4.14 | **4.19** | **+0.05** |
| Multi-step ratio (≥3 calls) | 59% | **64%** | **+5pp** |
| Multi-tool ratio (≥2 tools) | **83%** | 80% | −3pp |
| Joint (multi-step + multi-tool) | 59% | **64%** | **+5pp** |
| Mean tool calls / conversation | 2.85 | **3.20** | **+0.35** |
| Fallback chain samples (generation) | 0 | **14** | — |

**Reproduce:**

```bash
convfactory build --data-dir sample_data --output-dir output/run_a
convfactory generate --data-dir sample_data --output-dir output/run_a --num-conversations 100 --seed 42 --no-cross-conversation-steering
convfactory evaluate --data-dir sample_data --output-dir output/run_a

convfactory generate --data-dir sample_data --output-dir output/run_b --num-conversations 100 --seed 42
convfactory evaluate --data-dir sample_data --output-dir output/run_b
```

(Omit the `build` line for Run B if `registry.json` already exists in `output/run_b`; `generate` will build if missing.)

#### Fixture run (mock LLM, CI-friendly smoke test)

```bash
pytest tests/test_diversity_ab_metrics.py -v
WRITE_DIVERSITY_REPORT=1 pytest tests/test_diversity_ab_metrics.py -v  # writes output/diversity_ab_fixture_report.json
```

The fixture run uses **n = 6** conversations, **seed 42**, and a mock LLM (deterministic scores of 4.0). It validates that steering does not collapse coverage: **`category_coverage(B) ≥ 0.95 × category_coverage(A)`**.

### Analysis

**Diversity improves under steering.** Normalized endpoint entropy rose from **83.9% → 86.5%** (+2.6 pp), so usage spread more evenly over the graph. **Unique endpoints** increased from **71 → 85** out of **125** (~57% → ~68% of nodes touched), and **unique tools** from **10 → 22**. **Category coverage** reached **100%** (Run A missed one registry category—**Weather**—in the corpus-level counts); **category entropy** increased **1.94 → 2.26**, meaning mentions were more balanced across domains, not only “one hit per category.” **Mean pairwise Jaccard** ticked up **97.5% → 97.9%**, consistent with less overlap between conversations’ tool sets.

**Diversity–quality tradeoff:** On this run, **quality did not degrade.** Mean LLM-judge score rose **4.14 → 4.19** (+0.05 on 1–5). So steering improved diversity signals without paying a judge-score penalty here. (Other seeds or registries could differ.)

**Structural tradeoffs:** Unlike some earlier runs on different corpora, **multi-step share and mean tool calls increased** with steering (59% → 64% multi-step; 2.85 → 3.20 mean calls). **Multi-tool ratio** dipped slightly (83% → 80%). The clearest **operational cost** is **sampling effort**: Run B recorded **14 fallback chain samples** vs **0** for Run A—when constraints tighten, the sampler retries more often, but all **100** conversations still completed.

**Scaling:** With only **24** tools, Travel-heavy endpoints still dominate qualitative traces; steering’s job is to **redistribute** mass and **cover** thin categories (e.g. Weather). On a much larger ToolBench slice, endpoint entropy and hub concentration effects would likely show a larger A–B gap; see `scripts/scaling_benchmark.py` for automated sweeps.

A visual comparison of Run A vs Run B is saved as `run_ab_comparison.png` after generating **`output/run_a/metrics.json`** and **`output/run_b/metrics.json`**: run `python3 scripts/plot_ab_comparison.py` (optional: `--run-a PATH --run-b PATH -o run_ab_comparison.png`).

---

## 10. What I Would Do Next

Given more time, I would prioritize these improvements:

### Recently Implemented

- **Async batch generation** ✅: `pipeline.generate` now processes conversations in batches of `--concurrency` (default 5) via `asyncio.gather`. Steering is updated between batches so diversity constraints remain effective. Each coroutine gets its own `ConversationOrchestrator` to avoid shared state.

- **Response schema inference** ✅: `ToolBenchLoader._infer_response_schema` synthesizes `ResponseField` lists from endpoint name keywords and HTTP method (see §2). This increases data-flow edge density in the tool graph.

- **Conversation plan IR** ✅: `ConversationPlan` (see §4.1) provides a typed intermediate representation between scenario planning and execution, with step-level progress tracking and serialization to metadata.

- **Chain-bounded tool calls** ✅: The orchestrator converts tool-call actions to final responses once all planned endpoints have been called, preventing tool count inflation.

- **Assistant chain filtering** ✅: `AssistantAgent._parse_action` validates tool calls against the sampled chain’s endpoint set, rejecting hallucinated endpoints.

### High Priority

1. **Parallel tool calls**: The sampler’s **parallel** pattern is documented in **§4.4** (one assistant turn may batch **`tool_calls`**; executor order is sequential). Further work could add clearer **independent** batching rules or true async mock latency.

2. **Plan-guided assistant prompts**: The `ConversationPlan.progress_summary()` method is available but not yet injected into assistant prompts. Wiring it in would let the assistant make more targeted decisions about which tool to call next.

### Medium Priority

3. **Vector-backed deduplication**: Add embedding-based similarity checking for generated conversations. After generating each conversation, compute its embedding and check distance from existing conversations. This would catch *thematic* duplication that counter-based steering misses.

4. **Human evaluation calibration**: Collect human annotations on a sample of 50 conversations to calibrate the LLM judge. Map human scores to LLM scores and adjust thresholds accordingly.

5. **Conversation templates**: Pre-define a library of conversation patterns (clarify → search → detail → action, or search → compare → decide → book) that the sampler can draw from, ensuring structural diversity beyond just tool diversity.

### Lower Priority

6. **Cost optimization**: Track token usage per conversation and optimize prompts for cost. The scenario planner prompt, for example, could be shortened for simple 2-step chains.

7. **Streaming generation**: Stream conversations to disk as they’re generated rather than holding the full corpus in memory. Important for datasets of 10K+ conversations.

8. **Multi-language support**: Generate conversations in multiple languages for training multilingual tool-use agents.

---

## 11. Changelog

Iterative improvements to the tool graph, sampler, and tooling (see **§3** for full design rationale). Dates reflect development order, not release tags.

### Graph (`tool_graph.py`)

- **Same-tool edges:** Moved from bidirectional “all pairs” to **operation ranks** derived from endpoint names, with **declaration-order tie-breaks** and **rank-gap** weights (prefer adjacent phases search→get over skipping to book/purchase; weak backward edges retained for connectivity).
- **Data-flow edges:** **Entity-aware matching** for synthetic GET **`id`** → **`entity_id`** (entity substring must appear in **source** tool name) to cut cross-domain false positives.
- **Data-flow weights:** **Rank-gap** on forward steps: adjacent phase **1.0**, skipped phase **0.3** (tuned from an earlier **0.6**).
- **Shared-parameter edges:** New **`shared_params`** relation for meaningful shared inputs (`city`, `date`, …); **merges** onto an existing edge when `(u,v)` already exists.
- **Category edges:** **Rank-weighted targets** (prefer landing on discovery endpoints when jumping across tools in the same category).
- **`summary()`:** Adds **`data_flow_edges`** — count of edges with **`data_flow: true`**, independent of `relation`.

### Sampler (`sampler.py`)

- **`strict_graph_walk`:** Optional mode to **disable random “escape” jumps** when no out-neighbors (better ordering; may shorten chains).
- **Start nodes:** Prefer **search/list/discovery**-style endpoint names, then **GET**, then fallback.
- **Minimum chain length:** If the start has **≥2** same-tool **`data_flow`** successors, target length is bumped to **at least 3** (within `max_steps`) so two-hop chains do not always end after one intra-tool hop.

### CLI & artifacts

- **`convfactory sample`:** Samples chains from a built graph; writes **`sampled_tool_chains.json`** and optional **HTML** visualization.
- **`convfactory mock`:** Runs **schema-only** `MockExecutor` on a fixed **Hotels** chain (search → get_hotel_details → book_hotel), writes **`mock_executor_trace.json`**, and prints JSON — used to verify offline execution and ID threading without LLM mocks.
- **`generate`:** **`--no-strict-sampler-walk`** to allow escape jumps; pipeline passes **`strict_sampler_walk`** into constraints by default.

### Documentation

- **§3** expanded for edge taxonomy, weights, entity-aware `id`, `data_flow_edges`, category weighting, start-node bias, strict walk, and min-length bump.
- **§5** clarifies **entity vs per-response `id`** in schema-derived mocks and tightens the **session ID counter** vs **executor seed** wording.
- **§4** restructured: roles/protocol (**4.1**), JSON mode (**4.2**), **`generate`** multi-step/multi-tool split + CLI (**4.3**), parallel label vs execution (**4.4**), disambiguation (**4.5**).

### Pipeline / CLI

- **CI:** `.github/workflows/ci.yml` runs **`pytest tests/`** with **`CI=true`** so the **100-conversation** mock E2E runs on every push (see README).
- **Diversity A/B:** `tests/test_diversity_ab_metrics.py` + optional **`output/diversity_ab_fixture_report.json`**.
- **Smoke build:** `scripts/smoke_toolbench_build.py` for large ToolBench slices without generation.
- **`sampling_constraints_for_cohort`** — explicit builder for multi vs remainder cohort constraints.
- **`convfactory generate`:** **`--multi-step-ratio`** (default **0.65**), **`--short-conversation-ratio`** (default **0.15**), **`--min-multi-step-calls`** (default **3**), **`--concurrency`** (default **5**).
- **Orchestrator:** Aborts before a **3rd consecutive identical** tool call (same endpoint + grounded args); **`generate`** treats structural runaway as repair-eligible. Assistant/judge prompts tightened against **hallucinated tool details** and **lenient 5.0** scores.
- **§7 / repair:** **`repair_policy.should_attempt_repair`** — repair when mean below threshold, structural flags, or **issues +** low **tool_correctness** / **coherence** (not only low mean).
- **Orchestrator / grounding:** **`tool_call_validation.validate_before_execute`** blocks **Flights_API/book_flight** when **flight_id** matches an **event** id from prior **Events** results but not any **search_flights** id. **`ConversationGrounding`** adds **tool-chain domain hints** (flights vs events, verbatim restaurant/hotel names). **Repair** prompts require verbatim tool data and correct **flight_id** sourcing. **User agent** prompt: do not invent business names when choosing options.

### Architecture improvements (post-review)

- **Response schema inference** (`loader.py`): `_infer_response_schema` synthesizes `ResponseField` lists from endpoint name keywords, enriching data-flow edges in the tool graph. Executor ignores inferred schemas (uses keyword templates) so mock quality is unaffected.
- **Async batch generation** (`pipeline.py`): `generate` processes conversations in batches via `asyncio.gather` with `--concurrency` (default 5). Each coroutine gets its own `ConversationOrchestrator`. Steering updates between batches.
- **Conversation plan IR** (`conversation_plan.py`): Typed `ConversationPlan` / `PlannedStep` dataclasses between scenario planner and orchestrator. Tracks step-level execution progress; serialized to `metadata.plan` in output JSONL.
- **Chain-bounded orchestrator**: `generate_conversation` converts tool-call actions to final responses once `num_tool_calls >= len(tool_chain.endpoints)`, fixing bimodal tool count distribution (previously 2 or 10 calls).
- **Assistant chain filtering**: `_parse_action` validates endpoint FQNs against the sampled chain, rejecting hallucinated tool calls outside the planned set.
- **Diversity steering hardening**: Frequency budget floor for small corpora, tool-level safety in exclusions (never exclude all endpoints of a tool), category/tool nudges only for single-tool conversations (prevents sampler failures on multi-tool constraints).
- **Cross-domain grounding fix**: `_find_replacement` only returns IDs from the same domain entity (e.g. `hotel_id` won't get an event ID).
- **MockExecutor**: Switched from `llm.complete()` + manual JSON parse to `llm.complete_structured()` for LLM mock responses.
- **Dead code removal**: Removed duplicate `JUDGE_SYSTEM`, `JUDGE_PROMPT`, `REPAIR_SYSTEM`, `REPAIR_PROMPT` from `agents/prompts.py` (duplicates of `evaluation/prompts_judge.py`).
- **Session `_is_id_key`**: Token-based matching instead of substring (prevents `"description"` matching `"id"`).
