# Conversation Factory

Multi-Agent Tool-Use Conversation Generator — produces synthetic multi-turn conversations containing multi-step, multi-tool tool-use traces grounded in ToolBench tool schemas.

## CI and tests

- **GitHub Actions:** [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs the full test suite on push/PR to `main`/`master`. **`CI=true`** is set there so the **large E2E** test (`100` conversations, mock LLM) runs automatically — no API key required.
- **Locally:** By default the large E2E is skipped unless you set `RUN_LARGE_E2E=1` (Unix) or `$env:RUN_LARGE_E2E="1"` (PowerShell).

## Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key (set `OPENAI_API_KEY` environment variable)

### Installation

```bash
pip install -e ".[dev]"
```

### End-to-End Pipeline

Use bundled **`sample_data/`** (run `python3 create_sample_data.py` first; samples ~24 tools from `tools/`), or point `--data-dir` at any ToolBench layout (e.g. `data/toolbench`).

```bash
# 1. Build artifacts (registry + tool graph)
convfactory build --data-dir sample_data --output-dir output

# 2. Generate conversations
convfactory generate \
  --data-dir sample_data \
  --output-dir output \
  --num-conversations 100 \
  --seed 42

# 3. Evaluate the dataset
convfactory evaluate --data-dir sample_data --output-dir output
```

### Diversity Experiment (Run A vs Run B)

Same seed (**42**), same **`sample_data`**, steering off vs on. Results are summarized in **`DESIGN.md` §9**.

```bash
# Run A: steering disabled
convfactory generate \
  --data-dir sample_data \
  --output-dir output/run_a \
  --num-conversations 100 \
  --seed 42 \
  --no-cross-conversation-steering

convfactory evaluate --data-dir sample_data --output-dir output/run_a

# Run B: steering enabled
convfactory generate \
  --data-dir sample_data \
  --output-dir output/run_b \
  --num-conversations 100 \
  --seed 42

convfactory evaluate --data-dir sample_data --output-dir output/run_b
```

## ToolBench Data

The system supports two ToolBench data layouts:

1. **Single file**: A JSON file containing a list of tool definitions
2. **Category directories**: `data_dir/category_name/tool_name.json`

The sample fixtures in `tests/fixtures.py` provide 6 tools across 5 categories for development/testing.

To use real ToolBench data, download from the [ToolBench repository](https://github.com/OpenBMB/ToolBench) and point `--data-dir` to the tool definitions directory.

## CLI Reference

### `convfactory build`

Ingest ToolBench data and build derived artifacts.

| Flag | Description | Default |
|------|-------------|---------|
| `--data-dir`, `-d` | Path to ToolBench data directory | **required** |
| `--output-dir`, `-o` | Output directory for artifacts | `output` |
| `--max-tools`, `-m` | Maximum number of tools to load | all |

### `convfactory generate`

Generate synthetic conversations.

| Flag | Description | Default |
|------|-------------|---------|
| `--data-dir`, `-d` | Path to ToolBench data directory | **required** |
| `--output-dir`, `-o` | Output directory | `output` |
| `--num-conversations`, `-n` | Number of conversations | 10 |
| `--seed`, `-s` | Random seed | 42 |
| `--model` | LLM model name | `gpt-4o-mini` |
| `--max-tools`, `-m` | Max tools to load | all |
| `--min-steps` | Min tool calls per conversation | 2 |
| `--max-steps` | Max tool calls per conversation | 6 |
| `--quality-threshold` | Min quality score (1-5) | 3.0 |
| `--max-repair-attempts` | Max repair attempts | 3 |
| `--no-cross-conversation-steering` | Disable diversity steering | false |
| `--no-llm-mock` | Use schema-derived mocks only | false |

### `convfactory evaluate`

Evaluate generated conversations.

| Flag | Description | Default |
|------|-------------|---------|
| `--output-dir`, `-o` | Output directory | `output` |
| `--dataset` | Path to dataset JSONL | `output/dataset.jsonl` |
| `--model` | LLM model for judging | `gpt-4o-mini` |
| `--quality-threshold` | Min quality score | 3.0 |

## Output Format

The dataset is a JSONL file where each line is a conversation record:

```json
{
  "conversation_id": "conv_0042",
  "messages": [
    {"role": "user", "content": "Find me a hotel in Paris for next weekend"},
    {"role": "assistant", "content": "What's your budget range?"},
    {"role": "user", "content": "Under 200 euros per night"},
    {"role": "assistant", "content": null, "tool_calls": [
      {"endpoint": "Hotels_API/search_hotels", "arguments": {"city": "Paris", "max_price": 200, "currency": "EUR"}}
    ]},
    {"role": "tool", "content": {"results": [{"id": "htl_001", "name": "Hotel du Marais", "price": 175}]}},
    {"role": "assistant", "content": null, "tool_calls": [
      {"endpoint": "Hotels_API/book_hotel", "arguments": {"hotel_id": "htl_001", "check_in": "2026-04-11"}}
    ]},
    {"role": "tool", "content": {"booking_id": "bk_001", "status": "confirmed"}},
    {"role": "assistant", "content": "I've booked Hotel du Marais for Apr 11. Confirmation: bk_001."}
  ],
  "judge_scores": {
    "naturalness": 4.2,
    "tool_correctness": 4.8,
    "task_completion": 5.0,
    "coherence": 4.5
  },
  "metadata": {
    "seed": 42,
    "tools_used": ["Hotels_API/search_hotels", "Hotels_API/book_hotel"],
    "num_turns": 8,
    "num_tool_calls": 2,
    "num_distinct_tools": 1,
    "num_clarifications": 1,
    "categories": ["Travel"],
    "tool_chain_pattern": "sequential",
    "scenario": "Book a hotel in Paris"
  }
}
```

### Metadata Schema

| Field | Type | Description |
|-------|------|-------------|
| `seed` | int | Random seed used |
| `tools_used` | list[str] | Fully-qualified endpoint names used |
| `num_turns` | int | Total messages in conversation |
| `num_tool_calls` | int | Number of tool calls made |
| `num_distinct_tools` | int | Number of unique tools used |
| `num_clarifications` | int | Disambiguation questions asked |
| `categories` | list[str] | ToolBench categories represented |
| `tool_chain_pattern` | str | `sequential` or `mixed` |
| `scenario` | str | Scenario description |
| `repaired` | bool | Whether conversation was repaired |
| `repair_action` | str | `regenerated`, `repaired`, or `best_effort` |

## Running Tests

```bash
# All tests (includes large E2E when CI=true or RUN_LARGE_E2E=1)
python3 -m pytest tests/ -v

# Unit tests only
python3 -m pytest tests/test_registry.py tests/test_graph.py tests/test_executor.py tests/test_evaluation.py tests/test_context.py -v

# Integration test (repair loop)
python3 -m pytest tests/test_integration_repair.py -v

# E2E tests
python3 -m pytest tests/test_e2e.py -v

# Large E2E test (100 conversations) — locally
RUN_LARGE_E2E=1 python3 -m pytest tests/test_e2e.py::TestEndToEndLarge -v

# Diversity Run A vs B metrics (fixture corpus, mock LLM) + optional report file
python3 -m pytest tests/test_diversity_ab_metrics.py -v
WRITE_DIVERSITY_REPORT=1 python3 -m pytest tests/test_diversity_ab_metrics.py -v   # writes output/diversity_ab_fixture_report.json
```

## ToolBench smoke build (large slice)

Verify the loader and graph on a downloaded ToolBench tree without calling an LLM:

```bash
python scripts/smoke_toolbench_build.py --data-dir /path/to/toolbench --max-tools 200
```

## Project Structure

```
conversation_factory/
├── cli.py                 # CLI entry point
├── pipeline.py            # End-to-end pipeline orchestration
├── registry/              # Tool data models, ToolBench loader, registry
├── graph/                 # Tool graph construction + constrained sampler
├── execution/             # Offline mock executor + session state
├── agents/                # Multi-agent system (user, assistant, orchestrator)
├── evaluation/            # LLM-as-judge, repair, metrics
├── context/               # Within-conversation grounding + cross-conversation steering
└── utils/                 # LLM client wrapper
```
