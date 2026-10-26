# Setup Guide — Conversation Factory

## Directory Structure (After Flattening)

```
conversation-factory/                 ← ROOT (you are here for all commands)
├── .git/
├── .venv/                             ← Python virtual environment
├── conversation_factory/              ← Main Python package
│   ├── __init__.py
│   ├── cli.py                         ← CLI entry point (build, generate, evaluate)
│   ├── pipeline.py                    ← Main pipeline orchestration
│   ├── agents/                        ← Multi-agent system
│   ├── context/                       ← Grounding & steering
│   ├── evaluation/                    ← LLM judge & repair
│   ├── execution/                     ← Mock executor
│   ├── graph/                         ← Tool graph & sampling
│   ├── registry/                      ← Tool data models & loader
│   └── utils/                         ← LLM client wrapper
├── tests/                             ← Test suite
├── sample_data/                       ← ~24 ToolBench tools from tools/ (6 domains); regenerate with create_sample_data.py
├── output/                            ← Generated artifacts & conversations
├── README.md                          ← Project overview
├── DESIGN.md                          ← Architecture documentation
├── pyproject.toml                     ← Package configuration
├── requirements.txt                   ← Dependencies
├── visualize_graph.py                 ← Graph visualization script
├── create_sample_data.py              ← Script to generate sample_data/
└── SETUP.md                           ← This file
```

---

## Quick Start (3 Steps)

### **1️⃣ Install Package**

```bash
cd /Users/aayushchamria/Desktop/Projects/conversation-factory
pip install -e .
```

This installs the `convfactory` CLI command and makes the package editable.

### **2️⃣ Build Artifacts** (No API Key Required ✅)

```bash
convfactory build --data-dir sample_data --output-dir output
```

**Output:**
- `output/registry.json` — All tools and endpoints
- `output/tool_graph.json` — Tool relationships and edges

**Expected result:**
```
Tool graph built: 15 nodes, 96 edges
Build complete: {...}
Artifacts saved to output/
```

### **3️⃣ Visualize the Graph** (Optional)

```bash
pip install matplotlib  # (if not already installed)
python visualize_graph.py --graph output/tool_graph.json --stats-only
```

**Output:**
```
📊 Basic Metrics:
   Nodes: 15 endpoints
   Edges: 96 relationships
   Avg edges per node: 6.40

🔧 Tools (6 total):
   Hotels_API: 3 endpoints
   Flights_API: 2 endpoints
   ...

🔗 Edge Types:
   data_flow: 60
   same_tool: 24
   same_category: 12
```

**Generate PNG visualization:**
```bash
python visualize_graph.py --graph output/tool_graph.json --output graph_viz.png
```

---

## Full Pipeline (Requires OpenAI API Key 🔑)

### **Get API Key**
1. Go to https://platform.openai.com/api-keys
2. Create a new API key
3. Set environment variable:
```bash
export OPENAI_API_KEY="sk-your-actual-key"
```

### **Generate 20 Conversations**

```bash
convfactory generate \
  --data-dir sample_data \
  --output-dir output \
  --num-conversations 20 \
  --seed 42
```

**Output:**
- `output/dataset.jsonl` — Conversations with messages, tool calls, judge scores
- `output/metrics.json` — Corpus-level metrics

### **Evaluate Conversations**

```bash
convfactory evaluate --output-dir output
```

---

## Diversity Experiment (Run A vs Run B)

**Run A: Steering DISABLED**
```bash
convfactory generate \
  --data-dir sample_data \
  --output-dir output/run_a \
  --num-conversations 50 \
  --seed 42 \
  --no-cross-conversation-steering
```

**Run B: Steering ENABLED**
```bash
convfactory generate \
  --data-dir sample_data \
  --output-dir output/run_b \
  --num-conversations 50 \
  --seed 42
```

Compare metrics between `output/run_a/metrics.json` and `output/run_b/metrics.json` to measure diversity impact.

---

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Unit tests only
python -m pytest tests/test_registry.py tests/test_graph.py tests/test_executor.py -v

# Integration test (repair loop)
python -m pytest tests/test_integration_repair.py -v

# End-to-end test (100 conversations)
RUN_LARGE_E2E=1 python -m pytest tests/test_e2e.py::TestEndToEndLarge -v
```

---

## Common Commands Quick Reference

| Command | Purpose | API Key? |
|---------|---------|----------|
| `convfactory build --data-dir sample_data --output-dir output` | Build registry & graph | ❌ No |
| `python visualize_graph.py --graph output/tool_graph.json --stats-only` | Show graph stats | ❌ No |
| `python visualize_graph.py --graph output/tool_graph.json --output viz.png` | Generate visualization | ❌ No |
| `convfactory generate --data-dir sample_data --output-dir output --num-conversations 20` | Generate conversations | ✅ Yes |
| `convfactory evaluate --output-dir output` | Score conversations | ✅ Yes |
| `python -m pytest tests/ -v` | Run test suite | ✅ Yes |

---

## Troubleshooting

### "convfactory: command not found"
→ Run `pip install -e .` first

### "OPENAI_API_KEY not set"
→ Run `export OPENAI_API_KEY="sk-..."` before generate/evaluate

### "No such file or directory: sample_data"
→ Run `python3 create_sample_data.py` first (requires a `tools/` tree of ToolBench JSON). Adjust counts with `--tools 28 --num-domains 6`, etc.

### Path errors
→ Always run commands from **root directory**: `/Users/aayushchamria/Desktop/Projects/conversation-factory/`

---

## Project Overview

**Conversation Factory** generates synthetic multi-turn conversations with multi-step tool-use traces:

- **Input:** ToolBench tool schemas (sample: ~24 tools, 6 categories from `tools/`)
- **Output:** JSONL dataset of realistic conversations with LLM-as-judge scores
- **Key Features:**
  - Tool Graph with constrained sampling
  - Multi-agent conversation generation
  - Automatic quality evaluation & repair
  - Cross-conversation diversity steering

See README.md and DESIGN.md for full documentation.

---

## Next Steps

1. ✅ Verify build works: `convfactory build --data-dir sample_data --output-dir output`
2. ✅ Visualize graph: `python visualize_graph.py --graph output/tool_graph.json --stats-only`
3. 📋 Get OpenAI API key
4. 🚀 Generate conversations: `convfactory generate ...`
5. 📊 Run diversity experiment (Run A vs Run B)
6. 📈 Evaluate and analyze results
