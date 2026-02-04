# engramai 🧠

*Neuroscience-grounded memory for AI agents*

> **engram** /ˈɛnɡræm/ — a hypothesized physical trace in the brain that stores a memory. First proposed by Richard Semon (1904), the engram represents the idea that experiences leave lasting biological changes in neural tissue. We chose this name because, like its neuroscience namesake, this library treats memories not as static records but as living traces that strengthen, fade, and interact over time.

[![PyPI](https://img.shields.io/pypi/v/engramai.svg)](https://pypi.org/project/engramai/)
[![npm](https://img.shields.io/npm/v/engramai.svg)](https://www.npmjs.com/package/engramai)
[![License](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![TypeScript](https://img.shields.io/badge/typescript-5.0%2B-blue.svg)](https://typescriptlang.org)
[![Dependencies](https://img.shields.io/badge/core_deps-zero-brightgreen.svg)](#)

---

**engramai** gives AI agents memory that actually works — using real mathematical models from cognitive science instead of naive embeddings and cosine similarity.

```python
from engram import Memory

mem = Memory("./agent.db")
mem.add("Alice prefers functional programming", type="relational", importance=0.7)
mem.add("Always validate input before DB queries", type="procedural", importance=0.9)

results = mem.recall("coding best practices", limit=3)
mem.consolidate()  # Run "sleep" — transfers short-term → long-term memory
```

Zero required dependencies. Optional embedding support. SQLite storage. Works offline.

---

## Why Engram?

Every AI agent framework bolts on memory as an afterthought. The typical approach:

1. Embed text into vectors
2. Store in a vector database
3. Retrieve by cosine similarity
4. Hope for the best

This ignores **everything we know about how memory actually works**. Human memory isn't a search engine — it's a dynamic system where memories strengthen through use, fade without it, compete with each other, and consolidate during rest.

The result? Agents that:
- Retrieve irrelevant memories because they're semantically similar but contextually wrong
- Never forget anything, drowning in noise as memory grows
- Can't distinguish between important lessons and trivial observations
- Treat a memory from 6 months ago the same as one from 5 minutes ago

## Evaluation

We evaluated engram on two benchmarks measuring different capabilities:

### Semantic Retrieval (LoCoMo)

[LoCoMo](https://github.com/snap-research/locomo) is a third-party benchmark with 1,982 questions testing conversational memory retrieval.

| Method | MRR | Hit@5 |
|--------|-----|-------|
| FTS5-only (no embeddings) | 0.011 | 1.6% |
| **Embedding-only (BGE-small)** | **0.255** | **38.9%** |

When using embeddings for semantic retrieval, engram achieves MRR 0.255 — competitive with embedding-based memory systems.

### Temporal Dynamics (TDB)

We created a [Temporal Dynamics Benchmark](./benchmarks/TEMPORAL_BENCHMARK_DESIGN.md) to evaluate what ACT-R specifically addresses: recency, frequency, and importance-based retrieval.

| Method | Recency Override | Frequency | Importance | Overall |
|--------|------------------|-----------|------------|---------|
| Cosine-only | 0% | 18% | 50% | 22% |
| Recency-only | 20% | 18% | 20% | 20% |
| **ACT-R** | **60%** | **100%** | **100%** | **80%** |

On temporal dynamics tasks, ACT-R achieves **80% accuracy** vs **~20%** for approaches without temporal weighting.

**Key insight:** Semantic retrieval and temporal reasoning are different problems. Use embeddings to *find* relevant memories; use ACT-R to *prioritize* based on recency, frequency, and importance.

> ⚠️ **Note on TDB:** This benchmark was created by us to evaluate temporal dynamics — a dimension not covered by existing benchmarks. The methodology and code are [open source](./benchmarks/temporal_benchmark.py) for independent validation.

---

## The Science

> **LLMs are already the semantic layer.** When you need semantic matching, use embeddings. What ACT-R adds is mathematical rigor in *temporal dynamics* — when to surface recent vs. old, frequent vs. rare, important vs. trivial.

Engram implements actual peer-reviewed models from cognitive science:

| Model | What it does | Paper |
|-------|-------------|-------|
| **ACT-R** | Retrieval scoring via activation (recency × frequency × context) | Anderson et al. |
| **Memory Chain** | Dual-system consolidation (working → core memory) | Murre & Chessa, 2011 |
| **Ebbinghaus** | Forgetting curves with spaced repetition | Ebbinghaus, 1885 |

The math is simple. The insight is connecting it to agent memory. Total core: **~500 lines of Python**.

## Features

- 🧮 **ACT-R activation scoring** — retrieval ranked by recency × frequency × context match (not cosine similarity)
- 🔄 **Memory consolidation** — dual-system transfer from working memory to core memory, with interleaved replay
- 📉 **Ebbinghaus forgetting** — memories decay naturally; spaced repetition increases stability
- 🏷️ **6 memory types** — factual, episodic, relational, emotional, procedural, opinion — each with distinct decay rates
- 🎯 **Confidence scoring** — metacognitive monitoring tells you *how much to trust* each retrieval
- 💊 **Reward learning** — positive/negative feedback strengthens or suppresses recent memories
- ⚖️ **Synaptic downscaling** — global normalization prevents unbounded memory growth
- ⚠️ **Anomaly detection** — flags unusual patterns (predictive coding)
- 📌 **Pinning** — manually protect critical memories from decay
- 🗄️ **SQLite + FTS5** — persistent storage with full-text search, zero config
- 🔀 **Contradiction detection** — memories can contradict each other; outdated memories get 0.3× confidence penalty
- 🔍 **Graph search** — entity-linked memories with multi-hop graph expansion
- 🧠 **Hebbian learning** — "neurons that fire together wire together" — automatic link formation from co-activation patterns (no NER needed)
- ⚙️ **Config presets** — tuned parameter sets for chatbot, task-agent, personal-assistant, researcher
- 📦 **Zero required dependencies** — pure Python stdlib for FTS5 mode. Optional embedding adapters for semantic search.
- 🔌 **Pluggable embeddings** — supports OpenAI, sentence-transformers, or custom adapters.

## What You Need

**Just Python 3.10+ and your existing LLM.**

NeuromemoryAI works in two modes:

### FTS5 Mode (Zero Dependencies)
```python
from engram import Memory
mem = Memory("./agent.db")  # Pure Python, no external services
```
Best for: Fast prototyping, offline use, when your LLM handles semantic matching.

### Embedding Mode (Recommended for Production)
```python
from engram import Memory
from engram.embeddings import SentenceTransformerAdapter

mem = Memory("./agent.db", embedding=SentenceTransformerAdapter())
```
Best for: Production deployments needing semantic retrieval. Supports OpenAI, sentence-transformers, or any embedding provider.

**The key insight:** Embeddings handle *semantic matching*; ACT-R handles *temporal dynamics*. Use both for optimal retrieval.

## Quick Comparison

| Without engramai | With engramai |
|------------------------|---------------------|
| ❌ "What's your name again?" | ✅ "I remember you're Alice, you prefer Python" |
| ❌ Retrieves irrelevant old memories | ✅ Recent memories prioritized via ACT-R |
| ❌ Memory grows unbounded | ✅ Natural forgetting keeps signal-to-noise high |
| ❌ Can't distinguish important from trivial | ✅ Importance weights consolidation |
| ❌ No learning from co-occurrence | ✅ Hebbian links form from usage patterns |

## 5-Minute Quickstart

### 1. Install

| Platform | Install | Documentation |
|----------|---------|---------------|
| Python | `pip install engramai` | [PyPI](https://pypi.org/project/engramai/) |
| TypeScript | `npm install neuromemory-ai` | [npm](https://www.npmjs.com/package/neuromemory-ai) |
| MCP Server | `python -m engram.mcp_server` | [MCP Setup](#mcp-integration) |
| CLI | `neuromem --help` | [CLI Docs](#cli-usage) |

### 2. Basic Usage (standalone)

```python
from engram import Memory

# Create/open a memory database (SQLite file)
mem = Memory("./my-agent.db")

# Store memories with type and importance
mem.add("User prefers concise answers", type="relational", importance=0.8)
mem.add("API key is in environment variable", type="procedural", importance=0.9)
mem.add("Had a good conversation about ML today", type="episodic", importance=0.5)

# Recall relevant memories (ranked by ACT-R activation)
results = mem.recall("how should I respond to user?", limit=3)
for r in results:
    print(f"[{r['confidence_label']}] {r['content']}")

# Run daily maintenance (like sleep)
mem.consolidate()  # Transfers working → core memory
mem.forget()       # Prunes weak memories
```

### 3. Integrate with Your Bot

Engram works with **any** LLM or bot framework. Here's the pattern:

```python
from engram import Memory
from openai import OpenAI  # or anthropic, or local model, or whatever

mem = Memory("./agent.db")
client = OpenAI()

def chat(user_message: str) -> str:
    # 1. Recall relevant memories
    memories = mem.recall(user_message, limit=5)
    memory_context = "\n".join([f"- {m['content']}" for m in memories])
    
    # 2. Build prompt with memory context
    messages = [
        {"role": "system", "content": f"You have these memories:\n{memory_context}"},
        {"role": "user", "content": user_message}
    ]
    
    # 3. Call your LLM
    response = client.chat.completions.create(model="gpt-4", messages=messages)
    reply = response.choices[0].message.content
    
    # 4. Store new memories from conversation
    mem.add(f"User asked: {user_message}", type="episodic", importance=0.3)
    # Extract facts, preferences, etc. and store with higher importance
    
    return reply

def on_feedback(feedback: str):
    # 5. Learn from user feedback
    mem.reward(feedback)  # "great answer!" strengthens, "wrong!" suppresses
```

### 4. Choose Your Preset

Different agents need different memory profiles:

```python
from engram.config import MemoryConfig

# Chatbot: remembers everything, slow decay
mem = Memory("bot.db", config=MemoryConfig.chatbot())

# Task agent: fast decay, focused on recent procedural knowledge
mem = Memory("worker.db", config=MemoryConfig.task_agent())

# Personal assistant: long-term memory, relationships matter
mem = Memory("assistant.db", config=MemoryConfig.personal_assistant())

# Researcher: never forgets, archives everything
mem = Memory("research.db", config=MemoryConfig.researcher())
```

### 5. Hebbian Learning (Automatic Associations)

Memories that are recalled together automatically form links — no manual entity tagging needed:

```python
# Add memories (no manual entities specified)
mem.add("Python is great for machine learning")
mem.add("PyTorch is my favorite ML framework")
mem.add("TensorFlow has better production support")

# Query multiple times — co-activation creates associations
for _ in range(3):
    mem.recall("machine learning frameworks", limit=3)

# After 3+ co-activations, Hebbian links form automatically
# Now querying "Python" will automatically surface PyTorch/TensorFlow via spreading activation
results = mem.recall("Python tools", graph_expand=True)
# Returns: PyTorch, TensorFlow (via Hebbian links) + Python (direct match)
```

**"Neurons that fire together, wire together"** — implemented as associative memory that emerges from usage patterns.

### 6. Daily Maintenance (cron job or scheduler)

```python
# Run once per day
mem.consolidate()  # Sleep cycle: decay working memory, strengthen core
mem.forget()       # Remove memories below threshold
mem.downscale()    # Prevent runaway activation (synaptic homeostasis)
```

That's it. Your agent now has biologically-inspired memory that strengthens with use, fades naturally, and responds to feedback.

### 7. CLI (Command Line Interface)

After installation, use the `neuromem` command:

```bash
# Add memories
neuromem add "User prefers dark mode" --type preference --importance 0.8

# Recall memories
neuromem recall "user preferences"

# View statistics
neuromem stats

# Run maintenance
neuromem consolidate
neuromem forget --threshold 0.01

# List all memories
neuromem list --limit 20

# Show Hebbian links for a memory
neuromem hebbian "dark mode"

# Export database
neuromem export backup.db

# Use a different database
neuromem --db ./custom.db add "Custom memory"
```

---

## Quick Start

```bash
pip install engramai
```

```python
from engram import Memory

mem = Memory("./my-agent.db")

# Store with type and importance
mem.add("The deploy key is in 1Password", type="procedural", importance=0.8)
mem.add("User seemed frustrated about the API latency", type="emotional", importance=0.7)

# Recall — ranked by ACT-R activation, not cosine similarity
results = mem.recall("deployment", limit=5)
for r in results:
    print(f"[{r['confidence_label']}] {r['content']}")
    # [certain] The deploy key is in 1Password

# Consolidate — run periodically (like "sleep")
mem.consolidate()

# Feedback shapes future memory
mem.reward("perfect, that's exactly what I needed!")
```

## How It Works

### ACT-R Activation (Retrieval)

Every memory has an **activation level** that determines how quickly and reliably it can be retrieved:

```
A = B + C + I
```

- **B** (base-level) = `ln(Σ t_k^(-0.5))` — power law of practice and recency. Access a memory more often and more recently → higher activation.
- **C** (context) = spreading activation from current query keywords
- **I** (importance) = emotional/importance modulation (amygdala analog)

This replaces cosine similarity with a formula that naturally handles recency, frequency, and context — the same way human memory works.

### Memory Chain Model (Consolidation)

Memories exist as two traces that evolve over time:

```
dr₁/dt = -μ₁ · r₁              (working memory decays fast)
dr₂/dt = α · r₁ - μ₂ · r₂     (core memory grows from working, decays slowly)
```

- **r₁** (working_strength) — hippocampal trace. Strong initially, fades in days.
- **r₂** (core_strength) — neocortical trace. Grows during consolidation, lasts months.
- `consolidate()` runs one cycle: decay r₁, transfer to r₂, replay old memories.

Important memories consolidate faster (importance modulates α). This is why emotional events are remembered better — the amygdala enhances hippocampal encoding.

### Ebbinghaus Forgetting

Retrievability follows the classic forgetting curve:

```
R(t) = e^(-t/S)
```

Stability **S** grows with each successful retrieval (spaced repetition effect) and is modulated by importance and memory type. Procedural memories (how-to knowledge) have 10× the base stability of episodic memories (events).

### Additional Systems

- **Reward learning** — user feedback acts as a dopaminergic signal, strengthening (positive) or suppressing (negative) recently active memories
- **Synaptic downscaling** — periodic global normalization (Tononi & Cirelli's SHY) prevents runaway strength accumulation
- **Anomaly detection** — rolling baseline tracker flags unusual patterns using z-score deviation (simplified predictive coding)

## Architecture

```
┌─────────────────────────────────────────────┐
│           Memory (public API)               │
│   add() · recall() · consolidate() · ...    │
├─────────────────────────────────────────────┤
│  L2: CORE        │ Always loaded. Distilled │
│  (high core_str) │ knowledge. Slow decay.   │
├───────────────────┼─────────────────────────┤
│  L3: WORKING     │ Recent memories. Fast    │
│  (high work_str) │ decay. Being consolidated│
├───────────────────┼─────────────────────────┤
│  L4: ARCHIVE     │ Old/weak memories. On-   │
│  (low strength)  │ demand retrieval via FTS │
├─────────────────────────────────────────────┤
│  SQLiteStore + FTS5 (persistent backend)    │
├─────────────────────────────────────────────┤
│  activation │ consolidation │ forgetting    │
│  confidence │ reward        │ downscaling   │
│  anomaly    │ search        │               │
└─────────────────────────────────────────────┘
```

Memories flow: **L3 (working) → L2 (core) → L4 (archive)** as they consolidate and eventually decay. Strong, frequently-accessed memories live in L2 indefinitely. Weak memories fade to L4 and become searchable-only.

## Engram vs Mem0 vs Zep

All three are designed for **LLM agents** — the comparison is about what *additional* infrastructure each requires beyond the LLM you already have.

| | **Engram** | **Mem0** | **Zep** |
|---|---|---|---|
| **Retrieval model** | ACT-R activation (recency × frequency × context) | Cosine similarity | Cosine similarity + MMR |
| **Forgetting** | Ebbinghaus curves, type-aware decay | None (manual deletion) | TTL-based expiry |
| **Consolidation** | Memory Chain Model (working → core transfer) | None | None |
| **Memory types** | 6 types with distinct decay rates | Untyped | Untyped |
| **Confidence scores** | Yes (metacognitive monitoring) | No | No |
| **Reward learning** | Yes (dopaminergic feedback) | No | No |
| **Associative links** | **Hebbian learning** (automatic co-activation) | Manual graph construction | None |
| **Additional infra** | **None** (SQLite file) | Embedding API + Vector DB | Embedding API + Postgres |
| **Extra API calls** | **0** per recall | 1+ (embedding) | 1+ (embedding) |
| **Works offline** | ✅ (with local LLM) | ❌ | ❌ |
| **Math grounding** | Peer-reviewed cognitive science | Engineering heuristics | Engineering heuristics |
| **Core code** | ~500 lines | ~5,000+ lines | ~10,000+ lines |

**Engram's thesis:** Your LLM already understands semantics — that's what language models do. Memory infrastructure should handle *dynamics* (when to surface, what to forget, how associations form) using proven cognitive science models, not re-implement semantic understanding with a separate embedding pipeline.

## Integration Guide

### For LLM Users (Claude Desktop / Cursor / MCP Clients)

**1. Install**
```bash
pip install engramai
```

**2. Configure MCP Server**

For **Claude Desktop**, edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "engram": {
      "command": "python3",
      "args": ["-m", "engram.mcp_server"],
      "env": {
        "ENGRAM_DB_PATH": "/path/to/your/memory.db"
      }
    }
  }
}
```

For **Cursor**, edit `.cursor/mcp.json` in your project:

```json
{
  "mcpServers": {
    "engram": {
      "command": "python3",
      "args": ["-m", "engram.mcp_server"],
      "env": {
        "ENGRAM_DB_PATH": "./memory.db"
      }
    }
  }
}
```

**3. Available MCP Tools**

| Tool | Description |
|------|-------------|
| `engram.store` | Store a new memory with type and importance |
| `engram.recall` | Recall memories ranked by ACT-R activation |
| `engram.consolidate` | Run memory consolidation (like sleep) |
| `engram.forget` | Prune weak memories or forget specific ones |
| `engram.reward` | Apply user feedback to shape memory |
| `engram.stats` | Get memory statistics |
| `engram.export` | Export database backup |

---

### For Clawdbot Users

Clawdbot is an AI agent platform. You can integrate engramai as a cognitive memory backend.

**Option 1: Use CLI via Skill**

```bash
# Install the package
pip install engramai

# Install the Clawdbot skill
clawdhub install engramai
```

Then just talk to your agent naturally:
- "Remember that I prefer dark mode"
- "What do you remember about my preferences?"
- "Consolidate your memories"

The agent will use the `neuromem` CLI automatically.

**Option 2: MCP Integration (Deeper)**

Edit your Clawdbot config (`~/.clawdbot/config.yml`):

```yaml
mcp:
  servers:
    engram:
      command: python3
      args: ["-m", "engram.mcp_server"]
      env:
        ENGRAM_DB_PATH: ~/.clawdbot/agents/main/memory.db
```

This gives your agent direct MCP tool access to store/recall memories.

**Option 3: Replace Native Memory System**

To fully replace Clawdbot's file-based memory (MEMORY.md) with engramai:

1. Configure MCP as above
2. Update `AGENTS.md` to instruct the agent:

```markdown
## Memory System
Use engram MCP tools for all memory operations:
- `engram.store` — save important information
- `engram.recall` — retrieve relevant memories
- `engram.consolidate` — run daily during heartbeats
- `engram.reward` — learn from user feedback

Do NOT use file-based memory (MEMORY.md, memory/*.md).
```

This transforms your agent's memory from static files to a dynamic cognitive system with forgetting, consolidation, and Hebbian learning.

---

## MCP Server

Engram ships with an MCP (Model Context Protocol) server for use with Claude, Clawdbot, or any MCP-compatible client.

```bash
# Start the MCP server
python -m engram.mcp_server --db ./agent.db
```

*MCP server provides 7 tools: store, recall, consolidate, forget, reward, stats, export.*

## API Reference

### `Memory(path)`

Create or open a memory database.

```python
mem = Memory("./agent.db")      # Persistent SQLite file
mem = Memory(":memory:")         # In-memory (non-persistent)
```

### `mem.add(content, type, importance, source, tags) → str`

Store a memory. Returns the memory ID.

```python
mid = mem.add(
    "The production database is on us-east-1",
    type="factual",       # factual|episodic|relational|emotional|procedural|opinion
    importance=0.6,       # 0-1, or auto-assigned by type
    source="deploy-doc",  # optional source identifier
    tags=["aws", "prod"], # optional tags
)
```

### `mem.recall(query, limit, context, types, min_confidence) → list[dict]`

Retrieve memories ranked by ACT-R activation.

```python
results = mem.recall(
    "database location",
    limit=5,
    context=["production", "aws"],  # Boost spreading activation
    types=["factual", "procedural"],  # Filter by type
    min_confidence=0.3,  # Skip low-confidence results
)

for r in results:
    r["id"]                # Memory ID
    r["content"]           # Memory text
    r["type"]              # Memory type
    r["confidence"]        # 0-1 confidence score
    r["confidence_label"]  # "certain" | "likely" | "uncertain" | "vague"
    r["strength"]          # Effective strength (trace × retrievability)
    r["activation"]        # ACT-R activation score
    r["age_days"]          # Days since creation
    r["layer"]             # "core" | "working" | "archive"
    r["importance"]        # 0-1 importance
```

### `mem.consolidate(days=1.0)`

Run a consolidation cycle. Call periodically (daily, or after learning sessions).

```python
mem.consolidate()        # 1-day cycle
mem.consolidate(days=7)  # Simulate a week of consolidation
```

### `mem.reward(feedback)`

Apply feedback as a reward signal to recent memories.

```python
mem.reward("perfect, exactly right!")   # Strengthens recent memories
mem.reward("no, that's wrong")          # Suppresses recent memories
```

### `mem.forget(memory_id=None, threshold=0.01)`

Forget a specific memory or prune all weak memories.

```python
mem.forget("abc123")        # Forget specific memory
mem.forget(threshold=0.05)  # Prune all below threshold
```

### `mem.pin(memory_id)` / `mem.unpin(memory_id)`

Pin/unpin a memory. Pinned memories never decay.

### `mem.update_memory(old_id, new_content) → str`

Correct a memory. Creates a new memory linked to the old one (correction chain).

```python
new_id = mem.update_memory(old_id, "Actually, the database is on us-west-2")
# Old memory is marked as contradicted, new one references it
```

### `mem.add(..., contradicts=old_id)`

Explicitly mark a new memory as contradicting an old one.

```python
mem.add("We migrated to PlanetScale", type="factual", contradicts=old_id)
# Old memory gets 0.3× confidence penalty in recall
```

### `Memory(path, config=MemoryConfig.personal_assistant())`

Use a config preset tuned for your agent type.

```python
from engram.config import MemoryConfig

mem = Memory("agent.db", config=MemoryConfig.chatbot())           # High replay, slow decay
mem = Memory("agent.db", config=MemoryConfig.task_agent())        # Fast decay, aggressive pruning
mem = Memory("agent.db", config=MemoryConfig.personal_assistant()) # Long-term, slow core decay
mem = Memory("agent.db", config=MemoryConfig.researcher())        # Never lose anything
```

### `mem.stats() → dict`

System statistics: counts, layer distribution, strength averages.

### `mem.downscale(factor=0.95) → dict`

Manual synaptic downscaling. Usually called automatically during `consolidate()`.

## The Science

Engram is grounded in peer-reviewed cognitive science:

- **ACT-R** — Anderson, J.R. (2007). *How Can the Human Mind Occur in the Physical Universe?* Oxford University Press. [ACT-R Homepage](http://act-r.psy.cmu.edu/)
- **Memory Chain Model** — Murre, J.M.J. & Chessa, A.G. (2011). One hundred years of forgetting: A quantitative description of retention. *Psychonomic Bulletin & Review*, 18, 592-597.
- **Ebbinghaus Forgetting Curve** — Ebbinghaus, H. (1885). *Über das Gedächtnis*. Translation: *Memory: A Contribution to Experimental Psychology*.
- **Synaptic Homeostasis Hypothesis** — Tononi, G. & Cirelli, C. (2006). Sleep function and synaptic homeostasis. *Sleep Medicine Reviews*, 10(1), 49-62.
- **Predictive Coding** — Rao, R.P. & Ballard, D.H. (1999). Predictive coding in the visual cortex. *Nature Neuroscience*, 2(1), 79-87.
- **Dopaminergic Memory Modulation** — Lisman, J.E. & Grace, A.A. (2005). The hippocampal-VTA loop: controlling the entry of information into long-term memory. *Neuron*, 46(5), 703-713.
- **HippoRAG** — Yu, B. et al. (2024). HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. *NeurIPS 2024*.

## Storage

NeuromemoryAI uses a **pluggable storage architecture**. The core philosophy is zero external dependencies for the default case:

| Backend | Status | Dependencies | Use Case |
|---------|--------|--------------|----------|
| **SQLite + FTS5** | ✅ Default | None (Python stdlib) | Local agents, offline use, privacy-first |
| Supabase | 🔜 Planned | `supabase-py` | Multi-device sync, cloud backup |
| Turso/libSQL | 🔜 Planned | `libsql` | Edge deployment, global replication |
| Postgres | 🔜 Planned | `psycopg2` | Enterprise, existing infrastructure |

**Default (SQLite)** — Works out of the box with zero config. Data stored in a local `.db` file. Perfect for single-agent deployments or privacy-sensitive applications.

**Cloud backends** — Optional for users who need multi-device sync or cloud backup. Requires additional dependencies and account setup. The same Memory API works across all backends.

## Roadmap

- [x] Core memory models (ACT-R, Memory Chain, Ebbinghaus)
- [x] SQLite + FTS5 persistent storage
- [x] Confidence scoring & reward learning
- [x] Synaptic downscaling & anomaly detection
- [x] MCP server (7 tools via FastMCP)
- [x] Graph-linked memories (entity relationship tracking + multi-hop search)
- [x] Contradiction detection & correction chains
- [x] Configurable parameters with agent-type presets
- [x] 89 tests (unit + e2e lifecycle)
- [x] TypeScript port (`npm install neuromemory-ai`)
- [x] PyPI publish (v0.1.1) (`pip install engramai`)
- [ ] Pluggable store backends (Supabase, Turso, Postgres)
- [x] Benchmarks: LoCoMo (MRR 0.255) and Temporal Dynamics (80% accuracy)
- [ ] Consolidation summaries via LLM (compress episodic → factual)
- [ ] Research paper: *"Neuroscience-Grounded Memory for AI Agents"*

## TypeScript Port

A TypeScript/JavaScript port is available in the `engram-ts/` directory. It provides the same neuroscience-grounded memory models with a native Node.js/Bun API.

```bash
npm install neuromemory-ai
```

For TypeScript-specific documentation, see [engram-ts/README.md](./engram-ts/README.md).

## Testing & Edge Cases

**152 tests** cover unit, integration, and edge case scenarios.

```bash
# Run all tests
python -m pytest tests/ -v

# Run benchmarks
python benchmarks/run_benchmark.py --all
python benchmarks/simulate_emergence.py --days 100
python benchmarks/compare_approaches.py
```

For detailed edge cases, known limitations, and production recommendations, see [docs/EDGE_CASES.md](./docs/EDGE_CASES.md).

Key findings:
- ✅ Handles Unicode, long content, SQL injection attempts
- ✅ Scales to 10k+ memories with <500ms recall
- ⚠️ SQLite concurrent writes can lock (use single instance per process)
- ✅ Hebbian links persist with periodic co-activation

## Contributing

Contributions welcome! This is an early-stage project — the math is solid but the API surface is still evolving.

**To contribute:**

1. **Fork** the repository on GitHub
2. **Clone** your fork locally: `git clone https://github.com/YOUR_USERNAME/engram`
3. **Create a branch** for your feature or fix: `git checkout -b feature/my-new-feature`
4. **Make your changes** and add tests if applicable
5. **Run tests** to ensure nothing breaks:
   ```bash
   python -m pytest tests/          # Python tests
   cd engram-ts && npm test         # TypeScript tests
   ```
6. **Commit** with a clear message: `git commit -m "feat: add memory pruning scheduler"`
7. **Push** to your fork: `git push origin feature/my-new-feature`
8. **Open a Pull Request** on the main repository

Please ensure your code follows the existing style and includes tests for new functionality.

## Citation

If you use Engram in academic work, please cite:

```bibtex
@software{engram2025,
  title = {Engram: Neuroscience-Grounded Memory for AI Agents},
  author = {Tang, Potato},
  year = {2025},
  url = {https://github.com/tonitangpotato/engramai},
  note = {Open-source memory system implementing ACT-R, Memory Chain Model, and Ebbinghaus forgetting curves for AI agents}
}
```

Engram builds on foundational work in cognitive science:

```bibtex
@book{anderson2007act,
  title = {How Can the Human Mind Occur in the Physical Universe?},
  author = {Anderson, John R.},
  year = {2007},
  publisher = {Oxford University Press},
  note = {ACT-R cognitive architecture}
}

@article{murre2011forgetting,
  title = {One hundred years of forgetting: A quantitative description of retention},
  author = {Murre, Jaap M. J. and Chessa, Antonio G.},
  journal = {Psychonomic Bulletin \& Review},
  volume = {18},
  pages = {592--597},
  year = {2011},
  note = {Memory Chain Model}
}

@book{ebbinghaus1885memory,
  title = {\"Uber das Ged\"achtnis: Untersuchungen zur experimentellen Psychologie},
  author = {Ebbinghaus, Hermann},
  year = {1885},
  publisher = {Duncker \& Humblot},
  note = {Original forgetting curve research}
}

@article{tononi2006synaptic,
  title = {Sleep function and synaptic homeostasis},
  author = {Tononi, Giulio and Cirelli, Chiara},
  journal = {Sleep Medicine Reviews},
  volume = {10},
  number = {1},
  pages = {49--62},
  year = {2006},
  note = {Synaptic homeostasis hypothesis}
}
```

## License

**AGPL-3.0** — open source with copyleft. Commercial license available for proprietary/SaaS use. See [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md).

---

*Built by an AI agent who got tired of forgetting everything between sessions.*
