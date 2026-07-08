# MetricGuard 🛡️
### A Governed Semantic Layer for AI-Driven Product Analytics

> **Status:** 🟡 In Progress — actively building  
> **Stack:** Python · SQL · DuckDB · dbt · LLM APIs (Anthropic Claude) · sqlglot  
> **Motivation:** Solves the metric-consistency problem that every large analytics org faces at scale

---

## The Problem

At companies operating across multiple products and billions of users, a deceptively simple question — *"How many users were active yesterday?"* — can return three different answers depending on who runs the query.

```
PM asks: "DAU this week?"

Engineer A → Facebook pipeline  → 2.1B
Engineer B → Instagram pipeline → 2.4B  
Engineer C → Messenger pipeline → 1.9B

Which number is right? Nobody knows. The meeting derails.
```

This happens because every product team defines the same metric slightly differently — what counts as a "session," how to handle cross-posts, whether bot traffic is excluded, how to attribute cross-platform users. When AI tools start generating SQL automatically, this problem gets **worse**: models confidently return wrong numbers with no warning.

MetricGuard fixes this with three components:

1. **A governed semantic layer** — metrics defined once in YAML, referenced everywhere
2. **An LLM query agent** — translates plain English to SQL, but *only* through governed views
3. **A validation gate** — parses generated SQL and refuses to execute queries that would produce ambiguous or double-counted results

The key insight: **the dangerous failure is not "no answer." It's "confident wrong answer."**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Natural Language Query                    │
│           "How much policy-violating content was            │
│            actioned in Germany last month?"                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│          LLM Text-to-SQL Agent          │
│    (Anthropic API + semantic context)   │
└─────────────────────┬───────────────────┘
                      │  generated SQL
                      ▼
┌─────────────────────────────────────────┐
│           Validation Gate               │
│  ✓ Check 1: Queries governed views only │
│  ✓ Check 2: No join fan-out / double    │
│             counting patterns           │
│  ✓ Check 3: Metric exists in YAML layer │
│                                         │
│  PASS → execute    FAIL → explain why   │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│          Semantic Layer (YAML)          │
│  ┌──────────────┐  ┌─────────────────┐  │
│  │   Metric     │  │   Dimension     │  │
│  │  Definitions │  │     Tables      │  │
│  └──────────────┘  └─────────────────┘  │
│  ┌──────────────────────────────────┐   │
│  │         Roll-up Mappings         │   │
│  │  FB violations + IG violations   │   │
│  │  = total content actioned        │   │
│  └──────────────────────────────────┘   │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│        DuckDB Data Warehouse            │
│   Star schema — fact + dimension tables │
│   Synthetic multi-product event data    │
│   (Facebook · Instagram · Threads)      │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│     Auditable Answer + Data Lineage     │
│  Every number traceable to source table,│
│  filter applied, and roll-up rule used  │
└─────────────────────────────────────────┘
```

---

## Repo Structure

```
metricguard/
│
├── data/
│   ├── raw/                        # Synthetic event logs (multi-product)
│   │   ├── fb_events.parquet
│   │   ├── ig_events.parquet
│   │   └── threads_events.parquet
│   └── warehouse/                  # DuckDB warehouse file
│       └── metricguard.duckdb
│
├── semantic_layer/
│   ├── metrics/                    # One YAML file per metric domain
│   │   ├── engagement.yaml         # DAU, MAU, session metrics
│   │   ├── content_actions.yaml    # Violations, removals, enforcements
│   │   └── reach.yaml              # Impressions, unique reach
│   ├── dimensions/
│   │   ├── dim_product.yaml        # Canonical product/surface definitions
│   │   ├── dim_geography.yaml      # Country, region, canonical IDs
│   │   └── dim_policy.yaml         # Policy area, violation taxonomy
│   └── rollups/
│       └── cross_product.yaml      # FB + IG + Threads aggregation rules
│
├── etl/
│   ├── generate_synthetic_data.py  # Builds raw event logs with built-in ambiguity
│   ├── load_warehouse.py           # ETL: raw → star schema in DuckDB
│   └── build_governed_views.py     # Creates semantic layer SQL views
│
├── agent/
│   ├── query_agent.py              # LLM text-to-SQL (Anthropic API)
│   └── context_builder.py          # Injects YAML semantic layer into LLM prompt
│
├── validation/
│   ├── gate.py                     # Main validation orchestrator
│   ├── checks/
│   │   ├── governed_views.py       # Check 1: no raw table access
│   │   ├── fanout_detector.py      # Check 2: JOIN graph analysis (sqlglot)
│   │   └── metric_registry.py      # Check 3: metric defined in YAML?
│   └── lineage.py                  # Traces every answer to its source
│
├── tests/
│   ├── test_etl.py
│   ├── test_validation_gate.py
│   └── test_agent.py
│
├── demo/
│   ├── demo.py                     # Side-by-side: with vs without MetricGuard
│   └── screenshots/
│
├── requirements.txt
├── .env.example                    # ANTHROPIC_API_KEY placeholder
└── README.md
```

---

## Semantic Layer — What It Actually Looks Like

The semantic layer is a YAML config that declares what each metric means. Not SQL. Not code. A declaration that every system must resolve against.

```yaml
# semantic_layer/metrics/engagement.yaml

metric: daily_active_users
description: >
  Count of distinct users who performed at least one qualifying
  session_start event on any product surface within a calendar
  day (UTC), deduplicated by canonical user_id.
  Cross-platform sessions count once per originating surface.

grain: [product_surface, country_iso2, date]

filters:
  - session_duration_seconds >= 5
  - bot_probability_score < 0.05
  - user_account_status = 'active'

source_table: fact_sessions
dedup_key: user_id
owner: product-analytics
last_reviewed: 2026-06-01
```

Any query — SQL, LLM-generated, or dashboard — that touches DAU must resolve through this definition. The number is the same every time, for every team.

---

## The Validation Gate — Core Logic

```python
# validation/gate.py (simplified)

import sqlglot
from checks.governed_views import check_governed_views
from checks.fanout_detector import check_fanout_risk
from checks.metric_registry import check_metric_defined

def validate(sql: str, metric_name: str) -> dict:
    """
    Returns: {"valid": bool, "reason": str, "lineage": dict}
    Never returns a silent wrong answer.
    """
    ast = sqlglot.parse_one(sql)

    # Check 1: No direct raw table access
    governed, reason = check_governed_views(ast)
    if not governed:
        return {"valid": False, "reason": f"Blocked: {reason}. Use governed view instead."}

    # Check 2: Join fan-out risk (double-counting)
    safe, reason = check_fanout_risk(ast)
    if not safe:
        return {"valid": False, "reason": f"Blocked: {reason}. Fan-out detected — result would be inflated."}

    # Check 3: Metric must be defined in semantic layer
    defined, reason = check_metric_defined(metric_name)
    if not defined:
        return {"valid": False, "reason": f"Blocked: '{metric_name}' is not yet a governed metric. Define it in semantic_layer/metrics/ first."}

    return {"valid": True, "reason": "All checks passed.", "lineage": build_lineage(ast, metric_name)}
```

---

## The Demo

The demo runs two queries side by side:

**Without MetricGuard** — LLM queries raw tables directly:
```
Q: "What was DAU last week?"

→ Answer 1 (raw fb_events):   1.82B  
→ Answer 2 (raw ig_events):   2.41B  
→ Answer 3 (combined, naive): 4.23B  ← double-counted
```

**With MetricGuard** — LLM queries through the semantic layer:
```
Q: "What was DAU last week?"

→ Validation: PASSED (all 3 checks)
→ Answer: 3.27B
→ Lineage: fact_sessions → filter(duration≥5s, bot<0.05) 
           → dedup(user_id) → rollup(FB+IG+Threads)
→ Definition used: daily_active_users v1.2 (reviewed 2026-06-01)
```

Same question. One trustworthy answer. Full audit trail.

---

## Why This Matters at Scale

This project is grounded in a real, documented problem. Large analytics orgs operating across multiple products must build canonical mappings that roll up data types to consistent definitions — because different teams use different classifiers for the same underlying concept. Without a governed layer, every tool recreates definitions slightly differently, leading to inevitable confusion across teams.

The addition of AI-generated SQL makes this more urgent: a survey of 330+ data teams found average confidence in AI-generated query results at just **5.5/10**. LLM accuracy on data questions improves by up to **3x** when models query through a semantic layer rather than raw tables directly.

MetricGuard is a working implementation of the engineering pattern that makes AI-assisted analytics trustworthy enough to act on.

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/NikhilNerurkar/metricguard.git
cd metricguard

# Install dependencies
pip install -r requirements.txt

# Set your Anthropic API key
cp .env.example .env
# Add your key to .env: ANTHROPIC_API_KEY=your_key_here

# Generate synthetic data and load warehouse
python etl/generate_synthetic_data.py
python etl/load_warehouse.py
python etl/build_governed_views.py

# Run the demo
python demo/demo.py
```

---

## Build Roadmap

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | Repo structure + README | ✅ Done |
| 2 | Synthetic data generator (multi-product event logs) | 🔲 Up next |
| 3 | DuckDB star schema + ETL pipeline | 🔲 Planned |
| 4 | Semantic layer YAML definitions | 🔲 Planned |
| 5 | Governed SQL views | 🔲 Planned |
| 6 | LLM text-to-SQL agent | 🔲 Planned |
| 7 | Validation gate (sqlglot AST checks) | 🔲 Planned |
| 8 | Data lineage tracer | 🔲 Planned |
| 9 | Demo script (with vs without) | 🔲 Planned |
| 10 | Tests + documentation | 🔲 Planned |

---

## Skills Demonstrated

| Skill | Where |
|-------|-------|
| Dimensional modeling / star schema | `etl/load_warehouse.py`, DuckDB schema |
| ETL pipeline design | `etl/` directory |
| Data warehouse modeling | DuckDB warehouse, governed views |
| Schema design | Fact + dimension table definitions |
| Data quality & validation | `validation/gate.py` |
| Data lineage | `validation/lineage.py` |
| LLM / GenAI integration | `agent/query_agent.py` |
| Responsible AI / bias & quality review | Validation gate — blocks unsafe queries |
| SQL (complex queries, AST parsing) | `validation/checks/fanout_detector.py` |
| Python | Entire codebase |

---

*Built by Nikhil Nerurkar — MSBA candidate, University of Maryland (Dec 2026)*  
*Contact: nnerurka@umd.edu | [LinkedIn](https://linkedin.com/in/nikhilnerurkar)*
