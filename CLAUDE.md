# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

Interview take-home task. The **flexible** architecture (see `docs/ARCHITECTURE_FLEXIBLE.md`) is fully implemented: shared product pipeline, all three search-strategy phases, and the full `extract_entities -> assistant -> completion_check -> (analysis -> suggestion | END)` graph, wired up behind `uv run consultant-bot-web --arch flexible`. The **rigid** architecture (`docs/ARCHITECTURE_RIGID.md`) has not been started yet; its implementation checklist is `docs/TODO_RIGID.md` (local, gitignored). Live-LLM manual QA (the demo scenarios called for in the architecture doc) is still pending in whatever environment picks this up next, since no `OPENAI_API_KEY` was available while building the flexible variant.

## Configuration

All settings (OpenAI connection details, plus the `CONSULTANT_BOT_*` search/LLM knobs) are loaded via `pydantic-settings` from the environment or a `.env` file at the repo root — anchored to the repo, not the current directory (see `src/consultant_bot/common/config.py`; read them via `get_settings()`, not at import time). Copy `example.env` to `.env` and fill in `OPENAI_API_KEY` before running the web UI or any LLM-touching test; `OPENAI_BASE_URL` and `OPENAI_MODEL` are optional overrides (defaults: OpenAI's own endpoint, `gpt-4o-mini`).

## Commands

Package management is via `uv`.

- Install deps: `uv sync`
- Run the Gradio web UI (RTL-aware, for Persian; the only front end): `uv run consultant-bot-web`
- Run tests: `uv run pytest`
- Run a single test: `uv run pytest tests/flexible/test_graph.py::test_route_after_assistant_ends_when_not_yet_requested`
- Add a dependency: `uv add <package>` (dev-only: `uv add --dev <package>`)
- **Run the full local check pipeline (lint, format check, types, dependency vulnerability scan, tests): `./scripts/check.sh`** — run this before considering any change done. `./scripts/check.sh --fix` auto-applies `ruff check --fix` and `ruff format` first.
- Lint only: `uv run ruff check .` (add `--fix` to auto-fix)
- Format only: `uv run ruff format .` (add `--check` to check without writing)
- Type-check only: `uv run mypy` (checks `src`, `tests` and `scripts`, per `pyproject.toml`)
- Dependency vulnerability scan only: `uv run pip-audit`

## Repository map

- `docs/TASK_SPEC.md` — faithful English rendering of the original (Persian) task description. Treat as close to immutable; don't edit it to reflect scope decisions.
- `docs/CLARIFICATIONS.md` — open questions and observations about the task/data that haven't been resolved into a decision yet. Items move here first, then get resolved into `docs/DECISIONS.md` and removed from this file.
- `docs/DECISIONS.md` — the living log of scope/architecture decisions, dated and with rationale + alternatives considered. **This is the source of truth for how the assistant should be built.** Read it before implementing anything — it already answers most "how should this work" questions.
- `docs/ARCHITECTURE_FLEXIBLE.md` / `docs/ARCHITECTURE_RIGID.md` — implementation-level design for the two chatbot variants being built (see "What's being built" below): module layout, state schema, graph topology, and per-node behavior for each. Read the relevant one before touching `src/consultant_bot/flexible/` or `src/consultant_bot/rigid/`.
- `products.json` — raw WooCommerce REST API export of the store's product catalog (29 items, Persian-language digital-marketing products/services). `description` and `short_description` are raw HTML and need cleanup before use. Not all WooCommerce fields are relevant (see the "Products data" section of `docs/CLARIFICATIONS.md` for which ones).
- `src/consultant_bot/` — the Python package (uv-managed, src layout): `architectures.py` (`--arch` name -> graph builder registry), `webui.py` (Gradio RTL web UI, the only front end — the CLI was dropped), `common/` (shared product pipeline + search strategies + LLM factory), `flexible/` (fully built), `rigid/` (not yet started) per the architecture docs above.
- `tests/` — pytest suite, mirroring the `src/` package layout.

## What's being built (per docs/DECISIONS.md)

A chatbot with a Gradio web UI (LangChain + LangGraph, OpenAI as LLM provider) with two capabilities:

1. **Product search** over `products.json`, built in three phases as separate, comparable strategies (not a single final implementation):
   - Phase 1: filter/keyword search (substring match on name/description/category + explicit category filter).
   - Phase 2: TF-IDF similarity search (`scikit-learn` `TfidfVectorizer` + cosine similarity, category folded into the text corpus).
   - Phase 3: embedding/semantic search (local multilingual `sentence-transformers` model — not yet pinned).
   - A small comparison script (`uv run python scripts/compare_search.py`) runs all three strategies side-by-side on realistic Persian queries.
2. **Business consultation flow**: collects 4 entities (business type, customer type B2B/B2C, geographic location, virtual sales channel) across turns. Once all 4 are present, two separate LLM calls run: an analysis node (free-knowledge business recommendation, no product data in context) followed by a suggestion node (runs product search, then formats suggestions grounded in the retrieved results).

Built as **two parallel chatbot architectures**, not one — see `docs/ARCHITECTURE_FLEXIBLE.md` and `docs/ARCHITECTURE_RIGID.md`:
   - **Rigid**: a scripted state machine — entities collected one at a time in the spec's literal order, a fixed per-turn intent classifier, deterministic/template handling everywhere the task doesn't explicitly require an LLM call.
   - **Flexible**: a tool-calling LLM agent — entities extracted/merged opportunistically in any order, one general assistant node handles search/follow-ups/tangents/off-topic input, hardcoded logic reserved only for the parts the task requires without exception.
   - Both share the same product pipeline and pluggable search strategies (`src/consultant_bot/common/`).

Conversation state is in-memory only (LangGraph in-memory checkpointer, no cross-restart persistence). Acceptance bar is a working end-to-end web UI demo, not a test suite.

When implementing, keep search strategies as pluggable/swappable components and keep the analysis and suggestion LLM calls separate in both variants — these are explicit decisions in `docs/DECISIONS.md`, not incidental structure.
