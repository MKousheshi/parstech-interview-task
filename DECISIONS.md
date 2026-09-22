# Scope & Architecture Decisions

Living document for decisions we make together about the scope and architecture of the assistant, ahead of and during implementation. Add to this iteratively; keep entries dated and reasoned, not just conclusions.

## Status

In progress. See `CLARIFICATIONS.md` for the open questions driving these decisions.

## Decision log

### 2026-09-22 — Delivery interface: CLI chat loop

- **Decision:** Build the assistant as a terminal REPL (Python script) driving the LangGraph app.
- **Why:** Fastest to build and demo; appropriate for an interview take-home. Core logic will still live in a reusable module so an API/UI could be layered on later if needed.
- **Alternatives considered:** HTTP API; CLI+API combo. Deferred — not needed for the current scope.

### 2026-09-22 — LLM provider: OpenAI

- **Decision:** Use OpenAI chat models via `langchain-openai` (e.g. `gpt-4o-mini`) as the primary LLM.
- **Why:** Widely supported, cheap, solid tool-calling support in LangGraph.

### 2026-09-22 — Conversation persistence: in-memory only

- **Decision:** Conversation/entity state lives only for the duration of the running process, using LangGraph's in-memory checkpointer.
- **Why:** Simplest option; sufficient for a demo/interview task. No requirement surfaced for resuming conversations after a restart.

### 2026-09-22 — Product search: phased, multi-strategy (filter → TF-IDF → embeddings)

- **Decision:** Implement product search in three phases, each building on the last, as separate pluggable strategies so they can be compared against each other:
  1. **Phase 1 — Filter/keyword search:** naive substring/keyword matching over product name, description, and category, plus explicit category filter. Bare minimum, no ranking sophistication.
  2. **Phase 2 — TF-IDF similarity search:** vectorize product text (name + description) with category folded into the text corpus (so category terms naturally contribute to relevance) using `scikit-learn`'s `TfidfVectorizer` + cosine similarity for ranked relevance.
  3. **Phase 3 — Embedding/semantic search:** vector similarity search using text embeddings, to catch semantic/synonym matches that TF-IDF misses.
- **Why:** The user wants to empirically compare accuracy/usefulness of filter vs. TF-IDF vs. embedding-based (and possibly hybrid) search, rather than committing to one approach upfront. A phased build lets each strategy be evaluated before adding the next layer of complexity.

#### Sub-decisions

- **Embeddings for Phase 3:** local `sentence-transformers`, using a multilingual model (e.g. `paraphrase-multilingual-MiniLM-L12-v2` or similar — exact model to be pinned when Phase 3 is implemented) since product content is in Persian. Chosen over OpenAI embeddings to avoid API cost/calls for this step and keep semantic search runnable fully offline.
- **Category integration in Phase 1:** both — category name is folded into the searchable text (so free-text queries like "instagram" match products in that category) **and** exposed as an explicit `category` filter parameter for precise narrowing.
- **Strategy comparison:** a small eval script with a handful of realistic Persian sample queries, run against all three strategies (filter, TF-IDF, embeddings) with ranked results printed side-by-side for manual inspection. No labeled ground-truth set for now — can be upgraded to one later if needed.

### 2026-09-22 — Consultation flow: LLM-driven entity extraction, two-step analysis/suggestion

- **Decision:**
  - **Entity collection:** on each user turn, an LLM call (structured output / tool-calling against a Pydantic schema for the 4 entities: business type, customer type, geographic location, virtual sales channel) extracts and updates whatever entities are present in the message, merging into graph state. The graph routes to "ask for next missing entity" until all 4 are filled, then transitions to the analysis phase. This allows the user to answer out of order or provide multiple entities in one message, rather than forcing a rigid Q&A script.
  - **Analysis + suggestion:** implemented as two separate graph nodes/LLM calls rather than one combined call:
    1. **Analysis node:** LLM generates the free-knowledge business analysis/recommendation based on the 4 collected entities only (no product data in context).
    2. **Suggestion node:** runs product search (using the search tool from the phased search work) based on the entities/analysis, then a second LLM call formats the product/package suggestion from the retrieved shortlist.
- **Why:** Structured per-turn extraction keeps the conversation natural instead of a rigid script. Splitting analysis and suggestion into two steps keeps each LLM call focused (general business reasoning vs. grounding suggestions in actual retrieved products), makes it easier to debug/inspect each stage independently, and avoids conflating "free knowledge" reasoning with retrieval-grounded output in a single prompt.
- **Alternatives considered:** single combined LLM call for analysis+suggestion (simpler, fewer calls, but risks the model inventing products instead of using the real catalog); strict sequential entity prompts (simpler graph, less natural conversation).

### 2026-09-22 — Acceptance criteria: working CLI demo

- **Decision:** The task is considered done when the CLI can run end-to-end: product search works, and a full consultation conversation (entity collection → analysis → product suggestion) completes successfully.
- **Why:** Matches the chosen CLI-first delivery scope; keeps the bar concrete and demoable without requiring a formal test suite or write-up as a hard requirement.
- **Note:** The search-strategy eval script (from the phased search decision) is still being built as part of the search work itself, and a README is worth adding for an interview submission — but neither is a hard gate on "done" beyond the working demo.
