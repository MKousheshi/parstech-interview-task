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

### 2026-09-22 — Build two parallel chatbot architectures: scripted (flow-based) and agentic (tool-calling)

- **Decision:** Implement the assistant twice, as two structurally distinct LangGraph apps sharing the same product-search/data layer:
  - **Scripted** (`docs/ARCHITECTURE_SCRIPTED.md`): a literal, scripted state machine — entities collected one at a time in the exact order given in the task spec, a single per-turn intent classifier with fixed branches, deterministic template-based handling wherever the task doesn't explicitly require an LLM call, no correction/follow-up/tangent handling.
  - **Agentic** (`docs/ARCHITECTURE_AGENTIC.md`): a tool-calling, LLM-driven conversational agent — entities extracted and merged opportunistically in any order, one general-purpose assistant node handles search/follow-ups/tangents/off-topic input, with hardcoded logic reserved only for the parts the task requires without exception (analysis/suggestion firing once all 4 entities are known, kept as isolated, separate LLM calls).
- **Why:** Mirrors the empirical-comparison philosophy already applied to search (filter vs. TF-IDF vs. embeddings): rather than committing to one conversational design, build both and compare them concretely against the same scenarios worked through during design (follow-up questions, compound queries, off-topic input, mid-conversation corrections). This produces a stronger interview artifact than either alone — a real trade-off comparison instead of an unsupported claim that one design is better.
- **Shared infrastructure:** both variants reuse the same product data pipeline and pluggable search strategies (`consultant_bot/common/`) — they differ only in how the conversation itself is orchestrated (`consultant_bot/scripted/` vs `consultant_bot/agentic/`).
- **Alternatives considered:** picking one design and committing (half the implementation work, but loses the comparison value and forecloses the "which is actually better for this task" question the design discussion was building toward).

### 2026-09-22 — Agentic variant: consultation firing requires an explicit request, not just complete entities

- **Decision:** Refine the agentic architecture's consultation trigger: `completion_check` now fires analysis+suggestion only when all 4 entities are present **and** the user has explicitly asked for (or agreed to) a consultation (`consultation_requested`), not on entity completeness alone. If entities complete while `consultation_requested` is still false, the `assistant` node proactively offers to run the consultation once (tracked via `consultation_offered`, set by plain code so the offer isn't repeated every subsequent turn) instead of firing unasked.
- **Why:** Firing an unrequested business analysis and product recommendation the instant the 4th entity happens to be mentioned in casual conversation is presumptuous — entities can complete incidentally (e.g. during a product-search chat) without the user ever having asked for consultation. Gating on an explicit request keeps the assistant from talking over the user's actual intent, while still meeting the "fire immediately once appropriate" requirement from the earlier decision — "appropriate" now includes having been asked.
- **Scope:** Agentic variant only. The scripted variant already gates consultation on `route_intent` classifying the turn as `consultation` before any entity questions are asked, so it has no equivalent "complete but unrequested" state to handle.
- **Alternatives considered:** always firing on completion regardless of request (the prior behavior — simpler, but presumptuous per above); requiring an explicit request only, with no proactive offer (misses entities that complete silently mid-conversation, leaving the user without a natural next step).

### 2026-09-23 — Phase 3 embedding model: `paraphrase-multilingual-MiniLM-L12-v2`

- **Decision:** Pin `sentence-transformers`' `paraphrase-multilingual-MiniLM-L12-v2` as the Phase 3 embedding model.
- **Why:** Small (~118MB) and fast enough for local CPU inference in a demo, while covering Persian among its 50+ supported languages — sufficient to catch semantic/synonym matches TF-IDF misses without needing GPU inference or an API call.
- **Alternatives considered:** a larger multilingual model (e.g. `paraphrase-multilingual-mpnet-base-v2`) for better semantic accuracy, at the cost of slower local encoding — not worth it for a 29-product catalog where TF-IDF and embeddings are already being compared side by side rather than one being the final answer.

### 2026-09-23 — Drop the CLI; the Gradio web UI is the only front end

- **Decision:** Remove the terminal REPL (`cli.py` and the `consultant-bot` script). The Gradio web UI (`uv run consultant-bot-web`) is the only front end; the `--arch` graph registry moves to its own `architectures.py` module. This supersedes the "Delivery interface: CLI chat loop" decision, and the "working CLI demo" acceptance bar now means a working web UI demo.
- **Why:** The web UI already covers everything the CLI did, and renders Persian right-to-left properly, which a terminal doesn't. Keeping two front ends meant two copies of the "which replies did this turn append" logic, plus the web UI importing its graph registry from the CLI module.
- **Alternatives considered:** keeping the CLI as a lightweight debugging entry point — not worth the upkeep, since the web UI is the demo that actually gets shown.

### 2026-09-23 — Agentic variant: the message history is the agent's memory of shown products; drop `last_shown_products`

- **Decision:** Remove the agentic variant's `last_shown_products` state field and the system-prompt block that injected it. Tool results stay in `messages` as ordinary `AIMessage(tool_calls=...)`/`ToolMessage` pairs, and those are what the assistant uses to answer follow-ups about earlier results. The `suggestion` node, which searches outside the tool loop, now adds its retrieval to the history as a synthetic `search_products` call/result pair right before its formatted reply, so the products it recommends are stored in the same form.
- **Why:** The field repeated data that was already in the history, since tool results were never trimmed. Its prompt also told the model to answer follow-ups only from the latest result set and never to search again. That rule assumed the user only asks about the most recent search, which is the kind of scripted assumption the agentic variant is supposed to avoid. Without the field, a question about any earlier result can be answered, and the model decides for itself when a new search helps. The one case the field genuinely covered was suggestion-node products, which only reached the history as formatting-LLM prose that might drop a price or link. The recorded call/result pair covers that case.
- **Keeping tool results in the history — trade-offs considered:** For: the agent sees exactly what it saw, prices and links included, instead of relying on its own prose summaries; it matches the standard ReAct message shape; there are fewer pointless repeat searches; the checkpointed state is easy to debug; and the size cost is negligible here (29 products, `top_k` 5, one line per hit). Against: context grows for the life of the session; stale or irrelevant earlier results stay visible, and the explicit "latest" marker is gone, so the model has to work out which list "the second one" refers to; any node that reads the history has to filter tool traffic (`extract_entities` already does, `analysis` is isolated by design, and the web UI shows only reply text); and trimming later has to remove call/result pairs together or the chat API rejects the history.
- **If it needs limiting later:** trim or summarize older tool results at model-call time, for example with a `trim_messages` middleware or by replacing old results with a short placeholder, keeping the full history in state. Don't bring back a curated slot.
- **Scope:** Agentic variant only. The scripted variant keeps `last_search_results`, which fits its scripted, template-driven follow-up handling.
- **Alternatives considered:** keeping the field but softening its prompt wording (smaller change, but keeps duplicated state and a nudge toward the latest results); removing tool results from the history and keeping only the curated slot (smaller context, but follow-ups about anything but the latest search fail, and the model's prose becomes the only record of older results).

### 2026-09-23 — Agentic variant: the entity extractor sees only the latest exchange

- **Decision:** `extract_entities` now reads only the latest user message and the assistant reply just before it, instead of the last 6 conversational messages. Its prompt lists the values already stored and asks only for what the latest message states or corrects. A value that differs from the stored one only in spacing or Arabic/Persian letter variants no longer counts as a change.
- **Why:** With the wider window, the extractor re-returned answers given turns earlier, often reworded. In a live gpt-4o-mini check, an unrelated follow-up ("how much was the second one?") returned "کافی‌شاپ" for a stored "کافه" on 3 runs out of 3, even with the prompt told not to repeat old values. Any change clears `consultation_done`, and `consultation_requested` stays set, so this re-ran the whole analysis and suggestion on a turn that had nothing to do with it. With the two-message window, the same check returned no change, and short answers ("شیراز" after "which city?"), corrections and a "yes" to the offer were all still picked up, 3 runs out of 3.
- **Cost:** a "yes" to the consultation offer only counts when it directly answers it; an agreement that arrives turns later has to be an explicit request. A reference back to something said several turns earlier ("the city I mentioned") can't be resolved.
- **Alternatives considered:** a prompt-only fix (tried; it didn't stop the rewording); a stricter change test such as asking a second LLM call whether the value really changed (an extra call every turn to guard against a problem the smaller window removes).

### 2026-09-23 — Agentic variant: the `search_products` tool applies the relevance threshold too

- **Decision:** The strategy's `relevance_threshold` now applies to the assistant's `search_products` tool as well as the `suggestion` node, through one shared helper (`common/search/base.py:search_relevant()`). When a guessed category has no hit above the threshold, the helper retries across the whole catalog.
- **Why:** Only `suggestion` applied the threshold before. With the `tfidf` and `embedding` strategies the tool always handed the model its `top_k` best matches, however unrelated, and a model shown five products for an off-catalog question will usually recommend one. One shared floor also means the two search paths can't drift apart.
- **Scope:** Agentic variant only. The scripted variant still writes up whatever `top_k` returns, as its architecture doc says. The `filter` strategy's threshold is 0, so its results don't change.
- **Alternatives considered:** leaving relevance to the model's judgment (it has no score to judge by, only the list); returning scores in the tool output (more tokens, and it still leaves the call to the model).

### 2026-09-23 — Hybrid search: a weighted sum of the TF-IDF and embedding scores

- **Decision:** A fourth strategy, `hybrid` (`common/search/hybrid_search.py`), scores each product as `0.5 × TF-IDF + 0.5 × embedding`, using the two strategies' raw cosine similarities. Its `relevance_threshold` is the same blend of their floors (0.15). A product TF-IDF leaves out (zero similarity) scores 0 on that side. Selected with `CONSULTANT_BOT_SEARCH_STRATEGY=hybrid`, and shown as its own column in `scripts/compare_search.py`.
- **Why:** TF-IDF is precise on exact terms but blind to synonyms, and embeddings catch paraphrases but score unrelated text generously. On the real catalog the blend drops products that only the embedding liked: for "ارسال پیامک تبلیغاتی" an AI video-teaser product leaves the top 5 and the messaging products move up, and for the compound site + Google Ads + Instagram query the store-design product joins the top 5. The blended floor is also stricter on off-catalog queries: "تعمیر ماشین لباسشویی" gets 0.287 from the embedding strategy alone (above its 0.2 floor, so a product would be recommended) but 0.144 blended, under the 0.15 floor.
- **Weight:** 0.5 each. Embedding scores already spread wider (about 0.15–0.75 on this catalog vs. 0–0.3 for TF-IDF), so semantics still lead the ranking and exact terms mostly break near-ties. It's a constructor argument, not a setting; no labeled set exists to tune it against.
- **Alternatives considered:** Reciprocal Rank Fusion and per-query min-max normalization. Both are common, but both always give the best candidate a top score, even for a query nothing in the catalog matches. That would break the absolute relevance floor that `search_relevant()` depends on.

### 2026-09-23 — Rename the two architectures: rigid → scripted, flexible → agentic

- **Decision:** The variants are now `scripted` (formerly `rigid`) and `agentic` (formerly `flexible`) everywhere: the packages (`consultant_bot/scripted/`, `consultant_bot/agentic/`), their test folders, the `--arch` values, and the architecture docs (`ARCHITECTURE_SCRIPTED.md`, `ARCHITECTURE_AGENTIC.md`). Earlier entries in this log were updated to the new names.
- **Why:** The names should say what actually differs, which is who drives the conversation: the code or the LLM. "Rigid" read as a flaw, although predictability, low cost and full testability are what that variant is for. "Flexible" was accurate but didn't say what was flexible.
- **Alternatives considered:** `workflow` / `agent` (the common industry split, but both variants are LangGraph workflows, so it would confuse readers of the code); `deterministic` / `agentic` (overclaims, since the scripted variant still makes LLM calls to classify intent and to write the analysis and suggestion).
