# Vera Bot — magicpin AI Challenge Submission

**Team:** Rakshit Varma  
**Model:** Claude claude-sonnet-4-6  
**Version:** 1.0.0

---

## Approach

The bot is a **FastAPI server** that exposes the 5 required endpoints and uses Claude claude-sonnet-4-6 (temperature=0) as the composition engine.

### Core design decisions

**1. Trigger-aware prompt construction**  
Every `/v1/tick` call extracts the highest-signal fact from the trigger payload — the specific number, date, headline, or peer delta most relevant to that trigger kind — and anchors the prompt on it. A `research_digest` trigger foregrounds the trial's n and the percentage result; a `perf_dip` trigger foregrounds the exact call-count drop and peer median. This makes the judge's "Specificity" dimension the default, not an afterthought.

**2. Category voice enforcement**  
The prompt includes the category's `vocab_taboo` list and `tone` (e.g. `peer_clinical` for dentists) as hard constraints. The LLM is instructed to fail validation if taboo words appear — the fallback re-prompts with stricter constraints.

**3. Auto-reply detection**  
Two signals are combined: (a) exact phrase matching against 10 known Indian WhatsApp Business auto-reply patterns, and (b) verbatim repeat detection (same message ≥2 times from the same sender = auto-reply). On first auto-reply, one polite redirect is sent. On second, conversation ends gracefully.

**4. Intent routing**  
On every `/v1/reply`, the merchant's message is classified into: `accept / stop / question / hostile / neutral`. Accept → immediate action mode (no re-qualification). Stop/hostile → graceful exit. Question → answer specifically from context, then restate CTA.

**5. Suppression and idempotency**  
`/v1/context` is idempotent on `(scope, context_id, version)`. A higher version replaces atomically. Sent suppression keys are tracked in memory to prevent duplicate sends within the test window.

### What additional context would have helped most

- **Real merchant reply samples** per trigger kind — knowing what % of merchants reply YES vs auto-reply vs ask a question for each trigger type would let the bot tune its CTA phrasing better.
- **Actual suppression window rules** — the spec says "suppress duplicates" but doesn't define whether the window is per-session, per-day, or per-week.
- **Customer opt-in date per trigger** — knowing whether a customer opted in 6 months ago vs last week changes the re-engagement warmth level.

### Tradeoffs

- **Memory-only state**: fine for a 60-minute test window; would need Redis for production.
- **Single Claude call per action**: could be improved with retrieval (embed digest items, pull top-K for the trigger) but keeps latency well inside the 30s limit.
- **Temperature=0**: deterministic outputs at the cost of some phrasing variety across repeated ticks — acceptable given the judge rewards consistency.

---

## Deployment

```bash
# Set env var
export ANTHROPIC_API_KEY=sk-ant-...

# Run
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8080
```

Or deploy to Railway in one command:
```bash
railway up
```

Set `ANTHROPIC_API_KEY` in Railway's Variables tab before deploying.

---

## File structure

```
main.py                  # FastAPI server — all 5 endpoints
bot.py                   # Standalone compose() function
conversation_handlers.py # Multi-turn ConversationState + respond()
submission.jsonl         # 30 pre-composed test pair messages
requirements.txt
Dockerfile
railway.json             # Railway deploy config
render.yaml              # Render deploy config
```
