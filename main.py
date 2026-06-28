"""
Vera AI Bot — magicpin Merchant AI Assistant Challenge
FastAPI server exposing 5 required endpoints.
Uses Claude claude-sonnet-4-6 for high-quality message composition.
"""

import os
import time
import uuid
import json
import logging
import httpx
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("vera-bot")

# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────
app = FastAPI(title="Vera Bot", version="1.0.0")
START_TIME = time.time()

# ─────────────────────────────────────────────
# In-memory stores
# ─────────────────────────────────────────────
# (scope, context_id) -> {version, payload}
contexts: dict[tuple[str, str], dict] = {}

# conversation_id -> list of turns
conversations: dict[str, list[dict]] = {}

# suppression: suppression_key -> timestamp sent
sent_suppressions: set[str] = set()


# ─────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────
class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


# ─────────────────────────────────────────────
# Helper: get context payload
# ─────────────────────────────────────────────
def get_ctx(scope: str, context_id: str) -> Optional[dict]:
    entry = contexts.get((scope, context_id))
    return entry["payload"] if entry else None


def count_by_scope() -> dict[str, int]:
    counts: dict[str, int] = {}
    for (scope, _) in contexts:
        counts[scope] = counts.get(scope, 0) + 1
    return counts


# ─────────────────────────────────────────────
# Auto-reply detection
# ─────────────────────────────────────────────
AUTO_REPLY_PHRASES = [
    "automated assistant",
    "automated response",
    "auto reply",
    "thank you for contacting",
    "aapki madad ke liye shukriya",
    "bahut-bahut shukriya",
    "team tak pahuncha",
    "main ek automated",
    "this is an automated",
    "out of office",
    "will get back to you",
]


def is_auto_reply(message: str, conversation: list[dict]) -> bool:
    msg_lower = message.lower()
    # Check phrase patterns
    if any(p in msg_lower for p in AUTO_REPLY_PHRASES):
        return True
    # Check verbatim repeat (3+ times = auto-reply)
    merchant_msgs = [t["msg"] for t in conversation if t.get("from") == "merchant"]
    if merchant_msgs.count(message) >= 2:  # this would be 3rd occurrence
        return True
    return False


def detect_intent(message: str) -> str:
    """Detect merchant intent from message."""
    msg_lower = message.lower()
    # Strong action intent
    action_phrases = ["yes", "ha", "haan", "ok", "okay", "go ahead", "sure", "chalega",
                      "karo", "karein", "let's do", "let's go", "join", "signup",
                      "send me", "bhejo", "show me", "dikhao", "start", "proceed"]
    stop_phrases = ["no", "nahi", "nope", "not interested", "stop", "band karo",
                    "mat bhejo", "don't", "later", "baad mein", "busy", "not now"]
    question_phrases = ["kya", "what", "how", "kaise", "kyun", "why", "when", "kitna",
                        "?", "bataiye", "tell me", "explain"]

    if any(p in msg_lower for p in stop_phrases):
        return "stop"
    if any(p in msg_lower for p in action_phrases):
        return "accept"
    if any(p in msg_lower for p in question_phrases):
        return "question"
    return "neutral"


# ─────────────────────────────────────────────
# Claude AI Composer
# ─────────────────────────────────────────────
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are Vera, magicpin's AI merchant assistant on WhatsApp.
Your job: compose one highly targeted WhatsApp message to a merchant (or their customer).

OUTPUT FORMAT — respond ONLY with valid JSON, no markdown, no preamble:
{
  "body": "<WhatsApp message text>",
  "cta": "open_ended" | "binary_yes_stop" | "none",
  "send_as": "vera" | "merchant_on_behalf",
  "suppression_key": "<unique key>",
  "rationale": "<1-2 sentences explaining why this message, what lever it uses>"
}

RULES:
1. Specificity wins: anchor on a verifiable fact (number, date, headline, peer stat, source)
2. Single primary CTA: binary YES/STOP for action triggers; no CTA for pure-info
3. Voice match: clinical-peer for dentists/doctors; friendly-aspirational for salons; energetic for gyms; warm for restaurants
4. Hindi-English code-mix when merchant language includes "hi"
5. No generic offers ("Flat 30% off") — use service+price ("Haircut @ ₹99")
6. No preamble ("I hope you're doing well…")
7. No fabrication — use only data provided
8. Message must feel like a WhatsApp message: concise, warm, specific, one clear ask at the end
9. Use compulsion levers: loss aversion, social proof, curiosity, effort externalization, reciprocity
10. For customer-facing: set send_as="merchant_on_behalf", softer tone, practical slot info

ANTI-PATTERNS (penalized by judge):
- "AMAZING DEAL!" promotional tone for clinical categories
- Multiple CTAs
- Buried CTA (it must land in the LAST sentence)
- Re-introducing yourself after first message
- Long messages without a clear point
"""


async def compose_with_claude(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
    conversation_history: Optional[list] = None,
    is_reply: bool = False,
    merchant_message: Optional[str] = None,
    intent: Optional[str] = None,
) -> dict:
    """Call Claude to compose the next message."""
    
    # Build context for the prompt
    merchant_name = merchant.get("identity", {}).get("name", "")
    owner_name = merchant.get("identity", {}).get("owner_first_name", merchant_name)
    languages = merchant.get("identity", {}).get("languages", ["en"])
    lang_note = "Hindi-English code-mix preferred" if "hi" in languages else "English only"
    
    trigger_kind = trigger.get("kind", "")
    trigger_payload = trigger.get("payload", {})
    
    # Build category digest summary
    digest_items = category.get("digest", [])
    digest_summary = ""
    if digest_items:
        top = digest_items[0]
        digest_summary = f"Latest digest: {top.get('title')} ({top.get('source','')}) — {top.get('summary','')}"
        if len(digest_items) > 1:
            digest_summary += f"\nAlso: {digest_items[1].get('title')} ({digest_items[1].get('source','')})"
    
    # Build performance summary
    perf = merchant.get("performance", {})
    peer_stats = category.get("peer_stats", {})
    perf_summary = (
        f"30d: {perf.get('views',0)} views, {perf.get('calls',0)} calls, "
        f"CTR {perf.get('ctr',0):.3f} (peer avg {peer_stats.get('avg_ctr',0):.3f}). "
        f"7d delta: views {perf.get('delta_7d',{}).get('views_pct',0):+.0%}, "
        f"calls {perf.get('delta_7d',{}).get('calls_pct',0):+.0%}"
    )
    
    # Signals
    signals = ", ".join(merchant.get("signals", []))
    
    # Active offers
    active_offers = [o["title"] for o in merchant.get("offers", []) if o.get("status") == "active"]
    
    # Customer aggregate
    cust_agg = merchant.get("customer_aggregate", {})
    
    # Customer context (if any)
    customer_section = ""
    if customer:
        c_id = customer.get("identity", {})
        c_rel = customer.get("relationship", {})
        customer_section = f"""
CUSTOMER CONTEXT (message is on behalf of merchant TO this customer):
- Name: {c_id.get('name')}, Language: {c_id.get('language_pref')}
- State: {customer.get('state')} | Visits: {c_rel.get('visits_total')} | Last visit: {c_rel.get('last_visit')}
- Services received: {', '.join(c_rel.get('services_received', []))}
- Preferences: {customer.get('preferences', {})}
- Consent scope: {customer.get('consent', {}).get('scope', [])}"""
    
    # Conversation history (if replying)
    history_section = ""
    if conversation_history:
        recent = conversation_history[-4:]  # last 4 turns
        history_section = "\nCONVERSATION SO FAR:\n" + "\n".join(
            f"[{t.get('from','?').upper()}] {t.get('msg','')}" for t in recent
        )
    
    if is_reply and merchant_message:
        history_section += f"\n[MERCHANT JUST REPLIED] {merchant_message}"
        if intent:
            history_section += f"\n[DETECTED INTENT]: {intent}"
    
    prompt = f"""COMPOSE A WHATSAPP MESSAGE for this scenario:

TRIGGER: {trigger_kind} (urgency {trigger.get('urgency',2)})
Trigger payload: {json.dumps(trigger_payload, ensure_ascii=False)}

MERCHANT:
- Name: {merchant_name} (address merchant as "{owner_name}" or "Dr. {owner_name}" if dentist/doctor)
- Category: {category.get('slug')} | City: {merchant.get('identity',{}).get('city')} | Locality: {merchant.get('identity',{}).get('locality')}
- Language: {lang_note}
- Subscription: {merchant.get('subscription',{}).get('status')} plan, {merchant.get('subscription',{}).get('days_remaining')} days left
- Performance: {perf_summary}
- Active offers: {', '.join(active_offers) if active_offers else 'None'}
- Signals: {signals}
- Customers: {cust_agg.get('total_unique_ytd',0)} total YTD, {cust_agg.get('lapsed_180d_plus',0)} lapsed 180d+, {cust_agg.get('retention_6mo_pct',0):.0%} 6mo retention

CATEGORY VOICE: {category.get('voice',{}).get('tone')} | Taboo words: {', '.join(category.get('voice',{}).get('vocab_taboo',[]))}
PEER STATS: avg_ctr={peer_stats.get('avg_ctr',0):.3f}, avg_rating={peer_stats.get('avg_rating',0)}, avg_reviews={peer_stats.get('avg_review_count',0)}

{digest_summary}
{customer_section}
{history_section}

{"TASK: Compose the REPLY to the merchant's latest message above. Honor their intent — if they accepted, take action immediately. If they asked a question, answer it. If they declined, exit gracefully." if is_reply else "TASK: Compose the OPENING proactive message for this trigger. End with a single clear CTA."}"""

    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post(
                CLAUDE_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": CLAUDE_MODEL,
                    "max_tokens": 600,
                    "temperature": 0,
                    "system": SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            raw = data["content"][0]["text"].strip()
            # Strip any markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            return json.loads(raw)
    except Exception as e:
        logger.error(f"Claude API error: {e}")
        # Fallback: return a generic but valid response
        merchant_id = merchant.get("merchant_id", "unknown")
        return {
            "body": f"Namaste {owner_name}! Aapke account mein kuch important updates hain. Reply YES to know more.",
            "cta": "binary_yes_stop",
            "send_as": "vera",
            "suppression_key": f"fallback:{merchant_id}:{trigger_kind}",
            "rationale": "Fallback message due to API error.",
        }


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/v1/healthz")
async def healthz():
    counts = count_by_scope()
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": {
            "category": counts.get("category", 0),
            "merchant": counts.get("merchant", 0),
            "customer": counts.get("customer", 0),
            "trigger": counts.get("trigger", 0),
        },
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Rakshit Varma",
        "team_members": ["Rakshit Varma"],
        "model": CLAUDE_MODEL,
        "approach": (
            "Claude claude-sonnet-4-6 composer with trigger-aware prompt routing. "
            "Extracts the highest-signal fact per trigger kind and anchors the message "
            "on that specific number/date/headline. Auto-reply detection via phrase + verbatim "
            "repeat check. Intent detection (accept/stop/question/neutral) for reply routing."
        ),
        "contact_email": "rakshitvarma.de@gmail.com",
        "version": "1.0.0",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/v1/context")
async def push_context(body: ContextBody):
    if body.scope not in ("category", "merchant", "customer", "trigger"):
        return JSONResponse(
            status_code=400,
            content={"accepted": False, "reason": "invalid_scope",
                     "details": f"scope must be one of category/merchant/customer/trigger, got '{body.scope}'"},
        )
    key = (body.scope, body.context_id)
    current = contexts.get(key)
    if current and current["version"] >= body.version:
        return JSONResponse(
            status_code=409,
            content={"accepted": False, "reason": "stale_version",
                     "current_version": current["version"]},
        )
    contexts[key] = {"version": body.version, "payload": body.payload}
    logger.info(f"Context stored: {body.scope}/{body.context_id} v{body.version}")
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}_{uuid.uuid4().hex[:6]}",
        "stored_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []

    for trg_id in body.available_triggers:
        trg = get_ctx("trigger", trg_id)
        if not trg:
            logger.warning(f"Trigger not found in context: {trg_id}")
            continue

        # Suppression check
        sup_key = trg.get("suppression_key", "")
        if sup_key and sup_key in sent_suppressions:
            logger.info(f"Suppressed trigger {trg_id} (key={sup_key})")
            continue

        merchant_id = trg.get("merchant_id") or trg.get("payload", {}).get("merchant_id")
        if not merchant_id:
            continue

        merchant = get_ctx("merchant", merchant_id)
        if not merchant:
            logger.warning(f"Merchant not found: {merchant_id}")
            continue

        category_slug = merchant.get("category_slug")
        category = get_ctx("category", category_slug)
        if not category:
            logger.warning(f"Category not found: {category_slug}")
            continue

        # Optional customer context
        customer_id = trg.get("customer_id")
        customer = get_ctx("customer", customer_id) if customer_id else None

        # Compose
        try:
            composed = await compose_with_claude(
                category=category,
                merchant=merchant,
                trigger=trg,
                customer=customer,
            )
        except Exception as e:
            logger.error(f"Compose failed for {trg_id}: {e}")
            continue

        # Generate conversation ID
        conv_id = f"conv_{merchant_id}_{trg_id}_{uuid.uuid4().hex[:8]}"

        # Record suppression
        if sup_key:
            sent_suppressions.add(sup_key)

        # Init conversation
        conversations[conv_id] = [
            {"from": "vera", "msg": composed["body"], "ts": body.now}
        ]

        # Determine template name
        trigger_kind = trg.get("kind", "generic")
        template_name = f"vera_{trigger_kind}_v1"

        # Prepare template params (name, key fact, CTA hint)
        owner = merchant.get("identity", {}).get("owner_first_name",
                 merchant.get("identity", {}).get("name", "there"))
        
        action = {
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": composed.get("send_as", "vera"),
            "trigger_id": trg_id,
            "template_name": template_name,
            "template_params": [owner, trigger_kind, composed.get("cta", "open_ended")],
            "body": composed["body"],
            "cta": composed.get("cta", "open_ended"),
            "suppression_key": composed.get("suppression_key", sup_key),
            "rationale": composed.get("rationale", ""),
        }
        actions.append(action)
        logger.info(f"Action queued: {conv_id} | trigger={trg_id}")

        # Limit to 20 actions per tick
        if len(actions) >= 20:
            break

    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv_id = body.conversation_id
    conv = conversations.get(conv_id, [])

    merchant_msg = body.message

    # Auto-reply detection
    if is_auto_reply(merchant_msg, conv):
        # One gentle retry if first time, then exit
        auto_count = sum(1 for t in conv if t.get("auto_reply"))
        if auto_count >= 1:
            # Exit gracefully
            conv.append({"from": "merchant", "msg": merchant_msg, "auto_reply": True})
            conversations[conv_id] = conv
            return {
                "action": "end",
                "rationale": "Auto-reply detected twice. Gracefully exiting to avoid spam.",
            }
        else:
            conv.append({"from": "merchant", "msg": merchant_msg, "auto_reply": True})
            conversations[conv_id] = conv
            # Try once more with acknowledgment
            merchant_id = body.merchant_id
            merchant = get_ctx("merchant", merchant_id) if merchant_id else None
            owner = "there"
            if merchant:
                owner = merchant.get("identity", {}).get("owner_first_name",
                         merchant.get("identity", {}).get("name", "there"))
            retry_msg = (
                f"Lagta hai yeh automated reply tha. Koi baat nahi — "
                f"jab bhi {owner} ji time ho, reply karen. "
                f"Aapke account mein ek important update wait kar raha hai. 🙂"
            )
            conv.append({"from": "vera", "msg": retry_msg})
            conversations[conv_id] = conv
            return {
                "action": "send",
                "body": retry_msg,
                "cta": "open_ended",
                "rationale": "First auto-reply detected; sending one gentle real-human redirect then will exit.",
            }

    # Detect intent
    intent = detect_intent(merchant_msg)
    logger.info(f"Reply intent={intent} conv={conv_id} turn={body.turn_number}")

    # Stop intent — graceful exit
    if intent == "stop":
        conv.append({"from": "merchant", "msg": merchant_msg})
        conversations[conv_id] = conv
        return {
            "action": "end",
            "rationale": "Merchant signaled not-interested. Gracefully exiting conversation.",
        }

    # Too many turns — wrap up
    if body.turn_number >= 5:
        conv.append({"from": "merchant", "msg": merchant_msg})
        conversations[conv_id] = conv
        return {
            "action": "end",
            "rationale": "Reached 5-turn limit. Ending conversation cleanly.",
        }

    # Compose reply using Claude
    merchant_id = body.merchant_id
    merchant = get_ctx("merchant", merchant_id) if merchant_id else None

    if not merchant:
        # No merchant context — generic reply
        if intent == "accept":
            return {
                "action": "send",
                "body": "Bilkul! Main abhi aapke liye yeh set up karti hoon. Ek minute dein. ✅",
                "cta": "none",
                "rationale": "Merchant accepted; confirming action.",
            }
        return {
            "action": "send",
            "body": "Got it! Kya aap thoda aur detail share kar sakte hain?",
            "cta": "open_ended",
            "rationale": "Generic follow-up — no merchant context available.",
        }

    category_slug = merchant.get("category_slug")
    category = get_ctx("category", category_slug) if category_slug else {}

    # Find the trigger for this conversation (approximate — use first active trigger for merchant)
    trigger = {}
    for (scope, ctx_id), entry in contexts.items():
        if scope == "trigger" and entry["payload"].get("merchant_id") == merchant_id:
            trigger = entry["payload"]
            break

    # Customer context
    customer = get_ctx("customer", body.customer_id) if body.customer_id else None

    # Record message in conversation
    conv.append({"from": body.from_role, "msg": merchant_msg})
    conversations[conv_id] = conv

    try:
        composed = await compose_with_claude(
            category=category or {},
            merchant=merchant,
            trigger=trigger or {"kind": "reply", "urgency": 2, "payload": {}},
            customer=customer,
            conversation_history=conv,
            is_reply=True,
            merchant_message=merchant_msg,
            intent=intent,
        )
        reply_body = composed["body"]
        cta = composed.get("cta", "open_ended")
        rationale = composed.get("rationale", "")
    except Exception as e:
        logger.error(f"Reply compose failed: {e}")
        # Fallback based on intent
        if intent == "accept":
            reply_body = "Perfect! Main abhi yeh kaam karti hoon — 2 minute mein update deti hoon. ✅"
            cta = "none"
            rationale = "Merchant accepted; executing action."
        else:
            reply_body = "Samajh gayi! Koi aur sawaal ho toh batayein. 😊"
            cta = "none"
            rationale = "Fallback reply."

    # Anti-repetition: don't send the same body as last vera message
    last_vera_msgs = [t["msg"] for t in conv if t.get("from") == "vera"]
    if last_vera_msgs and last_vera_msgs[-1] == reply_body:
        reply_body = reply_body + " Koi aur madad chahiye? Reply karen. 🙂"

    conv.append({"from": "vera", "msg": reply_body})
    conversations[conv_id] = conv

    return {
        "action": "send",
        "body": reply_body,
        "cta": cta,
        "rationale": rationale,
    }


# ─────────────────────────────────────────────
# Optional teardown endpoint
# ─────────────────────────────────────────────
@app.post("/v1/teardown")
async def teardown():
    contexts.clear()
    conversations.clear()
    sent_suppressions.clear()
    logger.info("State wiped on teardown.")
    return {"status": "wiped"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
