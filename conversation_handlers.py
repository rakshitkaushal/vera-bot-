"""
conversation_handlers.py — Multi-turn conversation handler for Vera Bot
Implements respond() for optional multi-turn capability (tiebreaker per spec §7.4).
"""

import json
import httpx
from typing import Optional
from dataclasses import dataclass, field

CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = "claude-sonnet-4-6"


@dataclass
class ConversationState:
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str]
    turns: list[dict] = field(default_factory=list)  # [{"from": "vera"|"merchant", "msg": "..."}]
    intent_history: list[str] = field(default_factory=list)  # ["neutral", "accept", ...]
    auto_reply_count: int = 0
    ended: bool = False
    merchant_context: Optional[dict] = None
    category_context: Optional[dict] = None
    trigger_context: Optional[dict] = None
    customer_context: Optional[dict] = None


# ─────────────────────────────────────────────
# Auto-reply detection
# ─────────────────────────────────────────────
AUTO_REPLY_PHRASES = [
    "automated assistant", "automated response", "auto reply",
    "thank you for contacting", "aapki madad ke liye shukriya",
    "bahut-bahut shukriya", "team tak pahuncha", "main ek automated",
    "this is an automated", "out of office", "will get back to you",
]


def is_auto_reply(message: str, state: ConversationState) -> bool:
    msg_lower = message.lower()
    if any(p in msg_lower for p in AUTO_REPLY_PHRASES):
        return True
    merchant_msgs = [t["msg"] for t in state.turns if t.get("from") == "merchant"]
    if merchant_msgs.count(message) >= 2:
        return True
    return False


def detect_intent(message: str) -> str:
    msg_lower = message.lower()
    action = ["yes", "ha", "haan", "ok", "okay", "go ahead", "sure", "chalega",
              "karo", "karein", "let's do", "send me", "bhejo", "show me", "start", "join"]
    stop = ["no", "nahi", "nope", "not interested", "stop", "band karo",
            "mat bhejo", "don't", "later", "baad mein", "busy", "not now", "gst"]
    question = ["kya", "what", "how", "kaise", "kyun", "why", "when", "kitna", "?"]
    hostile = ["bakwas", "chup", "stupid", "idiot", "spam", "block", "report", "abuse"]
    
    if any(p in msg_lower for p in hostile):
        return "hostile"
    if any(p in msg_lower for p in stop):
        return "stop"
    if any(p in msg_lower for p in action):
        return "accept"
    if any(p in msg_lower for p in question):
        return "question"
    return "neutral"


# ─────────────────────────────────────────────
# Multi-turn reply composer
# ─────────────────────────────────────────────
MULTI_TURN_SYSTEM = """You are Vera, magicpin's WhatsApp merchant AI assistant.
You are in an ongoing conversation. Compose the next reply.

OUTPUT FORMAT — valid JSON only, no markdown:
{
  "action": "send" | "wait" | "end",
  "body": "<WhatsApp message — omit if action is end or wait>",
  "cta": "open_ended" | "binary_yes_stop" | "none",
  "wait_seconds": <int, only if action=wait>,
  "rationale": "<1 sentence>"
}

MULTI-TURN RULES:
- If merchant ACCEPTED: switch to action mode immediately, confirm what you'll do, do NOT re-qualify
- If merchant DECLINED: return action=end gracefully
- If merchant sent AUTO-REPLY twice: return action=end gracefully
- If merchant is HOSTILE or off-topic (GST, unrelated): stay on mission, politely redirect once, then end
- If merchant ASKED A QUESTION: answer it specifically using the context, then restate the CTA
- Keep replies shorter than the opener; merchant is engaged, don't over-explain
- Never re-introduce yourself; never repeat the exact same body you sent before
- After 5 turns, gracefully close the conversation
"""


def respond(state: ConversationState, merchant_message: str) -> dict:
    """
    Given the conversation state + the merchant's latest message, produce the next reply.
    Returns: {"action": "send"|"wait"|"end", "body": "...", "cta": "...", "rationale": "..."}
    """
    if state.ended:
        return {"action": "end", "rationale": "Conversation already ended."}
    
    # Auto-reply handling
    if is_auto_reply(merchant_message, state):
        state.auto_reply_count += 1
        state.turns.append({"from": "merchant", "msg": merchant_message, "auto_reply": True})
        
        if state.auto_reply_count >= 2:
            state.ended = True
            return {
                "action": "end",
                "rationale": "Auto-reply detected 2+ times. Exiting to avoid pollution.",
            }
        else:
            # One gentle redirect
            owner = "there"
            if state.merchant_context:
                owner = (state.merchant_context.get("identity", {})
                         .get("owner_first_name", "there"))
            body = (
                f"Lagta hai yeh automated message tha. "
                f"Koi baat nahi {owner} ji — jab bhi free hon, reply karen. "
                f"Aapke liye ek useful update wait kar raha hai. 🙂"
            )
            state.turns.append({"from": "vera", "msg": body})
            return {
                "action": "send",
                "body": body,
                "cta": "open_ended",
                "rationale": "First auto-reply; sending one human-redirect then will exit.",
            }
    
    # Intent detection
    intent = detect_intent(merchant_message)
    state.intent_history.append(intent)
    state.turns.append({"from": "merchant", "msg": merchant_message})
    
    # Hard exit conditions
    if intent == "stop":
        state.ended = True
        owner = "there"
        if state.merchant_context:
            owner = state.merchant_context.get("identity", {}).get("owner_first_name", "there")
        return {
            "action": "end",
            "rationale": "Merchant declined. Exiting gracefully.",
        }
    
    if intent == "hostile":
        state.ended = True
        return {
            "action": "end",
            "rationale": "Merchant sent hostile/abusive message. Exiting to preserve relationship.",
        }
    
    # Turn limit
    vera_turns = sum(1 for t in state.turns if t.get("from") == "vera")
    if vera_turns >= 5:
        state.ended = True
        return {
            "action": "end",
            "rationale": "5-turn limit reached. Closing conversation.",
        }
    
    # Build context for LLM
    merchant = state.merchant_context or {}
    category = state.category_context or {}
    trigger = state.trigger_context or {}
    
    owner_name = merchant.get("identity", {}).get("owner_first_name",
                 merchant.get("identity", {}).get("name", "there"))
    languages = merchant.get("identity", {}).get("languages", ["en"])
    lang_note = "Hindi-English code-mix preferred" if "hi" in languages else "English"
    
    history_text = "\n".join(
        f"[{t.get('from','?').upper()}] {t.get('msg','')}"
        for t in state.turns[-6:]  # last 6 turns
    )
    
    perf = merchant.get("performance", {})
    active_offers = [o["title"] for o in merchant.get("offers", []) if o.get("status") == "active"]
    trigger_kind = trigger.get("kind", "ongoing")
    
    prompt = f"""MULTI-TURN CONVERSATION — compose next reply.

CONVERSATION HISTORY:
{history_text}

MERCHANT: {owner_name} | Category: {category.get('slug','?')} | Language: {lang_note}
TRIGGER TYPE: {trigger_kind}
DETECTED INTENT OF LAST MERCHANT MESSAGE: {intent}
ACTIVE OFFERS: {', '.join(active_offers) if active_offers else 'none'}
PERFORMANCE: views={perf.get('views',0)}, calls={perf.get('calls',0)}, CTR={perf.get('ctr',0):.3f}
VOICE: {category.get('voice',{}).get('tone','friendly')}
VERA TURN #{vera_turns + 1} of max 5

TASK: Respond naturally to the merchant's last message above. 
{"The merchant ACCEPTED — take action immediately, don't re-qualify." if intent == "accept" else ""}
{"The merchant ASKED A QUESTION — answer it specifically." if intent == "question" else ""}
Keep the reply SHORT (1-3 sentences + CTA if needed)."""
    
    try:
        response = httpx.post(
            CLAUDE_API_URL,
            headers={
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": CLAUDE_MODEL,
                "max_tokens": 400,
                "temperature": 0,
                "system": MULTI_TURN_SYSTEM,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=25.0,
        )
        response.raise_for_status()
        data = response.json()
        raw = data["content"][0]["text"].strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        result = json.loads(raw)
        
        if result.get("action") in ("end", "wait"):
            state.ended = result["action"] == "end"
            return result
        
        body = result.get("body", "")
        
        # Anti-repetition
        last_vera = next(
            (t["msg"] for t in reversed(state.turns) if t.get("from") == "vera"), ""
        )
        if body and body == last_vera:
            body += " Koi sawaal ho toh batayein. 😊"
        
        state.turns.append({"from": "vera", "msg": body})
        return {
            "action": "send",
            "body": body,
            "cta": result.get("cta", "open_ended"),
            "rationale": result.get("rationale", ""),
        }
    
    except Exception as e:
        # Fallback
        if intent == "accept":
            body = "Bilkul! Main abhi aapke liye kaam start karti hoon. ✅ 2 minute mein update milega."
        elif intent == "question":
            body = "Great question! Iska jawab aapke account data mein hai — ek second dein."
        else:
            body = f"Samajh gayi, {owner_name} ji. Koi aur sawaal ho toh bataiye. 😊"
        
        state.turns.append({"from": "vera", "msg": body})
        return {
            "action": "send",
            "body": body,
            "cta": "none",
            "rationale": f"Fallback reply (error: {str(e)[:60]})",
        }
