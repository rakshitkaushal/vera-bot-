"""
bot.py — Vera Bot Composer
magicpin AI Challenge submission module.

Exposes:
  compose(category, merchant, trigger, customer=None) -> dict
  
This module is importable standalone (for submission.jsonl generation)
and is also used internally by the FastAPI server (main.py).
"""

import os
import json
import time
import httpx
from typing import Optional

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
6. No preamble ("I hope you're doing well...")
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


def build_compose_prompt(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
) -> str:
    merchant_name = merchant.get("identity", {}).get("name", "")
    owner_name = merchant.get("identity", {}).get("owner_first_name", merchant_name)
    languages = merchant.get("identity", {}).get("languages", ["en"])
    lang_note = "Hindi-English code-mix preferred" if "hi" in languages else "English only"
    
    trigger_kind = trigger.get("kind", "")
    trigger_payload = trigger.get("payload", {})
    
    # Digest summary
    digest_items = category.get("digest", [])
    digest_summary = ""
    if digest_items:
        top = digest_items[0]
        digest_summary = (
            f"Latest digest: {top.get('title')} ({top.get('source','')}) "
            f"— {top.get('summary','')}"
        )
        if len(digest_items) > 1:
            digest_summary += f"\nAlso: {digest_items[1].get('title')} ({digest_items[1].get('source','')})"
    
    # Performance
    perf = merchant.get("performance", {})
    peer_stats = category.get("peer_stats", {})
    perf_summary = (
        f"30d: {perf.get('views',0)} views, {perf.get('calls',0)} calls, "
        f"CTR {perf.get('ctr',0):.3f} (peer avg {peer_stats.get('avg_ctr',0):.3f}). "
        f"7d delta: views {perf.get('delta_7d',{}).get('views_pct',0):+.0%}, "
        f"calls {perf.get('delta_7d',{}).get('calls_pct',0):+.0%}"
    )
    
    signals = ", ".join(merchant.get("signals", []))
    active_offers = [o["title"] for o in merchant.get("offers", []) if o.get("status") == "active"]
    cust_agg = merchant.get("customer_aggregate", {})
    
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
- Trigger payload: {json.dumps(trigger_payload, ensure_ascii=False)}"""
    
    return f"""COMPOSE A WHATSAPP MESSAGE for this scenario:

TRIGGER: {trigger_kind} (urgency {trigger.get('urgency',2)})
Trigger payload: {json.dumps(trigger_payload, ensure_ascii=False)}

MERCHANT:
- Name: {merchant_name} (address as "{owner_name}" or "Dr. {owner_name}" if dentist/doctor)
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

TASK: Compose the OPENING proactive message for this trigger. End with a single clear CTA."""


def compose(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
) -> dict:
    """
    Main compose function.
    Inputs are the dicts loaded from the dataset JSON.
    Returns a dict with keys: body, cta, send_as, suppression_key, rationale.
    Temperature=0 for determinism.
    """
    prompt = build_compose_prompt(category, merchant, trigger, customer)
    
    try:
        response = httpx.post(
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
            timeout=28.0,
        )
        response.raise_for_status()
        data = response.json()
        raw = data["content"][0]["text"].strip()
        # Strip markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        return json.loads(raw)
    
    except Exception as e:
        # Fallback
        merchant_id = merchant.get("merchant_id", "unknown")
        trigger_kind = trigger.get("kind", "generic")
        sup_key = trigger.get("suppression_key", f"fallback:{merchant_id}:{trigger_kind}")
        owner = merchant.get("identity", {}).get("owner_first_name",
                 merchant.get("identity", {}).get("name", "there"))
        return {
            "body": f"Namaste {owner}! Aapke account mein ek important update hai. Reply YES janane ke liye.",
            "cta": "binary_yes_stop",
            "send_as": "vera",
            "suppression_key": sup_key,
            "rationale": f"Fallback message (error: {str(e)[:80]})",
        }


if __name__ == "__main__":
    # Quick smoke test with sample data
    sample_category = {
        "slug": "dentists",
        "display_name": "Dentists",
        "voice": {"tone": "peer_clinical", "vocab_taboo": ["guaranteed", "100% safe"]},
        "offer_catalog": [{"title": "Dental Cleaning @ ₹299"}],
        "peer_stats": {"avg_ctr": 0.030, "avg_rating": 4.4, "avg_review_count": 62},
        "digest": [{
            "id": "d_jida_fluoride",
            "title": "3-month fluoride recall cuts caries 38% better than 6-month",
            "source": "JIDA Oct 2026, p.14",
            "trial_n": 2100,
            "patient_segment": "high_risk_adults",
            "summary": "Multi-center Indian trial. No effect in low-risk patients.",
        }],
    }
    sample_merchant = {
        "merchant_id": "m_001_drmeera_dentist_delhi",
        "category_slug": "dentists",
        "identity": {
            "name": "Dr. Meera's Dental Clinic",
            "owner_first_name": "Meera",
            "city": "Delhi", "locality": "Lajpat Nagar",
            "languages": ["en", "hi"],
        },
        "subscription": {"status": "active", "plan": "Pro", "days_remaining": 82},
        "performance": {
            "views": 2410, "calls": 18, "ctr": 0.021,
            "delta_7d": {"views_pct": 0.18, "calls_pct": -0.05},
        },
        "offers": [{"title": "Dental Cleaning @ ₹299", "status": "active"}],
        "signals": ["stale_posts:22d", "ctr_below_peer_median", "high_risk_adult_cohort"],
        "customer_aggregate": {"total_unique_ytd": 540, "lapsed_180d_plus": 78, "retention_6mo_pct": 0.38},
    }
    sample_trigger = {
        "id": "trg_001",
        "kind": "research_digest",
        "scope": "merchant",
        "source": "external",
        "merchant_id": "m_001_drmeera_dentist_delhi",
        "urgency": 2,
        "suppression_key": "research:dentists:2026-W17",
        "payload": {"category": "dentists", "top_item_id": "d_jida_fluoride"},
    }
    
    print("Running smoke test...")
    result = compose(sample_category, sample_merchant, sample_trigger)
    print(json.dumps(result, indent=2, ensure_ascii=False))
