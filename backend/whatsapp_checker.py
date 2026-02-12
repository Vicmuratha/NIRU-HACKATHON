"""
WhatsApp Forward Checker

In Kenya, most misinformation spreads via WhatsApp forwards.
67% of Kenyans get news via WhatsApp (Reuters Institute 2024).

This module detects common patterns in forwarded messages:
- Urgency/fear language in English and Swahili
- Known Kenya-specific hoax templates (Safaricom, KRA, health scares)
- ALL CAPS, excessive exclamation marks
- "Forward to X people" chains
- Scam patterns (fake prizes, fake jobs)
"""

import re
from typing import Dict, Any, List

FORWARD_INDICATORS = [
    # Urgency patterns (English)
    r"forward\s*(this|to)",
    r"share\s*(this|with|to)",
    r"send\s*to\s*\d+\s*(people|contacts|groups)",
    r"breaking\s*news",
    r"confirmed\s*by\s*(state\s*house|government|ministry)",
    r"just\s*in",
    r"urgent",
    r"please\s*share",

    # Urgency patterns (Swahili)
    r"tuma\s*(kwa|hii)",          # send this
    r"sambaza",                    # spread/share
    r"hatari\s*kubwa",             # great danger
    r"onyo\s*la\s*dharura",       # emergency warning
    r"habari\s*za\s*mwisho",     # last/breaking news
    r"taarifa\s*ya\s*dharura",   # emergency notice
    r"shiriki\s*na",              # share with

    # Common hoax patterns
    r"safaricom\s*(is|will|ita)\s*(charg|block|shut|fung)",
    r"kra\s*(is|will|ita)\s*(tax|charg|levy)",
    r"government\s*(is|has)\s*(banned|announced|declared)",
    r"serikali\s*(ime|ita)",      # government has/will

    # Religious/emotional manipulation
    r"if\s*you\s*(love|believe|fear)\s*(god|jesus|allah)",
    r"type\s*amen",
    r"ignore\s*if\s*you\s*don",

    # Money/prize scams
    r"you\s*(have\s*)?won",
    r"umeshinda",                  # you have won (Swahili)
    r"free\s*(airtime|data|money|bundles)",
    r"click\s*(here|this\s*link)",
    r"bonyeza\s*hapa",            # click here (Swahili)
]

KENYA_HOAX_TEMPLATES = [
    {
        "pattern": r"safaricom.*(charg|tax|block|shut|fung)",
        "category": "Safaricom Hoax",
        "debunk": (
            "Safaricom does not announce policy changes via WhatsApp forwards. "
            "Verify at safaricom.co.ke or call 100."
        ),
    },
    {
        "pattern": r"(kra|government|serikali).*(tax|levy|fine).*(whatsapp|mpesa|airtime)",
        "category": "Tax/Government Hoax",
        "debunk": (
            "Official government policies are published in the Kenya Gazette "
            "and on mygov.go.ke. WhatsApp forwards are never official channels."
        ),
    },
    {
        "pattern": r"(county|governor|senator|mp|cs|cabinet\s*secretary).*(fired|arrested|resign|died|sacked)",
        "category": "Political Misinformation",
        "debunk": (
            "Verify political news through established Kenyan media: "
            "Nation, Standard, Citizen TV, or the official government portal."
        ),
    },
    {
        "pattern": r"(vaccine|dawa|medicine|chanjo).*(kill|poison|infertil|dangerous|hatari)",
        "category": "Health Misinformation",
        "debunk": (
            "Verify health claims with the Kenya Ministry of Health (health.go.ke) "
            "or WHO Kenya."
        ),
    },
    {
        "pattern": r"(job|kazi|internship|recruitment|hiring).*(apply|send\s*cv|deadline|haraka)",
        "category": "Fake Job Listing",
        "debunk": (
            "Verify job postings on official company websites or MyGov.go.ke for "
            "government positions. Never pay for a job application."
        ),
    },
    {
        "pattern": r"(earthquake|tetemeko|tsunami|floods|mafuriko).*(warning|onyo|alert|tahadhari)",
        "category": "Disaster Hoax",
        "debunk": (
            "Verify disaster alerts with Kenya Meteorological Department "
            "(meteo.go.ke) or the Kenya Red Cross (@KenyaRedCross)."
        ),
    },
    {
        "pattern": r"(mpesa|m-pesa).*(charg|fee|tax|ada).*(call|message|sms)",
        "category": "M-Pesa Fee Hoax",
        "debunk": (
            "M-Pesa tariff changes are communicated officially via Safaricom app, "
            "mysafaricom.co.ke, and gazetted by CA. Not via WhatsApp."
        ),
    },
]

# Swahili clickbait / sensationalism keywords
SWAHILI_CLICKBAIT = [
    "usisadiki",        # "you won't believe"
    "kumbe",            # "as it turns out" (sensational)
    "siri kubwa",       # "big secret"
    "imefichuliwa",     # "has been exposed"
    "utashangaa",       # "you will be surprised"
    "haraka sana",      # "very urgent"
    "taarifa ya dharura",
    "imethibitishwa",   # "has been confirmed"
    "kashfa kubwa",     # "big scandal"
    "habari za mwisho", # "final/breaking news"
    "ushahidi mpya",    # "new evidence"
    "imevuja",          # "has leaked"
]


def analyze_forward(text: str) -> Dict[str, Any]:
    """Analyse text for WhatsApp forward / misinformation patterns."""
    text_lower = (text or "").lower()

    if len(text_lower.strip()) < 10:
        return {
            "is_likely_forward": False,
            "forward_risk_score": 0,
            "indicators_count": 0,
            "indicators": [],
            "hoax_matches": [],
            "swahili_clickbait": [],
            "verdict": "TOO_SHORT",
            "advice": {},
        }

    # Count forward indicators
    indicators_found: List[str] = []
    for pattern in FORWARD_INDICATORS:
        if re.search(pattern, text_lower):
            indicators_found.append(pattern)

    # Check against known Kenya hoax templates
    hoax_matches: List[Dict[str, str]] = []
    for hoax in KENYA_HOAX_TEMPLATES:
        if re.search(hoax["pattern"], text_lower):
            hoax_matches.append({
                "category": hoax["category"],
                "debunk": hoax["debunk"],
            })

    # Swahili clickbait detection
    swahili_hits: List[str] = []
    for keyword in SWAHILI_CLICKBAIT:
        if keyword in text_lower:
            swahili_hits.append(keyword)

    # Calculate forward risk score
    forward_score = min(
        100,
        len(indicators_found) * 12
        + len(hoax_matches) * 25
        + len(swahili_hits) * 8,
    )

    # Long forwarded messages are more suspicious
    if len(text) > 1000:
        forward_score = min(100, forward_score + 10)

    # ALL CAPS detection
    alpha_chars = [c for c in text if c.isalpha()]
    if alpha_chars:
        caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if caps_ratio > 0.4:
            forward_score = min(100, forward_score + 15)
            indicators_found.append("EXCESSIVE_CAPS")

    # Exclamation marks
    if text.count("!") > 3:
        forward_score = min(100, forward_score + 10)
        indicators_found.append("EXCESSIVE_EXCLAMATION")

    # Multiple emojis (common in forwards)
    emoji_count = sum(1 for c in text if ord(c) > 0x1F600)
    if emoji_count > 5:
        forward_score = min(100, forward_score + 5)

    return {
        "is_likely_forward": forward_score > 30,
        "forward_risk_score": forward_score,
        "indicators_count": len(indicators_found),
        "indicators": indicators_found[:10],
        "hoax_matches": hoax_matches,
        "swahili_clickbait": swahili_hits,
        "verdict": (
            "LIKELY_MISINFORMATION"
            if forward_score > 65
            else "SUSPICIOUS_FORWARD"
            if forward_score > 35
            else "LOW_RISK"
        ),
        "advice": {
            "en": (
                "Before sharing: Is the source named? Can you verify it on a "
                "news site? If not, DON'T FORWARD."
            ),
            "sw": (
                "Kabla ya kushiriki: Je, chanzo kimetajwa? Je, unaweza kuthibitisha "
                "kwenye tovuti ya habari? Kama hapana, USISAMBAZE."
            ),
        },
        "fact_check_resources": [
            {"name": "PesaCheck", "url": "https://pesacheck.org/"},
            {"name": "Africa Check", "url": "https://africacheck.org/"},
            {"name": "Citizen Digital", "url": "https://www.citizen.digital/"},
        ],
    }
