"""
SafEye Election Shield — Kenya 2027 Election Integrity Module

Kenya's 2007/08 Post-Election Violence killed 1,500+ and displaced 600,000+.
Incitement spread via SMS and vernacular radio. By 2027, the weapon is AI —
deepfaked videos of politicians, fabricated "leaked audio", doctored news screenshots.

This module contextualises deepfake detection results with election-specific
intelligence to flag manipulated political content before it spreads.
"""

import re
from datetime import datetime, date
from typing import Optional, List, Dict, Any

# Kenyan political figures (public figures — election integrity monitoring)
POLITICAL_FIGURES: Dict[str, List[str]] = {
    "president": ["ruto", "william ruto", "president ruto"],
    "opposition": ["raila", "raila odinga", "odinga", "baba"],
    "dp": ["gachagua", "rigathi", "kindiki", "kithure kindiki"],
    "political_parties": [
        "kenya kwanza", "azimio", "uda", "odm", "jubilee",
        "wiper", "ford kenya", "anc", "dap-k",
    ],
    "institutions": [
        "iebc", "electoral commission", "tume ya uchaguzi",
        "judiciary", "supreme court", "chief justice",
    ],
    "counties": [
        "nairobi", "mombasa", "kisumu", "nakuru", "eldoret",
        "nyeri", "machakos", "kiambu", "uasin gishu",
    ],
}

# Ethnic incitement indicators — detecting their use in deepfake context
# is about PREVENTING incitement, not targeting communities
INCITEMENT_RISK_KEYWORDS: Dict[str, List[str]] = {
    "high_risk": [
        "madoadoa",         # derogatory: "spots" — used in 2007 incitement
        "kabila adui",      # "enemy tribe"
        "wapinzani ni",     # "opponents are..."
        "hawa watu",        # "these people" (dehumanising context)
        "funga mpaka",      # "close the border" (internal displacement)
        "rudi kwenu",       # "go back where you came from"
        "vita",             # "war"
        "mapanga",          # "machetes"
        "hatutaki",         # "we don't want [them]"
        "ondoa",            # "remove"
    ],
    "medium_risk": [
        "ukabila",          # "tribalism"
        "kabila",           # "tribe" (contextual)
        "mwizi",            # "thief" (common political accusation)
        "uchochezi",        # "incitement"
        "ghasia",           # "chaos/violence"
        "maandamano",       # "protests"
        "revolution",
        "kifua mbele",      # "chest forward" — protest call
    ],
}

# Kenyan media outlets (for impersonation detection)
KENYAN_MEDIA_OUTLETS = [
    "citizen tv", "citizen digital", "royal media",
    "ntv kenya", "ntv", "nation media group",
    "ktn news", "ktn home", "standard group",
    "daily nation", "nation.africa",
    "the star", "the-star.co.ke",
    "tv47", "k24 tv",
    "kbc", "radio maisha", "radio citizen",
    "classic 105", "kiss fm", "milele fm",
    "the east african", "tuko", "kenyans.co.ke",
    "pulse live kenya", "kahawa tungu", "mpasho",
]

ELECTION_DATE_2027 = date(2027, 8, 10)  # Approximate


def analyze_election_context(
    text: str,
    media_type: str,
    deepfake_score: float,
    metadata: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Analyse content in Kenya's election context.

    This is not censorship — it provides context so users can make
    informed decisions about political content they encounter.
    """
    text_lower = (text or "").lower()

    result: Dict[str, Any] = {
        "election_relevant": False,
        "risk_level": "NONE",
        "political_figures_mentioned": [],
        "incitement_indicators": [],
        "media_impersonation": None,
        "days_to_election": (ELECTION_DATE_2027 - date.today()).days,
        "warnings": [],
        "recommendations": [],
    }

    # 1. Political figure mentions
    for category, names in POLITICAL_FIGURES.items():
        for name in names:
            if name in text_lower:
                result["political_figures_mentioned"].append({
                    "name": name,
                    "category": category,
                })
                result["election_relevant"] = True

    # 2. Incitement keywords
    for risk_level, keywords in INCITEMENT_RISK_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                result["incitement_indicators"].append({
                    "keyword": keyword,
                    "risk": risk_level,
                })

    # 3. Media outlet impersonation
    for outlet in KENYAN_MEDIA_OUTLETS:
        if outlet in text_lower:
            result["media_impersonation"] = {
                "outlet": outlet,
                "warning": (
                    f"This content references {outlet.title()}. If manipulated "
                    f"(score: {deepfake_score:.0f}%), it may be impersonating "
                    f"this outlet to spread false information."
                ),
            }
            result["election_relevant"] = True
            break

    # 4. If not election-relevant, return early
    if not result["election_relevant"] and not result["incitement_indicators"]:
        return result

    # 5. Calculate election risk
    election_risk = deepfake_score

    high_risk_count = sum(
        1 for i in result["incitement_indicators"] if i["risk"] == "high_risk"
    )
    medium_risk_count = sum(
        1 for i in result["incitement_indicators"] if i["risk"] == "medium_risk"
    )
    election_risk += high_risk_count * 15 + medium_risk_count * 5

    # Boost if close to election
    days = result["days_to_election"]
    if days < 90:
        election_risk *= 1.3
    elif days < 180:
        election_risk *= 1.15

    if result["political_figures_mentioned"] and deepfake_score > 50:
        election_risk += 20

    election_risk = min(100, election_risk)

    # 6. Set risk level and warnings
    if election_risk > 75:
        result["risk_level"] = "CRITICAL"
        result["warnings"].append({
            "en": (
                "🚨 CRITICAL: This appears to be manipulated political content "
                "containing incitement indicators. DO NOT SHARE."
            ),
            "sw": (
                "🚨 HATARI: Maudhui haya yanaonekana kuwa ya kisiasa "
                "yaliyobadilishwa na yana dalili za uchochezi. USISAMBAZE."
            ),
        })
        result["recommendations"].extend([
            "Do not share this content on any platform",
            "Report to NCIC: complaints@cohesion.or.ke / 0800 720 607",
            "Report to DCI Cybercrime Unit: reportcrime@dci.go.ke",
            "Report to Communications Authority: complaints@ca.go.ke",
        ])
    elif election_risk > 50:
        result["risk_level"] = "HIGH"
        result["warnings"].append({
            "en": (
                "⚠️ HIGH RISK: Political content with manipulation indicators. "
                "Verify with official sources before sharing."
            ),
            "sw": (
                "⚠️ HATARI KUBWA: Maudhui ya kisiasa yenye dalili za "
                "kubadilishwa. Thibitisha na vyanzo rasmi kabla ya kusambaza."
            ),
        })
        result["recommendations"].extend([
            "Verify this content with at least 2 established Kenyan news outlets",
            "Check IEBC's official channels: iebc.or.ke",
            "Check PesaCheck.org for fact-checks on this claim",
        ])
    elif election_risk > 30:
        result["risk_level"] = "MEDIUM"
        result["warnings"].append({
            "en": (
                "ℹ️ This political content has some manipulation indicators. "
                "Exercise caution before sharing."
            ),
            "sw": (
                "ℹ️ Maudhui haya ya kisiasa yana dalili za kubadilishwa. "
                "Kuwa makini kabla ya kusambaza."
            ),
        })

    # 7. Media-type specific context
    if media_type == "image" and deepfake_score > 50:
        result["warnings"].append({
            "en": (
                "This image may be AI-generated or manipulated. Common tactics: "
                "placing politicians in fabricated scenarios, altering crowd sizes, "
                "creating fake campaign posters."
            ),
            "sw": (
                "Picha hii inaweza kuwa imetengenezwa au kubadilishwa na AI. "
                "Mbinu za kawaida: kuweka wanasiasa katika hali za uongo, "
                "kubadilisha ukubwa wa umati, kuunda posteri bandia."
            ),
        })
    elif media_type == "audio" and deepfake_score > 50:
        result["warnings"].append({
            "en": (
                "This audio may be manipulated. In Kenya, 'leaked audio' of "
                "politicians is a common manipulation tactic used before elections. "
                "Always verify with the person's official channels."
            ),
            "sw": (
                "Sauti hii inaweza kuwa imebadilishwa. Nchini Kenya, "
                "'sauti zilizovuja' za wanasiasa ni mbinu ya kawaida ya "
                "udanganyifu kabla ya uchaguzi."
            ),
        })

    result["election_risk_score"] = round(election_risk, 1)

    # 8. Legal context
    result["legal_framework"] = {
        "applicable_laws": [
            "Computer Misuse and Cybercrimes Act 2018 — Section 22 (false publications)",
            "Computer Misuse and Cybercrimes Act 2018 — Section 23 (publication of false information)",
            "National Cohesion and Integration Act 2008 — Section 13 (ethnic contempt)",
            "Elections Act 2011 — Section 14 (electoral offences)",
            "Penal Code — Section 66 (undermining authority of public officer)",
        ],
        "penalties": "Up to KES 5,000,000 fine and/or 2 years imprisonment under CMCA 2018",
        "reporting_bodies": {
            "NCIC": "complaints@cohesion.or.ke | 0800 720 607",
            "DCI Cybercrime": "reportcrime@dci.go.ke | 0800 722 203",
            "IEBC": "info@iebc.or.ke",
            "Communications Authority": "complaints@ca.go.ke | 0703 042 000",
        },
    }

    return result
