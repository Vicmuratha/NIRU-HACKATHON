"""
Audio Detection — Honest Kenyan Context

Current reality (2026):
- Swahili AI-generated audio is LOW quality and detectable by ear
- English AI-generated audio is HIGH quality and hard to detect
- Kenya is bilingual — political manipulation will use ENGLISH deepfakes
  targeting educated/urban audiences, not Swahili (yet)

Our detection focuses on:
1. English-language deepfakes featuring Kenyan political figures
2. Manipulated REAL audio (splicing, editing, context removal)
3. Building baseline detection for when Swahili AI improves (2027+)

The REAL audio threat in Kenya today is splicing/editing real recordings,
not AI voice cloning. "Leaked audio" of politicians is a known tactic.
"""

from typing import Dict, Any

AUDIO_CONTEXT: Dict[str, Dict[str, Any]] = {
    "english": {
        "threat_level": "HIGH",
        "description": (
            "English AI voice cloning is mature and dangerous. "
            "Kenyan politicians frequently speak English in press "
            "conferences, interviews, and parliament. These are "
            "prime targets for AI voice cloning."
        ),
        "examples": [
            "Fake 'leaked' phone call recordings of politicians",
            "AI-generated voice statements attributed to CS/PS officials",
            "Cloned voices used in fake news video narration",
        ],
    },
    "swahili": {
        "threat_level": "LOW_BUT_GROWING",
        "description": (
            "Swahili AI audio is currently poor quality and "
            "detectable by native speakers. However, this gap "
            "is closing rapidly. By 2027 elections, Swahili "
            "deepfakes will be significantly more convincing."
        ),
        "examples": [
            "Currently rare — but preparing for 2027",
            "Swahili TTS improving via Meta SeamlessM4T, Google USM",
        ],
    },
    "sheng": {
        "threat_level": "MINIMAL",
        "description": (
            "Sheng is virtually impossible for AI to generate "
            "convincingly. The vocabulary, code-switching patterns, "
            "and cultural context are too complex and undocumented."
        ),
        "examples": [],
    },
}


def get_audio_kenya_context(
    detected_language: str,
    deepfake_score: float,
) -> Dict[str, Any]:
    """
    Provide honest Kenya-specific context for audio analysis.
    """
    result: Dict[str, Any] = {
        "detection_focus": "AUDIO_MANIPULATION",
        "context": None,
        "primary_threat": None,
        "warnings": [],
    }

    if detected_language in AUDIO_CONTEXT:
        result["context"] = AUDIO_CONTEXT[detected_language]
    else:
        result["context"] = AUDIO_CONTEXT["english"]

    # The REAL audio threat in Kenya is splicing/editing, not AI generation
    result["primary_threat"] = {
        "type": "AUDIO_SPLICING",
        "description": (
            "The most common audio manipulation in Kenya is NOT AI voice "
            "cloning — it is editing REAL recordings: cutting statements "
            "out of context, splicing different speeches together, or "
            "adding misleading background audio."
        ),
        "examples": [
            "A politician's rally speech edited to remove context",
            "Two separate audio clips spliced to create a fake 'conversation'",
            "Real audio with fake subtitles/translation overlaid",
            "Background crowd sounds added/removed to alter perceived setting",
        ],
    }

    if deepfake_score > 30:
        result["warnings"].append({
            "en": (
                "This audio shows signs of digital manipulation. "
                "This could be splicing (cutting/rearranging real speech), "
                "not necessarily AI generation."
            ),
            "sw": (
                "Sauti hii inaonyesha dalili za kubadilishwa kidijitali. "
                "Hii inaweza kuwa kukata/kupanga upya hotuba halisi, "
                "si lazima kutengenezwa na AI."
            ),
        })

    if deepfake_score > 60:
        result["warnings"].append({
            "en": (
                "HIGH MANIPULATION RISK: Before sharing any 'leaked audio' "
                "of a political figure, verify with the person's official "
                "channels or at least 2 credible news outlets."
            ),
            "sw": (
                "HATARI KUBWA YA UDANGANYIFU: Kabla ya kushiriki 'sauti "
                "iliyovuja' ya mtu wa kisiasa, thibitisha kupitia njia "
                "rasmi za mtu huyo au angalau vyombo 2 vya habari vya kuaminika."
            ),
        })

    # Detection honesty note
    result["detection_note"] = (
        "This analysis detects audio manipulation patterns including "
        "splicing, noise inconsistencies, and AI generation markers. "
        "In the Kenyan context, edited real audio is a far more common "
        "threat than AI-generated audio (as of 2026)."
    )

    return result
