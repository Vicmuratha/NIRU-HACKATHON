"""
Kenya Document Forgery Detector

Detects forged:
- KRA PIN certificates
- HELB statements
- Government circulars (from CS offices)
- M-Pesa confirmation screenshots (edited transaction amounts)
- University certificates (fake degrees)
- National ID images

These are shared daily on WhatsApp/Telegram to scam Kenyans.
Forged KRA PINs are used for fake tenders.
Fake HELB clearances are used for job applications.
Edited M-Pesa screenshots are used to fake payment proof.
"""

import re
from typing import Optional, Dict, Any, List

KENYA_DOCUMENT_PATTERNS: Dict[str, Dict[str, Any]] = {
    "kra_pin": {
        "name": "KRA PIN Certificate",
        "pattern": r"[AP]\d{9}[A-Z]",
        "required_elements": ["kenya revenue authority", "taxpayer", "pin"],
        "description": "KRA PIN certificates are commonly forged for fraud and fake tenders",
        "verify_url": "https://itax.kra.go.ke/KRA-Portal/ (verify PIN status)",
    },
    "national_id": {
        "name": "Kenya National ID",
        "pattern": r"\d{7,8}",
        "required_elements": ["republic of kenya", "identity card"],
        "description": "Forged national IDs used for SIM registration fraud and identity theft",
        "verify_url": "https://www.ecitizen.go.ke/ (verify via eCitizen)",
    },
    "helb": {
        "name": "HELB Statement / Clearance",
        "required_elements": ["higher education loans board", "loan"],
        "description": "Fake HELB clearance letters used for job applications",
        "verify_url": "https://www.helb.co.ke/ (verify clearance status)",
    },
    "government_circular": {
        "name": "Government Circular / Letter",
        "required_elements": ["republic of kenya", "ministry"],
        "description": "Fake government letters announcing policies, tenders, or jobs",
        "verify_url": "https://www.mygov.go.ke/ (verify official communications)",
    },
    "university_cert": {
        "name": "University Certificate",
        "known_institutions": [
            "university of nairobi", "kenyatta university", "moi university",
            "jomo kenyatta university", "egerton university", "maseno university",
            "strathmore university", "usiu", "daystar university",
            "mount kenya university", "multimedia university", "dedan kimathi",
            "technical university of kenya", "cooperative university",
            "karatina university", "laikipia university", "chuka university",
            "machakos university", "kibabii university", "rongo university",
        ],
        "description": "Fake degree certificates — a massive problem in Kenya's job market",
        "verify_url": "https://www.knqa.go.ke/ (Kenya National Qualifications Authority)",
    },
    "mpesa_statement": {
        "name": "M-Pesa Confirmation Screenshot",
        "required_elements": ["m-pesa", "confirmed"],
        "pattern": r"[A-Z]{2,3}\d{7,10}",  # M-Pesa transaction ID format
        "description": "Edited M-Pesa screenshots used to fake payment proof",
        "verify_url": "Check your M-Pesa message history or the Safaricom App",
    },
}


def detect_document_type(text: str) -> Optional[Dict[str, Any]]:
    """Identify what type of Kenyan document this might be from OCR text."""
    if not text:
        return None

    text_lower = text.lower()

    # Special case: university certificate (match institution names)
    uni_info = KENYA_DOCUMENT_PATTERNS["university_cert"]
    for institution in uni_info["known_institutions"]:
        if institution in text_lower:
            return {
                "document_type": "university_cert",
                "document_name": uni_info["name"],
                "description": uni_info["description"],
                "confidence": 0.8,
                "institution": institution.title(),
            }

    # General document detection
    for doc_id, doc_info in KENYA_DOCUMENT_PATTERNS.items():
        if doc_id == "university_cert":
            continue

        required = doc_info.get("required_elements", [])
        if not required:
            continue

        matches = sum(1 for elem in required if elem in text_lower)
        threshold = max(1, len(required) * 0.6)

        if matches >= threshold:
            confidence = matches / len(required)

            # Also check if specific pattern matches (e.g. KRA PIN format)
            pattern = doc_info.get("pattern")
            if pattern and re.search(pattern, text):
                confidence = min(1.0, confidence + 0.2)

            return {
                "document_type": doc_id,
                "document_name": doc_info["name"],
                "description": doc_info["description"],
                "confidence": round(confidence, 2),
            }

    return None


def analyze_kenya_document(
    text: str,
    image_risk_score: float,
    ela_score: float,
) -> Dict[str, Any]:
    """
    Analyse a suspected Kenyan document for forgery indicators.

    Combines:
    - OCR text pattern matching
    - ELA from image analysis
    - AI deepfake score
    """
    doc_type = detect_document_type(text)

    if not doc_type:
        return {"is_document": False}

    warnings: List[str] = []
    risk_boost = 0
    doc_id = doc_type["document_type"]
    doc_info = KENYA_DOCUMENT_PATTERNS[doc_id]

    # M-Pesa transaction validation
    if doc_id == "mpesa_statement":
        pattern = doc_info.get("pattern", "")
        if pattern and not re.search(pattern, text or ""):
            warnings.append("⚠️ No valid M-Pesa transaction ID found")
            risk_boost += 15

        # Round amounts are suspicious in M-Pesa screenshots
        amounts = re.findall(r"(?:KES|Ksh)\s*([\d,]+)", text or "", re.IGNORECASE)
        for amt in amounts:
            try:
                clean_amt = int(amt.replace(",", ""))
                if clean_amt % 1000 == 0 and clean_amt > 5000:
                    warnings.append(f"⚠️ Suspiciously round amount: KES {amt}")
                    risk_boost += 5
            except ValueError:
                pass

    # KRA PIN format validation
    if doc_id == "kra_pin":
        pattern = doc_info.get("pattern", "")
        pins_found = re.findall(pattern, text or "") if pattern else []
        if not pins_found:
            warnings.append("⚠️ No valid KRA PIN format found in document")
            risk_boost += 20

    # High ELA on official documents = likely edited
    if ela_score > 40:
        warnings.append(
            f"⚠️ High editing indicators (ELA: {ela_score:.0f}%) — "
            f"document may be tampered"
        )
        risk_boost += 15

    # Combine scores
    final_score = min(100, image_risk_score + risk_boost)

    verdict = (
        "LIKELY_FORGED"
        if final_score > 60
        else "SUSPICIOUS"
        if final_score > 35
        else "APPEARS_GENUINE"
    )

    return {
        "is_document": True,
        "document_type": doc_type["document_type"],
        "document_name": doc_type["document_name"],
        "description": doc_type["description"],
        "confidence": doc_type.get("confidence", 0.5),
        "risk_score": round(final_score, 1),
        "warnings": warnings,
        "verdict": verdict,
        "kenya_context": {
            "message_en": (
                f"This appears to be a {doc_type['document_name']}. "
                f"Document forgery is a criminal offence under the "
                f"Kenya Penal Code, Section 349."
            ),
            "message_sw": (
                f"Hii inaonekana kuwa {doc_type['document_name']}. "
                f"Kughushi hati ni kosa la jinai chini ya Sheria ya "
                f"Adhabu ya Kenya, Kifungu 349."
            ),
            "verify_at": doc_info.get("verify_url", "Contact the issuing authority"),
            "report_to": "DCI: reportcrime@dci.go.ke | Hotline: 0800 722 203",
        },
    }
