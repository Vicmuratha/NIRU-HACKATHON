"""
Fake News Screenshot Detector

One of Kenya's most common misinformation vectors:
Someone takes a screenshot of Citizen TV / NTV / Nation breaking news banner,
edits the text using basic photo editing, and shares on WhatsApp.

This module:
1. Detects if an image looks like a Kenyan news screenshot
2. Runs ELA to detect pixel-level edits on text areas
3. Provides verification links to the actual news outlet

Real examples:
- Doctored Citizen TV "BREAKING NEWS" banners with fabricated headlines
- Edited NTV lower-third graphics with false political statements
- Fake Daily Nation article screenshots with altered text
"""

import re
import io
import numpy as np
from PIL import Image, ImageChops
from typing import Optional, Dict, Any

KENYAN_NEWS_BANNERS: Dict[str, Dict[str, Any]] = {
    "citizen_tv": {
        "identifiers": ["citizen tv", "citizen digital", "royal media"],
        "verify_url": "https://www.citizen.digital/",
        "description": "Citizen TV / Royal Media Services",
    },
    "ntv_kenya": {
        "identifiers": ["ntv", "ntv kenya", "nation media group"],
        "verify_url": "https://ntvkenya.co.ke/",
        "description": "NTV Kenya / Nation Media Group",
    },
    "ktn": {
        "identifiers": ["ktn news", "ktn home", "standard group"],
        "verify_url": "https://www.standardmedia.co.ke/",
        "description": "KTN News / Standard Group",
    },
    "daily_nation": {
        "identifiers": ["daily nation", "nation.africa", "nation media"],
        "verify_url": "https://nation.africa/kenya",
        "description": "Daily Nation / NMG",
    },
    "the_star": {
        "identifiers": ["the star", "the-star.co.ke"],
        "verify_url": "https://www.the-star.co.ke/",
        "description": "The Star Kenya",
    },
    "tv47": {
        "identifiers": ["tv47", "tv 47"],
        "verify_url": "https://tv47.digital/",
        "description": "TV47",
    },
    "k24": {
        "identifiers": ["k24 tv", "k24tv"],
        "verify_url": "https://www.k24tv.co.ke/",
        "description": "K24 TV",
    },
}


def run_ela(image_path: str, quality: int = 90) -> Dict[str, Any]:
    """
    Error Level Analysis — catches edited screenshots.

    How it works:
    1. Re-save the image at a known JPEG quality
    2. Compare pixel differences between original and re-saved
    3. Edited regions show HIGHER error levels
    4. For news screenshots: the edited TEXT will have different error
       than the original banner/background

    Returns:
        dict with ela_score, region analysis, and interpretation
    """
    try:
        img = Image.open(image_path).convert("RGB")

        # Re-save at known quality
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        resaved = Image.open(buffer).convert("RGB")

        # Compute difference
        diff = ImageChops.difference(img, resaved)
        diff_array = np.array(diff).astype(np.float32)

        # Split into grid and find high-error regions
        h, w = diff_array.shape[:2]
        grid_size = 8
        grid_h = max(h // grid_size, 1)
        grid_w = max(w // grid_size, 1)
        grid_scores = []

        for i in range(grid_size):
            for j in range(grid_size):
                region = diff_array[
                    i * grid_h : min((i + 1) * grid_h, h),
                    j * grid_w : min((j + 1) * grid_w, w),
                ]
                grid_scores.append(float(np.mean(region)))

        mean_error = float(np.mean(grid_scores))
        max_region_error = float(max(grid_scores)) if grid_scores else 0
        variance = float(np.var(grid_scores))

        # High variance = some parts edited, others not
        manipulation_indicator = min(100, variance * 2 + (max_region_error - mean_error) * 3)

        return {
            "ela_score": round(manipulation_indicator, 1),
            "mean_error": round(mean_error, 2),
            "max_region_error": round(max_region_error, 2),
            "region_variance": round(variance, 2),
            "interpretation": (
                "HIGH editing detected — specific regions show significantly more "
                "compression artifacts, indicating selective modification"
                if manipulation_indicator > 50
                else "MODERATE indicators — some inconsistency detected"
                if manipulation_indicator > 25
                else "LOW indicators — image appears uniformly compressed"
            ),
        }
    except Exception as e:
        return {
            "ela_score": 0,
            "mean_error": 0,
            "max_region_error": 0,
            "region_variance": 0,
            "interpretation": f"ELA failed: {str(e)}",
        }


def detect_news_screenshot(
    ocr_text: str,
    ela_score: float,
    ai_deepfake_score: float,
) -> Dict[str, Any]:
    """
    Determine if an image is a manipulated Kenyan news screenshot.

    Args:
        ocr_text: Text extracted via OCR from the image
        ela_score: Error Level Analysis score (0-100)
        ai_deepfake_score: AI model deepfake confidence score
    """
    text_lower = (ocr_text or "").lower()

    # Detect which outlet
    detected_outlet = None
    for outlet_id, outlet_info in KENYAN_NEWS_BANNERS.items():
        for identifier in outlet_info["identifiers"]:
            if identifier in text_lower:
                detected_outlet = {
                    "id": outlet_id,
                    "name": outlet_info["description"],
                    "verify_url": outlet_info["verify_url"],
                }
                break
        if detected_outlet:
            break

    if not detected_outlet:
        return {"is_news_screenshot": False}

    # Breaking news indicators
    breaking_indicators = [
        "breaking", "breaking news", "developing", "just in",
        "alert", "update", "habari", "taarifa",
        "exclusive", "confirmed",
    ]
    has_breaking = any(ind in text_lower for ind in breaking_indicators)

    # Calculate manipulation likelihood
    manipulation_score = 0
    warnings = []

    if ela_score > 35:
        manipulation_score += 30
        warnings.append(
            f"High editing detected (ELA: {ela_score:.0f}%) — "
            f"text on this screenshot may have been altered"
        )

    if ai_deepfake_score > 40:
        manipulation_score += 25
        warnings.append(
            f"AI detection flags manipulation ({ai_deepfake_score:.0f}%)"
        )

    if has_breaking:
        manipulation_score += 10
        warnings.append(
            "Contains 'breaking news' indicator — the most commonly forged format"
        )

    manipulation_score = min(100, manipulation_score)

    return {
        "is_news_screenshot": True,
        "detected_outlet": detected_outlet,
        "has_breaking_banner": has_breaking,
        "manipulation_score": manipulation_score,
        "warnings": warnings,
        "verdict": (
            "LIKELY_MANIPULATED"
            if manipulation_score > 60
            else "SUSPICIOUS"
            if manipulation_score > 30
            else "APPEARS_GENUINE"
        ),
        "action": {
            "en": (
                f"Verify this story directly at {detected_outlet['verify_url']} — "
                f"do not trust screenshots shared on WhatsApp or social media."
            ),
            "sw": (
                f"Thibitisha habari hii moja kwa moja kwa {detected_outlet['verify_url']} — "
                f"usiamini picha za skrini zinazoshirikiwa kwenye WhatsApp au mitandao ya kijamii."
            ),
        },
    }
