import re
from typing import Dict, List


MONEY_PATTERNS = [
    r"\$\d+",
    r"\d+\s?(usd|dollars?)",
    r"₹\s?\d+",
    r"\b(inr|rs)\b\.?\s?\d+",
    r"wire transfer",
    r"bank account",
    r"\bupi\b",
    r"processing fee",
]

URGENT_PATTERNS = [
    r"urgent",
    r"immediately",
    r"asap",
    r"within 24 hours",
    r"last chance",
]

SUSPICIOUS_KEYWORDS = [
    "lottery",
    "jackpot",
    "winner",
    "confidential",
    "western union",
    "moneygram",
    "investment",
    "crypto",
    "bitcoin",
]

RECRUITMENT_SUSPICIOUS = [
    "processing fee",
    "training fee",
    "visa fee",
    "sponsorship fee",
    "upfront payment",
    "security deposit",
    "refundable security deposit",
    "refundable deposit",
    "refundable",
    "work from home and earn",
]

TRUSTED_DOMAINS = [
    "microsoft.com",
    "google.com",
    "amazon.com",
    "linkedin.com",
]

FREE_EMAIL_DOMAINS = [
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "hotmail.com",
    "icloud.com",
    "aol.com",
    "proton.me",
    "protonmail.com",
]


def extract_suspicious_patterns(email_text: str, sender_email: str = "") -> Dict:
    lower_text = email_text.lower()
    reasons: List[str] = []
    keywords_highlighted: List[str] = []

    money_flag = False
    urgent_flag = False
    recruitment_flag = False
    free_email_flag = False
    domain_flagged = False

    for pattern in MONEY_PATTERNS:
        if re.search(pattern, lower_text):
            reasons.append("Money request or financial transfer language detected.")
            keywords_highlighted.append(pattern)
            money_flag = True
            break

    for pattern in URGENT_PATTERNS:
        if re.search(pattern, lower_text):
            reasons.append("Urgent or time-pressure phrasing detected.")
            keywords_highlighted.append(pattern)
            urgent_flag = True
            break

    for kw in SUSPICIOUS_KEYWORDS:
        if kw in lower_text:
            reasons.append("Suspicious high-risk keyword related to scams detected.")
            keywords_highlighted.append(kw)
            break

    for kw in RECRUITMENT_SUSPICIOUS:
        if kw in lower_text:
            reasons.append("Suspicious recruitment-related payment or fee language.")
            keywords_highlighted.append(kw)
            recruitment_flag = True
            break

    # Look for free email domains mentioned in text (common in scams)
    mentioned_emails = re.findall(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}", lower_text)
    for em in mentioned_emails:
        domain = em.split("@")[-1]
        if domain in FREE_EMAIL_DOMAINS:
            reasons.append("Free/public email domain used for payment or HR contact.")
            keywords_highlighted.append(domain)
            free_email_flag = True
            break

    if sender_email and "@" in sender_email:
        domain = sender_email.split("@")[-1].lower()
        if not any(domain.endswith(td) for td in TRUSTED_DOMAINS):
            domain_flagged = True
            reasons.append("Unverified or non-corporate sender domain.")

    if not reasons:
        reasons.append("No strongly suspicious linguistic patterns detected.")

    risk_score_bonus = 0.0
    if len(keywords_highlighted) >= 1:
        risk_score_bonus += 0.05
    if len(keywords_highlighted) >= 2:
        risk_score_bonus += 0.1
    if domain_flagged:
        risk_score_bonus += 0.1

    # High-risk heuristic: strong scam pattern like money + urgency + fees/free email
    high_risk = False
    if money_flag and (urgent_flag or recruitment_flag or free_email_flag):
        high_risk = True
    if recruitment_flag and free_email_flag:
        high_risk = True

    return {
        "reasons": list(dict.fromkeys(reasons)),
        "suspicious_keywords": list(dict.fromkeys(keywords_highlighted)),
        "risk_score_bonus": min(risk_score_bonus, 0.35),
        "high_risk": high_risk,
    }

