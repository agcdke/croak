"""
Guardrail rules — patterns, keywords, and limits used by the guard functions.

Centralising all rules here makes it easy to:
  - Add / remove patterns without touching logic code
  - Tune thresholds per environment via .env overrides
  - Audit exactly what is being blocked
"""
import re
from dataclasses import dataclass, field
from typing import List, Pattern


# ══════════════════════════════════════════════════════════════════════════════
# INPUT LIMITS
# ══════════════════════════════════════════════════════════════════════════════

MIN_QUESTION_LENGTH = 3        # chars — reject single-char or empty inputs
MAX_QUESTION_LENGTH = 2000     # chars — reject absurdly long prompts


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT INJECTION PATTERNS
# ══════════════════════════════════════════════════════════════════════════════
# These phrases attempt to override the LLM's system prompt or persona.

INJECTION_PATTERNS: List[Pattern] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)", re.I),
    re.compile(r"forget\s+(all\s+)?(previous|prior|above|your)\s+(instructions?|prompts?|context|training)", re.I),
    re.compile(r"you\s+are\s+now\s+(a\s+)?(?!an?\s+assistant)", re.I),  # "you are now a hacker"
    re.compile(r"act\s+as\s+(if\s+you\s+(were?|are)\s+)?(a\s+)?(?!an?\s+assistant)", re.I),
    re.compile(r"pretend\s+(you\s+are|to\s+be)", re.I),
    re.compile(r"override\s+(your\s+)?(system|safety|guidelines?|rules?|restrictions?)", re.I),
    re.compile(r"jailbreak", re.I),
    re.compile(r"DAN\s+mode", re.I),       # "Do Anything Now" jailbreak
    re.compile(r"developer\s+mode", re.I),
    re.compile(r"bypass\s+(your\s+)?(filters?|safety|restrictions?|guidelines?)", re.I),
    re.compile(r"system\s*prompt\s*[:=]", re.I),
    re.compile(r"<\s*system\s*>", re.I),   # XML injection attempt
    re.compile(r"\[\s*system\s*\]", re.I), # bracket injection attempt
]


# ══════════════════════════════════════════════════════════════════════════════
# HARMFUL CONTENT KEYWORDS
# ══════════════════════════════════════════════════════════════════════════════
# Block questions asking for dangerous real-world information.

HARMFUL_KEYWORDS: List[str] = [
    # Weapons & violence
    "how to make a bomb",
    "how to build a weapon",
    "how to synthesize drugs",
    "how to make explosives",
    "instructions to kill",
    "how to hack",
    "how to crack passwords",
    "how to ddos",
    "ransomware",
    "malware code",
    "exploit code",
    # Self-harm
    "how to hurt myself",
    "how to commit suicide",
    "methods of self-harm",
]

# Compile for fast matching
HARMFUL_PATTERNS: List[Pattern] = [
    re.compile(re.escape(kw), re.I) for kw in HARMFUL_KEYWORDS
]


# ══════════════════════════════════════════════════════════════════════════════
# OFF-TOPIC DETECTION
# ══════════════════════════════════════════════════════════════════════════════
# If a question matches NONE of the on-topic signals AND matches an off-topic
# pattern, it is flagged as out of scope for a document QA assistant.

OFF_TOPIC_PATTERNS: List[Pattern] = [
    re.compile(r"\b(weather|forecast|temperature|rain|sunny)\b", re.I),
    re.compile(r"\b(stock\s+price|cryptocurrency|bitcoin|ethereum|invest)\b", re.I),
    re.compile(r"\b(sports?\s+score|soccer|football|basketball|baseball)\b", re.I),
    re.compile(r"\b(recipe|cook|bake|ingredient|tablespoon)\b", re.I),
    re.compile(r"\b(movie|film|actor|actress|celebrity|TV\s+show)\b", re.I),
    re.compile(r"\b(joke|tell\s+me\s+a\s+story|write\s+a\s+poem)\b", re.I),
    re.compile(r"\b(translate\s+this\s+to|what\s+language\s+is)\b", re.I),
    re.compile(r"\b(my\s+horoscope|astrology|zodiac)\b", re.I),
]

# These phrases indicate a legitimate document question — short-circuit off-topic check
ON_TOPIC_SIGNALS: List[str] = [
    "what", "who", "when", "where", "why", "how", "list", "explain",
    "describe", "define", "summarize", "find", "show", "tell me about",
    "according to", "based on", "in the document", "in the file",
    "does the", "is there", "are there", "can you",
]


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT SAFETY
# ══════════════════════════════════════════════════════════════════════════════

# If the LLM answer contains these phrases it means it has no grounded answer
# and we should return a structured "not found" response instead of guessing.
HALLUCINATION_SIGNALS: List[str] = [
    "i don't have information",
    "i don't have access",
    "i cannot find",
    "not mentioned in",
    "not provided in",
    "no information available",
    "outside the scope",
    "i'm not sure",
    "i am not sure",
    "based on my training",        # LLM using outside knowledge
    "as of my knowledge cutoff",
    "according to my knowledge",
    "in general,",                 # general answer not grounded in docs
]

# Answer is suspiciously short — may be a non-answer
MIN_ANSWER_LENGTH = 10   # chars

# Phrases that indicate a harmful output slipped through
OUTPUT_HARMFUL_PATTERNS: List[Pattern] = [
    re.compile(r"\b(step\s+\d+.*?(bomb|weapon|explosive|malware))\b", re.I),
    re.compile(r"\b(sudo\s+rm\s+-rf|format\s+c:)\b", re.I),
]
