"""
Guard functions for the PDF & Turtle RAG Chatbot.

Each guard returns a GuardResult dataclass:
    passed  : bool   — True = safe to proceed, False = blocked
    reason  : str    — human-readable explanation (shown to user on block)
    code    : str    — machine-readable code for logging / metrics

INPUT GUARDS  (run before the RAG chain):
    check_input_length(question)      — too short / too long
    check_prompt_injection(question)  — injection / jailbreak attempts
    check_harmful_input(question)     — dangerous real-world requests
    check_off_topic(question)         — clearly out-of-scope questions
    validate_input(question)          — runs all input guards in order

OUTPUT GUARDS (run after the RAG chain returns):
    check_empty_answer(answer)        — blank or trivially short answer
    check_hallucination(answer)       — LLM signals it has no grounded answer
    check_harmful_output(answer)      — dangerous content in the response
    validate_output(answer)           — runs all output guards in order
"""
from dataclasses import dataclass
from typing import Optional
from loguru import logger

from src.guardrails.rules import (
    MIN_QUESTION_LENGTH,
    MAX_QUESTION_LENGTH,
    INJECTION_PATTERNS,
    HARMFUL_PATTERNS,
    OFF_TOPIC_PATTERNS,
    ON_TOPIC_SIGNALS,
    HALLUCINATION_SIGNALS,
    MIN_ANSWER_LENGTH,
    OUTPUT_HARMFUL_PATTERNS,
)


# ══════════════════════════════════════════════════════════════════════════════
# RESULT TYPE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GuardResult:
    """
    Returned by every guard function.

    Attributes:
        passed  : True = safe, False = blocked
        reason  : User-facing explanation when blocked
        code    : Snake_case code for logging, metrics, or API response field
        detail  : Optional extra info (matched pattern, etc.) for internal logs
    """
    passed: bool
    reason: str = ""
    code:   str = "ok"
    detail: str = ""

    def __bool__(self) -> bool:
        return self.passed


SAFE = GuardResult(passed=True, reason="", code="ok")


# ══════════════════════════════════════════════════════════════════════════════
# INPUT GUARDS
# ══════════════════════════════════════════════════════════════════════════════

def check_input_length(question: str) -> GuardResult:
    """
    Reject questions that are too short (likely noise) or too long
    (potential prompt-stuffing attack).
    """
    q = question.strip()

    if len(q) < MIN_QUESTION_LENGTH:
        logger.warning(f"[GUARDRAIL] Input too short: {len(q)} chars")
        return GuardResult(
            passed=False,
            reason="Your question is too short. Please provide more detail.",
            code="input_too_short",
            detail=f"length={len(q)} min={MIN_QUESTION_LENGTH}",
        )

    if len(q) > MAX_QUESTION_LENGTH:
        logger.warning(f"[GUARDRAIL] Input too long: {len(q)} chars")
        return GuardResult(
            passed=False,
            reason=(
                f"Your question is too long ({len(q)} characters). "
                f"Please keep it under {MAX_QUESTION_LENGTH} characters."
            ),
            code="input_too_long",
            detail=f"length={len(q)} max={MAX_QUESTION_LENGTH}",
        )

    return SAFE


def check_prompt_injection(question: str) -> GuardResult:
    """
    Detect prompt injection / jailbreak attempts that try to override
    the system prompt or change the assistant's behaviour.
    """
    for pattern in INJECTION_PATTERNS:
        match = pattern.search(question)
        if match:
            logger.warning(
                f"[GUARDRAIL] Prompt injection detected: '{match.group()}'"
            )
            return GuardResult(
                passed=False,
                reason=(
                    "Your message appears to contain instructions that attempt "
                    "to override the assistant's behaviour. "
                    "Please ask a genuine question about your documents."
                ),
                code="prompt_injection",
                detail=f"matched='{match.group()}'",
            )
    return SAFE


def check_harmful_input(question: str) -> GuardResult:
    """
    Block questions that ask for harmful real-world information
    (weapons, drugs, hacking, self-harm, etc.).
    """
    for pattern in HARMFUL_PATTERNS:
        match = pattern.search(question)
        if match:
            logger.warning(
                f"[GUARDRAIL] Harmful input detected: '{match.group()}'"
            )
            return GuardResult(
                passed=False,
                reason=(
                    "I'm not able to answer questions that involve harmful, "
                    "dangerous, or illegal content. "
                    "Please ask about your uploaded documents."
                ),
                code="harmful_input",
                detail=f"matched='{match.group()}'",
            )
    return SAFE


def check_off_topic(question: str) -> GuardResult:
    """
    Detect questions that are clearly unrelated to document analysis.

    Logic:
      1. If any on-topic signal is present → pass (benefit of the doubt).
      2. If an off-topic pattern matches → block.
      3. Otherwise → pass.
    """
    q_lower = question.lower()

    # Fast-pass: question contains a typical document-QA phrase
    if any(signal in q_lower for signal in ON_TOPIC_SIGNALS):
        return SAFE

    for pattern in OFF_TOPIC_PATTERNS:
        match = pattern.search(question)
        if match:
            logger.warning(
                f"[GUARDRAIL] Off-topic question detected: '{match.group()}'"
            )
            return GuardResult(
                passed=False,
                reason=(
                    "That question appears to be outside the scope of this "
                    "document assistant. I can only answer questions about "
                    "your uploaded PDF or Turtle RDF documents."
                ),
                code="off_topic",
                detail=f"matched='{match.group()}'",
            )

    return SAFE


def validate_input(question: str) -> GuardResult:
    """
    Run all input guards in priority order.
    Returns the first failed GuardResult, or SAFE if all pass.

    Order:
      1. Length check     — cheapest, no regex
      2. Injection check  — security-critical, always run early
      3. Harmful check    — safety-critical
      4. Off-topic check  — UX quality, run last
    """
    for guard_fn in [
        check_input_length,
        check_prompt_injection,
        check_harmful_input,
        check_off_topic,
    ]:
        result = guard_fn(question)
        if not result.passed:
            return result
    return SAFE


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT GUARDS
# ══════════════════════════════════════════════════════════════════════════════

def check_empty_answer(answer: str) -> GuardResult:
    """
    Catch blank or trivially short answers before returning them to the user.
    """
    if not answer or len(answer.strip()) < MIN_ANSWER_LENGTH:
        logger.warning(
            f"[GUARDRAIL] Answer too short: '{answer[:50]}'"
        )
        return GuardResult(
            passed=False,
            reason=(
                "I was unable to generate a meaningful answer from the "
                "available documents. Please try rephrasing your question "
                "or uploading more relevant files."
            ),
            code="empty_answer",
            detail=f"answer_length={len(answer.strip())}",
        )
    return SAFE


def check_hallucination(answer: str) -> GuardResult:
    """
    Detect when the LLM signals it has no grounded answer and is guessing
    from its training data instead of the retrieved context.

    These phrases mean the RAG retrieval did not return useful chunks and
    the model is falling back to general knowledge — which we want to block
    in a strict document-QA setting.
    """
    a_lower = answer.lower()
    for signal in HALLUCINATION_SIGNALS:
        if signal in a_lower:
            logger.warning(
                f"[GUARDRAIL] Hallucination signal detected: '{signal}'"
            )
            return GuardResult(
                passed=False,
                reason=(
                    "I could not find a confident answer in your uploaded documents. "
                    "The information may not be present in the indexed files. "
                    "Try uploading more relevant documents or rephrasing your question."
                ),
                code="hallucination_detected",
                detail=f"signal='{signal}'",
            )
    return SAFE


def check_harmful_output(answer: str) -> GuardResult:
    """
    Last-line-of-defence: catch harmful content that may have slipped
    through input checks and been generated in the LLM output.
    """
    for pattern in OUTPUT_HARMFUL_PATTERNS:
        match = pattern.search(answer)
        if match:
            logger.error(
                f"[GUARDRAIL] Harmful output detected and blocked: '{match.group()}'"
            )
            return GuardResult(
                passed=False,
                reason=(
                    "The generated response contained content that cannot be "
                    "displayed. Please rephrase your question."
                ),
                code="harmful_output",
                detail=f"matched='{match.group()}'",
            )
    return SAFE


def validate_output(answer: str) -> GuardResult:
    """
    Run all output guards in priority order.
    Returns the first failed GuardResult, or SAFE if all pass.

    Order:
      1. Empty check       — cheapest
      2. Hallucination     — quality gate
      3. Harmful output    — safety last-resort
    """
    for guard_fn in [
        check_empty_answer,
        check_hallucination,
        check_harmful_output,
    ]:
        result = guard_fn(answer)
        if not result.passed:
            return result
    return SAFE
