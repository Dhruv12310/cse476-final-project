#!/usr/bin/env python3
"""
Generate a placeholder answer file that matches the expected auto-grader format.

Replace the placeholder logic inside `build_answers()` with your own agent loop
before submitting so the ``output`` fields contain your real predictions.

Reads the input questions from cse_476_final_project_test_data.json and writes
an answers JSON file where each entry contains a string under the "output" key.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# API config – ONLY the class endpoint
API_DB_KEY = os.getenv("OPENAI_API_KEY", "cse476")
API_DB_BASE = os.getenv("API_BASE", "http://10.4.58.53:41701/v1")
MODEL_DB_NAME = os.getenv("MODEL_NAME", "bens_model")

# Paths for input / output
INPUT_DB_PATH = Path("cse_476_final_project_test_data.json")
OUTPUT_DB_PATH = Path("cse_476_final_project_answers.json")
# Debug: if you want to test on a subset first, set DEBUG_DB_N to a small int.
# For the real submission, set this to None.
DEBUG_DB_N: Optional[int] = None  # e.g., 20 for quick tests, None for full run

# Safety limits
MAX_OUTPUT_DB_CHARS = 4900  # keep well under 5000 to be safe
REQUEST_DB_TIMEOUT_SECONDS = 60  # per-request timeout
MAX_DB_RETRIES = 3  # number of retries per question on network/API errors
RETRY_DB_SLEEP_SECONDS = 3  # pause between retries


def call_model_chat_completions(
    prompt: str,
    system: str = (
        "You are a careful problem solver.\n"
        "ALWAYS follow all explicit instructions in the user message.\n"
        "When no explicit format is given, reply ONLY with the final answer "
        "in the most concise form possible. Do NOT show your reasoning."
    ),
    model: str = MODEL_DB_NAME,
    temperature: float = 0.0,
    timeout: int = REQUEST_DB_TIMEOUT_SECONDS,
    max_tokens: int = 128,
) -> Dict[str, Any]:
    """
    Thin wrapper around the class-provided /chat/completions endpoint.

    Returns a dict with fields:
        ok: bool
        text: str | None
        status: int
        error: str | None
        raw: full JSON response or None
        headers: dict
    """
    url_db = API_DB_BASE.rstrip("/") + "/chat/completions"
    headers_db = {
        "Authorization": f"Bearer {API_DB_KEY}",
        "Content-Type": "application/json",
    }
    payload_db: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        resp_db = requests.post(
            url_db, headers=headers_db, json=payload_db, timeout=timeout
        )
        status_db = resp_db.status_code
        hdrs_db = dict(resp_db.headers)
        if status_db == 200:
            data_db = resp_db.json()
            text_db = (
                data_db.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            return {
                "ok": True,
                "text": text_db,
                "raw": data_db,
                "status": status_db,
                "error": None,
                "headers": hdrs_db,
            }
        else:
            # Try to surface error text for debugging
            try:
                err_text_db = resp_db.json()
            except Exception:
                err_text_db = resp_db.text
            return {
                "ok": False,
                "text": None,
                "raw": None,
                "status": status_db,
                "error": str(err_text_db),
                "headers": hdrs_db,
            }
    except requests.RequestException as e_db:
        return {
            "ok": False,
            "text": None,
            "raw": None,
            "status": -1,
            "error": str(e_db),
            "headers": {},
        }


def classify_question(question: str) -> str:
    """
    VERY light heuristic classification based only on the text of the question.

    This is used to tweak the prompting style; it is not used directly for grading.
    """
    q_lower_db = question.lower()

    coding_keywords_db = [
        "python",
        "java",
        "c++",
        "c#",
        "function",
        "class ",
        "implement",
        "write code",
        "pseudocode",
        "pseudo-code",
        "time complexity",
        "runtime complexity",
        "big-o",
        "big o",
        "algorithm",
    ]
    if any(k_db in q_lower_db for k_db in coding_keywords_db):
        return "coding"

    math_keywords_db = [
        "integer",
        "prime",
        "factor",
        "divisible",
        "mod ",
        "equation",
        "solve for",
        "roots of",
        "triangle",
        "quadrilateral",
        "probability",
        "area",
        "perimeter",
        "sequence",
        "series",
        "sum",
        "product",
        "remainder",
        "inequality",
    ]
    if any(k_db in q_lower_db for k_db in math_keywords_db):
        return "math"

    # Future prediction / forecasting style
    if "you are an agent that can predict future events" in q_lower_db:
        return "future_prediction"

    # STRIPS-like planning problems
    if "(unstack " in q_lower_db or "(stack " in q_lower_db or "(pick-up " in q_lower_db:
        return "planning"

    return "other"


def question_has_strict_format_instructions(question: str) -> bool:
    """
    Detects if the question already specifies an explicit answer format.
    In that case, we do NOT add our own formatting instructions.
    """
    q_lower_db = question.lower()
    format_phrases_db = [
        "answer with just the integer",
        "answer with just the number",
        "answer with exactly one of",
        "your final answer must be",
        "answer with a single word",
        "return a python",
        "output should be",
        "output must be",
        "answer using the following format",
    ]
    return any(p_db in q_lower_db for p_db in format_phrases_db)


def build_user_prompt(question: str) -> str:
    """
    Build the user message we send to the model.

    We pass the question text and (when appropriate) append a short,
    domain-specific instruction about output style.
    """
    if question_has_strict_format_instructions(question):
        # Question already defines the output format clearly; trust it.
        return question

    q_type_db = classify_question(question)

    if q_type_db == "math":
        extra_db = (
            "\n\nIMPORTANT: Respond with ONLY the final numeric answer "
            "(just the number or simplest algebraic expression). "
            "Do NOT show your work or add any words."
        )
    elif q_type_db == "coding":
        extra_db = (
            "\n\nIMPORTANT: Provide ONLY the final code or final textual "
            "answer requested by the problem. Do NOT explain the code."
        )
    elif q_type_db in {"planning", "future_prediction"}:
        extra_db = (
            "\n\nIMPORTANT: Follow any output format implied by the problem "
            "exactly, and do NOT add any extra commentary."
        )
    else:
        extra_db = (
            "\n\nIMPORTANT: Respond with ONLY the final answer as a short "
            "phrase or sentence, with no explanation."
        )

    return question + extra_db


# ---------------------------------------------------------------------
# Single-call agent
# ---------------------------------------------------------------------
def agent_answer(question_text: str) -> str:
    """
    Main agent entry point: given the raw question text, return a single
    final answer string (no chain-of-thought, no intermediate results).

    Uses:
      * Domain-aware prompting
      * Output-format enforcement
      * Robust error handling + retries
    """
    user_prompt_db = build_user_prompt(question_text)

    last_error_db: Optional[str] = None
    for attempt_db in range(1, MAX_DB_RETRIES + 1):
        resp_db = call_model_chat_completions(user_prompt_db)
        if resp_db.get("ok"):
            text_db = resp_db.get("text") or ""
            answer_db = text_db.strip()
            if not answer_db:
                # Empty answer – treat as error and retry
                last_error_db = "empty_response"
            else:
                # Enforce length and type guarantees
                if len(answer_db) > MAX_OUTPUT_DB_CHARS:
                    answer_db = answer_db[:MAX_OUTPUT_DB_CHARS]
                return answer_db
        else:
            last_error_db = resp_db.get("error") or "unknown_error"

        # If we reach here, something went wrong; wait a bit then retry
        if attempt_db < MAX_DB_RETRIES:
            time.sleep(RETRY_DB_SLEEP_SECONDS)

    # If all retries failed, fall back to a short error marker instead of crashing
    fallback_db = "ERROR: model_call_failed"
    if last_error_db:
        # Keep it short just in case
        fallback_db = f"ERROR: {last_error_db}"[:MAX_OUTPUT_DB_CHARS]
    return fallback_db


# ---------------------------------------------------------------------
# IO helpers and validation
# ---------------------------------------------------------------------
def load_questions(path: Path, debug_n: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load the questions JSON list, optionally truncating to first debug_n."""
    with path.open("r", encoding="utf-8") as fp_db:
        data_db = json.load(fp_db)
    if not isinstance(data_db, list):
        raise ValueError("Input file must contain a list of question objects.")

    if debug_n is not None:
        return data_db[:debug_n]
    return data_db


def build_answers(questions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Core loop the autograder cares about.

    For each question object (with an 'input' field), call our agent and
    store the result under 'output'.

    This version also prints simple progress information while running.
    """
    answers_db: List[Dict[str, str]] = []
    total_db = len(questions)

    for idx_db, question_db in enumerate(questions, start=1):
        # Progress print every 20 questions (tweak if you want more/less spam)
        if idx_db == 1 or idx_db % 20 == 0 or idx_db == total_db:
            print(f"Processing question {idx_db}/{total_db}...")

        q_text_db = question_db.get("input", "")
        if not isinstance(q_text_db, str):
            q_text_db = str(q_text_db)

        answer_str_db = agent_answer(q_text_db)

        # Final safety checks
        if not isinstance(answer_str_db, str):
            answer_str_db = str(answer_str_db)
        if len(answer_str_db) > MAX_OUTPUT_DB_CHARS:
            answer_str_db = answer_str_db[:MAX_OUTPUT_DB_CHARS]

        answers_db.append({"output": answer_str_db})

    return answers_db


def validate_results(
    questions: List[Dict[str, Any]], answers: List[Dict[str, Any]]
) -> None:
    """Validate that outputs match the expected format for the grader."""
    if len(questions) != len(answers):
        raise ValueError(
            f"Mismatched lengths: {len(questions)} questions vs "
            f"{len(answers)} answers."
        )
    for idx_db, answer_db in enumerate(answers):
        if "output" not in answer_db:
            raise ValueError(f"Missing 'output' field for answer index {idx_db}.")
        if not isinstance(answer_db["output"], str):
            raise TypeError(
                f"Answer at index {idx_db} has non-string output: "
                f"{type(answer_db['output'])}"
            )
        if len(answer_db["output"]) >= 5000:
            raise ValueError(
                f"Answer at index {idx_db} exceeds 5000 characters "
                f"({len(answer_db['output'])} chars). Please make sure your "
                f"answer does not include intermediate reasoning."
            )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    questions_db = load_questions(INPUT_DB_PATH, debug_n=DEBUG_DB_N)
    print(f"Loaded {len(questions_db)} question(s) from {INPUT_DB_PATH}")

    answers_db = build_answers(questions_db)

    # Write answers
    with OUTPUT_DB_PATH.open("w", encoding="utf-8") as fp_db:
        json.dump(answers_db, fp_db, ensure_ascii=False, indent=2)

    # Re-load and validate
    with OUTPUT_DB_PATH.open("r", encoding="utf-8") as fp_db:
        saved_answers_db = json.load(fp_db)

    validate_results(questions_db, saved_answers_db)
    print(
        f"Wrote {len(answers_db)} answers to {OUTPUT_DB_PATH} "
        f"and validated format successfully."
    )


if __name__ == "__main__":
    main()
