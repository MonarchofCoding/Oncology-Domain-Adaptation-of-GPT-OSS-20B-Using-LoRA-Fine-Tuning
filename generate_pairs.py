"""
Mistral-only oncology QA dataset generator.

Workflow:
    1. Put website text in source.txt (keep headings, lists, and blank lines).
    2. Run: python generate_pairs.py
    3. The final generated_pairs.jsonl is replaced only after a full successful run.

Features:
- Structure-aware chunking with sentence/list/word fallbacks.
- Context-only carryover between chunks (context is not itself eligible for extraction).
- Bounded exponential-backoff retries for transient API failures.
- Automatic recursive splitting after truncated/invalid model output.
- Resume support using source/config hashes and per-chunk metadata.
- Atomic writes, raw responses, per-chunk JSONL, count history, and backups.
- Exact-pair deduplication only; no semantic deduplication.
"""

import csv
import hashlib
import json
import os
import random
import re
import shutil
import time
from datetime import datetime
from getpass import getpass
from pathlib import Path

from mistralai.client import Mistral


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SOURCE_FILE = BASE_DIR / "source.txt"
RUNS_DIR = BASE_DIR / "generation_runs"
JSONL_OUTPUT_FILE = BASE_DIR / "generated_pairs.jsonl"
PARTIAL_OUTPUT_FILE = BASE_DIR / "generated_pairs.partial.jsonl"
COUNT_LOG_FILE = BASE_DIR / "chunk_counts.csv"

MODEL = "ministral-14b-2512"
PIPELINE_VERSION = "2.0.0"

# Maximum primary source text per API call. Context is sent separately.
MAX_SOURCE_CHARS_PER_CHUNK = 6500
MAX_OUTPUT_TOKENS = 8192

# Retry only transient API failures; never retry forever.
MAX_API_ATTEMPTS = 5
INITIAL_RETRY_DELAY_SECONDS = 2.0
MAX_RETRY_DELAY_SECONDS = 45.0

# If JSON is malformed or output is truncated, retry the same chunk once,
# then automatically split it into smaller child chunks.
MAX_FORMAT_ATTEMPTS = 2
MIN_AUTO_SPLIT_CHARS = 900

# Context helps interpret a chunk's opening, but pairs must be extracted
# only from the current primary text.
MAX_CONTEXT_CHARS = 900

SYSTEM_PROMPT = r"""
You are an expert oncology knowledge extraction system generating a
source-grounded dataset for fine-tuning GPT-OSS-20B.

PROJECT GOAL
Introduce useful oncology knowledge, terminology, concepts, facts,
mechanisms, and relationships so the model performs better on an unseen
oncology knowledge test. This is not intended to make the model a doctor,
medical advisor, diagnostician, or comprehensive medical expert.

Prioritize information coverage, factual fidelity, and usefulness. Do not
discard useful source information merely to keep the output short. There is
no target number of pairs.

SOURCE AUTHORITY
The PRIMARY SOURCE TEXT in the current user message is the only source from
which pairs may be generated. The optional CONTEXT FROM PREVIOUS SOURCE TEXT
is copied verbatim from immediately preceding source material. Use that
context only to understand references or the continuation at the start of
the primary text. Do NOT generate a pair based solely on information that
appears only in the context section. Do not use pretrained knowledge,
other chunks, other files, previous conversations, or outside sources.

ABSOLUTE SOURCE-GROUNDING
Use only information explicitly stated in, or directly and unambiguously
derivable from, the PRIMARY SOURCE TEXT. Do not add outside medical
knowledge, assumptions, unstated mechanisms, examples, conclusions, causal
relationships, or recommendations. Do not correct, update, supplement, or
reinterpret the source using outside information.

SYSTEMATIC EXTRACTION — AVOID LOSING USEFUL INFORMATION
Examine the entire primary text, including its beginning, middle, and end.
Look for all useful source-supported information, including:
- definitions and terminology;
- biological mechanisms and processes;
- genes, genetic changes, biomarkers, and molecular features;
- disease characteristics, cancer types, subtypes, and classifications;
- risk and protective factors;
- prevention, screening, and diagnostic concepts;
- treatment concepts and principles;
- epidemiology, statistics, numbers, percentages, rates, dates, and projections;
- comparisons, distinctions, relationships, examples, and named entities;
- exceptions, limitations, uncertainty, and qualifications;
- outcomes, trends, and other useful source-specific details.

These categories are examples, not a restrictive list. Examine all content,
not just headings or the first few paragraphs.

COVERAGE REQUIREMENTS
1. Preserve meaningful details, examples, named entities, distinctions,
   numbers, dates, populations, units, and qualifications.
2. Do not replace several useful facts with one vague summary.
3. If a passage contains multiple independently useful facts, create
   separate pairs when that preserves information clearly.
4. Combine sentences only when they form one coherent concept and important
   details remain intact.
5. Do not omit useful information just because it seems basic, familiar,
   less prominent, or related to another fact.
6. Do not shorten a useful list by arbitrarily keeping only some entries.
   If a long list needs splitting, preserve all useful entries.
7. Keep statistics tied to relevant years, geography, populations, units,
   denominators, and whether they are estimates or projections.
8. Do not force every sentence into a pair. Ignore navigation text,
   image placeholders, promotional text, and material with no useful
   oncology knowledge.
9. Do not impose a maximum, minimum, or target pair count.
10. Do not discard a supported detail merely because another pair covers
    the same general topic. Avoid exact duplicates or pairs that truly add
    no new information, but prioritize preserving useful details.

PRESERVE FACTUAL MEANING
- Preserve the source's terminology and meaning.
- Preserve qualifiers such as "may", "can", "usually", "generally",
  "approximately", "in some cases", and "in general".
- Do not turn a possibility into a certainty.
- Do not infer causation from association unless the source explicitly
  states causation.
- Do not generalize a claim beyond the stated population or circumstances.
- Preserve meaningful exceptions, limitations, and contrasts.
- Do not turn descriptive information into advice unless the source
  explicitly provides that recommendation.
- Do not combine separate facts into a conclusion the source does not support.
- Do not fill gaps using pretrained or outside knowledge.

PAIR DESIGN
Each pair must contain:
- "input": a clear, concise question or knowledge prompt;
- "output": a direct answer preserving relevant source-supported facts.

Match the question to the information. Use definitions, mechanisms,
classifications, comparisons, statistics, examples, trends, or other
appropriate question forms. Do not make every pair a generic definition.
Each question must contain enough context to be understandable on its own.
Use one pair for one coherent question-answer task. Split passages when
they contain multiple independently useful facts; do not merge unrelated
facts into an unnecessarily long answer or fragment one simple fact into
artificial questions.

FINAL INTERNAL CHECK
Review the entire primary text for useful details that may have been missed.
Check the middle and end as carefully as the start. Ensure every factual
claim is supported by the primary text and numbers, dates, context,
certainty, and qualifications are preserved. If useful supported
information was omitted, add it. This audit should improve coverage, not
encourage unsupported additions.

OUTPUT FORMAT
Return exactly one valid JSON object:
{"pairs":[{"input":"Question","output":"Source-grounded answer"}]}
Every pair must contain exactly the string keys "input" and "output".
If no useful pairs exist, return {"pairs":[]}.
Do not include Markdown fences, headings, citations, counts, or commentary.
Do not generate fill-in-the-blank questions; that conversion happens later.
"""


# ============================================================
# GENERAL HELPERS
# ============================================================

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    """Write to a temporary sibling, then atomically replace the destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temp_path.replace(path)


def atomic_write_json(path: Path, data) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def read_source() -> str:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Source file not found: {SOURCE_FILE}")
    text = SOURCE_FILE.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError("source.txt is empty.")
    # Normalize line endings only; keep the text and its layout otherwise.
    return text.replace("\r\n", "\n").replace("\r", "\n")


def write_jsonl(path: Path, pairs: list[dict]) -> None:
    body = "".join(
        json.dumps(pair, ensure_ascii=False) + "\n"
        for pair in pairs
    )
    atomic_write_text(path, body)


def validate_pairs(content: str) -> list[dict]:
    data = json.loads(content)
    if not isinstance(data, dict) or set(data.keys()) != {"pairs"}:
        raise ValueError("Response must be a JSON object containing only 'pairs'.")
    if not isinstance(data["pairs"], list):
        raise ValueError("'pairs' must be a JSON array.")

    validated = []
    for index, pair in enumerate(data["pairs"], start=1):
        if not isinstance(pair, dict) or set(pair.keys()) != {"input", "output"}:
            raise ValueError(f"Pair {index} must contain exactly 'input' and 'output'.")
        if not isinstance(pair["input"], str) or not isinstance(pair["output"], str):
            raise ValueError(f"Pair {index}: input and output must be strings.")

        question = pair["input"].strip()
        answer = pair["output"].strip()
        if not question or not answer:
            raise ValueError(f"Pair {index} has an empty input or output.")
        validated.append({"input": question, "output": answer})
    return validated


def load_jsonl_pairs(path: Path) -> list[dict]:
    pairs = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict) or set(item.keys()) != {"input", "output"}:
                raise ValueError(f"{path.name}, line {line_number}: invalid JSONL pair.")
            if not isinstance(item["input"], str) or not isinstance(item["output"], str):
                raise ValueError(f"{path.name}, line {line_number}: input/output must be strings.")
            if not item["input"].strip() or not item["output"].strip():
                raise ValueError(f"{path.name}, line {line_number}: empty input/output.")
            pairs.append({"input": item["input"].strip(), "output": item["output"].strip()})
    return pairs


# ============================================================
# STRUCTURE-AWARE CHUNKING
# ============================================================

ABBREVIATIONS = {
    "e.g.", "i.e.", "etc.", "vs.", "dr.", "mr.", "mrs.", "ms.", "prof.",
    "fig.", "no.", "inc.", "ltd.", "u.s.", "u.k.", "et al.", "approx.",
    "est.", "vol.", "ref.", "st."
}


def split_sentences(text: str) -> list[str]:
    """Split on likely sentence endings while protecting common abbreviations."""
    text = text.strip()
    if not text:
        return []

    pieces = []
    start = 0
    # Boundary after punctuation and whitespace, before a likely new sentence.
    for match in re.finditer(r'(?<=[.!?])\s+(?=[A-Z0-9“‘"\(\[])', text):
        boundary = match.start()
        left = text[start:boundary].rstrip()
        token = left.split()[-1].lower() if left.split() else ""
        # Do not split after a known abbreviation.
        if token in ABBREVIATIONS:
            continue
        # Avoid splitting common multi-initial abbreviations such as U.S.
        if re.search(r"(?:\b[A-Za-z]\.){2,}$", left):
            continue
        piece = text[start:boundary].strip()
        if piece:
            pieces.append(piece)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        pieces.append(tail)
    return pieces or [text]


def split_oversized_unit(text: str, max_chars: int) -> list[str]:
    """
    Split a long block without arbitrary slicing where practical:
    list-like lines -> sentences -> clauses -> words -> hard split only for
    an exceptionally long unbroken token.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # Web copy often represents lists as short standalone lines. Preserve
    # these lines as individual units rather than gluing list items together.
    line_like = (
        len(lines) >= 3
        and sum(len(line) <= 220 for line in lines) / len(lines) >= 0.70
    )

    units: list[str] = []
    if line_like:
        for line in lines:
            if len(line) <= max_chars:
                units.append(line)
            else:
                units.extend(split_oversized_unit(line, max_chars))
    else:
        sentences = split_sentences(text)
        if len(sentences) > 1 and all(len(s) <= max_chars for s in sentences):
            units = sentences
        else:
            # Split at semicolons or commas only as a fallback for an
            # overlong sentence/list; retain delimiters in the preceding unit.
            clauses = re.split(r"(?<=[;,])\s+", text)
            if len(clauses) > 1 and all(len(c) <= max_chars for c in clauses):
                units = [c.strip() for c in clauses if c.strip()]
            elif len(sentences) > 1:
                for sentence in sentences:
                    units.extend(split_oversized_unit(sentence, max_chars))
            else:
                # Word-boundary split.
                words = text.split()
                current = ""
                for word in words:
                    candidate = word if not current else current + " " + word
                    if len(candidate) <= max_chars:
                        current = candidate
                    else:
                        if current:
                            units.append(current)
                        if len(word) > max_chars:
                            # Last resort for a single unbroken token.
                            for i in range(0, len(word), max_chars):
                                units.append(word[i:i + max_chars])
                            current = ""
                        else:
                            current = word
                if current:
                    units.append(current)

    # Defensive check: no returned unit should exceed the configured limit.
    result = []
    for unit in units:
        if len(unit) > max_chars:
            result.extend(split_oversized_unit(unit, max_chars))
        elif unit.strip():
            result.append(unit.strip())
    return result


def source_units(source_text: str, max_chars: int) -> list[str]:
    """Create semantic units from blank-line blocks, preserving headings/lists."""
    blocks = [
        block.strip()
        for block in re.split(r"\n[ \t]*\n+", source_text.strip())
        if block.strip()
    ]
    units = []
    for block in blocks:
        if len(block) <= max_chars:
            units.append(block)
        else:
            units.extend(split_oversized_unit(block, max_chars))
    return units


def split_source_into_chunks(source_text: str, max_chars: int = MAX_SOURCE_CHARS_PER_CHUNK) -> list[str]:
    """
    Pack related source units into bounded chunks. Headings and their following
    blocks normally stay together because adjacent blocks are packed in order.
    """
    units = source_units(source_text, max_chars)
    chunks = []
    current = ""

    for unit in units:
        candidate = unit if not current else current + "\n\n" + unit
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
        # source_units should already ensure this, but guard against surprises.
        if len(unit) > max_chars:
            smaller = split_oversized_unit(unit, max_chars)
            current = smaller[0]
            for part in smaller[1:]:
                chunks.append(current)
                current = part
        else:
            current = unit

    if current:
        chunks.append(current)

    return chunks


def context_tail(text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Return a short, verbatim, boundary-aware tail for interpretation only."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    tail = text[-max_chars:]
    # Prefer starting at a paragraph/line boundary.
    newline = tail.find("\n")
    if newline != -1 and newline < max_chars // 3:
        tail = tail[newline + 1:]
    # If possible, begin at a sentence boundary.
    match = re.search(r"(?<=[.!?])\s+", tail)
    if match and match.start() < len(tail) // 2:
        tail = tail[match.end():]
    return tail.strip()


def split_chunk_for_fallback(text: str) -> list[str]:
    """Split a failed chunk into smaller pieces at the best available boundaries."""
    target = max(MIN_AUTO_SPLIT_CHARS, len(text) // 2)
    pieces = split_source_into_chunks(text, target)
    if len(pieces) >= 2:
        return pieces

    # Final fallback: split near the midpoint at a whitespace boundary.
    midpoint = len(text) // 2
    candidates = [m.start() for m in re.finditer(r"\s+", text)]
    if not candidates:
        return []
    boundary = min(candidates, key=lambda position: abs(position - midpoint))
    left, right = text[:boundary].strip(), text[boundary:].strip()
    return [part for part in (left, right) if part]


# ============================================================
# RUN STATE / LOGGING
# ============================================================

def append_count_log(source_hash: str, chunk_id: str, chunk_text: str, pair_count: int) -> None:
    exists = COUNT_LOG_FILE.exists() and COUNT_LOG_FILE.stat().st_size > 0
    with COUNT_LOG_FILE.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow([
                "timestamp", "source_file", "source_sha256", "model",
                "chunk_id", "chunk_characters", "pair_count"
            ])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            SOURCE_FILE.name,
            source_hash,
            MODEL,
            chunk_id,
            len(chunk_text),
            pair_count,
        ])


def pair_key(pair: dict) -> tuple[str, str]:
    # Exact string-level deduplication only; do not merge near-duplicates.
    return pair["input"], pair["output"]


def read_completed_chunk(run_dir: Path, chunk_id: str, source_text: str,
                         context_text: str, config_hash: str):
    meta_path = run_dir / "meta" / f"{chunk_id}.json"
    pair_path = run_dir / "pairs" / f"{chunk_id}.jsonl"
    if not meta_path.exists() or not pair_path.exists():
        return None, None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("status") != "complete"
            or meta.get("source_chunk_sha256") != sha256_text(source_text)
            or meta.get("context_sha256") != sha256_text(context_text)
            or meta.get("config_sha256") != config_hash
        ):
            return None, None
        pairs = load_jsonl_pairs(pair_path)
        if len(pairs) != meta.get("pair_count"):
            return None, None
        return pairs, meta
    except (OSError, ValueError, json.JSONDecodeError):
        return None, None


def read_split_marker(run_dir: Path, chunk_id: str, source_text: str,
                      context_text: str, config_hash: str):
    meta_path = run_dir / "meta" / f"{chunk_id}.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("status") == "split"
            and meta.get("source_chunk_sha256") == sha256_text(source_text)
            and meta.get("context_sha256") == sha256_text(context_text)
            and meta.get("config_sha256") == config_hash
        ):
            return meta
    except (OSError, json.JSONDecodeError):
        pass
    return None


def write_chunk_meta(run_dir: Path, chunk_id: str, data: dict) -> None:
    path = run_dir / "meta" / f"{chunk_id}.json"
    atomic_write_json(path, data)


# ============================================================
# MISTRAL API
# ============================================================

def is_permanent_api_error(error: Exception) -> bool:
    """Avoid retry loops for clearly permanent HTTP errors."""
    status = getattr(error, "status_code", None)
    if status is None:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
    return status in {400, 401, 403, 404, 422}


def call_mistral_with_retries(client, messages):
    last_error = None
    for attempt in range(1, MAX_API_ATTEMPTS + 1):
        try:
            return client.chat.complete(
                model=MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=MAX_OUTPUT_TOKENS,
                response_format={"type": "json_object"},
            )
        except Exception as error:
            last_error = error
            if is_permanent_api_error(error) or attempt >= MAX_API_ATTEMPTS:
                raise
            delay = min(
                MAX_RETRY_DELAY_SECONDS,
                INITIAL_RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
            )
            delay += random.uniform(0, min(2.0, delay * 0.25))
            print(
                f"  API attempt {attempt}/{MAX_API_ATTEMPTS} failed: "
                f"{type(error).__name__}: {error}"
            )
            print(f"  Retrying in {delay:.1f} seconds...")
            time.sleep(delay)
    raise RuntimeError(f"Mistral API call failed: {last_error}")


def request_chunk(client, chunk_text: str, context_text: str, chunk_id: str,
                  run_dir: Path, format_attempt: int) -> tuple[list[dict], str]:
    user_message = (
        f"Chunk ID: {chunk_id}\n\n"
        "CONTEXT FROM PREVIOUS SOURCE TEXT (interpretation only; do not create "
        "pairs based solely on this context):\n"
        "----- BEGIN CONTEXT -----\n"
        f"{context_text if context_text else '[No preceding context]'}\n"
        "----- END CONTEXT -----\n\n"
        "PRIMARY SOURCE TEXT (the only text eligible for extracting pairs):\n"
        "----- BEGIN PRIMARY SOURCE TEXT -----\n"
        f"{chunk_text}\n"
        "----- END PRIMARY SOURCE TEXT -----\n\n"
        "Return only the required JSON object."
    )

    response = call_mistral_with_retries(
        client,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    choice = response.choices[0]
    content = choice.message.content
    finish_reason = str(getattr(choice, "finish_reason", "")).lower()

    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{chunk_id.replace('.', '_')}_format_attempt_{format_attempt}.txt"
    if isinstance(content, str):
        atomic_write_text(raw_path, content)

    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"Chunk {chunk_id}: model returned an empty/non-text response.")
    if "length" in finish_reason:
        raise RuntimeError(f"Chunk {chunk_id}: response reached the output-token limit.")
    try:
        return validate_pairs(content), finish_reason
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Chunk {chunk_id}: invalid model JSON: {error}") from error


def process_chunk(client, run_dir: Path, chunk_id: str, chunk_text: str,
                  context_text: str, source_hash: str, config_hash: str,
                  depth: int = 0) -> list[dict]:
    """Process or resume one chunk; recursively split output-limit failures."""
    pairs_dir = run_dir / "pairs"
    meta_dir = run_dir / "meta"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    cached, cached_meta = read_completed_chunk(
        run_dir, chunk_id, chunk_text, context_text, config_hash
    )
    if cached is not None:
        print(f"  Reusing completed chunk {chunk_id} ({len(cached)} pairs).")
        return cached

    split_marker = read_split_marker(
        run_dir, chunk_id, chunk_text, context_text, config_hash
    )
    if split_marker is not None:
        children = split_chunk_for_fallback(chunk_text)
        if len(children) < 2:
            raise RuntimeError(
                f"Chunk {chunk_id} was marked split, but cannot be split further. "
                "Inspect its raw response and source text."
            )
        print(f"  Resuming previously subdivided chunk {chunk_id}.")
        return process_children(
            client, run_dir, chunk_id, children, context_text,
            source_hash, config_hash, depth
        )

    last_format_error = None
    for format_attempt in range(1, MAX_FORMAT_ATTEMPTS + 1):
        try:
            pairs, finish_reason = request_chunk(
                client, chunk_text, context_text, chunk_id, run_dir, format_attempt
            )
            pair_path = pairs_dir / f"{chunk_id}.jsonl"
            write_jsonl(pair_path, pairs)
            write_chunk_meta(run_dir, chunk_id, {
                "status": "complete",
                "chunk_id": chunk_id,
                "source_chunk_sha256": sha256_text(chunk_text),
                "context_sha256": sha256_text(context_text),
                "config_sha256": config_hash,
                "source_characters": len(chunk_text),
                "pair_count": len(pairs),
                "finish_reason": finish_reason,
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            })
            append_count_log(source_hash, chunk_id, chunk_text, len(pairs))
            return pairs
        except (RuntimeError, ValueError, json.JSONDecodeError) as error:
            last_format_error = error
            print(f"  {error}")
            if format_attempt < MAX_FORMAT_ATTEMPTS:
                print(f"  Retrying the same chunk ({format_attempt + 1}/{MAX_FORMAT_ATTEMPTS})...")
                continue
            break

    # Only auto-split for content/format failures, not exhausted API/network errors.
    if len(chunk_text) <= MIN_AUTO_SPLIT_CHARS or depth >= 8:
        raise RuntimeError(
            f"Chunk {chunk_id} still fails validation and is too small to split safely: "
            f"{last_format_error}"
        )

    children = split_chunk_for_fallback(chunk_text)
    if len(children) < 2:
        raise RuntimeError(
            f"Chunk {chunk_id} failed validation and no safe split was found: "
            f"{last_format_error}"
        )

    write_chunk_meta(run_dir, chunk_id, {
        "status": "split",
        "chunk_id": chunk_id,
        "source_chunk_sha256": sha256_text(chunk_text),
        "context_sha256": sha256_text(context_text),
        "config_sha256": config_hash,
        "source_characters": len(chunk_text),
        "children_count": len(children),
        "reason": str(last_format_error),
        "split_at": datetime.now().isoformat(timespec="seconds"),
    })
    print(f"  Automatically subdividing chunk {chunk_id} into {len(children)} smaller chunks.")
    return process_children(
        client, run_dir, chunk_id, children, context_text,
        source_hash, config_hash, depth
    )


def process_children(client, run_dir: Path, parent_id: str, children: list[str],
                     parent_context: str, source_hash: str, config_hash: str,
                     depth: int) -> list[dict]:
    combined = []
    previous_text = ""
    for index, child_text in enumerate(children, start=1):
        child_id = f"{parent_id}.{index}"
        child_context = parent_context if index == 1 else context_tail(previous_text)
        child_pairs = process_chunk(
            client, run_dir, child_id, child_text, child_context,
            source_hash, config_hash, depth + 1
        )
        combined.extend(child_pairs)
        previous_text = child_text
    return combined


# ============================================================
# MAIN
# ============================================================

def backup_existing_output() -> None:
    if not JSONL_OUTPUT_FILE.exists() or JSONL_OUTPUT_FILE.stat().st_size == 0:
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BASE_DIR / f"generated_pairs_backup_{timestamp}.jsonl"
    shutil.copy2(JSONL_OUTPUT_FILE, backup_path)
    print(f"Previous successful dataset backed up to: {backup_path.name}")


def main() -> int:
    api_key = os.environ.get("MISTRAL_API_KEY", "").strip()
    if not api_key:
        api_key = getpass("Enter your Mistral API key: ").strip()
    if not api_key:
        print("No API key provided.")
        return 2

    try:
        source_text = read_source()
    except (OSError, UnicodeError, ValueError) as error:
        print(f"Could not read source: {error}")
        return 2

    source_hash = sha256_text(source_text)
    config_material = "\n".join([
        PIPELINE_VERSION,
        MODEL,
        str(MAX_SOURCE_CHARS_PER_CHUNK),
        str(MAX_OUTPUT_TOKENS),
        str(MAX_CONTEXT_CHARS),
        str(MIN_AUTO_SPLIT_CHARS),
        SYSTEM_PROMPT,
    ])
    config_hash = sha256_text(config_material)
    run_hash = sha256_text(source_hash + config_hash)
    run_dir = RUNS_DIR / run_hash[:20]
    run_dir.mkdir(parents=True, exist_ok=True)

    chunks = split_source_into_chunks(source_text)
    if not chunks:
        print("No source chunks were produced.")
        return 2
    if any(len(chunk) > MAX_SOURCE_CHARS_PER_CHUNK for chunk in chunks):
        print("Internal error: a source chunk exceeds MAX_SOURCE_CHARS_PER_CHUNK.")
        return 2

    manifest_path = run_dir / "manifest.json"
    manifest = {
        "source_file": SOURCE_FILE.name,
        "source_sha256": source_hash,
        "config_sha256": config_hash,
        "model": MODEL,
        "max_source_chars_per_chunk": MAX_SOURCE_CHARS_PER_CHUNK,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "total_top_level_chunks": len(chunks),
        "status": "running",
        "started_or_resumed_at": datetime.now().isoformat(timespec="seconds"),
    }
    atomic_write_json(manifest_path, manifest)

    print(f"Source: {SOURCE_FILE.name}")
    print(f"Characters: {len(source_text):,}")
    print(f"Model: {MODEL}")
    print(f"Chunk size limit: {MAX_SOURCE_CHARS_PER_CHUNK:,} characters")
    print(f"Top-level chunks: {len(chunks)}")
    print(f"Run state: {run_dir.relative_to(BASE_DIR)}")
    print("Starting generation (completed chunks will be reused on rerun)...")

    all_pairs = []
    seen = set()
    completed_top_level = 0

    try:
        with Mistral(api_key=api_key) as client:
            previous_chunk_text = ""
            for index, chunk_text in enumerate(chunks, start=1):
                chunk_id = f"{index:04d}"
                context_text = context_tail(previous_chunk_text)
                print(
                    f"\n--- Chunk {index}/{len(chunks)} "
                    f"({len(chunk_text):,} characters; ID {chunk_id}) ---"
                )
                try:
                    pairs = process_chunk(
                        client, run_dir, chunk_id, chunk_text, context_text,
                        source_hash, config_hash
                    )
                except Exception as error:
                    print(f"\nSTOPPED on chunk {chunk_id}: {type(error).__name__}: {error}")
                    print(f"Completed top-level chunks: {completed_top_level}/{len(chunks)}")
                    print(f"Run state and raw responses are saved in: {run_dir}")
                    print("Fix the underlying issue, then rerun; completed chunks will be reused.")
                    manifest["status"] = "interrupted"
                    manifest["last_error"] = f"{type(error).__name__}: {error}"
                    manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
                    atomic_write_json(manifest_path, manifest)
                    write_jsonl(PARTIAL_OUTPUT_FILE, all_pairs)
                    return 1

                for pair in pairs:
                    key = pair_key(pair)
                    if key not in seen:
                        seen.add(key)
                        all_pairs.append(pair)

                completed_top_level += 1
                previous_chunk_text = chunk_text
                write_jsonl(PARTIAL_OUTPUT_FILE, all_pairs)
                print(f"Pairs returned for this chunk/tree: {len(pairs)}")
                print(f"Unique exact pairs so far: {len(all_pairs)}")
                print(f"Progress: {completed_top_level}/{len(chunks)} top-level chunks")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Completed chunk results are saved; rerun to resume.")
        write_jsonl(PARTIAL_OUTPUT_FILE, all_pairs)
        manifest["status"] = "interrupted_by_user"
        manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
        atomic_write_json(manifest_path, manifest)
        return 130
    except Exception as error:
        print(f"\nMistral client/setup failure: {type(error).__name__}: {error}")
        print("Completed chunk results are saved where available; rerun to resume.")
        write_jsonl(PARTIAL_OUTPUT_FILE, all_pairs)
        manifest["status"] = "interrupted"
        manifest["last_error"] = f"{type(error).__name__}: {error}"
        manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
        atomic_write_json(manifest_path, manifest)
        return 1

    # The final output is updated only after every top-level chunk succeeds.
    write_jsonl(PARTIAL_OUTPUT_FILE, all_pairs)
    backup_existing_output()
    atomic_write_text(
        JSONL_OUTPUT_FILE,
        PARTIAL_OUTPUT_FILE.read_text(encoding="utf-8")
    )
    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["total_unique_exact_pairs"] = len(all_pairs)
    atomic_write_json(manifest_path, manifest)

    print("\n========================================")
    print("GENERATION COMPLETE")
    print(f"Chunks processed: {completed_top_level}/{len(chunks)}")
    print(f"TOTAL UNIQUE EXACT PAIRS: {len(all_pairs)}")
    print("========================================")
    print(f"Final JSONL: {JSONL_OUTPUT_FILE.name}")
    print(f"Partial checkpoint: {PARTIAL_OUTPUT_FILE.name}")
    print(f"Per-chunk outputs and state: {run_dir.relative_to(BASE_DIR)}")
    print(f"Raw responses: {(run_dir / 'raw').relative_to(BASE_DIR)}")
    print(f"Count history: {COUNT_LOG_FILE.name}")
    print(
        "\nValidation checks JSON structure, not oncology factual accuracy. "
        "Use the saved raw responses and source text for any later audit."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
