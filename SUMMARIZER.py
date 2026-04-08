"""
summary.py — Pharma Intelligence Brief Generator

Reads extracted article text (text.json), calls the LLM with guaranteed
JSON output, validates the schema, and writes structured news items to
a clean JSON file.

Pipeline:
1. Chunk articles (CHUNK_SIZE each) → per-chunk news items
2. Merge all chunk results into one unified list (zero signal loss)
3. Validate schema and write output

After each run, results are also appended to briefs_history.json:

  {
    "2026-04-02": {
      "bispecific antibodies": [ { company, modality, news, url }, ... ],
      "CAR-T":                 [ ... ]
    },
    "2026-04-01": { ... }
  }

Usage:
  python summary.py --query "PROTAC"
  python summary.py --query "CAR-T" --input text.json --output briefs.json
"""

import argparse
import json
import re
import sys
import time
import requests
from datetime import datetime
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
INVOKE_URL      = "https://integrate.api.nvidia.com/v1/chat/completions"
API_KEY         = "Bearer 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"
MODEL           = "qwen/qwen3.5-122b-a10b"
MAX_RETRIES     = 3
BACKOFF_BASE    = 1      # seconds — attempt 1: no wait, 2: 1s, 3: 2s, 4: 4s
REQUEST_TIMEOUT = 180    # seconds per attempt
CHUNK_SIZE      = 5      # articles per chunk

BRIEFS_HISTORY_FILE = "briefs_history.json"

HEADERS = {
    "Authorization": API_KEY,
    "Content-Type":  "application/json",
    "Accept":        "application/json",
}

VALID_MODALITIES = {
    "bispecific antibodies",
    "monoclonal antibodies",
    "gene editing",
    "molecular glues",
}

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a pharmaceutical news extraction assistant.

Your job is to scan pharmaceutical articles and extract structured news events
related to specific therapeutic modalities:
  - bispecific antibodies
  - monoclonal antibodies
  - gene editing
  - molecular glues

RULES:
- Extract ONLY real news events: clinical trials, approvals, partnerships,
  acquisitions, funding rounds, or scientific breakthroughs.
- Ignore background/explainer content and unrelated topics.
- Do NOT hallucinate company names, drug names, or URLs.
- If multiple companies are involved, choose the primary one.
- If URL is unavailable, set it to null.
- Avoid duplicate entries.
- Return ONLY valid JSON — no explanations, no markdown fences.

OUTPUT FORMAT (strict JSON, nothing else):
{
  "news": [
    {
      "company": "string",
      "modality": "one of the four predefined modalities",
      "news": "2-3 line description of the event",
      "url": "source article URL or null"
    }
  ]
}
"""

# ── MERGE SYSTEM PROMPT ───────────────────────────────────────────────────────

MERGE_SYSTEM_PROMPT = """You are merging multiple structured pharmaceutical news JSON outputs into one unified list.

STRICT MERGE RULES:
- Return ONLY JSON in the exact same schema as the input chunks.
- ZERO signal loss — preserve ALL news items from ALL chunks.
- Combine all "news" arrays into a single flat list.
- Remove only exact duplicates (same company + same news text).
- Do NOT summarise, compress, or rewrite any news item.
- Do NOT hallucinate new content.
- The output must contain every unique item from every input chunk.

OUTPUT FORMAT (strict JSON, nothing else):
{
  "news": [
    {
      "company": "string",
      "modality": "one of the four predefined modalities",
      "news": "2-3 line description of the event",
      "url": "source article URL or null"
    }
  ]
}
"""

# ── CHUNKING ──────────────────────────────────────────────────────────────────

def chunk_articles(articles, chunk_size=CHUNK_SIZE):
    for i in range(0, len(articles), chunk_size):
        yield articles[i:i + chunk_size]

# ── PROMPT BUILDERS ───────────────────────────────────────────────────────────

def build_chunk_prompt(articles: list, query: str) -> str | None:
    sections = []

    for i, art in enumerate(articles, 1):
        title  = art.get("title", "Untitled")
        source = art.get("url", "Unknown")
        date   = art.get("date") or art.get("period", "")
        body   = (art.get("body") or "").strip()

        if not body:
            continue

        sections.append(
            f"--- ARTICLE {i} ---\n"
            f"Source : {source}\n"
            f"Date   : {date}\n"
            f"Title  : {title}\n\n"
            f"{body}"
        )

    if not sections:
        return None

    return (
        f"Query focus: {query}\n\n"
        f"Below are {len(sections)} pharmaceutical news articles.\n"
        f"Extract all relevant news events matching the query focus.\n\n"
        + "\n\n".join(sections)
    )


def build_merge_prompt(chunk_results: list, query: str) -> str:
    """Build a prompt to merge multiple chunk JSON outputs into one unified list."""
    all_chunks = "\n\n".join(chunk_results)
    return (
        f"Query focus: {query}\n\n"
        f"Below are {len(chunk_results)} structured pharmaceutical news outputs "
        f"from different article chunks.\n"
        f"Merge them into ONE unified news list with ZERO signal loss — "
        f"preserve every unique item, remove only exact duplicates.\n\n"
        f"{all_chunks}"
    )

# ── LLM CALL ─────────────────────────────────────────────────────────────────

def call_llm(system_prompt: str, user_prompt: str) -> str:
    payload = {
        "model":   MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "max_tokens":      16384,
        "temperature":     0.3,
        "top_p":           0.95,
        "stream":          False,
        "response_format": {"type": "json_object"},
    }

    last_error: str = "unknown"

    for attempt in range(1, MAX_RETRIES + 1):

        if attempt > 1:
            wait = BACKOFF_BASE * (2 ** (attempt - 2))
            print(f"[LLM] Retry {attempt}/{MAX_RETRIES} — backing off {wait}s...")
            time.sleep(wait)

        print(f"[LLM] Attempt {attempt}/{MAX_RETRIES} — sending request...", flush=True)

        try:
            resp = requests.post(
                INVOKE_URL,
                headers=HEADERS,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()

        except requests.exceptions.Timeout:
            last_error = f"timed out after {REQUEST_TIMEOUT}s"
            print(f"[LLM] ✗ Attempt {attempt} failed: {last_error}")
            continue

        except requests.exceptions.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else 0
            last_error = f"HTTP {code}"
            if 400 <= code < 500:
                raise RuntimeError(
                    f"HTTP {code} — aborting retries (client error): {exc}"
                ) from exc
            print(f"[LLM] ✗ Attempt {attempt} failed: {last_error}")
            continue

        except requests.exceptions.RequestException as exc:
            last_error = str(exc)
            print(f"[LLM] ✗ Attempt {attempt} network error: {last_error}")
            continue

        try:
            envelope = resp.json()
        except json.JSONDecodeError as exc:
            last_error = f"response body not valid JSON: {exc}"
            print(f"[LLM] ✗ Attempt {attempt} failed: {last_error}")
            continue

        content = (
            envelope
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        if not content:
            last_error = "empty content in API response"
            print(f"[LLM] ✗ Attempt {attempt} failed: {last_error}")
            continue

        print(f"[LLM] ✓ Response received ({len(content):,} chars)")
        return content

    raise RuntimeError(
        f"LLM call failed after {MAX_RETRIES} attempts. Last error: {last_error}"
    )

# ── CHUNK PROCESSING ──────────────────────────────────────────────────────────

def generate_chunk_results(articles: list, query: str) -> list[str]:
    """Process articles in chunks of CHUNK_SIZE, return list of raw JSON strings."""
    chunks = list(chunk_articles(articles, CHUNK_SIZE))
    print(f"[INFO] Total article chunks : {len(chunks)}")

    results = []
    for idx, chunk in enumerate(chunks, 1):
        print(f"\n[INFO] Processing chunk {idx}/{len(chunks)} ({len(chunk)} articles)...")
        print("─" * 60)

        prompt = build_chunk_prompt(chunk, query)
        if not prompt:
            print(f"[WARN] Chunk {idx}: no valid article bodies — skipping")
            continue

        try:
            raw = call_llm(SYSTEM_PROMPT, prompt)
            if raw:
                results.append(raw)
        except RuntimeError as e:
            print(f"[ERROR] Chunk {idx} failed: {e} — skipping")

    return results


def merge_chunk_results(chunk_results: list[str], query: str) -> str:
    """
    If only one chunk result exists, return it directly.
    Otherwise send all chunk results to the LLM for lossless merging.
    """
    if len(chunk_results) == 1:
        print("\n[INFO] Single chunk — skipping merge step.")
        return chunk_results[0]

    print(f"\n[INFO] Merging {len(chunk_results)} chunk results...\n")
    print("─" * 60)

    merge_prompt = build_merge_prompt(chunk_results, query)

    try:
        merged = call_llm(MERGE_SYSTEM_PROMPT, merge_prompt)
        return merged
    except RuntimeError as e:
        print(f"[ERROR] Merge failed: {e} — falling back to concatenating all chunk items")
        # Fallback: manually concatenate all parsed items from every chunk
        all_items = []
        for raw in chunk_results:
            all_items.extend(parse_llm_response(raw))
        return json.dumps({"news": all_items})

# ── JSON PARSER ───────────────────────────────────────────────────────────────

def parse_llm_response(raw: str) -> list:
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj.get("news", [])
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            return obj.get("news", [])
        except json.JSONDecodeError:
            pass

    print("[WARN] parse_llm_response: could not extract JSON — returning []")
    return []

# ── SCHEMA VALIDATOR ──────────────────────────────────────────────────────────

REQUIRED_KEYS = {"company", "modality", "news", "url"}

def validate_items(raw_items: list) -> tuple[list, int]:
    valid   = []
    dropped = 0

    for idx, item in enumerate(raw_items, 1):
        if not isinstance(item, dict):
            print(f"[VALIDATE] Item {idx}: not a dict — dropped")
            dropped += 1
            continue

        missing = REQUIRED_KEYS - item.keys()
        if missing:
            print(f"[VALIDATE] Item {idx}: missing keys {missing} — dropped")
            dropped += 1
            continue

        if item["modality"] not in VALID_MODALITIES:
            print(f"[VALIDATE] Item {idx}: invalid modality '{item['modality']}' — dropped")
            dropped += 1
            continue

        if not isinstance(item["company"], str) or not item["company"].strip():
            print(f"[VALIDATE] Item {idx}: empty company — dropped")
            dropped += 1
            continue

        if not isinstance(item["news"], str) or not item["news"].strip():
            print(f"[VALIDATE] Item {idx}: empty news text — dropped")
            dropped += 1
            continue

        if item["url"] is not None and not isinstance(item["url"], str):
            item["url"] = str(item["url"])

        valid.append(item)

    return valid, dropped

# ── OUTPUT BUILDER ────────────────────────────────────────────────────────────

def build_output(news_items: list, query: str, article_count: int, dropped: int) -> dict:
    return {
        "meta": {
            "query":           query,
            "articles_used":   article_count,
            "items_extracted": len(news_items),
            "items_dropped":   dropped,
            "generated_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "news": news_items,
    }

# ── DATEWISE HISTORY ──────────────────────────────────────────────────────────

def append_to_history(
    news_items: list,
    query: str,
    history_file: str = BRIEFS_HISTORY_FILE,
) -> None:
    """
    Append today's validated news items into briefs_history.json.

    Structure:
      {
        "2026-04-02": {
          "bispecific antibodies": [ {company, modality, news, url}, ... ],
          "CAR-T": [ ... ]
        },
        "2026-04-01": { ... }
      }

    - Items are stored under their actual modality key (not the query),
      so you can look up "what bispecific news ran on 2026-04-02" directly.
    - Duplicate URLs within the same date+modality are skipped.
    - Dates are kept sorted newest-first.
    """
    today = datetime.now().strftime("%Y-%m-%d")

    p = Path(history_file)
    if p.exists():
        try:
            history = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            history = {}
    else:
        history = {}

    if today not in history:
        history[today] = {}

    added = 0
    for item in news_items:
        modality = item.get("modality", "unknown")
        url      = item.get("url")

        if modality not in history[today]:
            history[today][modality] = []

        existing_sigs = {
            (r.get("company"), r.get("news")) for r in history[today][modality]
        }
        if (item.get("company"), item.get("news")) in existing_sigs:
            continue

        history[today][modality].append({
            "company":  item.get("company"),
            "modality": modality,
            "news":     item.get("news"),
            "url":      url,
            "query":    query,
        })
        added += 1

    history = dict(sorted(history.items(), reverse=True))

    p.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[HISTORY] Appended {added} items → {history_file}")
    print(f"[HISTORY] Dates stored: {list(history.keys())[:5]} ...")


def print_history_summary(history_file: str = BRIEFS_HISTORY_FILE) -> None:
    """Pretty-print a summary table of briefs_history.json."""
    p = Path(history_file)
    if not p.exists():
        return

    try:
        history = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return

    print("\n" + "═" * 64)
    print("  BRIEFS HISTORY SUMMARY")
    print("═" * 64)
    print(f"  {'Date':<14}  {'Modality':<28}  {'Items':>5}")
    print("  " + "─" * 52)

    for date in sorted(history.keys(), reverse=True):
        modalities = history[date]
        for modality, items in modalities.items():
            print(f"  {date:<14}  {modality:<28}  {len(items):>5}")

    total = sum(
        len(items)
        for day in history.values()
        for items in day.values()
    )
    print("═" * 64)
    print(f"  Total items stored : {total}")
    print(f"  Total dates        : {len(history)}")
    print("═" * 64 + "\n")

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Pharma Intelligence Brief Generator")
    p.add_argument("--input",  "-i", default="text.json",   help="Extracted articles JSON")
    p.add_argument("--output", "-o", default="briefs.json", help="Output JSON file")
    p.add_argument("--query",  "-q", required=True,         help="Therapeutic focus e.g. 'PROTAC'")
    p.add_argument(
        "--no-history", action="store_true",
        help="Skip writing to briefs_history.json"
    )
    return p.parse_args()

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Load input ────────────────────────────────────────────────────────────
    src = Path(args.input)
    if not src.exists():
        sys.exit(f"[ERROR] Input file not found: {src}")

    try:
        with open(src, encoding="utf-8") as f:
            raw_data = json.load(f)
    except Exception as e:
        sys.exit(f"[ERROR] Failed to load JSON: {e}")

    if isinstance(raw_data, dict) and "articles" in raw_data:
        articles = raw_data["articles"]
    elif isinstance(raw_data, list):
        articles = raw_data
    else:
        sys.exit("[ERROR] Unexpected JSON format in input file.")

    valid_articles = [a for a in articles if (a.get("body") or "").strip()]
    skipped        = len(articles) - len(valid_articles)

    print(f"[INFO] Loaded   : {len(articles)} articles")
    print(f"[INFO] Valid    : {len(valid_articles)}")
    if skipped:
        print(f"[INFO] Skipped  : {skipped} (no body text)")
    print(f"[INFO] Query    : {args.query}")
    print(f"[INFO] Chunk sz : {CHUNK_SIZE} articles per chunk\n")

    if not valid_articles:
        sys.exit("[WARN] No valid articles found.")

    # ── STEP 1: Process articles in chunks ────────────────────────────────────
    print("[INFO] Generating per-chunk news items...\n")
    print("─" * 60)

    chunk_results = generate_chunk_results(valid_articles, args.query)

    if not chunk_results:
        sys.exit("[WARN] No chunk results generated.")

    # ── STEP 2: Merge all chunk results into one ──────────────────────────────
    raw_response = merge_chunk_results(chunk_results, args.query)

    if not raw_response:
        sys.exit("[WARN] Empty response from merge step.")

    # ── STEP 3: Parse + validate ──────────────────────────────────────────────
    raw_items            = parse_llm_response(raw_response)
    news_items, dropped  = validate_items(raw_items)

    print(f"\n[INFO] Chunk results         : {len(chunk_results)}")
    print(f"[INFO] Raw items parsed      : {len(raw_items)}")
    print(f"[INFO] Passed validation     : {len(news_items)}")
    if dropped:
        print(f"[WARN] Dropped bad items     : {dropped}")

    # ── STEP 4: Write briefs.json ─────────────────────────────────────────────
    output_data = build_output(news_items, args.query, len(valid_articles), dropped)
    out_path    = Path(args.output)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved → {out_path}")

    # ── STEP 5: Append to datewise history ────────────────────────────────────
    if not args.no_history and news_items:
        append_to_history(news_items, args.query)
        print_history_summary()

    # ── Preview ───────────────────────────────────────────────────────────────
    if news_items:
        print("\n── PREVIEW (first 3 items) ──────────────────────────────────")
        for item in news_items[:3]:
            print(f"  [{item.get('modality','?')}] {item.get('company','?')}")
            print(f"  {item.get('news','')[:110]}...")
        print("─────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
