"""
summarizer.py  —  Pharma Intelligence Brief Generator

Takes filtered articles JSON and summarizes ALL articles into a single
combined MONTHLY PHARMA INTELLIGENCE BRIEF returned as structured JSON.

JSON output schema:
{
  "query": "...",
  "generated_at": "...",
  "sections": [
    {
      "heading": "Overview",
      "points": [
        { "text": "...", "url": "https://..." },
        ...
      ]
    },
    ...
  ]
}

Usage:
  python summarizer.py --input filtered_files.json --query "PROTAC"
  python summarizer.py --input filtered_files.json --query "CAR-T" --output briefs.json
"""

import argparse, json, sys, requests, re
from datetime import datetime
from pathlib import Path

# ── NVIDIA CONFIG ─────────────────────────────────────────────────────────────

INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
API_KEY    = "Bearer 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"
MODEL      = "qwen/qwen3.5-122b-a10b"

HEADERS = {
    "Authorization": API_KEY,
    "Accept": "text/event-stream",
}

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a pharmaceutical intelligence analyst specializing in biotechnology, drug development, and clinical trial reporting.
Your task is to convert a collection of pharmaceutical news articles into a single unified MONTHLY PHARMA INTELLIGENCE BRIEF.

STRICT RULES:
- Return ONLY valid JSON — no markdown fences, no explanations, no preamble.
- Use only information explicitly present in the articles. Do NOT hallucinate.
- Do NOT invent statistics or trial results.
- Every point must reference a real entity (company, drug, trial, or regulator).
- Avoid generic language such as "the company aims to improve outcomes".
- Only include data related to the query topic.
- Write in a formal pharmaceutical intelligence tone.
- Each point must include the source URL from the article it came from (set to null if unknown).

OUTPUT FORMAT — strict JSON, nothing else:
{
  "sections": [
    {
      "heading": "Overview",
      "points": [
        { "text": "...", "url": "source article URL or null" }
      ]
    },
    {
      "heading": "Key Developments",
      "points": [
        { "text": "Company — Drug — Indication — Development", "url": "source article URL or null" }
      ]
    },
    {
      "heading": "Companies in Focus",
      "points": [
        { "text": "Company: strategic objective and associated drug or program.", "url": "source article URL or null" }
      ]
    },
    {
      "heading": "Clinical & Scientific Highlights",
      "points": [
        { "text": "Drug mechanism, trial phase, patient population, endpoints or results.", "url": "source article URL or null" }
      ]
    },
    {
      "heading": "Business & Deals",
      "points": [
        { "text": "Acquisition/partnership/licensing deal description.", "url": "source article URL or null" }
      ]
    }
  ]
}

SECTION RULES:
1. Overview — 5-10 sentences summarising the most important developments across all articles. Each sentence is a separate point with its source URL.
2. Key Developments — one bullet per major development. Format: Company — Drug — Indication — Development.
3. Companies in Focus — one bullet per company: name, strategic objective, associated drug/program.
4. Clinical & Scientific Highlights — drug mechanism, trial phase, patient population, comparator (if any), clinical endpoints/results. Only explicitly stated information.
5. Business & Deals — acquisitions, partnerships, licensing, pipeline positioning. If none, return a single point: { "text": "No relevant business developments reported.", "url": null }.
"""

# ── HELPERS ───────────────────────────────────────────────────────────────────

def build_chunk_prompt(articles: list, query: str) -> str:
    sections = []
    for i, art in enumerate(articles, 1):
        title  = art.get("title", "Untitled")
        source = art.get("url", "Unknown")
        date   = art.get("date", "")
        body   = (art.get("text") or "").strip()
        if not body:
            continue
        sections.append(
            f"--- ARTICLE {i} ---\n"
            f"Source : {source}\n"
            f"Date   : {date}\n"
            f"Title  : {title}\n\n"
            f"{body}"
        )
    return (
        f"Query focus: {query}\n\n"
        f"Below are {len(sections)} pharmaceutical news articles.\n"
        f"Synthesize ALL of them into a single unified MONTHLY PHARMA INTELLIGENCE BRIEF.\n"
        f"Return ONLY the JSON object described in your instructions — nothing else.\n\n"
        + "\n\n".join(sections)
    )


def build_merge_prompt(partial_jsons: list[str], query: str) -> str:
    joined = "\n\n".join(
        f"--- PARTIAL BRIEF {i+1} ---\n{p}" for i, p in enumerate(partial_jsons)
    )
    return (
        f"Query focus: {query}\n\n"
        f"You are given {len(partial_jsons)} partial pharmaceutical intelligence briefs, "
        f"each already in the required JSON format.\n"
        f"Merge them into ONE unified brief using the same JSON schema.\n"
        f"Deduplicate overlapping points. Keep the most specific version of each point.\n"
        f"Return ONLY the final merged JSON object — nothing else.\n\n"
        f"{joined}"
    )


def call_api_streaming(system_prompt: str, user_prompt: str) -> str:
    """Stream from NVIDIA API, return full accumulated text."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "max_tokens": 16384,
        "temperature": 0.30,
        "top_p": 0.95,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": True},
    }

    resp = requests.post(INVOKE_URL, headers=HEADERS, json=payload, stream=True)
    resp.raise_for_status()

    output = ""
    for line in resp.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8")
        if decoded.startswith("data: "):
            data_str = decoded[6:]
            if data_str.strip() == "[DONE]":
                break
            try:
                chunk   = json.loads(data_str)
                delta   = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    output += content
                    print(content, end="", flush=True)
            except json.JSONDecodeError:
                continue

    print()  # newline after streaming
    return output.strip()


def parse_json_response(raw: str) -> dict | None:
    """Strip markdown fences and parse JSON; return None on failure."""
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    # Try full parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fallback: find the outermost { ... }
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    print("[WARN] Could not parse JSON from model response.")
    return None


def normalise_sections(obj: dict) -> list:
    """
    Accept both top-level shapes:
      { "sections": [...] }          ← standard
      { "brief": { "sections": [...] } }  ← occasional wrapper
    Returns the sections list, or [] on failure.
    """
    if not obj:
        return []
    if "sections" in obj:
        return obj["sections"]
    if "brief" in obj and "sections" in obj["brief"]:
        return obj["brief"]["sections"]
    return []


def merge_section_lists(all_sections: list[list]) -> list:
    """
    Merge multiple section lists (from multiple chunks) into one,
    combining points under matching headings and deduplicating by text.
    """
    merged: dict[str, list] = {}  # heading -> list of points
    heading_order: list[str] = []

    for sections in all_sections:
        for sec in sections:
            h = sec.get("heading", "").strip()
            if not h:
                continue
            if h not in merged:
                merged[h] = []
                heading_order.append(h)
            seen_texts = {p.get("text","").lower() for p in merged[h]}
            for pt in sec.get("points", []):
                t = (pt.get("text") or "").strip()
                if t and t.lower() not in seen_texts:
                    merged[h].append(pt)
                    seen_texts.add(t.lower())

    return [{"heading": h, "points": merged[h]} for h in heading_order]


def chunk_articles(articles: list, chunk_size: int = 9):
    for i in range(0, len(articles), chunk_size):
        yield articles[i:i + chunk_size]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Summarize pharma articles into a structured JSON intelligence brief."
    )
    p.add_argument("--input",  "-i", default="filtered_files.json")
    p.add_argument("--output", "-o", default=None,
                   help="Save brief to a .json file.")
    p.add_argument("--query",  "-q", required=True,
                   help='Topic focus e.g. "PROTAC" or "CAR-T"')
    p.add_argument("--chunk-size", type=int, default=5,
                   help="Articles per LLM call (default 5)")
    p.add_argument("--no-merge-llm", action="store_true",
                   help="Merge sections locally instead of a final LLM merge pass")
    return p.parse_args()


def main():
    args = parse_args()

    path = Path(args.input)
    if not path.exists():
        sys.exit(f"[ERROR] File not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    articles = data.get("articles", data) if isinstance(data, dict) else data
    if not isinstance(articles, list):
        sys.exit("[ERROR] JSON must contain a list of articles")

    valid   = [a for a in articles if (a.get("text") or "").strip()]
    skipped = len(articles) - len(valid)

    print(f"[INFO] Loaded   : {len(articles)} articles")
    if skipped:
        print(f"[INFO] Skipped  : {skipped} (empty text)")
    print(f"[INFO] Using    : {len(valid)} articles")
    print(f"[INFO] Query    : {args.query}\n")

    if not valid:
        sys.exit("[WARN] No articles with usable text found.")

    chunks       = list(chunk_articles(valid, chunk_size=args.chunk_size))
    partial_raw  = []   # raw JSON strings from each chunk (for LLM merge)
    partial_secs = []   # parsed section lists (for local merge fallback)

    print(f"[INFO] Processing {len(chunks)} chunk(s)...\n")

    for i, chunk in enumerate(chunks, 1):
        print(f"[INFO] ── Chunk {i}/{len(chunks)} ──────────────────────")
        prompt = build_chunk_prompt(chunk, args.query)
        try:
            raw = call_api_streaming(SYSTEM_PROMPT, prompt)
        except Exception as e:
            print(f"[ERROR] Chunk {i} failed: {e}")
            continue

        obj  = parse_json_response(raw)
        secs = normalise_sections(obj)
        if secs:
            partial_secs.append(secs)
            partial_raw.append(json.dumps({"sections": secs}, ensure_ascii=False))
        else:
            print(f"[WARN] Chunk {i}: no valid sections extracted.")

    if not partial_secs:
        sys.exit("[FATAL] No usable output from any chunk.")

    # ── Merge ────────────────────────────────────────────────────────────────
    if len(partial_secs) == 1:
        final_sections = partial_secs[0]
        print("\n[INFO] Single chunk — no merge needed.")
    elif args.no_merge_llm:
        final_sections = merge_section_lists(partial_secs)
        print(f"\n[INFO] Local merge → {len(final_sections)} sections.")
    else:
        print("\n[INFO] ── Final LLM merge pass ──────────────────────")
        merge_prompt = build_merge_prompt(partial_raw, args.query)
        try:
            raw_merged = call_api_streaming(SYSTEM_PROMPT, merge_prompt)
            obj_merged = parse_json_response(raw_merged)
            final_sections = normalise_sections(obj_merged)
            if not final_sections:
                raise ValueError("Empty sections after LLM merge")
            print(f"[INFO] LLM merge → {len(final_sections)} sections.")
        except Exception as e:
            print(f"[WARN] LLM merge failed ({e}), falling back to local merge.")
            final_sections = merge_section_lists(partial_secs)

    # ── Build final output object ─────────────────────────────────────────────
    output = {
        "query":        args.query,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "article_count": len(valid),
        "sections":     final_sections,
    }

    out_str = json.dumps(output, indent=2, ensure_ascii=False)

    if args.output:
        Path(args.output).write_text(out_str, encoding="utf-8")
        print(f"\n[INFO] Saved → {args.output}")
    else:
        print("\n" + out_str)

    # ── Quick preview ─────────────────────────────────────────────────────────
    print("\n── PREVIEW ─────────────────────────────────────────────────────")
    for sec in final_sections:
        pts = sec.get("points", [])
        print(f"  {sec.get('heading','?')}  ({len(pts)} points)")
    print("─────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
