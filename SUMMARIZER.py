"""
summarizer.py  —  Pharma Intelligence Brief Generator

Takes filtered articles JSON and summarizes ALL articles into a single
combined MONTHLY PHARMA INTELLIGENCE BRIEF returned as structured JSON.

Pipeline:
1. Chunk articles (chunk_size each) → per-chunk partial briefs via SYSTEM_PROMPT
2. Merge all partial briefs into one unified report via MERGE_SYSTEM_PROMPT
   (zero signal loss — every point preserved, only exact duplicates removed)
3. Write final structured JSON output

JSON output schema:
{
  "query": "...",
  "generated_at": "...",
  "sections": [
    {
      "heading": "Overview",
      "paragraph": "Full flowing paragraph text...",
      "sources": [
        { "url": "https://...", "label": "Short source label" },
        ...
      ]
    },
    {
      "heading": "Key Developments",
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

INVOKE_URL  = "https://integrate.api.nvidia.com/v1/chat/completions"
API_KEY     = "Bearer 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"
MODEL       = "qwen/qwen3.5-122b-a10b"
CHUNK_SIZE  = 5     # articles per LLM call

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
      "paragraph": "A single cohesive, densely-written paragraph of 150–250 words that synthesises the most important developments across ALL articles. The paragraph must read as flowing editorial prose — not a list, not separate sentences. Weave together companies, drugs, indications, and regulatory/deal highlights into a unified narrative. Reference all major entities and developments. Do NOT use bullet points, numbered items, or line breaks inside this paragraph.",
      "sources": [
        { "url": "source article URL or null", "label": "Company or drug name" }
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
1. Overview — A single flowing editorial paragraph (150–250 words) that weaves together all major developments, companies, drugs, and deals into a unified narrative. This MUST be a paragraph field, not a points array. Additionally include a "sources" array listing every unique source URL referenced across all articles, each with a short label (company or drug name).
2. Key Developments — one bullet per major development. Format: Company — Drug — Indication — Development.
3. Companies in Focus — one bullet per company: name, strategic objective, associated drug/program.
4. Clinical & Scientific Highlights — drug mechanism, trial phase, patient population, comparator (if any), clinical endpoints/results. Only explicitly stated information.
5. Business & Deals — acquisitions, partnerships, licensing, pipeline positioning. If none, return a single point: { "text": "No relevant business developments reported.", "url": null }.
"""

# ── MERGE SYSTEM PROMPT ───────────────────────────────────────────────────────

MERGE_SYSTEM_PROMPT = """You are merging multiple partial pharmaceutical intelligence briefs into one unified report.

STRICT MERGE RULES:
- Return ONLY valid JSON in the exact same schema as the input chunks — no markdown, no explanations.
- ZERO signal loss — preserve ALL points from ALL partial briefs.
- For list sections (Key Developments, Companies in Focus, Clinical & Scientific Highlights, Business & Deals):
    - Combine all points into a single flat list per section.
    - Remove only exact duplicates (identical text). Keep the most specific version of near-duplicates.
    - Do NOT summarise, compress, or rewrite any point.
- For the Overview section:
    - Merge all partial paragraphs into ONE cohesive flowing paragraph (150–250 words).
    - Combine all unique sources into the sources array (deduplicate by URL).
- Do NOT hallucinate new content.
- The output must contain every unique point from every input brief.

OUTPUT FORMAT (strict JSON, nothing else):
{
  "sections": [
    {
      "heading": "Overview",
      "paragraph": "...",
      "sources": [ { "url": "...", "label": "..." } ]
    },
    {
      "heading": "Key Developments",
      "points": [ { "text": "...", "url": "..." } ]
    },
    {
      "heading": "Companies in Focus",
      "points": [ { "text": "...", "url": "..." } ]
    },
    {
      "heading": "Clinical & Scientific Highlights",
      "points": [ { "text": "...", "url": "..." } ]
    },
    {
      "heading": "Business & Deals",
      "points": [ { "text": "...", "url": "..." } ]
    }
  ]
}
"""

# ── CHUNKING ──────────────────────────────────────────────────────────────────

def chunk_articles(articles: list, chunk_size: int = CHUNK_SIZE):
    for i in range(0, len(articles), chunk_size):
        yield articles[i:i + chunk_size]

# ── PROMPT BUILDERS ───────────────────────────────────────────────────────────

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
        f"Below are {len(partial_jsons)} partial pharmaceutical intelligence briefs, "
        f"each already in the required JSON format.\n"
        f"Merge them into ONE unified brief with ZERO signal loss — "
        f"preserve every unique point, remove only exact duplicates.\n\n"
        f"{joined}"
    )

# ── LLM CALL (streaming) ──────────────────────────────────────────────────────

def call_api_streaming(system_prompt: str, user_prompt: str) -> str:
    """Stream from NVIDIA API using the given system prompt; return full accumulated text."""
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

# ── JSON HELPERS ──────────────────────────────────────────────────────────────

def parse_json_response(raw: str) -> dict | None:
    """Strip markdown fences and parse JSON; return None on failure."""
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    print("[WARN] Could not parse JSON from model response.")
    return None


def normalise_sections(obj: dict) -> list:
    if not obj:
        return []
    if "sections" in obj:
        return obj["sections"]
    if "brief" in obj and "sections" in obj["brief"]:
        return obj["brief"]["sections"]
    return []

# ── LOCAL MERGE FALLBACK ──────────────────────────────────────────────────────

def merge_section_lists(all_sections: list[list]) -> list:
    """
    Fallback merge: combine section lists locally in Python.
    Used when the LLM merge call fails. Zero signal loss — only exact
    duplicate point texts are dropped.
    """
    merged: dict[str, dict] = {}
    heading_order: list[str] = []

    for sections in all_sections:
        for sec in sections:
            h = sec.get("heading", "").strip()
            if not h:
                continue

            if h == "Overview":
                if h not in merged:
                    merged[h] = {"heading": h, "paragraph": "", "sources": []}
                    heading_order.append(h)
                existing = merged[h].get("paragraph", "")
                new_para = sec.get("paragraph", "")
                if new_para:
                    merged[h]["paragraph"] = (existing + " " + new_para).strip() if existing else new_para
                seen_urls = {s.get("url") for s in merged[h].get("sources", [])}
                for src in sec.get("sources", []):
                    if src.get("url") not in seen_urls:
                        merged[h].setdefault("sources", []).append(src)
                        seen_urls.add(src.get("url"))
            else:
                if h not in merged:
                    merged[h] = {"heading": h, "points": []}
                    heading_order.append(h)
                seen_texts = {p.get("text", "").lower() for p in merged[h].get("points", [])}
                for pt in sec.get("points", []):
                    t = (pt.get("text") or "").strip()
                    if t and t.lower() not in seen_texts:
                        merged[h].setdefault("points", []).append(pt)
                        seen_texts.add(t.lower())

    return [merged[h] for h in heading_order]

# ── CHUNK PROCESSING ──────────────────────────────────────────────────────────

def generate_chunk_results(articles: list, query: str, chunk_size: int) -> tuple[list[str], list[list]]:
    """
    Process articles in chunks, calling SYSTEM_PROMPT per chunk.
    Returns (partial_raw_jsons, partial_section_lists).
    """
    chunks = list(chunk_articles(articles, chunk_size))
    print(f"[INFO] Total article chunks : {len(chunks)}\n")

    partial_raw  = []
    partial_secs = []

    for idx, chunk in enumerate(chunks, 1):
        print(f"[INFO] ── Chunk {idx}/{len(chunks)} ({len(chunk)} articles) ──────────────────────")
        prompt = build_chunk_prompt(chunk, query)
        try:
            raw = call_api_streaming(SYSTEM_PROMPT, prompt)
        except Exception as e:
            print(f"[ERROR] Chunk {idx} failed: {e} — skipping")
            continue

        obj  = parse_json_response(raw)
        secs = normalise_sections(obj)
        if secs:
            partial_secs.append(secs)
            partial_raw.append(json.dumps({"sections": secs}, ensure_ascii=False))
        else:
            print(f"[WARN] Chunk {idx}: no valid sections extracted.")

    return partial_raw, partial_secs


def merge_chunk_results(
    partial_raw: list[str],
    partial_secs: list[list],
    query: str,
    use_llm_merge: bool,
) -> list:
    """
    If only one chunk result exists, return it directly.
    Otherwise merge via LLM (MERGE_SYSTEM_PROMPT) or local fallback.
    """
    if len(partial_secs) == 1:
        print("\n[INFO] Single chunk — skipping merge step.")
        return partial_secs[0]

    if not use_llm_merge:
        final = merge_section_lists(partial_secs)
        print(f"\n[INFO] Local merge → {len(final)} sections.")
        return final

    print(f"\n[INFO] ── Merging {len(partial_raw)} chunk results (LLM merge pass) ──────────────────────")
    merge_prompt = build_merge_prompt(partial_raw, query)
    try:
        raw_merged = call_api_streaming(MERGE_SYSTEM_PROMPT, merge_prompt)
        obj_merged = parse_json_response(raw_merged)
        final      = normalise_sections(obj_merged)
        if not final:
            raise ValueError("Empty sections after LLM merge")
        print(f"[INFO] LLM merge → {len(final)} sections.")
        return final
    except Exception as e:
        print(f"[WARN] LLM merge failed ({e}) — falling back to local merge.")
        return merge_section_lists(partial_secs)

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
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                   help=f"Articles per LLM call (default {CHUNK_SIZE})")
    p.add_argument("--no-merge-llm", action="store_true",
                   help="Merge sections locally instead of a final LLM merge pass")
    return p.parse_args()

# ── MAIN ──────────────────────────────────────────────────────────────────────

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
    print(f"[INFO] Query    : {args.query}")
    print(f"[INFO] Chunk sz : {args.chunk_size} articles per chunk\n")

    if not valid:
        sys.exit("[WARN] No articles with usable text found.")

    # ── STEP 1: Process articles in chunks ────────────────────────────────────
    print("[INFO] Generating per-chunk intelligence briefs...\n")
    print("─" * 60)

    partial_raw, partial_secs = generate_chunk_results(valid, args.query, args.chunk_size)

    if not partial_secs:
        sys.exit("[FATAL] No usable output from any chunk.")

    # ── STEP 2: Merge all chunk results into one ──────────────────────────────
    final_sections = merge_chunk_results(
        partial_raw,
        partial_secs,
        args.query,
        use_llm_merge=not args.no_merge_llm,
    )

    # ── STEP 3: Write output ──────────────────────────────────────────────────
    output = {
        "query":         args.query,
        "generated_at":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "article_count": len(valid),
        "sections":      final_sections,
    }

    out_str = json.dumps(output, indent=2, ensure_ascii=False)

    if args.output:
        Path(args.output).write_text(out_str, encoding="utf-8")
        print(f"\n[INFO] Saved → {args.output}")
    else:
        print("\n" + out_str)

    # ── Preview ───────────────────────────────────────────────────────────────
    print("\n── PREVIEW ─────────────────────────────────────────────────────")
    for sec in final_sections:
        if sec.get("heading") == "Overview":
            para_len  = len((sec.get("paragraph") or "").split())
            src_count = len(sec.get("sources") or [])
            print(f"  Overview  ({para_len} words, {src_count} sources)")
        else:
            pts = sec.get("points", [])
            print(f"  {sec.get('heading','?')}  ({len(pts)} points)")
    print("─────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
