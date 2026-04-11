"""
summarizer.py  —  Pharma Intelligence Brief Generator

Takes filtered articles JSON and summarizes ALL articles into a single
combined MONTHLY PHARMA INTELLIGENCE BRIEF returned as structured JSON.

Pipeline:
1. Chunk articles (chunk_size each) → per-chunk partial briefs
2. Merge all partial briefs into one unified report (zero signal loss)
3. Write final structured JSON output

Usage:
  python summarizer.py --input filtered_files.json --query "PROTAC"
  python summarizer.py --input filtered_files.json --query "CAR-T" --output briefs.json
"""

import argparse, json, sys, requests, re, time
from datetime import datetime
from pathlib import Path

# ── NVIDIA CONFIG ─────────────────────────────────────────────────────────────

INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
API_KEY    = "Bearer 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"
MODEL      = "qwen/qwen3.5-122b-a10b"

CHUNK_SIZE      = 2    # articles per LLM call
MAX_RETRIES     = 3    # retries per chunk on transient errors
RETRY_DELAY     = 5    # seconds to wait between retries
REQUEST_TIMEOUT = 180  # seconds — model is slow on large prompts, 3 min is safe
CHUNK_DELAY     = 2    # seconds to pause between chunk calls (avoids rate limiting)

HEADERS = {
    "Authorization": API_KEY,
    "Accept": "text/event-stream",
}

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
# NOTE: qwen/qwen3.5-122b-a10b does NOT support the "system" role in messages.
# System instructions are injected at the top of the user message instead.

SYSTEM_PROMPT = """You are a senior pharmaceutical intelligence analyst at a top-tier life sciences advisory firm. Your task is to synthesize pharmaceutical news articles into a structured MONTHLY PHARMA INTELLIGENCE BRIEF in JSON format.

═══════════════════════════════════════════════════════════
QUALITY STANDARD — READ THIS BEFORE WRITING A SINGLE WORD
═══════════════════════════════════════════════════════════

Study these two examples. The BAD version is what you must NEVER produce. The GOOD version is the MINIMUM acceptable standard.

BAD (rejected): "The company aims to improve patient outcomes for its lead candidate."
GOOD (required): "Viridian Therapeutics — Elegrobart — Thyroid Eye Disease — Phase 3 trial met primary endpoint with 54% reduction in proptosis vs 18% placebo (p<0.001); FDA submission planned Q3 2026."

BAD (rejected): "No specific developments regarding monoclonal antibodies were reported."
BAD (rejected): "The provided articles contain no information relevant to the query."
BAD (rejected): "Both source documents exclusively report on market research for electric taps."
→ If an article is irrelevant to the query, SKIP IT SILENTLY. Never explain why you are skipping it.

BAD (rejected): "The company continues to advance its pipeline program."
GOOD (required): "Invivyd — PEMGARDA (pemivibart) — COVID-19 pre-exposure prophylaxis — Q4 2025 net revenue $17.2M, +25% YoY; Phase 3 DECLARATION trial fully enrolled, topline data expected mid-2026."

BAD (rejected): "Key companies in this space include GSK, Insmed, and Gossamer Bio."
GOOD (required): "Gossamer Bio / Chiesi Farmaceutici — Seralutinib — Pulmonary Arterial Hypertension — Inhaled PDGFR/FGFR/CSF1R inhibitor; Phase 3 PROSERA trial ongoing."

═══════════════════════════════════════════════════════════
ABSOLUTE RULES
═══════════════════════════════════════════════════════════

1. Return ONLY valid JSON — no markdown fences, no preamble, no explanation. The very first character of your response must be { and the last must be }.
2. Use ONLY information explicitly stated in the provided articles. Do NOT hallucinate, extrapolate, or invent.
3. Do NOT produce filler points. Every single point must contain: company name + drug/program name + indication + specific data (trial phase, endpoint result, regulatory action, deal value, or mechanism).
4. If an article has no relevance to the query topic, skip it entirely. Do not write a point explaining its irrelevance.
5. Do not repeat the same fact in multiple sections. Each fact belongs in exactly one section.
6. Write in a terse, data-dense intelligence style. No hedging. No preamble. No soft language.
7. Include the exact source URL for every point. If no URL is available, use null.

═══════════════════════════════════════════════════════════
OUTPUT FORMAT — STRICT JSON
═══════════════════════════════════════════════════════════

{
  "sections": [
    {
      "heading": "Overview",
      "points": [
        { "text": "...", "url": "source URL or null" }
      ]
    },
    {
      "heading": "Key Developments",
      "points": [
        { "text": "...", "url": "source URL or null" }
      ]
    },
    {
      "heading": "Companies in Focus",
      "points": [
        { "text": "...", "url": "source URL or null" }
      ]
    },
    {
      "heading": "Clinical & Scientific Highlights",
      "points": [
        { "text": "...", "url": "source URL or null" }
      ]
    },
    {
      "heading": "Business & Deals",
      "points": [
        { "text": "...", "url": "source URL or null" }
      ]
    }
  ]
}

═══════════════════════════════════════════════════════════
SECTION-BY-SECTION WRITING RULES
═══════════════════════════════════════════════════════════

── OVERVIEW ──
Write 6–12 concise intelligence sentences covering the most significant developments across all articles.
Each sentence is its own point with its source URL.
Every sentence must contain at least one named company, drug, or regulatory body + one specific data point.
Format: "[Entity] [action/outcome] [specific data]. [Context if essential.]"
Example: "Merck received FDA approval for ENFLONSIA (clesrovimab-cfor), a fixed 105 mg dose RSV preventive monoclonal antibody for newborns and infants entering their first RSV season."
Example: "Invivyd's PEMGARDA (pemivibart) generated $17.2M net revenue in Q4 2025, a 25% YoY increase, as enrollment in the Phase 3 DECLARATION trial for VYD2311 was completed."

── KEY DEVELOPMENTS ──
One point per major development. Use this exact format:
"[Company] — [Drug/Program] — [Indication] — [Development with specific data]"
Example: "Viridian Therapeutics — Elegrobart — Thyroid Eye Disease — Phase 3 met primary endpoint: 54% proptosis reduction vs 18% placebo; BLA submission planned."
Example: "Samsung Biologics — Rockville Manufacturing Facility — Biologics CMO — Acquired GSK's 60,000L cGMP facility in Maryland, raising total global capacity to 845,000L."
Include trial phase, endpoints, regulatory status, deal value, or market size wherever stated in the source.

── COMPANIES IN FOCUS ──
One point per company. Use this exact format:
"[Company] — [Strategic objective] — [Associated drug/program] — [Current status or key data]"
Example: "Invivyd — Vaccine-alternative antibody commercialization — PEMGARDA (pemivibart) + VYD2311 pipeline — $200M+ raised H2 2025; Phase 3 DECLARATION trial fully enrolled."
Example: "Bio-Thera Solutions — Biosimilar market entry — Avzivi (bevacizumab biosimilar) — FDA and EMA approvals secured; positioned for competitive pricing vs Avastin."

── CLINICAL & SCIENTIFIC HIGHLIGHTS ──
One point per drug/mechanism. Use this exact format:
"[Drug] — Mechanism: [specific MOA] — Phase: [trial phase] — Population: [patient type] — Comparator: [if applicable] — Endpoints/Results: [specific data]"
Example: "Elegrobart — Mechanism: IGF-1R receptor blockade — Phase: 3 — Population: Active thyroid eye disease (Graves' disease) — Comparator: Placebo — Results: 54% proptosis reduction vs 18%; no contralateral eye damage reported."
Example: "VYD2311 — Mechanism: Serial molecular evolution-engineered mAb neutralizing contemporary SARS-CoV-2 lineages — Phase: 3 (DECLARATION) — Population: Adults and adolescents with/without severe COVID-19 risk factors — Comparator: Placebo — Endpoint: PCR-confirmed symptomatic COVID incidence at 3 months; topline data expected mid-2026."
Only include data explicitly stated in the source. Do not speculate on missing fields — omit them.

── BUSINESS & DEALS ──
One point per deal, partnership, acquisition, regulatory filing, or market development. Include financial figures where stated.
Example: "Samsung Biologics — Acquired GSK's Rockville, Maryland manufacturing facility for undisclosed sum; site adds 60,000L capacity and establishes Samsung's first US manufacturing presence."
Example: "Evotec SE / Just-Evotec Biologics — Secured $10M multi-year BARDA contract for Ebola monoclonal antibody manufacturing optimization."
Example: "Roivant Sciences — Approved $1B share repurchase program while advancing batoclimab (anti-FcRn mAb) in IgG-mediated autoimmune indications."
If no business developments are relevant to the query, return exactly: { "text": "No relevant business developments reported in this period.", "url": null }
"""

# ── MERGE SYSTEM PROMPT ───────────────────────────────────────────────────────

MERGE_SYSTEM_PROMPT = """You are merging multiple partial pharmaceutical intelligence briefs into one unified report.

STRICT MERGE RULES:
- Return ONLY valid JSON — no markdown, no explanations. First character must be {, last must be }.
- ZERO signal loss — preserve ALL substantive points from ALL partial briefs.
- Combine all points into a single flat list per section.
- Remove only exact duplicates (identical text). When two points cover the same fact but one has more detail, keep the more detailed version and discard the shorter one.
- Do NOT summarise, compress, or rewrite any point.
- Do NOT hallucinate new content.
- SILENTLY discard any point that is a filler or meta-commentary, including:
    • Points saying "No relevant information", "No data available", "The articles do not contain..."
    • Points that describe the source article rather than the drug/company/trial
    • Points about non-pharmaceutical topics (electric taps, logistics, unrelated markets)

OUTPUT FORMAT (strict JSON, nothing else):
{
  "sections": [
    { "heading": "Overview",                         "points": [ { "text": "...", "url": "..." } ] },
    { "heading": "Key Developments",                 "points": [ { "text": "...", "url": "..." } ] },
    { "heading": "Companies in Focus",               "points": [ { "text": "...", "url": "..." } ] },
    { "heading": "Clinical & Scientific Highlights", "points": [ { "text": "...", "url": "..." } ] },
    { "heading": "Business & Deals",                 "points": [ { "text": "...", "url": "..." } ] }
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
        # Cap each article at 3000 chars to prevent connection aborts on large payloads
        body   = (art.get("text") or "").strip()[:3000]
        if not body:
            continue
        sections.append(
            f"--- ARTICLE {i} ---\n"
            f"Source : {source}\n"
            f"Date   : {date}\n"
            f"Title  : {title}\n\n"
            f"{body}"
        )

    # System prompt prepended inside user message (model rejects system role)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"{'=' * 60}\n\n"
        f"Query focus: {query}\n\n"
        f"Below are {len(sections)} pharmaceutical news articles.\n"
        f"Extract all information relevant to '{query}' and synthesize into a MONTHLY PHARMA INTELLIGENCE BRIEF.\n"
        f"If an article has no relevance to '{query}', skip it silently — do not mention it.\n"
        f"Return ONLY the JSON object described above — nothing else.\n\n"
        + "\n\n".join(sections)
    )


def build_merge_prompt(partial_jsons: list[str], query: str) -> str:
    joined = "\n\n".join(
        f"--- PARTIAL BRIEF {i+1} ---\n{p}" for i, p in enumerate(partial_jsons)
    )
    return (
        f"{MERGE_SYSTEM_PROMPT}\n\n"
        f"{'=' * 60}\n\n"
        f"Query focus: {query}\n\n"
        f"Below are {len(partial_jsons)} partial pharmaceutical intelligence briefs "
        f"in the required JSON format.\n"
        f"Merge them into ONE unified brief — zero signal loss, remove only exact duplicates "
        f"and all filler/meta-commentary points.\n\n"
        f"{joined}"
    )

# ── LLM CALL (streaming) ──────────────────────────────────────────────────────

def call_api_streaming(user_prompt: str) -> str:
    """
    Stream from NVIDIA API.

    Key fixes:
      1. No 'system' role — qwen3.5-122b-a10b rejects it → system prompt inside user message.
      2. No chat_template_kwargs — not supported on integrate.api.nvidia.com.
      3. REQUEST_TIMEOUT = 180s — model streams reasoning tokens first; default 30s kills it.
      4. Prints actual error body on failure so you always know the real reason.
      5. Only captures delta.content — ignores delta.reasoning (internal thinking tokens).
    """
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 16384,
        "temperature": 0.20,   # lower = more deterministic, better for structured data
        "top_p": 0.90,
        "stream": True,
        # chat_template_kwargs intentionally omitted — causes 400 on hosted API
    }

    resp = requests.post(
        INVOKE_URL,
        headers=HEADERS,
        json=payload,
        stream=True,
        timeout=REQUEST_TIMEOUT,  # 180s — critical for large prompts
    )

    # Always print the real error body so debugging is easy
    if resp.status_code != 200:
        print(f"\n[ERROR {resp.status_code}] {resp.text[:600]}")
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
                # Only capture actual response content — skip reasoning/thinking tokens
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
    # Fallback: extract first {...} block
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

# ── FILLER POINT FILTER ───────────────────────────────────────────────────────

FILLER_PATTERNS = [
    "no relevant",
    "no specific",
    "no data",
    "no information",
    "not contain",
    "does not contain",
    "no monoclonal",
    "no pharmaceutical",
    "no clinical trial",
    "no explicit",
    "exclusively report",
    "electric tap",
    "unrelated",
    "market research for",
    "the provided article",
    "the source document",
    "both source document",
    "neither article",
    "the body text exclusively",
    "despite the title",
    "this is irrelevant",
]

def is_filler(text: str) -> bool:
    t = text.lower()
    return any(pattern in t for pattern in FILLER_PATTERNS)

# ── LOCAL MERGE FALLBACK ──────────────────────────────────────────────────────

def merge_section_lists(all_sections: list[list]) -> list:
    """
    Merge section lists locally — zero signal loss.
    Drops exact duplicate texts and filler/meta-commentary points.
    """
    merged: dict[str, dict] = {}
    heading_order: list[str] = []

    for sections in all_sections:
        for sec in sections:
            h = sec.get("heading", "").strip()
            if not h:
                continue
            if h not in merged:
                merged[h] = {"heading": h, "points": []}
                heading_order.append(h)
            seen_texts = {p.get("text", "").lower() for p in merged[h].get("points", [])}
            for pt in sec.get("points", []):
                t = (pt.get("text") or "").strip()
                if t and not is_filler(t) and t.lower() not in seen_texts:
                    merged[h].setdefault("points", []).append(pt)
                    seen_texts.add(t.lower())

    return [merged[h] for h in heading_order]

# ── CHUNK PROCESSING ──────────────────────────────────────────────────────────

def generate_chunk_results(
    articles: list, query: str, chunk_size: int
) -> tuple[list[str], list[list]]:
    """
    Process articles in chunks.
    Returns (partial_raw_jsons, partial_section_lists).
    """
    chunks = list(chunk_articles(articles, chunk_size))
    print(f"[INFO] Total article chunks : {len(chunks)}\n")

    partial_raw  = []
    partial_secs = []

    for idx, chunk in enumerate(chunks, 1):
        print(f"[INFO] ── Chunk {idx}/{len(chunks)} ({len(chunk)} articles) ──────────────────────")
        prompt = build_chunk_prompt(chunk, query)

        raw = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw = call_api_streaming(prompt)
                break  # success — exit retry loop
            except requests.exceptions.Timeout:
                print(f"[WARN] Chunk {idx} attempt {attempt}/{MAX_RETRIES}: timeout after {REQUEST_TIMEOUT}s")
            except Exception as e:
                print(f"[WARN] Chunk {idx} attempt {attempt}/{MAX_RETRIES}: {e}")

            if attempt < MAX_RETRIES:
                print(f"[INFO] Waiting {RETRY_DELAY}s before retry...")
                time.sleep(RETRY_DELAY)

        if not raw:
            print(f"[ERROR] Chunk {idx} failed after {MAX_RETRIES} attempts — skipping\n")
            continue

        obj  = parse_json_response(raw)
        secs = normalise_sections(obj)
        if secs:
            # Filter filler points at chunk level before storing
            cleaned_secs = []
            for sec in secs:
                clean_pts = [p for p in sec.get("points", []) if not is_filler(p.get("text", ""))]
                if clean_pts:
                    cleaned_secs.append({"heading": sec["heading"], "points": clean_pts})
            if cleaned_secs:
                partial_secs.append(cleaned_secs)
                partial_raw.append(json.dumps({"sections": cleaned_secs}, ensure_ascii=False))
            else:
                print(f"[INFO] Chunk {idx}: all points were filler — skipping.")
        else:
            print(f"[WARN] Chunk {idx}: no valid sections extracted.")

        # Polite pause between chunks to avoid rate limiting
        if idx < len(chunks):
            time.sleep(CHUNK_DELAY)

    return partial_raw, partial_secs


def merge_chunk_results(
    partial_raw: list[str],
    partial_secs: list[list],
    query: str,
    use_llm_merge: bool,
) -> list:
    """
    Merge all chunk results into one final brief.
    Falls back to local Python merge if LLM merge fails.
    """
    if len(partial_secs) == 1:
        print("\n[INFO] Single chunk result — skipping merge step.")
        return partial_secs[0]

    if not use_llm_merge:
        final = merge_section_lists(partial_secs)
        print(f"\n[INFO] Local merge → {len(final)} sections.")
        return final

    print(f"\n[INFO] ── Merging {len(partial_raw)} chunk results (LLM merge pass) ──────────────────────")
    merge_prompt = build_merge_prompt(partial_raw, query)
    try:
        raw_merged = call_api_streaming(merge_prompt)
        obj_merged = parse_json_response(raw_merged)
        final      = normalise_sections(obj_merged)
        if not final:
            raise ValueError("Empty sections after LLM merge")
        # Final filler filter pass on merged result
        for sec in final:
            sec["points"] = [p for p in sec.get("points", []) if not is_filler(p.get("text", ""))]
        final = [s for s in final if s.get("points")]
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
    print(f"[INFO] Model    : {MODEL}")
    print(f"[INFO] Chunk sz : {args.chunk_size} articles per chunk")
    print(f"[INFO] Timeout  : {REQUEST_TIMEOUT}s per request")
    print(f"[INFO] Retries  : {MAX_RETRIES} per chunk\n")

    if not valid:
        sys.exit("[WARN] No articles with usable text found.")

    # ── STEP 1: Process articles in chunks ────────────────────────────────────
    print("[INFO] Generating per-chunk intelligence briefs...\n")
    print("─" * 60)

    partial_raw, partial_secs = generate_chunk_results(valid, args.query, args.chunk_size)

    if not partial_secs:
        sys.exit("[FATAL] No usable output from any chunk. Check your API key and model status.")

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
        pts = sec.get("points", [])
        print(f"  {sec.get('heading', '?')}  ({len(pts)} points)")
    print("─────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
