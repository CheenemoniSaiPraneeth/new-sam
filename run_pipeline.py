"""
run_pipeline.py
===============
Entry point for the Pharma News Intelligence Pipeline (extraction mode).

Runs the full pipeline in three stages:
  Stage A -- Extraction  : search + scrape all installed portals  (extraction.py)
  Stage B -- Merge       : flatten per-domain JSONs → merged_articles.json
  Stage C -- Summarize   : generate a structured intelligence brief  (SUMMARIZER.py)

Usage (terminal):
  python run_pipeline.py
  python run_pipeline.py --query crispr --days 14
  python run_pipeline.py --domain biopharmadive.com --query protac
  python run_pipeline.py --limit 5 --no-enrich
  python run_pipeline.py --no-summarize          (skip summarizer step)

Usage (Jupyter):
  await run_pipeline()
  await run_pipeline(query="crispr", days=14)
  await run_pipeline(domain="biopharmadive.com")
  await run_pipeline(query="protac", summarize=False)
"""

# =============================================================================
#  CONFIG
# =============================================================================

QUERY            = "protac"
DATE_WINDOW_DAYS = 7
ENRICH_ARTICLES  = True
OUTPUT_DIR       = "extraction_output"
MERGED_FILE      = "merged_articles.json"
BRIEF_FILE       = "pharma_brief.json"

# =============================================================================
#  IMPORTS
# =============================================================================

import argparse, asyncio, json, time, logging
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# =============================================================================
#  MERGE
# =============================================================================

def merge_results(query: str, days: int) -> list:
    """
    Read all *_results.json from OUTPUT_DIR and flatten into a single list
    of article dicts.  Writes merged_articles.json alongside the domain files.

    Each article gets a top-level 'domain' field so the summarizer can cite
    its source.

    Returns the flat list (may be empty if no per-domain files exist).
    """
    output_path = Path(OUTPUT_DIR)
    all_articles: list = []

    for fpath in sorted(output_path.glob("*_results.json")):
        domain_key = fpath.stem.replace("_results", "").replace("_", ".")
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning(f"[merge] Could not read {fpath.name}: {e}")
            continue

        # data shape: {month: {article_count, articles: [...]}, ...}
        for month, month_data in data.items():
            for art in month_data.get("articles", []):
                art.setdefault("domain", domain_key)
                art.setdefault("period", month)
                all_articles.append(art)

    merged_path = output_path / MERGED_FILE
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "query":          query,
                "date_window":    f"last {days} days",
                "merged_at":      datetime.now(timezone.utc).isoformat(),
                "total_articles": len(all_articles),
                "articles":       all_articles,
            },
            f, indent=2,
        )

    print(f"\n[merge] {len(all_articles)} articles → {merged_path}")
    return all_articles


# =============================================================================
#  SUMMARIZER RUNNER
# =============================================================================

def run_summarizer(articles: list, query: str, output_path: str) -> dict | None:
    """
    Calls SUMMARIZER.py logic to produce a structured JSON intelligence brief.
    Returns the final output dict, or None on failure.
    """
    if not articles:
        print("[summarizer] No articles to summarize — skipping.")
        return None

    import importlib.util, sys as _sys

    spec = importlib.util.spec_from_file_location("SUMMARIZER", "SUMMARIZER.py")
    if spec is None:
        print("[summarizer] ERROR: SUMMARIZER.py not found in working directory.")
        return None

    mod = importlib.util.module_from_spec(spec)
    _sys.modules["SUMMARIZER"] = mod
    spec.loader.exec_module(mod)

    chunks = list(mod.chunk_articles(articles, chunk_size=5))
    print(f"[summarizer] {len(articles)} articles in {len(chunks)} chunk(s) — streaming...\n")
    print("─" * 65)

    partial_secs = []
    partial_raw  = []

    for idx, chunk in enumerate(chunks, 1):
        print(f"[summarizer] Chunk {idx}/{len(chunks)}")
        prompt = mod.build_chunk_prompt(chunk, query)
        try:
            raw = mod.call_api_streaming(mod.SYSTEM_PROMPT, prompt)
            obj  = mod.parse_json_response(raw)
            secs = mod.normalise_sections(obj)
            if secs:
                partial_secs.append(secs)
                partial_raw.append(json.dumps({"sections": secs}, ensure_ascii=False))
            else:
                print(f"[summarizer] Chunk {idx}: no valid sections extracted.")
        except Exception as e:
            print(f"[summarizer] Chunk {idx} failed: {e}")

    if not partial_secs:
        print("[summarizer] All chunks failed — no brief produced.")
        return None

    # Merge
    if len(partial_secs) == 1:
        final_sections = partial_secs[0]
        print("\n[summarizer] Single chunk — no merge needed.")
    else:
        print("\n[summarizer] ── Final LLM merge pass ──")
        merge_prompt = mod.build_merge_prompt(partial_raw, query)
        try:
            raw_merged     = mod.call_api_streaming(mod.SYSTEM_PROMPT, merge_prompt)
            obj_merged     = mod.parse_json_response(raw_merged)
            final_sections = mod.normalise_sections(obj_merged)
            if not final_sections:
                raise ValueError("Empty sections after LLM merge")
        except Exception as e:
            print(f"[summarizer] LLM merge failed ({e}), falling back to local merge.")
            final_sections = mod.merge_section_lists(partial_secs)

    output = {
        "query":         query,
        "generated_at":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "article_count": len(articles),
        "sections":      final_sections,
    }

    Path(output_path).write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[summarizer] Brief saved → {output_path}")

    print("\n── PREVIEW ─────────────────────────────────────────────────────")
    for sec in final_sections:
        print(f"  {sec.get('heading','?')}  ({len(sec.get('points', []))} points)")
    print("─────────────────────────────────────────────────────────────\n")

    return output


# =============================================================================
#  MAIN
# =============================================================================

async def run_pipeline(
    url:         str | None = None,
    query:       str        = QUERY,
    days:        int        = DATE_WINDOW_DAYS,
    enrich:      bool       = ENRICH_ARTICLES,
    limit:       int | None = None,
    domain:      str | None = None,
    summarize:   bool       = True,
):
    """
    Run extraction → merge → summarize.

    Args:
        url       : alias for domain (convenience)
        query     : search query
        days      : date filter window in days
        enrich    : scrape full article body text
        limit     : cap number of domains processed
        domain    : process only this domain (overrides limit)
        summarize : run SUMMARIZER after merge
    """
    t0 = time.time()
    started = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    target_domain = domain or url

    print(f"\n{'='*65}")
    print(f"  PHARMA PIPELINE -- EXTRACTION + MERGE + SUMMARIZE")
    print(f"  Query       : {query!r}")
    print(f"  Date window : last {days} days")
    print(f"  Domain      : {target_domain or 'all installed portals'}")
    print(f"  Enrich      : {enrich}")
    print(f"  Summarize   : {summarize}")
    print(f"  Started     : {started}")
    print(f"{'='*65}")

    # ── Stage A: Extraction ───────────────────────────────────────────────────
    try:
        from extraction import main as run_extraction
    except ImportError:
        print("  ERROR: extraction.py not found in working directory.")
        return

    await run_extraction(
        query  = query,
        domain = target_domain,
        limit  = limit,
        enrich = enrich,
        days   = days,
    )

    # ── Stage B: Merge ────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  STAGE B -- MERGE")
    print(f"{'='*65}")
    articles = merge_results(query=query, days=days)

    if not articles:
        print("\n[merge] No articles found — nothing to summarize.")
        elapsed = round((time.time() - t0) / 60, 1)
        print(f"\n  Total time: {elapsed} minutes")
        return

    # ── Stage C: Summarize ────────────────────────────────────────────────────
    if summarize:
        print(f"\n{'='*65}")
        print(f"  STAGE C -- SUMMARIZE  ({len(articles)} articles, query={query!r})")
        print(f"{'='*65}")
        brief_path = str(Path(OUTPUT_DIR) / BRIEF_FILE)
        run_summarizer(articles=articles, query=query, output_path=brief_path)
    else:
        print("\n[summarize] Skipped (--no-summarize).")

    elapsed = round((time.time() - t0) / 60, 1)
    print(f"\n  Total time: {elapsed} minutes")


# =============================================================================
#  CLI
# =============================================================================

def _build_parser():
    p = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="Pharma News Intelligence Pipeline — extraction + merge + summarize.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py
  python run_pipeline.py --query crispr --days 14
  python run_pipeline.py --domain biopharmadive.com --query protac
  python run_pipeline.py --limit 5 --no-enrich
  python run_pipeline.py --no-summarize
        """,
    )
    p.add_argument("--url",    "-u", default=None,   help="Alias for --domain")
    p.add_argument("--query",  "-q", default=QUERY,  help=f"Search query (default: {QUERY!r})")
    p.add_argument("--limit",  "-n", type=int, default=None, help="Cap number of domains")
    p.add_argument("--domain", "-d", default=None,   help="Process only this domain")
    p.add_argument("--days",         type=int, default=DATE_WINDOW_DAYS,
                   help=f"Date filter in days (default: {DATE_WINDOW_DAYS})")
    p.add_argument("--enrich", dest="enrich",
                   action=argparse.BooleanOptionalAction, default=ENRICH_ARTICLES,
                   help="Scrape full article bodies (default: on)")
    p.add_argument("--summarize", dest="summarize",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Run SUMMARIZER after merge (default: on)")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    asyncio.run(run_pipeline(
        url       = args.url,
        query     = args.query,
        days      = args.days,
        enrich    = args.enrich,
        limit     = args.limit,
        domain    = args.domain,
        summarize = args.summarize,
    ))
else:
    print("OK  run_pipeline.py loaded.")
    print()
    print("  EXTRACT + MERGE + SUMMARIZE:")
    print("    await run_pipeline()")
    print("    await run_pipeline(query='crispr', days=14)")
    print("    await run_pipeline(domain='biopharmadive.com')")
    print("    await run_pipeline(query='protac', summarize=False)")
