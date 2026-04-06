"""
pharma_server.py — Pharma Intelligence Dashboard Backend
Reads brief_*.json files pushed by GitHub Actions and serves them via API
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# ─── CONFIG ──────────────────────────────────────────────────────────────────

STATIC_DIR    = Path("static")
FRONTEND_FILE = STATIC_DIR / "index.html"

MODALITIES = [
    "bispecific_antibodies",
    "monoclonal_antibodies",
    "molecular_glues",
    "gene_editing",
]

MODALITY_LABELS = {
    "bispecific_antibodies": "Bispecific Antibodies",
    "monoclonal_antibodies": "Monoclonal Antibodies",
    "molecular_glues":       "Molecular Glues",
    "gene_editing":          "Gene Editing",
}

MODALITY_QUERIES = {
    "bispecific_antibodies": "bispecific antibody BiTE T-cell engager CrossMab",
    "monoclonal_antibodies": "monoclonal antibody mAb therapeutic humanized",
    "molecular_glues":       "molecular glue PROTAC TPD degrader E3 ligase",
    "gene_editing":          "CRISPR gene editing Cas9 base editing prime editing",
}

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def load_brief(modality: str):
    path = Path(f"brief_{modality}.json")
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def get_today_modality() -> str:
    today = datetime.today().date()
    return MODALITIES[today.toordinal() % len(MODALITIES)]


def get_next_run_date(modality: str) -> str:
    today = datetime.today().date()
    for i in range(1, 5):
        d   = today + timedelta(days=i)
        idx = d.toordinal() % len(MODALITIES)
        if MODALITIES[idx] == modality:
            return d.isoformat()
    return "Unknown"


# ─── APP ─────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("✅ Pharma Intelligence server started")
    yield

app = FastAPI(title="Pharma Intelligence", lifespan=lifespan)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ─── API ROUTES ──────────────────────────────────────────────────────────────

@app.get("/api/modalities")
def get_modalities():
    today_mod = get_today_modality()
    result = []
    for key in MODALITIES:
        brief = load_brief(key)
        result.append({
            "key":           key,
            "label":         MODALITY_LABELS[key],
            "query":         MODALITY_QUERIES[key],
            "is_today":      key == today_mod,
            "next_run":      get_next_run_date(key),
            "available":     brief is not None,
            "generated_at":  brief.get("generated_at")  if brief else None,
            "article_count": brief.get("article_count", 0) if brief else 0,
            "section_count": len(brief.get("sections", [])) if brief else 0,
        })
    return result


@app.get("/api/modalities/{modality}")
def get_modality_brief(modality: str):
    if modality not in MODALITIES:
        raise HTTPException(404, "Unknown modality")
    brief = load_brief(modality)
    if not brief:
        return {
            "modality":  modality,
            "label":     MODALITY_LABELS[modality],
            "available": False,
            "next_run":  get_next_run_date(modality),
            "sections":  [],
        }
    return {
        "modality":      modality,
        "label":         MODALITY_LABELS[modality],
        "available":     True,
        "query":         brief.get("query", ""),
        "generated_at":  brief.get("generated_at", ""),
        "article_count": brief.get("article_count", 0),
        "sections":      brief.get("sections", []),
    }


@app.get("/api/stats")
def get_stats():
    today_mod  = get_today_modality()
    available  = [m for m in MODALITIES if load_brief(m) is not None]
    return {
        "today_modality":  today_mod,
        "today_label":     MODALITY_LABELS[today_mod],
        "available_count": len(available),
        "available":       available,
    }


# ─── SPA CATCH-ALL ───────────────────────────────────────────────────────────

@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    if FRONTEND_FILE.exists():
        return FileResponse(str(FRONTEND_FILE))
    return HTMLResponse("<h1>Frontend not found</h1>", 503)
