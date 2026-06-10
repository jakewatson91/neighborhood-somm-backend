"""Re-embed the wine catalog with gte-small into the new `embedding_v2` column.

The Edge Function `find-wine` embeds the query with Supabase's in-runtime gte-small
model, so the catalog must use the matching model: `thenlper/gte-small` (384-dim,
normalized). The original MiniLM `embedding` column is left untouched.

Run once after the migration:

    cd ~/src/neighbourhood-somm/packages/backend
    source .venv/bin/activate
    python -m src.reembed_gte
"""
import os
import time

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client

load_dotenv()

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_ROLE_KEY"],
)

model = SentenceTransformer("thenlper/gte-small")  # 384-dim, matches the Edge runtime


def search_text(w: dict) -> str:
    """Same recipe build_vector.py used, rebuilt from the stored columns."""
    f = w.get("features") or {}
    pairings = f.get("pairings") or []
    if not isinstance(pairings, list):
        pairings = []
    tags = w.get("tags") or []
    parts = [
        w.get("title", ""),
        w.get("description", ""),
        " ".join(tags) if isinstance(tags, list) else str(tags),
        f.get("grape", ""),
        f.get("type", ""),
        f.get("body", ""),
        f.get("acidity", ""),
        " ".join(pairings),
    ]
    return " ".join(p for p in parts if p).strip()


def main():
    rows = (
        supabase.table("wines")
        .select("id, title, description, tags, features")
        .limit(5000)
        .execute()
        .data
    )
    print(f"Fetched {len(rows)} wines.")

    texts = [search_text(w) for w in rows]
    t0 = time.perf_counter()
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    print(f"Embedded {len(embs)} wines in {time.perf_counter() - t0:.1f}s ({embs.shape[1]}-dim)")
    assert embs.shape[1] == 384, f"expected 384-dim, got {embs.shape[1]}"

    # Per-row UPDATE (not upsert): the table has NOT NULL columns, so a partial
    # upsert would fail the insert path before ON CONFLICT can fire.
    written = 0
    for w, e in zip(rows, embs):
        supabase.table("wines").update({"embedding_v2": e.tolist()}).eq("id", w["id"]).execute()
        written += 1
        if written % 100 == 0:
            print(f"  wrote {written}/{len(rows)}")
    print(f"Done. embedding_v2 written for {written} wines.")


if __name__ == "__main__":
    main()
