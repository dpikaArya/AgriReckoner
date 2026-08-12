"""Phase 20 STEP 20 — RAG knowledge-store sync (append-only, idempotent).

Existing embedding ids in outputs/phase14_5a/Embedding_Store/ids.json are
read-only and preserved. Phase 20 only emits NEW incremental chunk metadata +
an idempotent sync manifest under outputs/phase20/rag_sync/; it never rebuilds
or deletes the protected embedding store.
"""

from __future__ import annotations

import pandas as pd

try:
    from . import p20_common as C
except ImportError:
    import p20_common as C


def existing_ids(cfg: dict) -> set:
    p = C.PROJECT_ROOT / cfg.get("rag", {}).get(
        "store", "outputs/phase14_5a/Embedding_Store/ids.json"
    )
    ids = C.read_json(p, {})
    if isinstance(ids, dict):
        return set(ids.keys())
    if isinstance(ids, list):
        return set(str(i) for i in ids)
    return set()


def new_chunks(cfg: dict) -> pd.DataFrame:
    matrix_p = C.OUT / "model_ready_observation_matrix.parquet"
    if not matrix_p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(matrix_p)
    existing = existing_ids(cfg)
    max_chunks = int(cfg.get("rag", {}).get("max_chunks", 200))
    chunk_types = set(cfg.get("rag", {}).get("chunk_types_to_add", []))

    rows = []
    for _, r in df.head(max_chunks).iterrows():
        oid = str(r.get("canonical_observation_id") or r.get("ObservationID_ML") or "")
        if not oid or f"obs::{oid}" in existing:
            continue
        if "model_ready_observation" not in chunk_types:
            continue
        text = (
            f"Phase 20 model-ready observation {oid}: crop "
            f"{r.get('Crop_Normalized')} year {r.get('year')} location "
            f"{r.get('canonical_location_id')}; "
            f"readiness {r.get('QualityGrade')}."
        )
        rows.append(
            {
                "chunk_id": f"obs::{oid}",
                "chunk_type": "model_ready_observation",
                "source": "phase20",
                "text": text,
                "link_observation": oid,
                "sha256": C.sha256_bytes(text.encode("utf-8")),
            }
        )

    ig_p = C.OUT / "information_gain_metrics.json"
    if ig_p.exists() and "linkage_metric" in chunk_types:
        ig = C.read_json(ig_p, {})
        scored = {v: None for v in (ig.get("highest_gain_variables") or [])}
        for name, val in (scored if isinstance(scored, dict) else {}).items():
            cid = f"metric::infogain::{str(name).replace(' ', '_')}"
            if cid in existing:
                continue
            text = f"Phase 20 information gain for predictor {name}: {val}."
            rows.append(
                {
                    "chunk_id": cid,
                    "chunk_type": "linkage_metric",
                    "source": "phase20",
                    "text": text,
                    "link_observation": None,
                    "sha256": C.sha256_bytes(text.encode("utf-8")),
                }
            )

    mr_p = C.OUT / "model_readiness.json"
    if mr_p.exists() and "phase20_metric" in chunk_types:
        mr = C.read_json(mr_p, {})
        for d, v in mr.get("per_target", {}).items():
            if not isinstance(v, dict):
                continue
            cid = f"metric::readiness::{d}"
            if cid in existing:
                continue
            text = (
                f"Phase 20 model readiness for {d}: score {v.get('readiness_score')}, "
                f"observations {v.get('total_observations')}."
            )
            rows.append(
                {
                    "chunk_id": cid,
                    "chunk_type": "phase20_metric",
                    "source": "phase20",
                    "text": text,
                    "link_observation": None,
                    "sha256": C.sha256_bytes(text.encode("utf-8")),
                }
            )
    return pd.DataFrame(rows)


def run(force: bool = False):
    cfg = C.load_config()
    existing = existing_ids(cfg)
    chunks = new_chunks(cfg)
    n_existing = len(existing)
    n_new = len(chunks)

    C.to_parquet(chunks, C.RAG_OUT / "new_chunks.parquet")
    manifest = {
        "generated_at": C.now_full_iso(),
        "store_unchanged": True,
        "existing_ids_count": n_existing,
        "new_chunk_count": n_new,
        "new_chunk_ids": chunks["chunk_id"].tolist() if len(chunks) else [],
        "mode": "append_only",
        "note": "phase14_5a embedding store NOT modified; only manifest emitted",
    }
    C.write_json(manifest, C.RAG_OUT / "rag_sync_manifest.json")
    C.write_json(
        chunks.to_dict("records") if len(chunks) else [], C.RAG_OUT / "rag_sync_chunks.json"
    )
    C.mark_done("rag_sync", manifest)
    C.log_msg(f"STEP20 rag sync: {n_existing} existing, {n_new} new chunks (append-only)")
    return manifest


if __name__ == "__main__":
    import sys

    run(force="--force" in sys.argv)
