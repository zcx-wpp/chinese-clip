from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import DEFAULT_IMAGE_DIR, DEFAULT_MODEL_PATH, INDEX_KIND
from .encoder import ChineseClipEncoder, encode_image_path_pooled
from .image_io import image_id_from_path, iter_images, safe_embedding_filename, validate_image_decodable
from .index_build import sync_faiss_index
from .metadata_store import PictureMetadataStore
from .profile_paths import default_metadata_db_path, default_output_dir, resolve_path
from .vector_utils import l2_normalize


def parse_args():
    p = argparse.ArgumentParser(description="Index images with Chinese-CLIP.")
    p.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    p.add_argument("--output-dir")
    p.add_argument("--metadata-db")
    p.add_argument("--profile")
    p.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--rebuild-faiss", action="store_true")
    return p.parse_args()


def _collect_done(store, output_dir):
    ids, rows = [], []
    for row in store.list_images(status="done"):
        rel = row.get("embedding_path")
        if not rel:
            continue
        f = output_dir / rel
        if f.exists():
            ids.append(row["image_id"])
            rows.append(l2_normalize(np.load(f).astype(np.float32)))
    if not rows:
        return [], np.zeros((0, 0), dtype=np.float32)
    return ids, np.vstack(rows)


def main():
    args = parse_args()
    image_dir = Path(args.image_dir).resolve()
    if not image_dir.exists():
        raise SystemExit(f"image directory not found: {image_dir}")

    output_dir = resolve_path(args.output_dir, default_output_dir(args.profile))
    db_path = resolve_path(args.metadata_db, default_metadata_db_path(args.profile))
    emb_dir = output_dir / "embeddings"
    faiss_dir = output_dir / "faiss"
    emb_dir.mkdir(parents=True, exist_ok=True)

    paths = list(iter_images(image_dir))
    if args.limit > 0:
        paths = paths[: args.limit]

    store = PictureMetadataStore(db_path)
    encoder = ChineseClipEncoder(model_path=args.model_path, device=args.device, batch_size=args.batch_size)

    pending = []
    for p in paths:
        iid = image_id_from_path(p, image_dir)
        if not args.force and store.get_status(iid) == "done":
            continue
        pending.append((iid, p))

    print(f"[ingest] pending={len(pending)} total={len(paths)}", flush=True)
    new_ids = []
    for iid, path in pending:
        rel = path.relative_to(image_dir).as_posix()
        ok, err, size = validate_image_decodable(path)
        if not ok:
            store.mark_failed(iid, rel, err or "decode failed")
            print(f"[failed] {iid}: {err}", flush=True)
            continue
        try:
            vector, norm = encode_image_path_pooled(encoder, path)
            vector = l2_normalize(vector)
            emb_name = safe_embedding_filename(iid) + ".npy"
            np.save(emb_dir / emb_name, vector)
            (emb_dir / (safe_embedding_filename(iid) + ".json")).write_text(
                json.dumps({"image_id": iid, "path": rel, "index_kind": INDEX_KIND}, ensure_ascii=False),
                encoding="utf-8",
            )
            w, h = size or (None, None)
            store.upsert_image(
                image_id=iid, path=rel, width=w, height=h,
                embedding_path=str((emb_dir / emb_name).relative_to(output_dir).as_posix()),
                embedding_norm=float(norm), status="done",
            )
            new_ids.append(iid)
            print(f"[done] {iid}", flush=True)
        except Exception as exc:
            store.mark_failed(iid, rel, str(exc))
            print(f"[failed] {iid}: {exc}", flush=True)

    all_ids, matrix = _collect_done(store, output_dir)
    if matrix.size == 0:
        print("[ingest] no embeddings", flush=True)
        store.close()
        return

    mode = sync_faiss_index(
        faiss_dir=faiss_dir,
        index_filename="image_index.faiss",
        meta_filename="image_index.meta.json",
        index_kind=INDEX_KIND,
        dim=encoder.embedding_dim,
        model_name=encoder.model_name,
        all_ids=all_ids,
        all_vectors=matrix,
        new_ids=new_ids,
        force_full_rebuild=bool(args.rebuild_faiss or (args.force and pending)),
        manifest_path=output_dir / "index_manifest.json",
        extra_manifest={"modality": "image", "model_path": encoder.model_source_path},
    )
    store.close()
    print(f"[ingest] faiss mode={mode} count={len(all_ids)}", flush=True)


if __name__ == "__main__":
    main()
