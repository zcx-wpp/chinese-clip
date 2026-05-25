from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .config import (
    DEFAULT_BGE_MODEL_NAME,
    DEFAULT_IMAGE_DIR,
    INDEX_KIND_CAPTION,
    PICTURE_BGE_MODEL_ENV,
)
from .env_loader import env_first, load_default_dotenv_files
from .image_io import image_id_from_path, iter_images, validate_image_decodable
from .index_build import sync_faiss_index
from .mllm_caption import ImageMllmCaptioner, resolve_mllm_config
from .profile_paths import default_caption_metadata_db_path, default_output_dir, resolve_path
from .text_metadata_store import CaptionMetadataStore
from .vector_utils import l2_normalize


def _should_caption(status: str | None, *, force: bool, retry_failed: bool) -> bool:
    if force:
        return True
    if status is None:
        return True
    if status in {"caption_done", "done"}:
        return False
    if status == "failed":
        return retry_failed
    if status == "captioning":
        return True
    return True


def _run_mllm_phase(
    *,
    captioner: ImageMllmCaptioner,
    store: CaptionMetadataStore,
    image_dir: Path,
    paths: list[Path],
    force: bool,
    retry_failed: bool,
    workers: int,
) -> None:
    jobs: list[tuple[int, int, str, Path, str]] = []
    total = len(paths)
    for n, path in enumerate(paths, start=1):
        iid = image_id_from_path(path, image_dir)
        status = store.get_status(iid)
        if not _should_caption(status, force=force, retry_failed=retry_failed):
            continue
        rel = path.relative_to(image_dir).as_posix()
        ok, err, _ = validate_image_decodable(path)
        if not ok:
            store.mark_failed(iid, rel, err or "bad image")
            print(f"[failed] ({n}/{total}) {iid}: {err}", flush=True)
            continue
        jobs.append((n, total, iid, path, rel))

    if not jobs:
        print("[mllm] no pending images (all captioned or done)", flush=True)
        return

    print(
        f"[mllm] pending={len(jobs)} workers={workers} (API may take 10–120s per image)", flush=True
    )

    def _one(job: tuple[int, int, str, Path, str]) -> tuple[str, str | None, str | None]:
        n, tot, iid, path, rel = job
        t0 = time.perf_counter()
        try:
            store.mark_captioning(iid, rel)
            structured = captioner.caption_image_path(path)
            text = structured.to_embedding_text()
            store.upsert_caption(
                image_id=iid,
                path=rel,
                subject=structured.subject,
                color=structured.color,
                action=structured.action,
                style=structured.style,
                description=structured.description,
                caption_text=text,
                structured_json=json.dumps(structured.to_dict(), ensure_ascii=False),
                caption_model=captioner.config.model,
                status="caption_done",
            )
            elapsed = time.perf_counter() - t0
            line = f"[caption] ({n}/{tot}) {iid} {elapsed:.1f}s {structured.to_display_line()}"
            return iid, line, None
        except Exception as exc:
            store.mark_failed(iid, rel, str(exc))
            return iid, None, f"[failed] ({n}/{tot}) {iid}: {exc}"

    if workers <= 1:
        for job in jobs:
            _n, _t, iid, _p, _r = job
            print(f"[mllm] ({_n}/{_t}) {iid} calling API ...", flush=True)
            _, ok_line, err_line = _one(job)
            if ok_line:
                print(ok_line, flush=True)
            if err_line:
                print(err_line, flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_one, job): job for job in jobs}
            for fut in as_completed(futures):
                job = futures[fut]
                print(f"[mllm] done {job[2]}", flush=True)
                _, ok_line, err_line = fut.result()
                if ok_line:
                    print(ok_line, flush=True)
                if err_line:
                    print(err_line, flush=True)


def _run_bge_phase(
    *,
    store: CaptionMetadataStore,
    out_dir: Path,
    embedder,
    rebuild_faiss: bool,
) -> None:
    rows = store.list_needing_bge_embedding()
    if not rows:
        print("[bge] no rows need BGE embedding", flush=True)
        return

    texts = [str(r["caption_text"]) for r in rows]
    ids = [str(r["image_id"]) for r in rows]
    print(f"[bge] encoding {len(texts)} passages ...", flush=True)
    vectors = embedder.encode_passages(texts, progress_desc="BGE")
    new_ids: list[str] = []
    for iid, vec in zip(ids, vectors, strict=False):
        store.upsert_embedding(
            iid, embedding=l2_normalize(vec), embedding_model=embedder.model_name
        )
        new_ids.append(iid)
    print(f"[bge] embedded {len(new_ids)}", flush=True)

    done = store.list_done_with_embeddings()
    if not done:
        return
    all_ids = [x[0] for x in done]
    mat = np.vstack([l2_normalize(x[1]) for x in done]).astype(np.float32)
    mode = sync_faiss_index(
        faiss_dir=out_dir / "faiss",
        index_filename="caption_index.faiss",
        meta_filename="caption_index.meta.json",
        index_kind=INDEX_KIND_CAPTION,
        dim=embedder.vector_dim,
        model_name=embedder.model_name,
        all_ids=all_ids,
        all_vectors=mat,
        new_ids=new_ids,
        force_full_rebuild=rebuild_faiss,
        manifest_path=out_dir / "caption_index_manifest.json",
    )
    print(f"[faiss] mode={mode} count={len(all_ids)}", flush=True)


def main():
    p = argparse.ArgumentParser(description="MLLM caption -> BGE -> FAISS for pictures.")
    p.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    p.add_argument("--output-dir")
    p.add_argument("--metadata-db")
    p.add_argument("--profile")
    p.add_argument("--force", action="store_true")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument(
        "--skip-mllm", action="store_true", help="Only BGE+faiss from existing caption_done."
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--workers", type=int, default=2, help="Parallel MLLM API calls (mind rate limits)."
    )
    p.add_argument("--rebuild-faiss", action="store_true")
    p.add_argument("--bge-device", default="cuda")
    args = p.parse_args()
    load_default_dotenv_files()

    image_dir = Path(args.image_dir).resolve()
    out_dir = resolve_path(args.output_dir, default_output_dir(args.profile))
    db = resolve_path(args.metadata_db, default_caption_metadata_db_path(args.profile))
    store = CaptionMetadataStore(db)

    paths = list(iter_images(image_dir))
    if args.limit > 0:
        paths = paths[: args.limit]
    print(f"[text_ingest] images={len(paths)} profile={args.profile}", flush=True)

    if not args.skip_mllm:
        cfg = resolve_mllm_config()
        captioner = ImageMllmCaptioner(cfg)
        print(f"[text_ingest] phase 1/2: MLLM model={cfg.model}", flush=True)
        _run_mllm_phase(
            captioner=captioner,
            store=store,
            image_dir=image_dir,
            paths=paths,
            force=args.force,
            retry_failed=args.retry_failed,
            workers=max(1, args.workers),
        )

    print("[text_ingest] phase 2/2: loading BGE (after MLLM)", flush=True)
    from video_retrieval.hybrid.dense_embeddings import HuggingFaceBgeTextEmbedder

    embedder = HuggingFaceBgeTextEmbedder(
        model_name=env_first(PICTURE_BGE_MODEL_ENV) or DEFAULT_BGE_MODEL_NAME,
        device=args.bge_device,
        local_files_only=bool(os.environ.get("HF_LOCAL_FILES_ONLY")),
    )
    embedder.save(out_dir / "bge_embedder")
    _run_bge_phase(
        store=store, out_dir=out_dir, embedder=embedder, rebuild_faiss=args.rebuild_faiss
    )

    store.close()
    print("[text_ingest] all done", flush=True)


if __name__ == "__main__":
    main()
