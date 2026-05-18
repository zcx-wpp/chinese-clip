from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run the minimal video retrieval demo flow.")
    parser.add_argument("--video-dir", required=True, help="Directory containing source videos.")
    parser.add_argument("--work-dir", required=True, help="Directory for generated index files.")
    parser.add_argument("--model-path", required=True, help="Local Chinese-CLIP model directory.")
    parser.add_argument("--labels", required=True, help="Evaluation labels JSON.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vector-backend", choices=["faiss", "milvus"], default="faiss")
    parser.add_argument("--milvus-uri", default="http://127.0.0.1:19530")
    parser.add_argument("--milvus-token", default="")
    parser.add_argument("--milvus-collection", default="video_frame_embeddings")
    parser.add_argument("--enable-ocr", action="store_true")
    parser.add_argument("--enable-asr", action="store_true")
    parser.add_argument("--skip-index", action="store_true", help="Skip offline indexing if index already exists.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation.")
    parser.add_argument("--skip-validate", action="store_true", help="Skip environment validation.")
    return parser.parse_args()


def run_command(command: list[str]):
    print("")
    print("Running:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def module_command(python_exe: str, module: str, *args: str) -> list[str]:
    return [
        python_exe,
        "-m",
        module,
        *args,
    ]


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    python_exe = sys.executable

    if not args.skip_validate:
        run_command(module_command(python_exe, "video_processing.validate_env", "--model-path", args.model_path))

    if not args.skip_index:
        index_command = module_command(
            python_exe,
            "video_processing.offline_pipeline",
            "--video-dir",
            args.video_dir,
            "--work-dir",
            args.work_dir,
            "--model-path",
            args.model_path,
            "--device",
            args.device,
            "--vector-backend",
            args.vector_backend,
            "--milvus-uri",
            args.milvus_uri,
            "--milvus-token",
            args.milvus_token,
            "--milvus-collection",
            args.milvus_collection,
        )
        if args.enable_ocr:
            index_command.append("--enable-ocr")
        if args.enable_asr:
            index_command.append("--enable-asr")
        run_command(index_command)

    if not args.skip_eval:
        eval_command = module_command(
            python_exe,
            "video_processing.evaluate",
            "--work-dir",
            args.work_dir,
            "--model-path",
            args.model_path,
            "--labels",
            args.labels,
            "--device",
            args.device,
            "--vector-backend",
            args.vector_backend,
            "--milvus-uri",
            args.milvus_uri,
            "--milvus-token",
            args.milvus_token,
            "--milvus-collection",
            args.milvus_collection,
        )
        run_command(eval_command)

    print("")
    print("Demo flow finished.")
    print(f"Workspace: {root}")
    print(f"Index dir: {Path(args.work_dir).resolve()}")


if __name__ == "__main__":
    main()
