from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run the minimal video retrieval demo flow.")
    parser.add_argument("--video-dir", required=True, help="Directory containing source videos.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated pipeline output.")
    parser.add_argument("--metadata-db", required=True, help="SQLite metadata DB path.")
    parser.add_argument("--model-path", required=True, help="Local Chinese-CLIP model directory.")
    parser.add_argument("--labels", required=True, help="Evaluation labels JSON.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N videos. 0 means all.")
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
    package_name = __package__ or "project.src.video_processing"

    if not args.skip_validate:
        run_command(module_command(python_exe, f"{package_name}.validate_env", "--model-path", args.model_path))

    if not args.skip_index:
        index_command = module_command(
            python_exe,
            f"{package_name}.minimal_pipeline",
            "--video-dir",
            args.video_dir,
            "--output-dir",
            args.output_dir,
            "--metadata-db",
            args.metadata_db,
            "--model-path",
            args.model_path,
            "--device",
            args.device,
        )
        if args.limit > 0:
            index_command.extend(["--limit", str(args.limit)])
        run_command(index_command)

    if not args.skip_eval:
        eval_command = module_command(
            python_exe,
            f"{package_name}.evaluate",
            "--output-dir",
            args.output_dir,
            "--metadata-db",
            args.metadata_db,
            "--model-path",
            args.model_path,
            "--labels",
            args.labels,
            "--device",
            args.device,
        )
        run_command(eval_command)

    print("")
    print("Demo flow finished.")
    print(f"Workspace: {root}")
    print(f"Output dir: {Path(args.output_dir).resolve()}")
    print(f"Metadata DB: {Path(args.metadata_db).resolve()}")


if __name__ == "__main__":
    main()
