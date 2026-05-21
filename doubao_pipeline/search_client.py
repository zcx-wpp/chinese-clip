from __future__ import annotations

import argparse
import json
from typing import Any
from urllib import error, request


def parse_args():
    parser = argparse.ArgumentParser(description="Terminal client for the video search API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--query", help="Run one query and exit.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--json", action="store_true", help="Print raw JSON response.")
    return parser.parse_args()


def build_search_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/search"


def search_once(url: str, query_text: str, top_k: int, timeout: float) -> list[dict[str, Any]]:
    payload = json.dumps({"query": query_text, "top_k": top_k}).encode("utf-8")
    req = request.Request(
        url=url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    result = json.loads(body)
    if not isinstance(result, list):
        raise ValueError("API response is not a list.")
    return result


def format_segment(segment: dict[str, Any]) -> str:
    start = float(segment.get("start", 0.0))
    end = float(segment.get("end", 0.0))
    score = float(segment.get("score", 0.0))
    return f"{start:7.2f}s - {end:7.2f}s  score={score:.4f}"


def print_results(query_text: str, results: list[dict[str, Any]]) -> None:
    print("")
    print(f'Query: "{query_text}"')
    print(f"Returned: {len(results)}")
    print("")
    if not results:
        print("No results.")
        return

    for idx, item in enumerate(results, start=1):
        video_id = str(item.get("video_id", ""))
        score = float(item.get("score", 0.0))
        video_path = str(item.get("video_path", ""))
        tags = item.get("tags") or []
        description = str(item.get("description") or item.get("caption") or "")
        print(f"[{idx:02d}] {video_id}")
        print(f"     score: {score:.4f}")
        print(f"     path : {video_path}")
        if tags:
            print(f"     tags : {', '.join(str(tag) for tag in tags)}")
        if description:
            print(f"     desc : {description}")
        segments = item.get("segments") or []
        if segments:
            print("     segments:")
            for segment in segments:
                print(f"       - {format_segment(segment)}")
        else:
            print("     segments: none")
        print("")


def run_interactive(url: str, top_k: int, timeout: float, raw_json: bool) -> None:
    print("Video search terminal client")
    print("Type a query and press Enter. Type exit or quit to stop.")
    print("")
    while True:
        try:
            query_text = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            break
        if not query_text:
            continue
        if query_text.lower() in {"exit", "quit"}:
            break
        try:
            results = search_once(url, query_text, top_k, timeout)
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            print(f"HTTP error {exc.code}: {body}")
            continue
        except error.URLError as exc:
            print(f"Connection error: {exc}")
            continue
        except Exception as exc:
            print(f"Search failed: {exc}")
            continue

        if raw_json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            print_results(query_text, results)


def main():
    args = parse_args()
    url = build_search_url(args.host, args.port)
    if args.query:
        try:
            results = search_once(url, args.query, args.top_k, args.timeout)
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise SystemExit(f"HTTP error {exc.code}: {body}") from exc
        except error.URLError as exc:
            raise SystemExit(f"Connection error: {exc}") from exc

        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            print_results(args.query, results)
        return

    run_interactive(url, args.top_k, args.timeout, args.json)


if __name__ == "__main__":
    main()
