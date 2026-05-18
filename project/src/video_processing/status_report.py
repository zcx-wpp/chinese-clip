from __future__ import annotations

import argparse
from pathlib import Path

from .config import PROJECT_ROOT
from .metadata_store import MetadataStore


STATUS_KEYS = ("pending", "processing", "done", "failed")


def parse_args():
    parser = argparse.ArgumentParser(description="Show video processing status from metadata.db.")
    parser.add_argument("--metadata-db", default=str(PROJECT_ROOT / "metadata" / "metadata.db"))
    parser.add_argument("--view", choices=["videos", "tasks"], default="videos")
    parser.add_argument("--status", choices=["pending", "processing", "done", "failed", "all"], default="all")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--show-errors", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def fetch_summary(store: MetadataStore) -> list[tuple[str, int]]:
    rows = store.conn.execute(
        """
        SELECT COALESCE(status, 'pending') AS status, COUNT(*) AS count
        FROM videos
        GROUP BY COALESCE(status, 'pending')
        ORDER BY status
        """
    ).fetchall()
    return [(row["status"], row["count"]) for row in rows]


def fetch_rows(store: MetadataStore, status: str, limit: int) -> list[dict]:
    if status == "all":
        rows = store.conn.execute(
            """
            SELECT video_id, path, duration, status, error_message, updated_at
            FROM videos
            ORDER BY
                CASE status
                    WHEN 'failed' THEN 0
                    WHEN 'processing' THEN 1
                    WHEN 'pending' THEN 2
                    WHEN 'done' THEN 3
                    ELSE 4
                END,
                updated_at DESC,
                video_id ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    else:
        rows = store.conn.execute(
            """
            SELECT video_id, path, duration, status, error_message, updated_at
            FROM videos
            WHERE COALESCE(status, 'pending') = ?
            ORDER BY updated_at DESC, video_id ASC
            LIMIT ?
            """,
            (status, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def fetch_task_summary(store: MetadataStore) -> list[tuple[str, str, int]]:
    rows = store.conn.execute(
        """
        SELECT stage, COALESCE(status, 'pending') AS status, COUNT(*) AS count
        FROM tasks
        GROUP BY stage, COALESCE(status, 'pending')
        ORDER BY stage, status
        """
    ).fetchall()
    return [(row["stage"], row["status"], row["count"]) for row in rows]


def fetch_task_rows(store: MetadataStore, status: str, limit: int) -> list[dict]:
    if status == "all":
        rows = store.conn.execute(
            """
            SELECT task_id, video_id, stage, status, max_retry, retry_count, updated_at, error_message
            FROM tasks
            ORDER BY stage ASC, updated_at DESC, video_id ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    else:
        rows = store.conn.execute(
            """
            SELECT task_id, video_id, stage, status, max_retry, retry_count, updated_at, error_message
            FROM tasks
            WHERE COALESCE(status, 'pending') = ?
            ORDER BY stage ASC, updated_at DESC, video_id ASC
            LIMIT ?
            """,
            (status, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def print_status_totals(pairs: list[tuple[str, int]]):
    summary_map = dict(pairs)
    print(
        "pending={pending} processing={processing} done={done} failed={failed}".format(
            **{key: summary_map.get(key, 0) for key in STATUS_KEYS}
        )
    )


def print_grouped_status_totals(rows: list[tuple[str, str, int]]):
    buckets = {key: 0 for key in STATUS_KEYS}
    for _, status, count in rows:
        buckets[status] = buckets.get(status, 0) + count
    print(
        "pending={pending} processing={processing} done={done} failed={failed}".format(**buckets)
    )


def main():
    args = parse_args()
    store = MetadataStore(Path(args.metadata_db))

    if args.view == "tasks":
        summary_rows = fetch_task_summary(store)
        if args.summary_only:
            print_grouped_status_totals(summary_rows)
            return

        print("Task Summary")
        for stage, status, count in summary_rows:
            print(f"{stage} | {status}: {count}")

        print("")
        print(f"Task Details status={args.status} limit={args.limit}")
        rows = fetch_task_rows(store, status=args.status, limit=args.limit)
        if not rows:
            print("No rows.")
            return

        for row in rows:
            print(
                f"{row['video_id']} | stage={row['stage']} | status={row['status']} | retry={row['retry_count']}/{row['max_retry']} | updated_at={row['updated_at']}"
            )
            if args.show_errors and row["error_message"]:
                print(f"  error: {row['error_message']}")
        return

    summary_rows = fetch_summary(store)
    if args.summary_only:
        print_status_totals(summary_rows)
        return

    print("Summary")
    for status, count in summary_rows:
        print(f"{status}: {count}")

    print("")
    print(f"Details status={args.status} limit={args.limit}")
    rows = fetch_rows(store, status=args.status, limit=args.limit)
    if not rows:
        print("No rows.")
        return

    for row in rows:
        print(
            f"{row['video_id']} | status={row['status']} | updated_at={row['updated_at']} | path={row['path']}"
        )
        if args.show_errors and row["error_message"]:
            print(f"  error: {row['error_message']}")


if __name__ == "__main__":
    main()
