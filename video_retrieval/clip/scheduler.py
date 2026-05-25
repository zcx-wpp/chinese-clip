from __future__ import annotations

from pathlib import Path

from ..logging_utils import utc_now_iso
from .metadata_store import MetadataStore

VIDEO_STAGES = ("segment", "frame_extract", "embedding", "topk")
GLOBAL_FAISS_VIDEO_ID = "__global__"
GLOBAL_FAISS_STAGE = "faiss"
DEFAULT_STAGE_MAX_RETRIES = {
    "segment": 2,
    "frame_extract": 2,
    "embedding": 3,
    "topk": 2,
    "faiss": 3,
}

DOWNSTREAM_STAGE_MAP = {
    "segment": ("frame_extract", "embedding", "topk"),
    "frame_extract": ("embedding", "topk"),
    "embedding": ("topk",),
    "topk": (),
    "faiss": (),
}


class PipelineScheduler:
    def __init__(self, store: MetadataStore, stage_max_retries: dict[str, int] | None = None):
        self.store = store
        self.stage_max_retries = {**DEFAULT_STAGE_MAX_RETRIES, **(stage_max_retries or {})}

    def _now(self) -> str:
        return utc_now_iso()

    def _update_task(
        self,
        video_id: str,
        stage: str,
        status: str,
        *,
        error_message: str | None = None,
        increment_retry: bool = False,
    ):
        self.store.update_task(
            video_id,
            stage,
            status,
            updated_at=self._now(),
            error_message=error_message,
            increment_retry=increment_retry,
        )

    def _mark_video(self, video_id: str, status: str, *, error_message: str | None = None):
        self.store.mark_video_status(
            video_id, status=status, error_message=error_message, updated_at=self._now()
        )

    def ensure_video(self, video_id: str, path: str, duration: float):
        if self.store.get_video_status(video_id) is None:
            self.store.upsert_video(
                video_id=video_id,
                duration=duration,
                path=path,
                status="pending",
                error_message=None,
                updated_at=self._now(),
            )
        self.store.ensure_video_tasks(
            video_id,
            VIDEO_STAGES,
            max_retries=self.stage_max_retries,
            updated_at=self._now(),
        )

    def ensure_global_tasks(self):
        self.store.ensure_task(
            GLOBAL_FAISS_VIDEO_ID,
            GLOBAL_FAISS_STAGE,
            max_retry=self.stage_max_retries.get(GLOBAL_FAISS_STAGE, 3),
            updated_at=self._now(),
        )

    def start_video(self, video_id: str, path: str, duration: float):
        self.store.upsert_video(
            video_id=video_id,
            duration=duration,
            path=path,
            status="processing",
            error_message=None,
            updated_at=self._now(),
        )

    def complete_video(self, video_id: str):
        self._mark_video(video_id, "done")

    def fail_video(self, video_id: str, error_message: str):
        self._mark_video(video_id, "failed", error_message=error_message)

    def task_status(self, video_id: str, stage: str) -> str | None:
        return self.store.get_task_status(video_id, stage)

    def task_record(self, video_id: str, stage: str) -> dict | None:
        return self.store.get_task(video_id, stage)

    def can_retry_task(self, video_id: str, stage: str) -> bool:
        task = self.task_record(video_id, stage)
        if task is None:
            return True
        max_retry = int(task.get("max_retry") or self.stage_max_retries.get(stage, 2))
        retry_count = int(task.get("retry_count") or 0)
        return retry_count < max_retry

    def should_skip_task(self, video_id: str, stage: str) -> bool:
        task = self.task_record(video_id, stage)
        if task is None:
            return False
        if task.get("status") != "failed":
            return False
        return not self.can_retry_task(video_id, stage)

    def start_task(self, video_id: str, stage: str):
        if self.should_skip_task(video_id, stage):
            raise RuntimeError(
                f"task retry exhausted: video_id={video_id} stage={stage} "
                f"retry_count={self.task_record(video_id, stage).get('retry_count')} "
                f"max_retry={self.task_record(video_id, stage).get('max_retry')}"
            )
        self._update_task(video_id, stage, "processing")

    def complete_task(self, video_id: str, stage: str):
        self._update_task(video_id, stage, "done")

    def fail_task(self, video_id: str, stage: str, error_message: str):
        self._update_task(
            video_id, stage, "failed", error_message=error_message, increment_retry=True
        )

    def reset_downstream_tasks(self, video_id: str, stages: tuple[str, ...] | list[str]):
        self.store.reset_video_tasks(video_id, stages, updated_at=self._now())

    def reset_task(self, video_id: str, stage: str, include_downstream: bool = True):
        targets = [stage]
        if include_downstream:
            targets.extend(DOWNSTREAM_STAGE_MAP.get(stage, ()))
        for target_stage in targets:
            self.store.ensure_task(
                video_id=video_id,
                stage=target_stage,
                max_retry=self.stage_max_retries.get(target_stage, 2),
                updated_at=self._now(),
            )
            self.store.conn.execute(
                """
                UPDATE tasks
                SET status = 'pending',
                    retry_count = 0,
                    updated_at = ?,
                    error_message = NULL
                WHERE video_id = ? AND stage = ?
                """,
                (self._now(), video_id, target_stage),
            )
        self.store.conn.commit()

    def reset_video_status(self, video_id: str):
        self._mark_video(video_id, "pending")

    def segment_outputs_ready(self, segments_dir: Path, video_id: str) -> bool:
        return any((segments_dir / video_id).glob(f"{video_id}_seg_*.mp4"))

    def frame_outputs_ready(
        self, frames_dir: Path, video_id: str, segment_ids: list[str], image_format: str
    ) -> bool:
        if not segment_ids:
            return False
        for segment_id in segment_ids:
            if not any((frames_dir / video_id / segment_id).glob(f"*.{image_format}")):
                return False
        return True
