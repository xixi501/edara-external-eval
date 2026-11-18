"""Per-request ownership and restartable responses for Stage B."""

import json
from dataclasses import asdict

from .llm_client import LLMRunner
from .stage_b_evidence import digest, write_json


class RecordedRunner(LLMRunner):
    def __init__(self, endpoint, root, *, task_id, arm, stage, dry_run=False):
        super().__init__(endpoint, dry_run=dry_run)
        self.root = root
        self.owner = {"task_id": task_id, "arm": arm, "stage": stage, "model": endpoint.model}
        self.index = 0

    def chat_text(self, system, user, **kwargs):
        self.index += 1
        path = self.root / f"{self.index:04d}.json"
        key = digest([self.owner, system, user, kwargs, self.dry_run])
        if path.exists():
            record = json.loads(path.read_text())
            if record["request_sha256"] != key:
                raise RuntimeError(f"Resume request differs from recorded request at {path}")
            if record["status"] == "error":
                raise RuntimeError(record["error"])
            return record["response"]

        record = {**self.owner, "tag": kwargs.get("tag"), "request_sha256": key,
                  "request_number": self.index}
        before = asdict(self.usage)
        try:
            response = super().chat_text(system, user, **kwargs)
        except Exception as exc:
            record.update(status="error", error=type(exc).__name__, usage=None)
            write_json(path, record)
            raise
        change = {k: v - before[k] for k, v in asdict(self.usage).items()}
        record.update(status="ok", response=response, usage=change)
        write_json(path, record)
        return response
