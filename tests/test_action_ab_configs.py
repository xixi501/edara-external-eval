"""Offline unit tests for Action A (Cross-Backbone) & Action B (Slow-Thinking 8192) configurations."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

from edara_mvp_eval.llm_client import LLMRunner, ModelEndpoint, extract_json_dict
from edara_mvp_eval.paired_trial import TrialRunner

PILOTS_DIR = Path(__file__).resolve().parents[1] / "pilots"
if str(PILOTS_DIR) not in sys.path:
    sys.path.insert(0, str(PILOTS_DIR))

import run_action_a_cross_backbone as action_a
import run_action_b_thinking_ablation as action_b


def test_extract_json_dict_strips_inline_think_block() -> None:
    raw = "<think>\nLet me check the numbers carefully: {not_final: 1}\n</think>\n{\"winner\": \"A\", \"ok\": true}"
    parsed = extract_json_dict(raw)
    assert parsed == {"winner": "A", "ok": True}


def test_llm_runner_thinking_allowance_and_enable_thinking_false(monkeypatch) -> None:
    captured: list[dict] = []

    class DummyCompletions:
        def create(self, **kwargs):
            captured.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
                usage=SimpleNamespace(
                    prompt_tokens=100,
                    completion_tokens=250,
                    completion_tokens_details=SimpleNamespace(reasoning_tokens=180),
                ),
            )

    dummy_client = SimpleNamespace(chat=SimpleNamespace(completions=DummyCompletions()))

    # 1) Qwen enable_thinking=False -> thinking_disabled=True -> max_tokens stays exact (4096)
    ep_qwen_fast = ModelEndpoint(model="qwen3.8-flash", extra_body={"enable_thinking": False})
    r_fast = LLMRunner(ep_qwen_fast)
    monkeypatch.setattr(r_fast, "get_openai_client", lambda: dummy_client)
    r_fast.chat_text("sys", "usr", max_tokens=4096)
    assert captured[-1]["max_tokens"] == 4096

    # 2) DeepSeek thinking enabled + output_tokens=8192 -> effective_max_tokens=16384 & reasoning_tokens metered
    ep_ds_think = ModelEndpoint(model="deepseek-v4.1-flash-1017", extra_body={"thinking": {"type": "enabled"}})
    r_think = LLMRunner(ep_ds_think)
    monkeypatch.setattr(r_think, "get_openai_client", lambda: dummy_client)
    r_think.chat_text("sys", "usr", max_tokens=8192)
    assert captured[-1]["max_tokens"] == 16384
    assert r_think.usage.reasoning_tokens == 180


def test_trial_runner_custom_output_tokens_cap(tmp_path: Path) -> None:
    ep = ModelEndpoint(model="dry-run")
    tr = TrialRunner(ep, tmp_path / "logs", task_id="t1", arm="E", stage="editor", limit=4, output_tokens=8192, dry_run=True)
    out = tr.chat_text("sys", "usr", max_tokens=12000, tag="paired_edit")
    assert out
    rec = json.loads((tmp_path / "logs" / "0001.json").read_text(encoding="utf-8"))
    assert rec["status"] == "ok"


def test_action_a_and_action_b_dry_run_end_to_end(tmp_path: Path) -> None:
    out_a = tmp_path / "action_a"
    rc_a = action_a.main(["--out", str(out_a), "--limit-topics", "1", "--workers", "2", "--dry-run"])
    assert rc_a == 0
    sum_a = json.loads((out_a / "action_a_cross_backbone_summary.json").read_text(encoding="utf-8"))
    assert sum_a["n_manuscripts"] == 3
    assert "independent_2judge" in sum_a["judge_summaries"]

    out_b = tmp_path / "action_b"
    rc_b = action_b.main(["--out", str(out_b), "--limit-topics", "1", "--workers", "2", "--dry-run"])
    assert rc_b == 0
    sum_b = json.loads((out_b / "action_b_thinking_ablation_summary.json").read_text(encoding="utf-8"))
    assert sum_b["n_manuscripts"] == 3
    assert sum_b["output_tokens"] == 8192
    assert sum_b["direct_think_generation_probe"]["n_topics"] == 1
