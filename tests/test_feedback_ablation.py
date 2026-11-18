"""Regression checks for the treatment contrast and partial-collision scoring."""

from dataclasses import dataclass
import json
from pathlib import Path

import pytest

from edara_mvp_eval.schemas import IdeaGovernanceResult


@dataclass
class Usage:
    calls: int = 1
    cost_usd: float = 0.0


class Runner:
    def __init__(self, response):
        self.response = response
        self.requests = []
        self.usage = Usage()

    def chat_json(self, system, text, **_):
        self.requests.append(json.loads(text))
        return self.response

    def chat_text(self, system, text, **kwargs):
        self.requests.append((system, json.loads(text), kwargs))
        return json.dumps(self.response)


@pytest.fixture
def protocol(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "pilots"))
    import run_feedback_ablation as pilot

    return pilot


def case():
    return {
        "case_id": "controlled", "kind": "partial", "topic": "Research question",
        "cutoff_date": "2025-01-01", "draft": "The old mechanism and a new hypothesis.",
        "evidence": [{
            "id": "prior", "title": "Existing method", "year": "2024-02-01",
            "abstract": "We present the original mechanism.",
        }],
        "evaluation_key": {
            "prior_id": "prior", "covered_mechanism": "only for the scorer", "preserve_residual": "reference hypothesis",
        },
    }


def test_highlight_and_feedback_share_identical_evidence_and_settings(protocol):
    audit = IdeaGovernanceResult.model_validate({
        "action": "correct_attribution",
        "claims": [{
            "role": "component", "status": "covered", "proposal_span": "The old mechanism",
            "rationale": "This mechanism is established.",
            "evidence": [{"paper_id": "prior", "quote": "We present the original mechanism."}],
        }],
    })
    highlights, diagnosis = protocol.treatment(audit)
    source = case()
    task = protocol.task_for(source)
    ordinary, selected, feedback = (Runner({}) for _ in range(3))
    protocol.regenerate_candidate(task, source, ordinary, system=protocol._REVISE)
    protocol.regenerate_candidate(task, source, selected, highlights=highlights, system=protocol._REVISE)
    protocol.regenerate_candidate(task, source, feedback, diagnosis, highlights=highlights, system=protocol._REVISE)
    ordinary_prompt, highlighted_prompt, feedback_prompt = (runner.requests[0] for runner in (ordinary, selected, feedback))
    assert ordinary_prompt[0] == highlighted_prompt[0] == feedback_prompt[0]
    assert ordinary_prompt[2] == highlighted_prompt[2] == feedback_prompt[2]
    assert highlighted_prompt[1] == {**ordinary_prompt[1], "evidence_highlights": highlights}
    assert feedback_prompt[1] == {**highlighted_prompt[1], "diagnosis": diagnosis}
    assert "only for the scorer" not in json.dumps(feedback_prompt)
    assert "quote" not in json.dumps(diagnosis)


def test_partial_success_requires_preserving_the_residual_and_attribution(protocol):
    source = case()
    task = protocol.task_for(source)
    base = {
        "replaces_covered_core": True, "preserves_residual": True, "attributes_covered_mechanism": True,
        "tests_hypothesis": True, "unsupported_claim": False, "rationale": "Reference contrast tested.",
        "prior_paper_ids": ["prior"],
    }
    runner = Runner(base)
    generation = {"idea": {"title": "Alternative"}}
    assert protocol.evaluate(task, source, generation, runner)["success"]
    assert "diagnosis" not in runner.requests[0]
    for field in ("preserves_residual", "attributes_covered_mechanism", "tests_hypothesis"):
        assert not protocol.evaluate(task, source, generation, Runner({**base, field: False}))["success"]
    assert protocol.evaluate(task, {**source, "kind": "collision"}, generation,
                             Runner({**base, "preserves_residual": False}))["success"]
