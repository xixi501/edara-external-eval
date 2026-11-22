"""Offline unit tests for the one-click paper table & SVG figure generator."""
from __future__ import annotations

import json
from pathlib import Path

from edara_mvp_eval.paper_artifacts import generate_paper_artifacts


def test_generate_paper_artifacts_produces_latex_markdown_and_svg(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    w1_dir = results_dir / "benchmark50_wave1_test"
    evo_dir = results_dir / "benchmark50_evosci_wave1_test"
    w1_dir.mkdir(parents=True)
    evo_dir.mkdir(parents=True)

    mh_bundle = {
        "schema_version": "edara.wave-compact-summary/v1",
        "wave": "wave1",
        "cohort_kind": "multihost",
        "n_topics": 2,
        "n_manuscripts": 2,
        "mechanism_stats": {
            "n_manuscripts": 2,
            "E1_orig_keys_kept_all": 2,
            "E2_process_hype": {"A": {"occurrences": 2}, "D": {"occurrences": 4}, "E": {"occurrences": 0}},
            "E2_unverified_entities": {"A": {"total": 10}, "D": {"total": 12}, "E": {"total": 3}},
            "E3_confounds": {"targets_required_n": 1, "D_addressed": 0, "E_addressed": 1},
        },
        "usage": {"E": {"cost_usd": 0.25}, "blind": {"cost_usd": 0.50}},
        "manuscripts": [
            {
                "wave": "wave1",
                "task_id": "survey_01_llm_reasoning_cot",
                "host": "ai_scientist",
                "e_status": "changed",
                "e_actions": ["guided_revise"],
                "judges": {
                    "qwen3.8-flash": {"status": "win", "score_sum_delta": 3.0, "dimension_deltas": {}},
                    "deepseek-v4.1-flash": {"status": "win", "score_sum_delta": 2.5, "dimension_deltas": {}},
                    "GLM-5.3-flash": {"status": "win", "score_sum_delta": 2.0, "dimension_deltas": {}},
                    "majority": {"status": "win", "score_sum_delta": 2.5, "dimension_deltas": {}},
                },
            },
            {
                "wave": "wave1",
                "task_id": "survey_02_vlm_hallucination",
                "host": "direct",
                "e_status": "changed",
                "e_actions": ["guided_revise"],
                "judges": {
                    "qwen3.8-flash": {"status": "tie", "score_sum_delta": 0.5, "dimension_deltas": {}},
                    "deepseek-v4.1-flash": {"status": "tie", "score_sum_delta": 0.0, "dimension_deltas": {}},
                    "GLM-5.3-flash": {"status": "win", "score_sum_delta": 1.0, "dimension_deltas": {}},
                    "majority": {"status": "tie", "score_sum_delta": 0.5, "dimension_deltas": {}},
                },
            },
        ],
    }
    evo_bundle = {
        "schema_version": "edara.wave-compact-summary/v1",
        "wave": "wave1",
        "cohort_kind": "evoscientist",
        "n_topics": 1,
        "n_manuscripts": 1,
        "mechanism_stats": {
            "n_manuscripts": 1,
            "E1_orig_keys_kept_all": 1,
            "E2_process_hype": {"A": {"occurrences": 1}, "D": {"occurrences": 3}, "E": {"occurrences": 0}},
            "E2_unverified_entities": {"A": {"total": 5}, "D": {"total": 6}, "E": {"total": 1}},
        },
        "usage": {"A": {"cost_usd": 0.03}, "E": {"cost_usd": 0.05}, "blind": {"cost_usd": 0.18}},
        "manuscripts": [
            {
                "wave": "wave1",
                "task_id": "survey_01_llm_reasoning_cot",
                "host": "evoscientist",
                "e_status": "changed",
                "e_actions": ["guided_revise"],
                "judges": {
                    "qwen3.8-flash": {"status": "win", "score_sum_delta": 2.0, "dimension_deltas": {}},
                    "deepseek-v4.1-flash": {"status": "tie", "score_sum_delta": 0.5, "dimension_deltas": {}},
                    "GLM-5.3-flash": {"status": "win", "score_sum_delta": 1.5, "dimension_deltas": {}},
                    "majority": {"status": "win", "score_sum_delta": 1.33, "dimension_deltas": {}},
                },
            }
        ],
    }

    (w1_dir / "wave1_compact_summary.json").write_text(json.dumps(mh_bundle), encoding="utf-8")
    (evo_dir / "wave1_evoscientist_compact.json").write_text(json.dumps(evo_bundle), encoding="utf-8")

    out_dir = tmp_path / "paper_out"
    summary = generate_paper_artifacts(results_dir=results_dir, out_dir=out_dir)

    assert summary["n_manuscripts"] == 3
    assert summary["n_topics"] == 2
    assert summary["mechanism_totals"]["orig_keys_kept"] == 3
    assert summary["mechanism_totals"]["process_hype_occurrences"]["E"] == 0

    for fname in (
        "table1_main_multihost.tex",
        "table1_main_multihost.md",
        "table2_domain_breakdown.tex",
        "table3_mechanism_ablation.tex",
        "fig1_host_winrates.svg",
        "fig2_epistemic_hygiene.svg",
        "fig3_domain_heatmap.svg",
        "PAPER_SUMMARY_REPORT.md",
    ):
        fpath = out_dir / fname
        assert fpath.is_file()
        content = fpath.read_text(encoding="utf-8")
        assert len(content) > 100
        if fname.endswith(".tex"):
            assert "\\toprule" in content and "\\bottomrule" in content
        if fname.endswith(".svg"):
            assert content.startswith("<svg ") and content.strip().endswith("</svg>")
