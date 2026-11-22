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

    act_a_dir = results_dir / "action_a_qwen_backbone_test"
    act_b_dir = results_dir / "action_b_thinking_ablation_test"
    act_a_dir.mkdir(parents=True)
    act_b_dir.mkdir(parents=True)
    stub_j = {"counts": {"win": 6, "loss": 2, "tie": 2}, "topic_net_wins": 0.4, "topic_net_wins_ci95": [0.1, 0.7], "score_sum_delta": 1.2, "score_sum_delta_ci95": [0.4, 2.0]}
    (act_a_dir / "action_a_compact_summary.json").write_text(
        json.dumps({
            "n_manuscripts": 10,
            "judge_summaries": {"deepseek-v4.1-flash": stub_j, "GLM-5.3-flash": stub_j, "independent_2judge": stub_j, "qwen3.8-flash": stub_j, "majority_3way": stub_j},
            "by_host_summaries": {"evoscientist": {"deepseek-v4.1-flash": stub_j, "GLM-5.3-flash": stub_j, "independent_2judge": stub_j, "majority_3way": stub_j}},
            "mechanism_stats": {"E1_orig_keys_kept_all": 10, "E2_process_hype": {"A": {"occurrences": 0}, "D": {"occurrences": 3}, "E": {"occurrences": 0}}, "E3_confounds": {"targets_required_n": 4, "A_addressed": 1, "D_addressed": 1, "E_addressed": 4}},
            "total_cost_usd": 3.3,
        }),
        encoding="utf-8",
    )
    (act_b_dir / "action_b_compact_summary.json").write_text(
        json.dumps({
            "n_manuscripts": 10,
            "output_tokens": 8192,
            "judge_summaries": {"qwen3.8-flash": stub_j, "deepseek-v4.1-flash": stub_j, "GLM-5.3-flash": stub_j, "majority_3way": stub_j},
            "by_host_summaries": {"evoscientist": {"qwen3.8-flash": stub_j, "deepseek-v4.1-flash": stub_j, "GLM-5.3-flash": stub_j, "majority_3way": stub_j}},
            "mechanism_stats": {"E1_orig_keys_kept_all": 10, "E2_process_hype": {"A": {"occurrences": 2}, "D": {"occurrences": 12}, "E": {"occurrences": 0}}, "E2_unverified_entities": {"A": {"total": 10}, "D": {"total": 11}, "E": {"total": 2}}, "E3_confounds": {"targets_required_n": 4, "A_addressed": 1, "D_addressed": 1, "E_addressed": 4}},
            "direct_think_generation_probe": {"n_topics": 10, "fast_A_process_hype_total": 1, "think_A_process_hype_total": 4, "fast_A_unverified_entities_total": 10, "think_A_unverified_entities_total": 14},
            "wave1_fast_4096_paired_reference": {"judge_summaries": {"qwen3.8-flash": stub_j, "deepseek-v4.1-flash": stub_j, "GLM-5.3-flash": stub_j, "majority_3way": stub_j}, "hygiene": {"D_process_hype": 5, "E_process_hype": 0}},
            "total_cost_usd": 4.5,
        }),
        encoding="utf-8",
    )

    out_dir = tmp_path / "paper_out"
    summary = generate_paper_artifacts(results_dir=results_dir, out_dir=out_dir)

    assert summary["n_manuscripts"] == 3
    assert summary["n_topics"] == 2
    assert summary["mechanism_totals"]["orig_keys_kept"] == 3
    assert summary["mechanism_totals"]["process_hype_occurrences"]["E"] == 0
    assert summary["action_a_cross_backbone"]["n_manuscripts"] == 10
    assert summary["action_b_thinking_ablation"]["output_tokens"] == 8192

    for fname in (
        "table1_main_multihost.tex",
        "table1_main_multihost.md",
        "table2_domain_breakdown.tex",
        "table3_mechanism_ablation.tex",
        "table4_cross_backbone.tex",
        "table5_thinking_ablation.tex",
        "fig1_host_winrates.svg",
        "fig2_epistemic_hygiene.svg",
        "fig3_domain_heatmap.svg",
        "fig4_reasoning_hygiene_paradox.svg",
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
