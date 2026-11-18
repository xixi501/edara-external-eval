from edara_mvp_eval.canonical_render import render

DIRECT = {"title": "T", "literature_gap": "Gap text", "hypothesis": "H text", "method": "M text",
          "minimal_test": "Test text", "decisive_metric": "Metric", "supporting_result": "S",
          "falsifying_result": "F", "related_work": "RW", "limitations": "L"}


def test_direct_render_has_all_sections():
    text = render("direct", DIRECT)
    for header in ("# T", "## Problem", "## Proposed approach", "## Contribution claims",
                   "## Validation plan", "## Risks and limitations"):
        assert header in text


def test_empty_sections_are_omitted():
    text = render("ai_scientist", {"Title": "Only title", "Abstract": "Abs"})
    assert "## Problem" in text
    assert "## Validation plan" not in text
    assert "None" not in text


def test_edara_proposal_and_plan_render_differently():
    proposal = {"title": "P", "problem": "Problem", "idea": "Idea", "method": "Method",
                "claims": [{"text": "claim one"}], "validation": "Validation", "_edara": {"stage": "proposal"}}
    plan = {"objective": "Objective", "purpose": "Purpose", "hypothesis": "Hypothesis",
            "conditions": ["C1", "C2"], "falsification": "Falsify",
            "_edara": {"stage": "experiment_plan"}}
    assert "## Contribution claims" in render("edara", proposal)
    assert "## Validation plan" in render("edara", plan)
    assert "C1" in render("edara", plan)


def test_rendering_is_order_stable():
    assert render("direct", DIRECT) == render("direct", dict(reversed(list(DIRECT.items()))))
