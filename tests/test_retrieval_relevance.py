from edara_mvp_eval.governance import admissible
from edara_mvp_eval.schemas import TaskSpec
from edara_mvp_eval.stage_b_evidence import _relevant, native_prior_work


def test_cross_domain_polysemy_is_rejected():
    assert not _relevant("KV cache eviction for long-context inference",
                         "The Eviction Economy",
                         "A review of housing eviction and reform in American cities")
    assert not _relevant("KV cache compression for long-context inference",
                         "Generative Adversarial Network Based on Self-Attention Mechanism for "
                         "Automatic Page Layout Generation",
                         "We generate page layouts with a conditional GAN")
    assert not _relevant("world models simulation for embodied agents",
                         "A Proposal for Legalizing Small-Scale Physical Copyright Piracy",
                         "Book publishing, video films and music in developing countries")


def test_genuine_neighbour_still_passes():
    assert _relevant("KV cache compression token eviction",
                     "H2O: Heavy-Hitter Oracle for Efficient Generative Inference",
                     "We propose a KV cache eviction policy that keeps heavy hitters for efficient inference.")
    assert _relevant("Towards Agentic AI for Science: Hypothesis Generation and Validation",
                     "Agentic hypothesis generation",
                     "A method for research proposals.")


def test_single_token_query_needs_a_non_ambiguous_match():
    assert not _relevant("eviction", "The Eviction Economy", "housing")
    assert _relevant("cache", "KV cache compression", "cache")


def test_native_prior_work_marks_undated_sources_as_provenance():
    citation_map = {
        "S1": {"paper_id": "p1", "title": "Dated paper", "year": "2024",
               "evidence_ids": ["ev-a"], "quotes": {"ev-a": "known quote"}, "section_id": "s1"},
        "S2": {"paper_id": "p2", "title": "Undated paper", "year": "",
               "evidence_ids": ["ev-b"], "quotes": {"ev-b": "another quote"}, "section_id": "s2"},
    }
    items, provenance = native_prior_work(citation_map, stage="proposal", source_db="/tmp/research.db",
                                          cutoff_date="2025-06-10")
    assert [item.id for item in items] == ["native:p1", "native:p2"]
    assert items[0].date_basis == "publication" and items[0].year == "2024"
    assert items[1].date_basis == "participant_provenance" and items[1].year is None
    assert items[0].version == "native-export/proposal"
    assert items[0].date_source == "/tmp/research.db#S1"
    assert items[0].abstract == "known quote"
    assert provenance[1]["date_basis"] == "participant_provenance"


def test_undated_native_source_is_admitted_but_post_cutoff_is_not():
    task = TaskSpec(task_id="t", source_track="lit2test", title="T", field="ML",
                    prompt_context="ctx", cutoff_date="2025-06-10", strict_evidence=True)
    citation_map = {
        "S1": {"paper_id": "p1", "title": "Dated paper", "year": "2024",
               "evidence_ids": ["ev-a"], "quotes": {"ev-a": "quote one"}},
        "S2": {"paper_id": "p2", "title": "Undated paper", "year": "",
               "evidence_ids": ["ev-b"], "quotes": {"ev-b": "quote two"}},
    }
    items, _ = native_prior_work(citation_map, stage="proposal", source_db="/tmp/research.db",
                                 cutoff_date="2025-06-10")
    assert admissible(task, items[0])
    assert admissible(task, items[1])
    late = items[0].model_copy(update={"year": "2026", "publication_date": None})
    assert not admissible(task, late)
    wrong_source = items[1].model_copy(update={"source": "arxiv"})
    assert not admissible(task, wrong_source)


def test_same_year_source_falls_back_to_provenance():
    citation_map = {"S1": {"paper_id": "p1", "title": "Same year paper", "year": "2025",
                           "evidence_ids": ["ev-a"], "quotes": {"ev-a": "quote one"}}}
    items, _ = native_prior_work(citation_map, stage="proposal", source_db="/tmp/research.db",
                                 cutoff_date="2025-06-10")
    assert items[0].date_basis == "participant_provenance"
