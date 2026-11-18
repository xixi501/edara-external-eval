import json, sys
from pathlib import Path
sys.path.insert(0,"src")
from edara_mvp_eval.paired_trial import TrialRunner, Evidence, public_task, run_arm, audit_claims
from edara_mvp_eval.paired_inputs import TaskSpec
from edara_mvp_eval.llm_client import ModelEndpoint
from edara_mvp_eval.stage_b_evidence import digest, write_json

ROOT=Path("."); SRC=ROOT/"outputs/paired_v1_20261118_v12"; H=ROOT/"outputs/honesty_rejudge_20261118"
OUT=ROOT/"outputs/step1c_claim_audit_20261118"
core=json.loads((SRC/"inputs_core.json").read_text())
crit=ModelEndpoint.from_env_or_profile(profile_path=Path("/workspace/edara/data/p1b-critic-profile.json"))
CASES=[
 ("aiib-7144","direct","E_vs_D","c5"),
 ("smallari_expansion40_001_cache_compression_llms_long_context__v02_expansion40","ai_scientist","E_vs_D","c13"),
 ("smallari_expansion40_002_unlearning_machine_language_use_algorithms__v02_expansion40","direct","C_vs_B","c7"),
 ("smallari_expansion40_002_unlearning_machine_language_use_algorithms__v02_expansion40","ai_scientist","C_vs_B","c1"),
 ("smallari_expansion40_002_unlearning_machine_language_use_algorithms__v02_expansion40","ai_scientist","E_vs_D","c1"),
 ("ws_02_agentic_ai","ai_scientist","E_vs_D","c8"),
 ("ws_02_agentic_ai","edara","C_vs_B","c3"),
 ("survey_08_kv_cache_compression","ai_scientist","C_vs_B","c10"),
 ("survey_08_kv_cache_compression","ai_scientist","E_vs_D","c10"),
]
def one(case):
    t,h,cmp,cid=case
    try:
        seed=core[t]; task=TaskSpec.model_validate(seed["task"]); original=seed["originals"][h]["draft"]
        claims=json.loads((H/t/h/"ledger.json").read_text())["claims"]
        root=OUT/t/h; root.mkdir(parents=True,exist_ok=True)
        auditor=TrialRunner(crit, root/"audit", task_id=f"{t}/{h}", arm="C", stage="audit", limit=64)
        editor=TrialRunner(crit, root/"editor", task_id=f"{t}/{h}", arm="C", stage="editor", limit=1)
        cache=SRC/"versions"/t/h/"structured_audits"
        out=run_arm("C", original, task, seed["evidence"], auditor, editor, root/"lookup",
                    audit_cache=cache, claims=claims, claim_audit=True)
        findings=(out["history"][0].get("diagnosis") or {}).get("factual_findings") or []
        target=next((f for f in findings if f["id"]==cid), None)
        revised=out.get("draft") or original
        ev=Evidence.restore(task, seed["evidence"])
        re_runner=TrialRunner(crit, root/"reaudit", task_id=f"{t}/{h}", arm="C", stage="audit", limit=8)
        after={f["id"]:f for f in audit_claims(claims, revised, ev, public_task(task), re_runner)}.get(cid)
        row={"task_id":t,"host":h,"cmp":cmp,"claim":cid,
             "flagged": bool(target and target["label"]=="defect"),
             "target_before": target["label"] if target else None,
             "target_after": after["label"] if after else None,
             "implemented": bool(after and after["label"]!="defect"),
             "edits":[{"field":e.get("field"),"status":e.get("status")} for e in (out.get("edits") or [])],
             "labels":{f["id"]:f["label"] for f in findings}}
        write_json(root/"step1.json", row); return row
    except Exception as e:
        return {"task_id":t,"host":h,"cmp":cmp,"claim":cid,"error":f"{type(e).__name__}:{e}"}
if __name__=="__main__":
    rows=[one(c) for c in CASES]
    write_json(OUT/"step1_results.json", {"results":rows})
    for r in rows: print(json.dumps(r,ensure_ascii=False))
