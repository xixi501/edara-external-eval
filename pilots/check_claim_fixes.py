import json, sys, re
from pathlib import Path
sys.path.insert(0,"src")
from edara_mvp_eval.paired_trial import TrialRunner, run_arm
from edara_mvp_eval.paired_inputs import TaskSpec
from edara_mvp_eval.llm_client import ModelEndpoint
from edara_mvp_eval.stage_b_evidence import write_json
SRC=Path("outputs/paired_v1_20261118_v12"); H=Path("outputs/honesty_rejudge_20261118"); OUT=Path("outputs/fixcheck2_20261118")
core=json.loads((SRC/"inputs_core.json").read_text())
CASES=[("aiib-10541","ai_scientist"),("aiib-33","ai_scientist"),("aiib-4717","direct")]
crit=ModelEndpoint.from_env_or_profile(profile_path=Path("/workspace/edara/data/p1b-critic-profile.json"))
def one(item):
    t,h=item
    try:
        seed=core[t]; task=TaskSpec.model_validate(seed["task"]); original=seed["originals"][h]["draft"]
        claims=json.loads((H/t/h/"ledger.json").read_text())["claims"]
        root=OUT/t/h; root.mkdir(parents=True,exist_ok=True)
        au=TrialRunner(crit, root/"audit", task_id=f"{t}/{h}", arm="C", stage="audit", limit=64)
        ed=TrialRunner(crit, root/"editor", task_id=f"{t}/{h}", arm="C", stage="editor", limit=1)
        cf=TrialRunner(crit, root/"editor_confound", task_id=f"{t}/{h}", arm="C", stage="editor", limit=1)
        out=run_arm("C", original, task, seed["evidence"], au, ed, root/"lookup",
                    audit_cache=SRC/"versions"/t/h/"structured_audits", claims=claims,
                    claim_audit=True, editor_confound=cf)
        new=out["draft"]; enf=(out.get("history") or [{}])[0].get("confound_enforcement") or {}
        text="\n".join(str(v) for v in new.values())
        banned=[p for p in ("pre-registered","we have already","calibrated on","which dilutes") if p in text.lower()]
        row={"task_id":t,"host":h,"status":out["status"],"enforcement":{k:enf.get(k) for k in ("required","retried","addressed","edits")},
             "banned":banned,
             "new_sentences":[s.strip() for f,v in new.items() for s in str(v).replace("\n"," ").split(". ") if s.strip() and s.strip() not in " ".join(original.values())][:6]}
        write_json(root/"fixcheck.json", row); return row
    except Exception as e:
        return {"task_id":t,"host":h,"error":f"{type(e).__name__}:{e}"}
if __name__=="__main__":
    rows=[one(c) for c in CASES]
    write_json(OUT/"fixcheck_results.json",{"results":rows})
    for r in rows: print(json.dumps({k:(v if k!="enforcement" else {kk:(str(vv)[:80]) for kk,vv in v.items()}) for k,v in r.items()},ensure_ascii=False)[:700])