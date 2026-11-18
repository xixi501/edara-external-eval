import json, sys, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0,"src")
from edara_mvp_eval.paired_trial import (TrialRunner, Evidence, public_task, run_arm,
                                         review_pair, original_honesty)
from edara_mvp_eval.paired_inputs import TaskSpec
from edara_mvp_eval.llm_client import ModelEndpoint
from edara_mvp_eval.stage_b_evidence import write_json

SRC=Path("outputs/paired_v1_20261118_v12"); H=Path("outputs/honesty_rejudge_20261118")
OUT=Path("outputs/step2_20261118")
core=json.loads((SRC/"inputs_core.json").read_text())
crit=ModelEndpoint.from_env_or_profile(profile_path=Path("/workspace/edara/data/p1b-critic-profile.json"))

def one(item):
    t,h=item
    try:
        seed=core[t]; task=TaskSpec.model_validate(seed["task"]); original=seed["originals"][h]["draft"]
        claims=json.loads((H/t/h/"ledger.json").read_text())["claims"]
        if not claims: return {"task_id":t,"host":h,"skip":"empty_ledger"}
        versions={a: json.loads((SRC/"versions"/t/h/f"{a}.json").read_text())["draft"] for a in ("B","C")}
        root=OUT/t/h; root.mkdir(parents=True,exist_ok=True)
        auditor=TrialRunner(crit, root/"audit", task_id=f"{t}/{h}", arm="C", stage="audit", limit=64)
        editor=TrialRunner(crit, root/"editor", task_id=f"{t}/{h}", arm="C", stage="editor", limit=1)
        conf=TrialRunner(crit, root/"editor_confound", task_id=f"{t}/{h}", arm="C", stage="editor", limit=1)
        out=run_arm("C", original, task, seed["evidence"], auditor, editor, root/"lookup",
                    audit_cache=SRC/"versions"/t/h/"structured_audits", claims=claims,
                    claim_audit=True, editor_confound=conf)
        if out.get("status") not in ("changed","retained"): return {"task_id":t,"host":h,"error":out.get("status")}
        new=out["draft"]; write_json(root/"C_new.json",{"draft":new})
        ev=Evidence.restore(task, seed["evidence"]); packet=ev.packet(); pub=public_task(task)
        b_runner=TrialRunner(crit, H/t/h/"baseline_local", task_id=f"{t}/{h}", arm="blind", stage="original", limit=8)
        baseline=original_honesty(original, claims, packet, b_runner)
        def runners(axis,cmp):
            r=[TrialRunner(crit, root/"review"/cmp/axis/o, task_id=f"{t}/{h}", arm="blind", stage=axis, limit=6)
               for o in ("forward","reverse","repeat")]
            r[2].temperature=0.2; return r
        row={"task_id":t,"host":h,
             "C_vs_B":{"honesty":review_pair("honesty", original, {"B":versions["B"],"C":new}, pub, packet,
                                             runners("honesty","C_vs_B"), claims=claims, baseline=baseline),
                       "credibility":review_pair("credibility", original, {"B":versions["B"],"C":new}, pub, packet,
                                                 runners("credibility","C_vs_B")),
                       "novelty":review_pair("novelty", original, {"B":versions["B"],"C":new}, pub, packet,
                                             runners("novelty","C_vs_B"))},
             "confound_enforcement":(out.get("history") or [{}])[0].get("confound_enforcement")}
        write_json(root/"step2.json", row); return row
    except Exception as e:
        return {"task_id":t,"host":h,"error":f"{type(e).__name__}:{e}"}

if __name__=="__main__":
    items=[(t,h) for t,s in core.items() if not s.get("development")
           for h,r in s["originals"].items() if r["status"]=="ready" and (H/t/h/"ledger.json").exists()]
    if len(sys.argv)>1: items=items[:1]
    print("items",len(items),flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows=list(pool.map(one,items))
    write_json(OUT/"step2_results.json",{"results":rows})
    print("done",len([r for r in rows if "error" not in r and "skip" not in r]),"of",len(rows),flush=True)
