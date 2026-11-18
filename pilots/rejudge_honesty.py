import json, sys, traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
sys.path.insert(0,"src")
from edara_mvp_eval.paired_trial import TrialRunner, Evidence, shared_full_pool, public_task, AXES
from edara_mvp_eval.paired_review import review_pair, original_honesty, freeze_claims
from edara_mvp_eval.paired_inputs import TaskSpec
from edara_mvp_eval.llm_client import ModelEndpoint
from edara_mvp_eval.stage_b_evidence import digest, write_json

SRC=Path("outputs/paired_v1_20261118_v12")
OUT=Path("outputs/honesty_rejudge_20261118")
core=json.loads((SRC/"inputs_core.json").read_text())
crit=ModelEndpoint.from_env_or_profile(profile_path=Path("/workspace/edara/data/p1b-critic-profile.json"))

def one(item):
    t,h=item
    try:
        seed=core[t]; task=TaskSpec.model_validate(seed["task"])
        original=seed["originals"][h]["draft"]
        versions={a: json.loads((SRC/"versions"/t/h/f"{a}.json").read_text()) for a in "BCDE"}
        full_packet, full_drafts = shared_full_pool(task, seed["evidence"], versions)
        local_packet = Evidence.restore(task, seed["evidence"]).packet()
        root=OUT/t/h; root.mkdir(parents=True,exist_ok=True)
        ledger_path=root/"ledger.json"
        if ledger_path.is_file():
            claims=json.loads(ledger_path.read_text())["claims"]
        else:
            ledger_runner=TrialRunner(crit, root/"ledger", task_id=f"{t}/{h}", arm="blind", stage="original", limit=8)
            claims=freeze_claims(original, ledger_runner)
            write_json(ledger_path, {"claims": claims})
        def runners(cmp, base):
            r=[]
            for o in ("forward","reverse","repeat"):
                run=TrialRunner(crit, root/cmp/o, task_id=f"{t}/{h}", arm="blind", stage="honesty", limit=6)
                r.append(run)
            r[2].temperature=0.2
            return r
        base_local=TrialRunner(crit, root/"baseline_local", task_id=f"{t}/{h}", arm="blind", stage="original", limit=8)
        baseline_local=original_honesty(original, claims, local_packet, base_local)
        if full_packet == local_packet:
            baseline_full=baseline_local
        else:
            base_full=TrialRunner(crit, root/"baseline_full", task_id=f"{t}/{h}", arm="blind", stage="original", limit=8)
            baseline_full=original_honesty(original, claims, full_packet, base_full)
        out={}
        cb=review_pair("honesty", original, {"B":versions["B"]["draft"],"C":versions["C"]["draft"]},
                       public_task(task), local_packet, runners("C_vs_B",None), claims=claims, baseline=baseline_local)
        ed=review_pair("honesty", original, {"D":full_drafts["D"],"E":full_drafts["E"]},
                       public_task(task), full_packet, runners("E_vs_D",None), claims=claims, baseline=baseline_full)
        out={"task_id":t,"host":h,"stage":seed["originals"][h].get("stage"),
             "C_vs_B":{k:cb.get(k) for k in ("status","detail","delta","deltas","arms")},
             "E_vs_D":{k:ed.get(k) for k in ("status","detail","delta","deltas","arms")}}
        write_json(root/"result.json", out)
        return out
    except Exception as e:
        return {"task_id":t,"host":h,"error":f"{type(e).__name__}:{e}"}

items=[(t,h) for t,seed in core.items() for h,rec in seed["originals"].items() if rec["status"]=="ready"]
print("jobs",len(items),flush=True)
with ThreadPoolExecutor(max_workers=12) as pool:
    rows=list(pool.map(one,items))
write_json(OUT/"honesty_results.json", {"results":rows})
print("done",len([r for r in rows if "error" not in r]),"ok",flush=True)
