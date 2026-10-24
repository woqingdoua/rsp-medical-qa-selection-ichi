#!/usr/bin/env python3
"""Freeze entropy-prefiltered Drop-only selections for B=64,128,256."""
from __future__ import annotations
import argparse, hashlib, json, math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

DATASETS=("afrimedqa","medexpqa","medexqa","pubmedqa","medmcqa")
SEEDS=(13,23,37); BUDGETS=(64,128,256); SENTINEL=-1e30
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(2**20),b""):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rows(p):
 with Path(p).open(encoding="utf-8") as f:return [json.loads(x) for x in f if x.strip()]
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open("x",encoding="utf-8") as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
def writel(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open("x",encoding="utf-8") as f:
  for r in x:f.write(json.dumps(r,sort_keys=True,allow_nan=False)+"\n")
def uid_hash(pool,idx):return hashlib.sha256("\n".join(pool[i]["uid"] for i in idx).encode()).hexdigest()

def main():
 a=argparse.ArgumentParser();a.add_argument("--run_root",type=Path,required=True);a.add_argument("--project_root",type=Path,required=True);a.add_argument("--source_run",type=Path,required=True);x=a.parse_args()
 run,project,source=x.run_root.resolve(),x.project_root.resolve(),x.source_run.resolve()
 if Path(__file__).resolve()!=run/"source"/Path(__file__).name:raise ValueError("Execute frozen source")
 if (run/"protocol.json").exists():raise ValueError("Already prepared")
 audit=read(source/"artifact_audit.json");manifest=[];selections=[];diagnostics=[];hashes={}
 for rec in audit["coverage"]:
  ds,seed=rec["dataset"],int(rec["seed"]);pool=rows(project/"data"/"additional_medical"/ds/"pool.jsonl");score=rows(rec["source"])
  if [r["uid"] for r in pool]!=[r["uid"] for r in score]:raise ValueError(f"UID mismatch {ds}/{seed}")
  hashes[rec["source"]]=sha(rec["source"])
  supplement=source/"supplement"/ds/f"seed_{seed}.npz"
  if supplement.is_file():
   z=np.load(supplement);hashes[str(supplement)]=sha(supplement)
   for i,f,d in zip(z["candidate_indices"],z["flip"],z["drop"]):score[int(i)]["rsp_flip"]=float(f);score[int(i)]["rsp_drop"]=float(d)
  for budget in BUDGETS:
   col=f"preentropy_drop_only_b{budget}";count=min(4*budget,len(pool))
   candidate=sorted(range(len(pool)),key=lambda i:(-float(score[i]["entropy"]),i))[:count]
   d=np.asarray([score[i]["rsp_drop"] for i in candidate],float)
   if not np.isfinite(d).all():raise ValueError(f"Missing D in candidate pool {ds}/{seed}/B{budget}")
   eligible=set(candidate)
   for i,r in enumerate(score):r[col]=float(r["rsp_drop"]) if i in eligible else SENTINEL
   selected=sorted(sorted(candidate,key=lambda i:(-float(score[i]["rsp_drop"]),i))[:budget]);h=uid_hash(pool,selected)
   out=run/"branches"/ds/f"seed_{seed}"/f"b{budget}"
   manifest.append({"dataset":ds,"seed":seed,"budget":budget,"method_label":"preentropy_drop_only","score_method":col,
    "data_dir":str(project/"data"/"additional_medical"/ds),"expected_selected_indices":selected,"expected_selected_uid_sha256":h,"output_dir":str(out)})
   selections.append({"dataset":ds,"seed":seed,"budget":budget,"candidate_N":count,"candidate_uid_sha256":uid_hash(pool,sorted(candidate)),"selected_indices":selected,"selected_uid_sha256":h,"drop_nonzero":int((d>0).sum()),"drop_unique":len(np.unique(d))})
  derived=run/"scores"/ds/f"seed_{seed}"/"preentropy_drop_only_scores.jsonl";writel(derived,score);digest=sha(derived);hashes[str(derived)]=digest
  for r in manifest:
   if r["dataset"]==ds and r["seed"]==seed:r["score_file"]=str(derived);r["score_file_sha256"]=digest
  diagnostics.append({"dataset":ds,"seed":seed,"pool_N":len(pool),"source":rec["source"],"supplement":str(supplement) if supplement.is_file() else None})
 for split_ds in DATASETS:
  for split in ("pool","dev","test"):
   p=project/"data"/"additional_medical"/split_ds/f"{split}.jsonl";hashes[str(p)]=sha(p)
 writel(run/"training_manifest.jsonl",manifest);write(run/"selection_audit.json",selections);write(run/"score_source_audit.json",diagnostics)
 protocol={"name":"Predictive-entropy prefilter plus raw Drop-only reranking","created_utc":datetime.now(timezone.utc).isoformat(),"datasets":DATASETS,"seeds":SEEDS,"budgets":BUDGETS,
  "method":"preentropy_drop_only","candidate_rule":"top min(4B,N) by question-only predictive entropy","reranking_rule":"descending raw D; pool-index tie break","selector_visibility":"question/options only; no rationale or label",
  "noise_layer":-1,"mc_passes":8,"noise_sigma":.02,"scoring_batch_size":4,"lambda_search":False,"training_order":"ascending original pool index","backbone":"google/medgemma-4b-it",
  "training":{"epochs":2.0,"learning_rate":1e-4,"lora_rank":16,"lora_alpha":32,"max_length":1024},"source_run":str(source),"input_sha256":hashes,"expected_branches":45,
  "training_manifest_sha256":sha(run/"training_manifest.jsonl"),"selection_audit_sha256":sha(run/"selection_audit.json"),"score_source_audit_sha256":sha(run/"score_source_audit.json"),
  "source_snapshot_hashes":{p.name:sha(p) for p in (run/"source").iterdir() if p.is_file()}}
 write(run/"protocol.json",protocol);print(json.dumps({"run_root":str(run),"branches":len(manifest),"status":"frozen"},indent=2))
if __name__=="__main__":main()
