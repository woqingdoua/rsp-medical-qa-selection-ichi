#!/usr/bin/env python3
"""Extend current batch-4 RSP scores to the full pool and freeze Drop-only jobs."""

from __future__ import annotations

import argparse, hashlib, json, math, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

DATASETS=("afrimedqa","medexpqa","medexqa","pubmedqa","medmcqa")
SEEDS=(13,23,37); BUDGETS=(64,128,256); COL="full_pool_drop_only_k8"

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(2**20),b""): h.update(b)
    return h.hexdigest()
def read(p): return json.loads(Path(p).read_text())
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

def setup(run,project,source):
    if (run/"setup_manifest.json").exists():raise ValueError("Already set up")
    audit=read(source/"artifact_audit.json"); tasks=[]; provenance={}
    for rec in audit["coverage"]:
        ds,seed=rec["dataset"],int(rec["seed"]); score=Path(rec["source"]); retained=score.parent/"retained"
        meta=read(retained/"score_metadata.json"); a=meta["arguments"]
        if not (a["batch_size"]==4 and a["mc_passes"]==8 and a["noise_layer"]==-1 and a["noise_sigma"]==.02):
            raise ValueError(f"Not current canonical scorer: {ds}/{seed}")
        for p in (score,retained/"clean_scores.npz",retained/"rsp_scores.npz",retained/"score_metadata.json"):
            provenance[str(p)]=sha(p)
        if ds=="afrimedqa":
            supplement=source/"supplement"/ds/f"seed_{seed}.npz"
            if not supplement.is_file():raise ValueError(f"Missing B=256 supplement: {supplement}")
            provenance[str(supplement)]=sha(supplement)
            tasks.append({"dataset":ds,"seed":seed,"source_score":str(score),"retained":str(retained),
                "prefix_supplement":str(supplement),"output":str(run/"full_pool_supplement"/ds/f"seed_{seed}.npz"),
                "data_dir":str(project/"data"/"additional_medical"/ds)})
    write(run/"score_tasks.json",tasks)
    write(run/"setup_manifest.json",{"created_utc":datetime.now(timezone.utc).isoformat(),"project":str(project),
        "budget_source":str(source),"score_tasks":len(tasks),"source_sha256":provenance,
        "extension_rule":"retain canonical entropy-ordered batch-4 prefix; replay its Gaussian RNG tensor shapes; score only the remaining AfriMed rows"})
    print(json.dumps({"tasks":len(tasks),"status":"setup"},indent=2))

def score(run,index):
    import torch
    sys.path.insert(0,str(run/"source"))
    from common import load_model,load_tokenizer,get_transformer_blocks,set_seed,question_block,option_letters
    from original_rsp_raw_utils import AbsoluteGaussianHook,next_letter_probabilities,sanitize_for_acquisition,_truncate_prompt,rsp_components
    task=read(run/"score_tasks.json")[index];out=Path(task["output"])
    if out.exists():raise ValueError("Refusing overwrite")
    pool=rows(Path(task["data_dir"])/"pool.jsonl"); visible=[sanitize_for_acquisition(x) for x in pool]
    retained=Path(task["retained"]); clean=np.load(retained/"clean_scores.npz")
    entropy=clean["entropy"]; order=np.argsort(-entropy,kind="stable"); prefix=list(map(int,order[:1024])); missing=list(map(int,order[1024:]))
    base=np.load(retained/"rsp_scores.npz"); first=np.load(task["prefix_supplement"])
    if list(map(int,base["candidate_indices"]))!=prefix[:512] or list(map(int,first["candidate_indices"]))!=prefix[512:]:
        raise ValueError("Canonical prefix mismatch")
    if len({tuple(option_letters(visible[i])) for i in order})!=1:raise ValueError("Mixed answer spaces")
    set_seed(task["seed"]);tok=load_tokenizer("google/medgemma-4b-it","model_cache")
    model=load_model("google/medgemma-4b-it","model_cache",None,trainable=False)
    device=next(model.parameters()).device;dtype=next(model.parameters()).dtype
    hidden=getattr(model.config,"text_config",model.config).hidden_size;bs=4;shapes=[]
    for start in range(0,len(prefix),bs):
        batch=prefix[start:start+bs]
        lens=[len(_truncate_prompt(tok.encode(question_block(visible[i])+"\nChoose one option. Return only the final option letter.\nFinal Answer:",add_special_tokens=False),1024)) for i in batch]
        shapes.append((len(batch),max(lens),hidden))
    noisy=[]
    for k in range(8):
        hook=AbsoluteGaussianHook(.02,task["seed"]*1000003+k*104729);probe=torch.empty(1,device=device,dtype=dtype);gen=hook._generator(probe)
        for shape in shapes:torch.randn(shape,generator=gen,device=device,dtype=dtype)
        handle=get_transformer_blocks(model)[-1].register_forward_hook(hook)
        try:noisy.append(next_letter_probabilities(model,tok,[visible[i] for i in missing],bs,1024))
        finally:handle.remove()
        print(task["dataset"],task["seed"],"pass",k+1,flush=True)
    noisy=np.stack(noisy);comp=rsp_components(clean["probabilities"][missing],noisy,.5)
    out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,candidate_indices=np.array(missing),perturbed_probabilities=noisy,**comp)
    write(out.with_suffix(".json"),{"completed_utc":datetime.now(timezone.utc).isoformat(),"task":task,"sha256":sha(out),"gpu":torch.cuda.get_device_name(),"dtype":str(dtype),"prefix_rows":len(prefix),"new_rows":len(missing)})

def freeze(run,project,source):
    if (run/"protocol.json").exists():raise ValueError("Already frozen")
    audit=read(source/"artifact_audit.json"); manifest=[]; selections=[]; diagnostics=[]; hashes={}
    for rec in audit["coverage"]:
        ds,seed=rec["dataset"],int(rec["seed"]); pool=rows(project/"data"/"additional_medical"/ds/"pool.jsonl"); score=rows(rec["source"])
        if [x["uid"] for x in pool]!=[x["uid"] for x in score]:raise ValueError("UID mismatch")
        supplements=[]
        p=source/"supplement"/ds/f"seed_{seed}.npz"
        if p.is_file():supplements.append(p)
        p=run/"full_pool_supplement"/ds/f"seed_{seed}.npz"
        if p.is_file():supplements.append(p)
        for p in supplements:
            z=np.load(p);hashes[str(p)]=sha(p)
            for i,f,d in zip(z["candidate_indices"],z["flip"],z["drop"]):score[int(i)]["rsp_flip"]=float(f);score[int(i)]["rsp_drop"]=float(d)
        d=np.array([x["rsp_drop"] for x in score],float)
        if not np.isfinite(d).all():raise ValueError(f"Incomplete full pool: {ds}/{seed}")
        for x,v in zip(score,d):x[COL]=float(v)
        derived=run/"scores"/ds/f"seed_{seed}"/"full_pool_drop_only_scores.jsonl";writel(derived,score);hashes[str(derived)]=sha(derived)
        diagnostics.append({"dataset":ds,"seed":seed,"pool_N":len(pool),"drop_nonzero":int((d>0).sum()),"drop_unique":len(np.unique(d)),"drop_max":float(d.max()),"supplements":[str(x) for x in supplements]})
        for budget in BUDGETS:
            idx=sorted(sorted(range(len(pool)),key=lambda i:(-d[i],i))[:budget]);h=uid_hash(pool,idx)
            row={"dataset":ds,"seed":seed,"budget":budget,"method_label":"full_pool_drop_only_current","score_method":COL,
                "score_file":str(derived),"score_file_sha256":hashes[str(derived)],"data_dir":str(project/"data"/"additional_medical"/ds),
                "expected_selected_indices":idx,"expected_selected_uid_sha256":h,"output_dir":str(run/"branches"/ds/f"seed_{seed}"/f"b{budget}")}
            manifest.append(row);selections.append({k:row[k] for k in ("dataset","seed","budget","expected_selected_indices","expected_selected_uid_sha256")})
    writel(run/"training_manifest.jsonl",manifest);write(run/"selection_audit.json",selections);write(run/"score_diagnostics.json",diagnostics)
    protocol={"name":"Current canonical batch-4 full-pool Drop-only budget sensitivity","created_utc":datetime.now(timezone.utc).isoformat(),
        "datasets":DATASETS,"seeds":SEEDS,"budgets":BUDGETS,"method":"full_pool_drop_only_current","candidate_rule":"complete pool; entropy only fixes the inherited Gaussian traversal, never eligibility or ranking",
        "score_definition":"raw non-Flip positive margin contraction D","score_column":COL,"noise_layer":-1,"mc_passes":8,"noise_sigma":.02,"scoring_batch_size":4,
        "training_order":"ascending original pool index","backbone":"google/medgemma-4b-it","training":{"epochs":2.0,"learning_rate":1e-4,"lora_rank":16,"lora_alpha":32,"max_length":1024},
        "budget_source":str(source),"setup_manifest_sha256":sha(run/"setup_manifest.json"),"input_sha256":hashes,"training_manifest_sha256":sha(run/"training_manifest.jsonl"),
        "selection_audit_sha256":sha(run/"selection_audit.json"),"score_diagnostics_sha256":sha(run/"score_diagnostics.json"),"expected_branches":45,
        "source_snapshot_hashes":{p.name:sha(p) for p in (run/"source").iterdir() if p.is_file()}}
    write(run/"protocol.json",protocol);print(json.dumps({"status":"frozen","branches":45,"diagnostics":diagnostics},indent=2))

def main():
    a=argparse.ArgumentParser();a.add_argument("phase",choices=("setup","score","freeze"));a.add_argument("--run",type=Path,required=True);a.add_argument("--project",type=Path);a.add_argument("--source",type=Path);a.add_argument("--index",type=int);x=a.parse_args()
    run=x.run.resolve()
    if x.phase=="setup":setup(run,x.project.resolve(),x.source.resolve())
    elif x.phase=="score":score(run,x.index)
    else:freeze(run,x.project.resolve(),x.source.resolve())
if __name__=="__main__":main()
