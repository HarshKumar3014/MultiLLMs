#!/usr/bin/env python3
"""
14_pipeline_ablation.py — What each pipeline error did to the results (CPU)
==========================================================================
Runs the paired analysis of 09_reanalysis on four configurations: v1 or v2
scorer (results/ vs results/v2/) x raw or corrected StereoSet/BBQ roles, and
writes results/reanalysis/pipeline_ablation.csv (Table "pipeline" in the paper).
"""
import json, numpy as np, pandas as pd, importlib
from config import RESULTS_DIR, SCORES_DIR, DATA_DIR
re9 = importlib.import_module("09_reanalysis"); import label_fix
P = json.load(open(DATA_DIR/"prompts.json")); PARA = json.load(open(DATA_DIR/"noise_floor_paraphrases.json"))
def fixp(prompts, para):
    pr=[q for q in map(label_fix.fix_prompt, prompts) if q is not None]
    fp={s:{pid:{k:q[k] for k in v} for pid,v in vs.items() if (q:=label_fix.fix_prompt({**v,"id":pid})) is not None} for s,vs in para.items()}
    return pr, fp
rows=[]
for scorer, src in [("v1", RESULTS_DIR), ("v2", SCORES_DIR)]:
    df0=pd.read_csv(src/"all_results.csv"); nz0=pd.read_csv(src/"noise_floor_all_results.csv")
    for labels in ["raw","fixed"]:
        if labels=="fixed":
            pr,pa=fixp(P,PARA); df=label_fix.fix_scores(df0); nz=label_fix.fix_scores(nz0)
        else:
            pr,pa,df,nz=P,PARA,df0,nz0
        col,cp,_=re9.audit_collapses(pr,pa)
        p=re9.build_paired(df,col); n=re9.build_noise_paired(nz,cp)
        rng=np.random.default_rng(0)
        cells=re9.cell_tests(p,rng)
        en=df[(df.layer=="A")&(df.language=="en")&~df.prompt_id.isin(col)]
        x=p.ss_en-.5; y=p.stereotype_score-.5; xn=n.ss_orig-.5; yn=n.stereotype_score-.5
        rows.append({"scorer":scorer,"labels":labels,"en_ss":en.stereotype_score.mean(),
            "en_bin":(en.logprob_stereotype>en.logprob_anti_stereotype).mean(),
            "neg":int((cells.mean_d<0).sum()),"sig":int((cells.q_bh<.05).sum()),"sig_raw":int((cells.p_perm<.05).sum()),
            "gap_lo":cells.groupby("model").dfg.mean().min(),"gap_hi":cells.groupby("model").dfg.mean().max(),
            "slope_tr":np.polyfit(x,y,1)[0],"slope_rw":np.polyfit(xn,yn,1)[0],
            "inf_rows":int(np.isneginf(df0[["logprob_stereotype","logprob_anti_stereotype"]]).any(axis=1).sum())})
t=pd.DataFrame(rows); print(t.round(3).to_string()); t.to_csv(RESULTS_DIR/"reanalysis"/"pipeline_ablation.csv",index=False)
