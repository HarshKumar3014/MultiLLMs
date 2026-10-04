#!/usr/bin/env python3
"""
18_review_gpu_runs.py — GPU runs requested in review (round 3)
==============================================================
  (1) Target-language rewording. For Hindi and Spanish, an LLM that is not
      among the audited models (default: Aya-Expanse-32B) rewords every
      translated item in the same language, keeping the words that differ
      between the stereotyped and counter-stereotyped continuation. Each
      rewording is validated (pair still differs, wording changed, meaning
      close by LaBSE similarity) and retried once with sampling if it fails.
      All ten models then score the rewordings, so the within-language
      carry-over slope (Hindi -> reworded Hindi) can be compared with the
      cross-language one (English -> Hindi).
  (2) StereoSet without the blank context. Every StereoSet sentence (s+, s-,
      unrelated) is scored on its own, with no conditioning on the
      "... BLANK ..." template, in all eight languages.

Usage (GPU):
    python 18_review_gpu_runs.py --reword                 # build data/target_rewordings.json
    python 18_review_gpu_runs.py --score --models qwen2.5-7b --resume
    python 18_review_gpu_runs.py --test-cpu               # tiny smoke test on CPU
Outputs: data/target_rewordings.json, results/v2/review_runs/<model>.csv
"""

import argparse
import gc
import json
import re
from importlib import import_module

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import DATA_DIR, SCORES_DIR, MODEL_REGISTRY, MODEL_NAMES

OUT_DIR = SCORES_DIR / "review_runs"
REWORD_PATH = DATA_DIR / "target_rewordings.json"
TARGET_LANGS = {"hi": "Hindi", "es": "Spanish"}
FIELDS = ["context", "stereotype", "anti_stereotype", "unrelated"]
REWRITER = "CohereLabs/aya-expanse-32b"

PROMPT = """You are helping with a study of language models. Below is a test item written in {lang}. \
Reword it in {lang}: change the wording and sentence structure, but keep exactly the same meaning.

Rules:
- "a" and "b" currently differ only in a few words. Those differing words must stay exactly as they are, \
and "a" and "b" must still differ only in those words. Reword the rest of "a" and "b" in the same way.
- If the context contains the word "BLANK" or its translation, keep it as it is.
- Do not add or remove information. Keep names unchanged.
- Answer with JSON only, with the keys "context", "a", "b", "u".

Item:
{item}"""


# ══════════════════════════════════════════════════════════════════════════════
# (1a) Rewording with an LLM that is not audited
# ══════════════════════════════════════════════════════════════════════════════

def load_items():
    prompts = json.load(open(DATA_DIR / "prompts.json"))
    return [p for p in prompts if p["layer"] == "A" and p["language"] in TARGET_LANGS]


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not all(isinstance(d.get(k), str) and d.get(k).strip() for k in ["context", "a", "b", "u"]):
        return None
    return {"context": d["context"].strip(), "stereotype": d["a"].strip(),
            "anti_stereotype": d["b"].strip(), "unrelated": d["u"].strip()}


def _words(text):
    """Whitespace words with punctuation stripped (keeps Devanagari vowel signs)."""
    import unicodedata
    out = set()
    for w in text.split():
        w = "".join(ch for ch in w if unicodedata.category(ch)[0] != "P").casefold()
        if w:
            out.add(w)
    return out


def validate(orig, new, sim_model):
    re9 = import_module("09_reanalysis")
    if new is None:
        return False, "unparsable"
    if re9.is_collapsed(new["stereotype"], new["anti_stereotype"]):
        return False, "pair collapsed"
    # the words that distinguish s+ from s- must survive in the rewording
    wa, wb = _words(orig["stereotype"]), _words(orig["anti_stereotype"])
    na, nb = _words(new["stereotype"]), _words(new["anti_stereotype"])
    if not ((wa - wb) <= na and (wb - wa) <= nb):
        return False, "contrast words changed"
    if all(re9.is_collapsed(orig[f], new[f]) for f in ["context", "stereotype", "anti_stereotype"]):
        return False, "unchanged"
    e1 = sim_model.encode([orig[f] for f in FIELDS[:3]], normalize_embeddings=True)
    e2 = sim_model.encode([new[f] for f in FIELDS[:3]], normalize_embeddings=True)
    sims = (e1 * e2).sum(1)
    if sims.min() < 0.7:
        return False, f"meaning drift ({sims.min():.2f})"
    return True, f"ok ({sims.min():.2f})"


def reword(rewriter=REWRITER, batch_size=16, limit=None, device_map="auto"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from sentence_transformers import SentenceTransformer
    items = load_items()[:limit] if limit else load_items()
    done = json.load(open(REWORD_PATH)) if REWORD_PATH.exists() else {}
    todo = [p for p in items if p["id"] not in done]
    print(f"  rewording {len(todo)} items with {rewriter}")
    tok = AutoTokenizer.from_pretrained(rewriter)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"device_map": device_map}
    if torch.cuda.is_available():
        free = torch.cuda.get_device_properties(0).total_memory / 1e9
        if free < 70 and "32b" in rewriter.lower():
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.bfloat16)
        else:
            kw["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(rewriter, **kw).eval()
    sim = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    def generate(batch, sample):
        texts = []
        for p in batch:
            item = json.dumps({"context": p["context"], "a": p["stereotype"], "b": p["anti_stereotype"],
                               "u": p["unrelated"]}, ensure_ascii=False)
            msg = [{"role": "user", "content": PROMPT.format(lang=TARGET_LANGS[p["language"]], item=item)}]
            texts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=400, do_sample=sample, temperature=0.7 if sample else None,
                                 top_p=0.95 if sample else None, pad_token_id=tok.pad_token_id)
        return [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True) for o in out]

    for i in tqdm(range(0, len(todo), batch_size), desc="  reword"):
        batch = todo[i:i + batch_size]
        outs = generate(batch, sample=False)
        retry = []
        for p, o in zip(batch, outs):
            new = parse_json(o)
            ok, why = validate(p, new, sim)
            if ok:
                done[p["id"]] = {**new, "status": why, "attempt": 1}
            else:
                retry.append((p, why))
        if retry:
            outs2 = generate([p for p, _ in retry], sample=True)
            for (p, why1), o in zip(retry, outs2):
                new = parse_json(o)
                ok, why = validate(p, new, sim)
                done[p["id"]] = ({**new, "status": why, "attempt": 2} if ok
                                 else {"status": f"failed: {why1} / {why}", "attempt": 2})
        json.dump(done, open(REWORD_PATH, "w"), ensure_ascii=False, indent=1)
    ok = sum(1 for v in done.values() if "context" in v)
    print(f"  ✓ {ok}/{len(done)} rewordings passed validation → {REWORD_PATH}")
    del model
    gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# (1b, 2) Scoring with the audited models
# ══════════════════════════════════════════════════════════════════════════════

def sentence_logprob(model, tokenizer, text):
    """Average log-probability of `text` on its own (no context). All tokens
    after the BOS token are scored; tokenizers without a BOS token get one
    prepended when available, otherwise the first token is left unscored for
    every sentence alike."""
    import torch
    import torch.nn.functional as F
    ids = tokenizer.encode(text, add_special_tokens=True)
    bos = tokenizer.bos_token_id
    if bos is not None and (not ids or ids[0] != bos):
        ids = [bos] + ids
    if len(ids) < 2:
        return float("-inf")
    x = torch.tensor([ids], device=next(model.parameters()).device)
    with torch.no_grad():
        logits = model(x).logits[0].float()
    lp = F.log_softmax(logits[:-1], dim=-1).gather(1, x[0, 1:, None])
    return lp.mean().item()


def score(models, resume):
    audit = import_module("02_run_audit")
    prompts = json.load(open(DATA_DIR / "prompts.json"))
    rw = json.load(open(REWORD_PATH)) if REWORD_PATH.exists() else {}
    by_id = {p["id"]: p for p in prompts}
    jobs = [("rw_target", by_id[k], v) for k, v in rw.items() if "context" in v]
    jobs += [("nocontext", p, None) for p in prompts if p["layer"] == "A" and p["base_id"].startswith("ss_")]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for mk in models:
        ck = OUT_DIR / f"{mk}.csv"
        rows, done = [], set()
        if resume and ck.exists():
            old = pd.read_csv(ck)
            rows, done = old.to_dict("records"), set(zip(old["condition"], old["prompt_id"]))
        todo = [j for j in jobs if (j[0], j[1]["id"]) not in done]
        if not todo:
            continue
        model, tok = audit.load_model(mk)
        try:
            for i, (cond, p, new) in enumerate(tqdm(todo, desc=f"  {mk}")):
                if cond == "rw_target":
                    s = audit.compute_bias_scores(model, tok, {**p, **{f: new[f] for f in FIELDS}})
                else:
                    ls = sentence_logprob(model, tok, p["stereotype"])
                    la = sentence_logprob(model, tok, p["anti_stereotype"])
                    lu = sentence_logprob(model, tok, p["unrelated"])
                    s = {"logprob_stereotype": round(ls, 6), "logprob_anti_stereotype": round(la, 6),
                         "logprob_unrelated": round(lu, 6)}
                rows.append({"model": mk, "condition": cond, "prompt_id": p["id"], "base_prompt_id": p["base_id"],
                             "language": p["language"], **s})
                if i % 200 == 0:
                    pd.DataFrame(rows).to_csv(ck, index=False)
        finally:
            pd.DataFrame(rows).to_csv(ck, index=False)
            audit.unload_model(model, tok)


def test_cpu():
    """Smoke test with a tiny model: rewording parse/validate path and both scorers."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sentence_transformers import SentenceTransformer
    items = load_items()[:2]
    sim = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    fake = {"context": items[0]["context"] + " ", "a": items[0]["stereotype"], "b": items[0]["anti_stereotype"],
            "u": items[0]["unrelated"]}
    print("parse:", parse_json("```json\n" + json.dumps(fake, ensure_ascii=False) + "\n```") is not None)
    print("validate unchanged ->", validate(items[0], parse_json(json.dumps(fake, ensure_ascii=False)), sim))
    hf = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(hf); m = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.float32).eval()
    print("sentence logprob:", round(sentence_logprob(m, tok, items[0]["stereotype"]), 3))
    audit = import_module("02_run_audit")
    print("pair score:", audit.compute_bias_scores(m, tok, items[0])["stereotype_score"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reword", action="store_true")
    ap.add_argument("--rewriter", default=REWRITER)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--test-cpu", action="store_true")
    a = ap.parse_args()
    if a.test_cpu:
        test_cpu()
    if a.reword:
        reword(a.rewriter, limit=a.limit)
    if a.score:
        score(a.models or MODEL_NAMES, a.resume)


if __name__ == "__main__":
    main()
