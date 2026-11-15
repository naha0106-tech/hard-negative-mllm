"""2단계: 오답 난이도(m) 측정 + 관문 A 점검 + D1 검수표

하는 일
  1) 정답 글자 형식 확정: 모델이 "A"로 답하는지 " A"(앞에 공백)로 답하는지 확인
  2) 학습 전 모델의 val500 정확도 측정 (관문 A1, A2)
  3) train의 모든 (정답, 오답) 쌍에 대해 순서를 바꾼 2개 프롬프트로 m 계산
  4) m을 5구간(D1=가장 어려움 ~ D5=가장 쉬움)으로 나눔
  5) 관문 A 4개 기준 점검 → data/difficulty_report_{model}.json
  6) D1에서 50쌍 뽑아 수작업 검수표 생성 → data/d1_audit.csv

실행: python scripts/02_score_difficulty.py --model smolvlm
      (코랩이 끊길 때를 대비해 --partial 경로를 구글 드라이브로 주면 이어서 계산합니다)
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from common import (DATA, LETTERS, Logger, build_prompt, load_config, load_json,
                    load_scienceqa, save_json)


# ------------------------------------------------------------------ 모델 불러오기
def load_model(cfg, name):
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoM

    m = cfg["models"][name]
    kw = {"revision": m["revision"]} if m.get("revision") else {}
    pkw = dict(kw)
    if name == "qwen" and m.get("max_pixels"):
        pkw["max_pixels"] = m["max_pixels"]
    processor = AutoProcessor.from_pretrained(m["id"], **pkw)
    if name == "smolvlm" and m.get("longest_edge"):
        processor.image_processor.size = {"longest_edge": m["longest_edge"]}
    processor.tokenizer.padding_side = "left"  # 마지막 위치 = 진짜 마지막 토큰이 되도록

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() \
        else torch.float16
    model = AutoM.from_pretrained(m["id"], torch_dtype=dtype, **kw).to("cuda").eval()
    return model, processor


class Runner:
    def __init__(self, model, processor, name):
        self.model, self.processor, self.name = model, processor, name
        tok = processor.tokenizer
        self.ids = {"nospace": {}, "space": {}}
        for L in LETTERS:
            for key, s in [("nospace", L), ("space", " " + L)]:
                t = tok.encode(s, add_special_tokens=False)
                self.ids[key][L] = t[0] if len(t) == 1 else None
        self.letter_format = None

    def _inputs(self, prompts, images):
        texts = []
        for p in prompts:
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}]
            texts.append(self.processor.apply_chat_template(msgs, add_generation_prompt=True))
        imgs = [[im] for im in images] if self.name == "smolvlm" else list(images)
        inp = self.processor(text=texts, images=imgs, return_tensors="pt", padding=True)
        return {k: v.to("cuda") for k, v in inp.items()}

    @torch.no_grad()
    def last_logprobs(self, prompts, images):
        out = self.model(**self._inputs(prompts, images))
        return torch.log_softmax(out.logits[:, -1, :].float(), dim=-1).cpu()

    @torch.no_grad()
    def first_token(self, prompts, images):
        inp = self._inputs(prompts, images)
        gen = self.model.generate(**inp, max_new_tokens=1, do_sample=False)
        return gen[:, inp["input_ids"].shape[1]].cpu().tolist()

    def letter_score(self, lp_row, L):
        a, b = self.ids["nospace"][L], self.ids["space"][L]
        if self.letter_format == "nospace":
            return lp_row[a].item()
        if self.letter_format == "space":
            return lp_row[b].item()
        return torch.logsumexp(lp_row[[a, b]], dim=0).item()


def batches(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def get_image(ds, row_idx):
    return ds[int(row_idx)]["image"].convert("RGB")


# ------------------------------------------------------------------ 단계별 함수
def decide_letter_format(run, val, ds_val, bs, log):
    bad = [L for L in LETTERS if run.ids["nospace"][L] is None or run.ids["space"][L] is None]
    log(f"[글자 토큰] 'A'형식 id={run.ids['nospace']}, ' A'형식 id={run.ids['space']}")
    if bad:
        raise SystemExit(f"글자 {bad}가 토큰 1개로 안 쪼개집니다. 코드 수정이 필요합니다.")
    inv = {}
    for L in LETTERS:
        inv[run.ids["nospace"][L]] = "nospace"
        inv[run.ids["space"][L]] = "space"
    cnt = {"nospace": 0, "space": 0, "other": 0}
    for b in batches(val, bs):
        toks = run.first_token([build_prompt(r.question, r.hint, r.choices) for r in b],
                               [get_image(ds_val, r.row_idx) for r in b])
        for t in toks:
            cnt[inv.get(t, "other")] += 1
    total = cnt["nospace"] + cnt["space"]
    share_space = cnt["space"] / total if total else 0.5
    fmt = "space" if share_space >= 0.9 else "nospace" if share_space <= 0.1 else "both"
    log(f"[글자 형식] 생성 결과 집계 {cnt} → 사용할 형식: {fmt}")
    return fmt, cnt


def val_accuracy(run, val, ds_val, bs):
    correct, chance = [], []
    for b in batches(val, bs):
        lp = run.last_logprobs([build_prompt(r.question, r.hint, r.choices) for r in b],
                               [get_image(ds_val, r.row_idx) for r in b])
        for row, r in zip(lp, b):
            scores = [run.letter_score(row, L) for L in LETTERS[:r.K]]
            correct.append(int(np.argmax(scores)) == r.answer)
            chance.append(1 / r.K)
    return float(np.mean(correct)), float(np.mean(chance))


def score_pairs(run, pairs, items, ds_train, bs, save_every, partial_path, log):
    item = items.set_index("item_id")
    done = pd.read_parquet(partial_path) if os.path.exists(partial_path) else \
        pd.DataFrame(columns=["pair_id", "m_A", "m_B"])
    todo = pairs[~pairs.pair_id.isin(set(done.pair_id))]
    log(f"[m 계산] 전체 {len(pairs)}쌍 / 이미 계산 {len(done)} / 남은 {len(todo)}")

    half = max(1, bs // 2)  # 쌍 1개당 프롬프트 2개
    results = [done]
    buf = []
    for bi, b in enumerate(batches(list(todo.itertuples()), half)):
        prompts, images = [], []
        for p in b:
            r = item.loc[p.item_id]
            correct, wrong = r.choices[r.answer], r.choices[p.j]
            img = get_image(ds_train, r.row_idx)
            prompts += [build_prompt(r.question, r.hint, [correct, wrong]),   # P_A: A=정답
                        build_prompt(r.question, r.hint, [wrong, correct])]   # P_B: B=정답
            images += [img, img]
        lp = run.last_logprobs(prompts, images)
        for k, p in enumerate(b):
            ra, rb = lp[2 * k], lp[2 * k + 1]
            m_A = run.letter_score(ra, "A") - run.letter_score(ra, "B")
            m_B = run.letter_score(rb, "B") - run.letter_score(rb, "A")
            buf.append({"pair_id": p.pair_id, "m_A": m_A, "m_B": m_B})
        if (bi + 1) % save_every == 0:
            results.append(pd.DataFrame(buf)); buf = []
            pd.concat(results).to_parquet(partial_path, index=False)
            log(f"  ... 중간 저장 {sum(len(x) for x in results)}/{len(pairs)}")
    results.append(pd.DataFrame(buf))
    out = pd.concat(results, ignore_index=True)
    out.to_parquet(partial_path, index=False)
    return out


def spearman(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def make_audit(scored, items, ds_train, seed, log):
    d1 = scored[scored.bin == 1].sort_values("pair_id")
    pick = d1.sample(n=min(50, len(d1)), random_state=seed)
    item = items.set_index("item_id")
    img_dir = f"{DATA}/d1_audit_images"
    os.makedirs(img_dir, exist_ok=True)
    rows = []
    for p in pick.itertuples():
        r = item.loc[p.item_id]
        get_image(ds_train, r.row_idx).save(f"{img_dir}/{p.pair_id}.png")
        rows.append({"pair_id": p.pair_id, "item_id": p.item_id, "question": r.question,
                     "hint": r.hint, "choices": " | ".join(r.choices),
                     "correct": r.choices[r.answer], "distractor": r.choices[p.j],
                     "m": round(p.m, 4), "image_file": f"d1_audit_images/{p.pair_id}.png",
                     "label(1정상/2라벨오류/3판독불가/4복수정답)": "", "note": ""})
    pd.DataFrame(rows).to_csv(f"{DATA}/d1_audit.csv", index=False, encoding="utf-8-sig")
    log(f"[D1 검수표] {len(rows)}쌍 → data/d1_audit.csv (label 칸을 직접 채우세요)")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["smolvlm", "qwen"], required=True)
    ap.add_argument("--partial", default=None, help="중간 저장 파일 경로(구글 드라이브 추천)")
    args = ap.parse_args()

    cfg = load_config()
    log = Logger(f"02_score_{args.model}.log")
    bs = cfg["scoring"]["batch_size"]
    gate = cfg["gate_a"]
    partial = args.partial or f"{DATA}/partial_{args.model}.parquet"

    items = pd.read_parquet(f"{DATA}/items.parquet")
    items["choices"] = items["choices"].apply(list)
    pairs = pd.read_parquet(f"{DATA}/pairs.parquet")
    val_ids = set(load_json(f"{DATA}/val500_ids.json"))
    val = list(items[items.item_id.isin(val_ids)].itertuples())
    ds_train = load_scienceqa(cfg, "train")
    ds_val = load_scienceqa(cfg, "validation")

    model, processor = load_model(cfg, args.model)
    run = Runner(model, processor, args.model)

    # 1) 글자 형식
    fmt, cnt = decide_letter_format(run, val, ds_val, bs, log)
    run.letter_format = fmt
    save_json({"letter_format": fmt, "greedy_counts": cnt,
               "token_ids": run.ids}, f"{DATA}/letter_format_{args.model}.json")

    # 2) val500 정확도 (A1, A2)
    acc, chance = val_accuracy(run, val, ds_val, bs)
    log(f"[val500] 학습 전 정확도 {acc:.3f} / 찍기 정확도 {chance:.3f}")

    # 3) m 계산
    sc = score_pairs(run, pairs, items, ds_train, bs, cfg["scoring"]["save_every"], partial, log)
    scored = pairs.merge(sc, on="pair_id")
    scored["m"] = (scored.m_A + scored.m_B) / 2

    # 4) 5구간
    q = np.percentile(scored.m, [20, 40, 60, 80])
    scored["bin"] = np.digitize(scored.m, q) + 1  # m<q20 → 1(D1) ... m>=q80 → 5(D5)
    scored.to_parquet(f"{DATA}/pairs_scored_{args.model}.parquet", index=False)

    # 5) 관문 A
    rho = spearman(scored.m_A, scored.m_B)
    per_bin = scored.groupby("bin").item_id.nunique().to_dict()
    checks = {
        "A1_기저실력": {"value": acc, "기준": f">= {chance + gate['a1_margin_over_chance']:.3f}",
                    "pass": acc >= chance + gate["a1_margin_over_chance"]},
        "A2_천장": {"value": acc, "기준": f"<= {gate['a2_max_acc']}",
                  "pass": acc <= gate["a2_max_acc"]},
        "A3_순서일치": {"value": rho, "기준": f">= {gate['a3_min_spearman']}",
                    "pass": rho >= gate["a3_min_spearman"]},
        "A4_구간크기": {"value": {f"D{k}": int(v) for k, v in per_bin.items()},
                    "기준": f"모든 구간 >= {gate['a4_min_items_per_bin']}",
                    "pass": min(per_bin.values()) >= gate["a4_min_items_per_bin"]},
    }
    report = {"model": args.model, "letter_format": fmt, "val500_acc": acc,
              "chance_acc": chance, "quantiles_q20_q40_q60_q80": q.tolist(),
              "m_summary": scored.m.describe().to_dict(),
              "mean_m_by_bin": scored.groupby("bin").m.mean().to_dict(),
              "checks": checks, "all_pass": all(c["pass"] for c in checks.values())}
    save_json(report, f"{DATA}/difficulty_report_{args.model}.json")
    for k, c in checks.items():
        log(f"[관문 {k}] {'통과' if c['pass'] else '미통과'}  값={c['value']}  기준 {c['기준']}")
    log("==> 관문 A " + ("전부 통과. 3단계로 가도 됩니다." if report["all_pass"]
                       else "미통과 항목이 있습니다. 3단계로 가지 말고 계획서의 대응 방안을 따르세요."))

    # 6) D1 검수표 (smolvlm 1차 실험에서만)
    if args.model == "smolvlm":
        make_audit(scored, items, ds_train, cfg["seeds"]["d1_audit"], log)


if __name__ == "__main__":
    main()
