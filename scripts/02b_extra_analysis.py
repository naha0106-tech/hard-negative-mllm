"""2단계 보완: 계획서 2단계 항목 중 빠진 것을 채웁니다.
(이미 만든 구간 경계·D1~D5·학습 세트는 절대 바꾸지 않습니다)

  --task bindist   [GPU 불필요] 구간별 과목·학년·K·hint 분포 → difficulty_report에 추가
  --task letter    [GPU 불필요] 확정된 글자 형식을 config.yaml에도 기록
  --task items     [GPU, 약 30분] val·test 문항 난이도(분석 전용) → item_difficulty_{model}.parquet
  --task lp4       [GPU, 약 3시간] 오답쌍마다 로그확률 4개 값 저장 → pairs_scored에 열 추가
                   (기존 m_A, m_B와 같은 값이 나오는지 자동 비교. 구간은 바꾸지 않음)

실행 예: python scripts/02b_extra_analysis.py --task bindist --model smolvlm
"""
import argparse
import importlib
import os

import numpy as np
import pandas as pd

from common import (DATA, LETTERS, ROOT, Logger, build_prompt, load_config,
                    load_json, load_scienceqa, save_json)


# ------------------------------------------------------------------ GPU 불필요
def task_bindist(model, log):
    items = pd.read_parquet(f"{DATA}/items.parquet")
    scored = pd.read_parquet(f"{DATA}/pairs_scored_{model}.parquet")
    meta = items[items.split == "train"][["item_id", "subject", "grade_band", "has_hint"]]
    df = scored.merge(meta, on="item_id")
    out = {}
    for col in ["subject", "grade_band", "K", "has_hint"]:
        tab = pd.crosstab(df["bin"], df[col], normalize="index").round(4)
        base = df[col].value_counts(normalize=True).round(4)
        out[col] = {"by_bin": {f"D{int(k)}": v for k, v in tab.T.to_dict().items()},
                    "all_pairs": base.to_dict()}
        out[col]["by_bin"] = {b: {str(c): float(p) for c, p in d.items()}
                              for b, d in out[col]["by_bin"].items()}
        out[col]["all_pairs"] = {str(c): float(p) for c, p in out[col]["all_pairs"].items()}
        log(f"\n[구간별 {col} 비율]\n{tab.to_string()}\n전체: {base.to_dict()}")
    # 한 구간이 한 층에 쏠리는지: 구간별로 가장 큰 차이(구간 비율 − 전체 비율)
    skew = {}
    for col in ["subject", "grade_band"]:
        for b, d in out[col]["by_bin"].items():
            for c, p in d.items():
                gap = p - out[col]["all_pairs"].get(c, 0)
                if abs(gap) > abs(skew.get(b, ("", 0))[1]):
                    skew[b] = (f"{col}={c}", round(gap, 4))
    out["max_skew_by_bin"] = {b: {"category": k, "gap_vs_all": g} for b, (k, g) in skew.items()}
    log(f"\n[구간별 가장 큰 쏠림] {out['max_skew_by_bin']}")

    path = f"{DATA}/difficulty_report_{model}.json"
    rep = load_json(path)
    rep["bin_distribution"] = out
    save_json(rep, path)
    log(f"==> {path}에 bin_distribution 추가")


def task_letter(model, log):
    fmt = load_json(f"{DATA}/letter_format_{model}.json")["letter_format"]
    path = os.path.join(ROOT, "config.yaml")
    text = open(path, encoding="utf-8").read()
    key = f"  {model}: {fmt}"
    if "letter_format:" not in text:
        text += ("\n# ---- 2단계에서 확정한 정답 글자 형식 (학습 타깃·채점에 동일 적용) ----\n"
                 "letter_format:\n")
    if f"  {model}:" not in text.split("letter_format:")[1]:
        text = text.rstrip("\n") + "\n" + key + "\n"
    open(path, "w", encoding="utf-8").write(text)
    log(f"==> config.yaml에 letter_format {model}: {fmt} 기록")


# ------------------------------------------------------------------ GPU 필요
def setup_runner(cfg, model, log):
    s02 = importlib.import_module("02_score_difficulty")
    m, proc = s02.load_model(cfg, model)
    run = s02.Runner(m, proc, model)
    run.letter_format = load_json(f"{DATA}/letter_format_{model}.json")["letter_format"]
    log(f"[모델] {model}, 글자 형식 {run.letter_format}")
    return s02, run


def task_items(cfg, model, log):
    """val·test 문항을 원래 선택지 그대로 물어서 문항 난이도를 기록 (분석 전용, 학습 데이터 선택에 쓰지 않음)."""
    s02, run = setup_runner(cfg, model, log)
    bs = cfg["scoring"]["batch_size"]
    items = pd.read_parquet(f"{DATA}/items.parquet")
    items["choices"] = items["choices"].apply(list)
    rows = []
    for split in ["validation", "test"]:
        ds = load_scienceqa(cfg, split)
        sub = list(items[items.split == split].itertuples())
        log(f"[{split}] {len(sub)}문항 계산 시작")
        for bi, b in enumerate(s02.batches(sub, bs)):
            lp = run.last_logprobs([build_prompt(r.question, r.hint, r.choices) for r in b],
                                   [s02.get_image(ds, r.row_idx) for r in b])
            for row, r in zip(lp, b):
                sc = [run.letter_score(row, L) for L in LETTERS[:r.K]]
                wrong = [s for j, s in enumerate(sc) if j != r.answer]
                rows.append({"item_id": r.item_id, "split": split, "K": r.K,
                             "lp_letters": [round(x, 5) for x in sc],
                             "lp_correct": sc[r.answer], "lp_best_wrong": max(wrong),
                             "margin": sc[r.answer] - max(wrong),
                             "base_correct": int(np.argmax(sc)) == r.answer})
            if (bi + 1) % 100 == 0:
                log(f"  ... {split} {min((bi + 1) * bs, len(sub))}/{len(sub)}")
    df = pd.DataFrame(rows)
    df.to_parquet(f"{DATA}/item_difficulty_{model}.parquet", index=False)
    v = df[df.split == "validation"]
    log(f"[validation] 학습 전 정확도 {v.base_correct.mean():.3f}, 평균 margin {v.margin.mean():.3f}")
    log("[test] 계산만 하고 결과는 보지 않습니다 (본실험 분석 때 사용)")
    log(f"==> data/item_difficulty_{model}.parquet 저장")


def task_lp4(cfg, model, partial, log):
    """오답쌍마다 lp_PA(A), lp_PA(B), lp_PB(A), lp_PB(B)를 다시 계산해 저장. 구간(bin)은 그대로 둡니다."""
    s02, run = setup_runner(cfg, model, log)
    bs = cfg["scoring"]["batch_size"]
    save_every = cfg["scoring"]["save_every"]
    items = pd.read_parquet(f"{DATA}/items.parquet")
    items["choices"] = items["choices"].apply(list)
    item = items.set_index("item_id")
    scored = pd.read_parquet(f"{DATA}/pairs_scored_{model}.parquet")
    ds = load_scienceqa(cfg, "train")

    done = pd.read_parquet(partial) if os.path.exists(partial) else pd.DataFrame(
        columns=["pair_id", "lpA_A", "lpA_B", "lpB_A", "lpB_B"])
    todo = scored[~scored.pair_id.isin(set(done.pair_id))]
    log(f"[lp4] 전체 {len(scored)}쌍 / 이미 계산 {len(done)} / 남은 {len(todo)}")
    half = max(1, bs // 2)
    results, buf = [done], []
    for bi, b in enumerate(s02.batches(list(todo.itertuples()), half)):
        prompts, images = [], []
        for p in b:
            r = item.loc[p.item_id]
            c, w = r.choices[r.answer], r.choices[p.j]
            img = s02.get_image(ds, r.row_idx)
            prompts += [build_prompt(r.question, r.hint, [c, w]), build_prompt(r.question, r.hint, [w, c])]
            images += [img, img]
        lp = run.last_logprobs(prompts, images)
        for k, p in enumerate(b):
            ra, rb = lp[2 * k], lp[2 * k + 1]
            buf.append({"pair_id": p.pair_id,
                        "lpA_A": run.letter_score(ra, "A"), "lpA_B": run.letter_score(ra, "B"),
                        "lpB_A": run.letter_score(rb, "A"), "lpB_B": run.letter_score(rb, "B")})
        if (bi + 1) % save_every == 0:
            results.append(pd.DataFrame(buf)); buf = []
            pd.concat(results).to_parquet(partial, index=False)
            log(f"  ... 중간 저장 {sum(len(x) for x in results)}/{len(scored)}")
    results.append(pd.DataFrame(buf))
    lp4 = pd.concat(results, ignore_index=True)
    lp4.to_parquet(partial, index=False)

    merged = scored.drop(columns=[c for c in ["lpA_A", "lpA_B", "lpB_A", "lpB_B"] if c in scored],
                         errors="ignore").merge(lp4, on="pair_id")
    diffA = (merged.lpA_A - merged.lpA_B - merged.m_A).abs()
    diffB = (merged.lpB_B - merged.lpB_A - merged.m_B).abs()
    log(f"[일치 확인] 기존 m_A와 차이: 최대 {diffA.max():.4f}, 평균 {diffA.mean():.5f}")
    log(f"[일치 확인] 기존 m_B와 차이: 최대 {diffB.max():.4f}, 평균 {diffB.mean():.5f}")
    if max(diffA.max(), diffB.max()) > 0.1:
        log("⚠️ 차이가 큽니다(>0.1). 저장은 하지만 GPU 종류·설정이 2단계와 같은지 확인하세요.")
    merged.to_parquet(f"{DATA}/pairs_scored_{model}.parquet", index=False)
    rep_path = f"{DATA}/difficulty_report_{model}.json"
    rep = load_json(rep_path)
    rep["lp4_check"] = {"max_abs_diff_m_A": float(diffA.max()), "max_abs_diff_m_B": float(diffB.max()),
                        "note": "lp 원값은 2단계 이후 재계산. m·구간은 2단계 값을 그대로 유지"}
    save_json(rep, rep_path)
    log("==> pairs_scored에 lpA_A, lpA_B, lpB_A, lpB_B 열 추가 (m, bin은 변경 없음)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["bindist", "letter", "items", "lp4"], required=True)
    ap.add_argument("--model", default="smolvlm")
    ap.add_argument("--partial", default=None, help="lp4 중간 저장 경로(구글 드라이브 추천)")
    args = ap.parse_args()
    cfg = load_config()
    log = Logger(f"02b_{args.task}_{args.model}.log")
    if args.task == "bindist":
        task_bindist(args.model, log)
    elif args.task == "letter":
        task_letter(args.model, log)
    elif args.task == "items":
        task_items(cfg, args.model, log)
    else:
        task_lp4(cfg, args.model, args.partial or f"{DATA}/partial_lp4_{args.model}.parquet", log)


if __name__ == "__main__":
    main()
