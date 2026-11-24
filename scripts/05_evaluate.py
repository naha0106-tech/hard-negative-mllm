"""5단계: 실행 1개 평가 (본실험)

하는 일
  1) runs/{run_id}/adapter 의 최고 LoRA를 불러옴 (--base 이면 LoRA 없이 = 무학습)
  2) 사전 점검: 글자 토큰, Yes/No 토큰, AI2D 정답 번호가 0부터인지 1부터인지, POPE 라벨 값
  3) 평가 세트 채점 (파일럿과 같은 규칙: 마지막 위치 log-softmax에서 유효 글자 중 최댓값)
       scienceqa : ScienceQA-IMG test 2,017문항, 원래 보기·원래 순서 (주 지표)
                   --perm 이면 보기를 한 칸씩 K번 돌려 내용별 확률 평균 → 순환 순열 정확도
       ai2d      : AI2D 고정 1,000문항 (Context 줄 없이 같은 프롬프트)
       pope      : POPE random 500 + adversarial 500, " Yes"·"Yes" vs " No"·"No"
       val       : val500 (무학습 실행에서만 씀)
  4) runs/{run_id}/ 에 preds_*.csv, eval_checks.json, eval_summary.json, EVAL_DONE 저장

실행 예
  python scripts/05_evaluate.py --run_id smolvlm_main-Hard_N100_s0 --sets scienceqa,ai2d,pope --perm
  python scripts/05_evaluate.py --run_id smolvlm_ZS --base --sets val,scienceqa,ai2d,pope --perm
  python scripts/05_evaluate.py --prefetch      # 모델·데이터셋 미리 내려받기 (노트북 설치 셀)
EVAL_DONE이 있으면 아무것도 하지 않습니다.
"""
import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LETTERS, load_config, load_json  # noqa: E402
import train_common as tc  # noqa: E402

POPE_SUFFIX = "\nAnswer the question using a single word or phrase."


# ------------------------------------------------------------------ 지표
def restricted_probs(lp, K):
    """유효 글자 K개 안에서의 softmax (나머지는 0)."""
    p = np.zeros(len(LETTERS))
    x = lp[:K] - lp[:K].max()
    e = np.exp(x)
    p[:K] = e / e.sum()
    return p


def ece_score(conf, correct, n_bins):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= 0) & (conf <= edges[1])
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def yes_no_metrics(df):
    tp = int(((df.pred == "yes") & (df.answer == "yes")).sum())
    fp = int(((df.pred == "yes") & (df.answer == "no")).sum())
    fn = int(((df.pred == "no") & (df.answer == "yes")).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"n": int(len(df)), "acc": float(df.correct.mean()), "f1": f1, "precision": prec,
            "recall": rec, "yes_ratio": float((df.pred == "yes").mean())}


# ------------------------------------------------------------------ 객관식 채점 (ScienceQA·AI2D 공용)
def mc_score(model, processor, recs, images, lids, bs):
    texts = [tc.chat_text(processor, r) for r in recs]
    ims = [images[r["img_key"]] for r in recs]
    return tc.score_tokens(model, processor, texts, ims, [lids[L] for L in LETTERS], bs)


def mc_table(recs, lp):
    rows = []
    for r, row in zip(recs, lp):
        K = r["K"]
        p = restricted_probs(row, K)
        pred = LETTERS[int(np.argmax(row[:K]))]
        d = {"item_id": r["item_id"], "K": K, "answer": r["answer_letter"], "pred": pred,
             "correct": pred == r["answer_letter"], "conf": float(p.max())}
        for k in ("subject", "grade_band"):
            if k in r:
                d[k] = r[k]
        for j, L in enumerate(LETTERS):
            d[f"lp_{L}"] = float(row[j]) if j < K else None
            d[f"p_{L}"] = float(p[j]) if j < K else None
        rows.append(d)
    return pd.DataFrame(rows)


def rotate(rec, r):
    """보기를 r칸 돌림: 새 위치 p에는 원래 보기 (p + r) % K."""
    K = rec["K"]
    ans = LETTERS.index(rec["answer_letter"])
    return dict(rec, options=[rec["options"][(p + r) % K] for p in range(K)],
                answer_letter=LETTERS[(ans - r) % K], rot=r)


def perm_scores(model, processor, recs, base_lp, images, lids, bs):
    """순환 순열: 돌리지 않은 것(r=0)은 표준 채점을 그대로 쓰고, r=1..K-1만 추가 채점.
    내용 c의 확률 = r번 돌렸을 때 위치 (c - r) % K의 제한 softmax 확률. K번 평균 → 최댓값."""
    rot_recs = [rotate(r, k) for r in recs for k in range(1, r["K"])]
    rot_lp = mc_score(model, processor, rot_recs, images, lids, bs) if rot_recs else np.zeros((0, 5))
    acc = {r["item_id"]: restricted_probs(base_lp[i], r["K"])[:r["K"]].copy() for i, r in enumerate(recs)}
    for rr, row in zip(rot_recs, rot_lp):
        K, k = rr["K"], rr["rot"]
        p = restricted_probs(row, K)
        for c in range(K):
            acc[rr["item_id"]][c] += p[(c - k) % K]
    out = []
    for r in recs:
        pc = acc[r["item_id"]] / r["K"]
        pred = LETTERS[int(np.argmax(pc))]
        out.append({"item_id": r["item_id"], "pred_perm": pred, "correct_perm": pred == r["answer_letter"],
                    **{f"pp_{L}": (float(pc[j]) if j < r["K"] else None) for j, L in enumerate(LETTERS)}})
    return pd.DataFrame(out), len(rot_recs)


# ------------------------------------------------------------------ 데이터 불러오기
def scienceqa_recs(cfg, items, limit):
    recs = tc.test_records(items)
    if limit:
        recs = recs[:limit]
    images = tc.load_images(cfg, items, [r["item_id"] for r in recs], allow_test=True)
    for r in recs:
        r["img_key"] = r["item_id"]
    return recs, images


def val_recs(cfg, items, limit):
    recs = tc.val500_records(items)
    if limit:
        recs = recs[:limit]
    images = tc.load_images(cfg, items, [r["item_id"] for r in recs])
    for r in recs:
        r["img_key"] = r["item_id"]
    return recs, images


def ai2d_answer_base(answers, n_options):
    """정답 번호가 0부터(0..K-1)인지 1부터(1..K)인지 자동 판단."""
    a = [int(str(x).strip()) for x in answers]
    has0 = any(x == 0 for x in a)
    hasK = any(x == k for x, k in zip(a, n_options))
    if has0 and not hasK:
        return 0, "0이 있고 K가 없음"
    if hasK and not has0:
        return 1, "K가 있고 0이 없음"
    raise SystemExit(f"[멈춤] AI2D 정답 번호 해석 불가 (0 있음={has0}, K 있음={hasK})")


def ai2d_recs(cfg, limit):
    from datasets import load_dataset
    meta = load_json(os.path.join(tc.DATA, "ai2d1000_ids.json"))
    ds = load_dataset(meta["repo"], split=meta["split"]).select(meta["row_idx"])
    light = ds.select_columns(["answer", "options"])
    base, why = ai2d_answer_base(light["answer"], [len(o) for o in light["options"]])
    if limit:
        ds = ds.select(range(limit))
    recs, images = [], {}
    for i, (row_idx, row) in enumerate(zip(meta["row_idx"], ds)):
        K = len(row["options"])
        assert 2 <= K <= 5, f"AI2D {row_idx}: 보기 {K}개"
        ans = int(str(row["answer"]).strip()) - base
        assert 0 <= ans < K, f"AI2D {row_idx}: 정답 {row['answer']} / 보기 {K}개"
        key = f"ai2d_{row_idx}"
        images[key] = row["image"].convert("RGB")
        recs.append({"item_id": key, "img_key": key, "question": row["question"], "hint": "",
                     "options": [str(o) for o in row["options"]], "answer_letter": LETTERS[ans], "K": K})
    return recs, images, {"answer_base": base, "reason": why}


def pope_recs(cfg, limit):
    from datasets import load_dataset
    meta = load_json(os.path.join(tc.DATA, "pope1000_ids.json"))
    ds = load_dataset(meta["repo"], split=meta["split"])
    recs, images, labels = [], {}, set()
    for cat, idx in meta["row_idx"].items():
        if limit:
            idx = idx[:max(1, limit // len(meta["row_idx"]))]
        for row_idx, row in zip(idx, ds.select(idx)):
            assert row["category"] == cat, f"POPE {row_idx}: category {row['category']} != {cat}"
            lab = str(row["answer"]).strip().lower()
            labels.add(lab)
            key = f"pope_{row_idx}"
            images[key] = row["image"].convert("RGB")
            recs.append({"item_id": key, "img_key": key, "category": cat, "question": row["question"],
                         "answer": lab})
    bad = labels - {"yes", "no"}
    if bad:
        raise SystemExit(f"[멈춤] POPE 라벨 값이 yes/no가 아님: {sorted(labels)}")
    return recs, images, {"labels": sorted(labels)}


# ------------------------------------------------------------------ 메인
def prefetch(cfg):
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    m = cfg["models"]["smolvlm"]
    t = tc.timer()
    snapshot_download(m["id"], revision=m["revision"])
    print(f"[내려받기] 모델 {m['id']} ({t() / 60:.1f}분)", flush=True)
    from common import load_scienceqa
    for split in ("train", "validation", "test"):
        load_scienceqa(cfg, split)
    print(f"[내려받기] ScienceQA ({t() / 60:.1f}분)", flush=True)
    for name in ("ai2d1000_ids.json", "pope1000_ids.json"):
        meta = load_json(os.path.join(tc.DATA, name))
        load_dataset(meta["repo"], split=meta["split"])
        print(f"[내려받기] {meta['repo']} ({t() / 60:.1f}분)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id")
    ap.add_argument("--base", action="store_true", help="LoRA 없이 (무학습)")
    ap.add_argument("--sets", default="scienceqa,ai2d,pope")
    ap.add_argument("--perm", action="store_true", help="ScienceQA 순환 순열 정확도도")
    ap.add_argument("--limit", type=int, default=0, help="세트별 문항 수 제한 (스모크)")
    ap.add_argument("--prefetch", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    if args.prefetch:
        return prefetch(cfg)
    assert args.run_id, "--run_id가 필요합니다"
    sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    rdir = tc.runs_dir(cfg)
    out = os.path.join(rdir, args.run_id)
    if os.path.exists(os.path.join(out, "EVAL_DONE")):
        print(f"[건너뜀] {args.run_id}는 이미 평가했습니다 (EVAL_DONE 있음).")
        return
    if not args.base and not os.path.exists(os.path.join(out, "DONE")):
        raise SystemExit(f"[멈춤] {args.run_id}: 학습 DONE이 없습니다.")
    os.makedirs(out, exist_ok=True)
    bs = cfg["train"]["eval_batch"]
    total = tc.timer()
    log = lambda m: print(m, flush=True)  # noqa: E731
    log(f"===== 평가 {args.run_id} ({'무학습' if args.base else 'LoRA'}) sets={sets} perm={args.perm}"
        f"{' limit=' + str(args.limit) if args.limit else ''} =====")

    model, processor = tc.load_model_and_processor(cfg)
    if not args.base:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, os.path.join(out, "adapter"))
    model.eval()
    lids, _ = tc.letter_ids(processor)
    yn, yn_notes = tc.yes_no_ids(processor)
    checks = {"letter_ids": lids, "yes_no_tokens": yn_notes}
    items = tc.load_items()

    summ = {"run_id": args.run_id, "base": args.base, "sets": sets, "perm": args.perm,
            "limit": args.limit or None, "minutes": {}}
    sp = os.path.join(out, "summary.json")
    if os.path.exists(sp):
        s = load_json(sp)
        summ["val_best"] = {"acc": s.get("best_val_acc"), "step": s.get("best_step")}
        summ["diverged"] = s.get("diverged")

    if "val" in sets:
        t = tc.timer()
        recs, images = val_recs(cfg, items, args.limit)
        df = mc_table(recs, mc_score(model, processor, recs, images, lids, bs))
        df.to_csv(os.path.join(out, "preds_val.csv"), index=False)
        summ["val_acc"] = float(df.correct.mean())
        summ.setdefault("val_best", {"acc": summ["val_acc"], "step": 0})
        summ["minutes"]["val"] = round(t() / 60, 2)
        log(f"[val500] 정확도 {summ['val_acc']:.4f}  ({summ['minutes']['val']}분)")

    if "scienceqa" in sets:
        t = tc.timer()
        recs, images = scienceqa_recs(cfg, items, args.limit)
        lp = mc_score(model, processor, recs, images, lids, bs)
        df = mc_table(recs, lp)
        summ["minutes"]["scienceqa"] = round(t() / 60, 2)
        summ["test_n"] = int(len(df))
        summ["test_acc"] = float(df.correct.mean())
        summ["test_acc_by_K"] = {str(k): float(g.correct.mean()) for k, g in df.groupby("K")}
        summ["test_acc_by_subject"] = {k: float(g.correct.mean()) for k, g in df.groupby("subject")}
        summ["test_acc_K3plus"] = float(df[df.K >= 3].correct.mean()) if (df.K >= 3).any() else None
        summ["test_ece"] = ece_score(df.conf, df.correct, cfg["experiment"]["ece_bins"])
        summ["test_pred_dist"] = {L: float((df.pred == L).mean()) for L in LETTERS}
        log(f"[ScienceQA test] 정확도 {summ['test_acc']:.4f}  ECE {summ['test_ece']:.4f}  "
            f"K별 {summ['test_acc_by_K']}  ({summ['minutes']['scienceqa']}분)")
        if args.perm:
            t = tc.timer()
            pdf, n_extra = perm_scores(model, processor, recs, lp, images, lids, bs)
            df = df.merge(pdf, on="item_id")
            summ["test_acc_perm"] = float(df.correct_perm.mean())
            summ["perm_gap"] = summ["test_acc_perm"] - summ["test_acc"]
            summ["perm_extra_items"] = n_extra
            summ["minutes"]["perm"] = round(t() / 60, 2)
            log(f"[ScienceQA 순환 순열] 정확도 {summ['test_acc_perm']:.4f}  "
                f"(표준과 차이 {summ['perm_gap']:+.4f}, 추가 채점 {n_extra}회, {summ['minutes']['perm']}분)")
        df.to_csv(os.path.join(out, "preds_scienceqa.csv"), index=False)
        del images

    if "ai2d" in sets:
        t = tc.timer()
        recs, images, chk = ai2d_recs(cfg, args.limit)
        checks["ai2d"] = chk
        df = mc_table(recs, mc_score(model, processor, recs, images, lids, bs))
        df.to_csv(os.path.join(out, "preds_ai2d.csv"), index=False)
        summ["ai2d_n"] = int(len(df))
        summ["ai2d_acc"] = float(df.correct.mean())
        summ["ai2d_answer_base"] = chk["answer_base"]
        summ["minutes"]["ai2d"] = round(t() / 60, 2)
        log(f"[AI2D] 정확도 {summ['ai2d_acc']:.4f}  (정답 번호 {chk['answer_base']}부터: {chk['reason']}, "
            f"{summ['minutes']['ai2d']}분)")
        del images

    if "pope" in sets:
        t = tc.timer()
        recs, images, chk = pope_recs(cfg, args.limit)
        checks["pope"] = chk
        texts = [tc.chat_text_prompt(processor, r["question"] + POPE_SUFFIX) for r in recs]
        ids = [yn[" Yes"], yn["Yes"], yn[" No"], yn["No"]]
        sc = tc.score_tokens(model, processor, texts, [images[r["img_key"]] for r in recs], ids, bs)
        yes_ids, no_ids = sorted({0, 1} if ids[0] != ids[1] else {0}), sorted({2, 3} if ids[2] != ids[3] else {2})
        s_yes = np.logaddexp.reduce(sc[:, yes_ids], axis=1)
        s_no = np.logaddexp.reduce(sc[:, no_ids], axis=1)
        df = pd.DataFrame([{k: r[k] for k in ("item_id", "category", "answer")} for r in recs])
        df["s_yes"], df["s_no"] = s_yes, s_no
        df["pred"] = np.where(s_yes > s_no, "yes", "no")
        df["correct"] = df.pred == df.answer
        df.to_csv(os.path.join(out, "preds_pope.csv"), index=False)
        for cat, g in df.groupby("category"):
            summ[f"pope_{cat}"] = yes_no_metrics(g)
        summ["minutes"]["pope"] = round(t() / 60, 2)
        log("[POPE] " + "  ".join(f"{c}: acc {summ['pope_' + c]['acc']:.4f} F1 {summ['pope_' + c]['f1']:.4f} "
                                  f"yes {summ['pope_' + c]['yes_ratio']:.2f}" for c in sorted(df.category.unique()))
            + f"  ({summ['minutes']['pope']}분)")
        del images

    summ["minutes"]["total"] = round(total() / 60, 2)
    summ["env"] = tc.env_info(cfg["models"]["smolvlm"]["id"], cfg["models"]["smolvlm"]["revision"])
    summ["git"] = tc.git_info()
    summ["finished"] = datetime.now().isoformat(timespec="seconds")
    tc.save_json(checks, os.path.join(out, "eval_checks.json"))
    tc.save_json(summ, os.path.join(out, "eval_summary.json"))
    if args.base and not os.path.exists(os.path.join(out, "run_config.json")):
        tc.save_json({"run_id": args.run_id, "base": True, "env": summ["env"], "git": summ["git"]},
                     os.path.join(out, "run_config.json"))
    open(os.path.join(out, "EVAL_DONE"), "w").close()
    log(f"===== 평가 끝: {args.run_id} 총 {summ['minutes']['total']}분 =====")


if __name__ == "__main__":
    main()
