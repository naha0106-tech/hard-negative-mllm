"""4단계 파일럿 성적표: 관문 B1~B6 (파일럿 실행 가이드 8장)

  --stage lr    P1~P3만 보고 학습률을 고름 → runs/pilot_lr.json
  (기본)        P0~P7 전부 보고 B1~B6 판정 → runs/pilot_report.json

GPU가 필요 없습니다. 각 run의 summary.json, val_preds_best.csv만 읽습니다.
실행: python scripts/04b_pilot_report.py --stage lr
      python scripts/04b_pilot_report.py
"""
import argparse
import math
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, load_json  # noqa: E402
import train_common as tc  # noqa: E402

N_TEST = 2017
SIZES_MAIN = [100, 200, 400, 800, 1200]


def run_dir(rdir, rid):
    d = os.path.join(rdir, rid)
    if not os.path.exists(os.path.join(d, "DONE")):
        raise SystemExit(f"[멈춤] {rid}가 아직 끝나지 않았습니다 (DONE 없음).")
    return d


def summary(rdir, rid):
    return load_json(os.path.join(run_dir(rdir, rid), "summary.json"))


def preds(rdir, rid):
    return pd.read_csv(os.path.join(run_dir(rdir, rid), "val_preds_best.csv")).set_index("item_id")


def choose_lr(cfg, rdir):
    runs = cfg["pilot"]["runs"]
    rows = []
    for k in ("P1", "P2", "P3"):
        lr = runs[k]["lr"]
        s = summary(rdir, tc.pilot_run_id(k, runs[k], lr))
        rows.append({"key": k, "lr": lr, "best_val_acc": s["best_val_acc"], "best_step": s["best_step"],
                     "diverged": s["diverged"]})
    ok = [r for r in rows if not r["diverged"]] or rows
    top = max(r["best_val_acc"] for r in ok)
    tol = cfg["pilot"]["tie_items"] / 500 + 1e-9
    chosen = min((r for r in ok if top - r["best_val_acc"] <= tol), key=lambda r: r["lr"])
    return {"chosen_lr": chosen["lr"], "rule": f"최고값과 {cfg['pilot']['tie_items']}문항 이내면 작은 학습률",
            "candidates": rows}


def disagreement(a, b):
    j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    assert len(j) == 500, f"val500이 아닙니다: {len(j)}"
    return {"disagree_rate": float((j.pred_a != j.pred_b).mean()),
            "acc_a": float(j.correct_a.mean()), "acc_b": float(j.correct_b.mean()),
            "only_a_correct": int((j.correct_a & ~j.correct_b).sum()),
            "only_b_correct": int((~j.correct_a & j.correct_b).sum())}


def time_estimate(cfg, sums):
    tcfg = cfg["train"]
    step_min = [s["train_min"] / s["final_step"] for s in sums if s["final_step"]]
    eval_min = [s["eval_min"] / s["n_evals"] for s in sums if s["n_evals"]]
    per_step, per_eval = sum(step_min) / len(step_min), sum(eval_min) / len(eval_min)

    def run_min(n):  # 조기 종료 없는 최대치
        st = tc.max_steps_for(n, tcfg)
        return st * per_step + math.ceil(st / tcfg["eval_every"]) * per_eval

    te = per_eval * cfg["pilot"]["test_eval_factor"]
    main = 12 * sum(run_min(n) for n in SIZES_MAIN)          # 4조건 x 3시드 x 5N
    aux = 21 * (run_min(100) + run_min(1200))                 # 7조건 x 3시드 x N 100·1200
    reused = run_min(400) + 3 * run_min(1200)                 # 파일럿 재사용 (B4 통과 시)
    evals = 103 * te
    return {"min_per_step": round(per_step, 4), "min_per_val500": round(per_eval, 3),
            "T150_min": round(run_min(400), 1), "T200_min": round(run_min(800), 1),
            "T300_min": round(run_min(1200), 1), "Te_min": round(te, 1),
            "main_train_h": round(main / 60, 1), "aux_train_h": round(aux / 60, 1),
            "eval_h": round(evals / 60, 1), "reused_h": round(reused / 60, 1),
            "total_h_if_reused": round((main + aux + evals - reused) / 60, 1),
            "total_h_no_reuse": round((main + aux + evals) / 60, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["lr", "all"], default="all")
    args = ap.parse_args()
    cfg = load_config()
    pc = cfg["pilot"]
    rdir = tc.runs_dir(cfg)

    lr_info = choose_lr(cfg, rdir)
    tc.save_json(lr_info, os.path.join(rdir, "pilot_lr.json"))
    print("[B2 학습률]")
    for r in lr_info["candidates"]:
        print(f"  {r['key']} lr={r['lr']:g}  best_val={r['best_val_acc']:.4f} @ step {r['best_step']}"
              f"{'  (발산)' if r['diverged'] else ''}")
    print(f"  ==> 채택 학습률 {lr_info['chosen_lr']:g}  ({lr_info['rule']})")
    if args.stage == "lr":
        return

    runs, lr = pc["runs"], lr_info["chosen_lr"]
    rid = {k: tc.pilot_run_id(k, runs[k], runs[k]["lr"] if k in ("P1", "P2", "P3") else lr) for k in runs}
    zs_id = "pilot_P0K_ZS" if os.path.exists(os.path.join(rdir, "pilot_P0K_ZS", "DONE")) else "pilot_P0_ZS"
    zs = summary(rdir, zs_id)["val_acc"]
    sums = {k: summary(rdir, rid[k]) for k in rid}
    kaggle_keys = [k for k in ("P3K", "P4", "P5", "P6", "P7") if k in sums]

    b1_val = sums["P4"]["best_val_acc"] - zs
    b1 = {"pass": b1_val >= pc["b1_margin"], "P4_best": sums["P4"]["best_val_acc"], "ZS": zs,
          "gain": round(b1_val, 4), "기준": f">= +{pc['b1_margin']}"}

    he = disagreement(preds(rdir, rid["P5"]), preds(rdir, rid["P6"]))
    rep = disagreement(preds(rdir, rid["P4"]), preds(rdir, rid["P7"]))
    d, noise = he["disagree_rate"], rep["disagree_rate"]
    b4_pass = d >= pc["b4_min_disagree"] and d >= pc["b4_noise_ratio"] * noise
    b4 = {"pass": b4_pass, "Hard_vs_Easy": he, "repeat_P4_vs_P7_disagree": noise,
          "detectable_diff_pp": round(100 * 2.8 * math.sqrt(d * N_TEST) / N_TEST, 2) if d > 0 else None,
          "기준": f"d >= {pc['b4_min_disagree']} 그리고 d >= {pc['b4_noise_ratio']} x 반복 불일치율"}

    b5_diff = abs(sums["P4"]["best_val_acc"] - sums["P7"]["best_val_acc"])
    b5 = {"pass": True, "acc_diff": round(b5_diff, 4), "disagree_rate": noise,
          "warn": b5_diff > pc["b5_warn_diff"], "기준": "기록만 (1%p 넘으면 보고서에 명시)"}
    lr_key = next(k for k in ("P1", "P2", "P3") if runs[k]["lr"] == lr)
    plat = None
    if "P3K" in sums:  # 코랩(P1~P3)과 Kaggle 환경이 같은 결과를 내는지
        pd_ = disagreement(preds(rdir, rid[lr_key]), preds(rdir, rid["P3K"]))
        pdiff = abs(sums[lr_key]["best_val_acc"] - sums["P3K"]["best_val_acc"])
        plat = {"colab": rid[lr_key], "kaggle": rid["P3K"], "colab_acc": sums[lr_key]["best_val_acc"],
                "kaggle_acc": sums["P3K"]["best_val_acc"], "acc_diff": round(pdiff, 4),
                "disagree_rate": pd_["disagree_rate"], "warn": pdiff > pc["platform_warn_diff"]}

    lr_rates = {k: sums[k]["letter_rate"] for k in sums}
    b6 = {"pass": all(v is not None and v >= pc["b6_min_letter_rate"] for v in lr_rates.values()),
          "letter_rate": lr_rates, "기준": f"모두 >= {pc['b6_min_letter_rate']}"}

    n_gpu = int(os.environ.get("HN_N_GPU", "2"))
    est = time_estimate(cfg, [sums[k] for k in (kaggle_keys or list(sums))])
    est["n_gpu_parallel"] = n_gpu
    est["wall_h_parallel"] = round(est["total_h_if_reused"] / n_gpu, 1)
    b3 = {"pass": None, "estimate": est,
          "기준": "가용 GPU 시간과 비교해 사람이 판단 (넘치면 Qwen 생략 → 보조 실험 시드 3→2)"}

    diverged = {k: s["diverged"] for k, s in sums.items() if s["diverged"]}
    n400 = rid["P3K"] if "P3K" in sums else rid[lr_key]  # 본실험과 같은 환경(Kaggle)의 run을 재사용
    if b4_pass:
        reuse = {n400: "smolvlm_main-Rand_N400_s0", rid["P4"]: "smolvlm_main-Rand_N1200_s0",
                 rid["P5"]: "smolvlm_main-Hard_N1200_s0", rid["P6"]: "smolvlm_main-Easy_N1200_s0"}
    else:
        reuse = {}  # 대안 세트로 바뀌면 재사용하지 않음 (P5·P6은 민감도 분석용)
    decisions = {
        "learning_rate": lr,
        "main_trainset": "기존 smolvlm_matched 유지" if b4_pass else "대안 세트(순서 일치 + 차이 >= 1)로 교체",
        "reuse_as_main": reuse,
        "go_to_main": bool(b1["pass"] and b6["pass"] and not diverged),
    }
    report = {"B1": b1, "B2": lr_info, "B3": b3, "B4": b4, "B5": b5, "B6": b6, "platform_check": plat,
              "zs_run": zs_id,
              "diverged_runs": diverged, "run_ids": rid, "decisions": decisions}
    tc.save_json(report, os.path.join(rdir, "pilot_report.json"))

    def mark(x):
        return "통과" if x is True else "미통과" if x is False else "판단 필요"
    print(f"[B1 학습 가능성] {mark(b1['pass'])}  P4 {b1['P4_best']:.4f} - ZS({zs_id}) {zs:.4f} = {b1['gain']:+.4f}")
    e = b3["estimate"]
    print(f"[B3 시간] {mark(None)}  N=400 {e['T150_min']}분, N=1200 {e['T300_min']}분, 평가 {e['Te_min']}분/회"
          f" → 본실험 GPU 시간 약 {e['total_h_if_reused']}시간, GPU {n_gpu}개 동시 실행 시 약 {e['wall_h_parallel']}시간")
    print(f"[B4 조작 확인] {mark(b4_pass)}  Hard-Easy 불일치 {d:.3f} (반복 실행 {noise:.3f}),"
          f" 정확도 Hard {he['acc_a']:.4f} / Easy {he['acc_b']:.4f}, 검출 가능 차이 약 {b4['detectable_diff_pp']}%p")
    print(f"[B5 결정성] 기록  P4-P7 정확도 차 {b5_diff:.4f}{'  ⚠ 1%p 초과' if b5['warn'] else ''}")
    print(f"[B6 글자 출력] {mark(b6['pass'])}  " + ", ".join(f"{k} {v:.3f}" for k, v in lr_rates.items() if v is not None))
    if plat:
        print(f"[환경 확인] 코랩 {plat['colab']} {plat['colab_acc']:.4f} vs Kaggle {plat['kaggle_acc']:.4f}"
              f" (차이 {plat['acc_diff']:.4f}, 답 불일치 {plat['disagree_rate']:.3f})"
              f"{'  ⚠ 1%p 초과: 본실험은 Kaggle 결과만 사용' if plat['warn'] else ''}")
    if diverged:
        print(f"[발산] {list(diverged)}")
    print(f"==> 학습률 {lr:g} / 주 실험 세트: {decisions['main_trainset']} / "
          f"본실험 진행 {'가능' if decisions['go_to_main'] else '보류'}")


if __name__ == "__main__":
    main()
