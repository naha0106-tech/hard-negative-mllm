"""7단계: 세션 결과 모으기 (코랩 CPU, GPU 필요 없음)

하는 일
  1) results/sessions/session_*.zip 을 모두 풀어 실행별 결과를 한곳에 모음
     (같은 실행이 여러 세션에 있으면 EVAL_DONE이 있는 것 중 먼저 끝난 것을 쓰고 기록)
  2) jobs/all_jobs.json(103개)과 비교해 빠진 실행·실패·발산 확인 → results/leftover_ALL.json (SPILL Input으로 쓸 수 있음)
  3) 세션별 환경(torch 등)이 모두 같은지 확인
  4) 저장
       results/all_runs.csv          실행 1개 = 1줄 (모든 지표)
       results/item_level.parquet    실행 × 문항 정답 여부 (ScienceQA test 표준·순열, AI2D, POPE) → 06_analyze.py 입력
       results/summary_main.csv      주 실험 조건 × N별 평균·표준편차 (test, 순열, AI2D, POPE F1)
       results/summary_aux.csv       보조 실험 조건 × N별 평균·표준편차 (test)
       results/collect_report.json   완료 수, 빠진 실행, 환경 점검

실행
  python scripts/07_collect.py                       # results/sessions/*.zip
  python scripts/07_collect.py --zips 경로1.zip 경로2.zip
결론(가설 판정)은 여기서 내지 않습니다. 06_analyze.py에서만 냅니다.
"""
import argparse
import glob
import json
import os
import sys
import zipfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_json, save_json  # noqa: E402

RES = os.path.join(ROOT, "results")


def unpack(zips, dst):
    os.makedirs(dst, exist_ok=True)
    envs, sessions = {}, []
    for zp in sorted(zips):
        name = os.path.basename(zp).replace(".zip", "")
        sess = name.replace("session_", "")
        out = os.path.join(dst, sess)
        with zipfile.ZipFile(zp) as z:
            z.extractall(out)
        sessions.append(sess)
        ep = os.path.join(out, f"env_{sess}.json")
        if os.path.exists(ep):
            envs[sess] = load_json(ep)
    return sessions, envs


def flatten(e, s, rc):
    row = {}
    if s:
        for k in ("best_val_acc", "best_step", "final_step", "max_steps", "stopped_early", "diverged",
                  "attempt", "letter_rate", "train_min", "reused_from"):
            row[k] = s.get(k)
    for k in ("val_acc", "test_n", "test_acc", "test_acc_K3plus", "test_ece", "test_acc_perm", "perm_gap",
              "ai2d_n", "ai2d_acc", "ai2d_answer_base"):
        row[k] = e.get(k)
    for k, v in (e.get("test_acc_by_K") or {}).items():
        row[f"test_acc_K{k}"] = v
    for k, v in (e.get("test_acc_by_subject") or {}).items():
        row[f"test_acc_{k.split()[0]}"] = v
    for cat in ("random", "adversarial"):
        for m, v in (e.get(f"pope_{cat}") or {}).items():
            row[f"pope_{cat}_{m}"] = v
    row["eval_min"] = (e.get("minutes") or {}).get("total")
    row["torch"] = (e.get("env") or {}).get("torch")
    row["git"] = (e.get("git") or {}).get("describe")
    row["smoke"] = bool((rc or {}).get("smoke")) or e.get("limit") is not None
    return row


def item_rows(run_dir, rid):
    parts = []
    p = os.path.join(run_dir, "preds_scienceqa.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        cols = ["item_id", "K", "subject", "grade_band", "answer", "pred", "correct", "conf"]
        x = d[[c for c in cols if c in d.columns]].assign(eval_set="scienceqa")
        parts.append(x)
        if "correct_perm" in d.columns:
            parts.append(d[["item_id", "K", "subject", "grade_band", "answer"]]
                         .assign(pred=d.pred_perm, correct=d.correct_perm, eval_set="scienceqa_perm"))
    for name in ("ai2d", "pope"):
        p = os.path.join(run_dir, f"preds_{name}.csv")
        if os.path.exists(p):
            d = pd.read_csv(p)
            keep = [c for c in ("item_id", "K", "category", "answer", "pred", "correct", "conf") if c in d.columns]
            parts.append(d[keep].assign(eval_set=name))
    if not parts:
        return None
    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "run_id", rid)
    out["correct"] = out["correct"].astype(bool).astype(int)
    return out


def mean_sd(df, index, values):
    g = df.groupby(index)[values]
    m, s, n = g.mean(), g.std(ddof=1), g.count()
    out = m.add_suffix("_mean").join(s.add_suffix("_sd")).join(n[[values[0]]].rename(columns={values[0]: "n_seeds"}))
    return out.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips", nargs="*", default=None)
    ap.add_argument("--include_smoke", action="store_true")
    args = ap.parse_args()
    zips = args.zips or sorted(glob.glob(os.path.join(RES, "sessions", "session_*.zip")))
    zips = [z for z in zips if args.include_smoke or "SMOKE" not in os.path.basename(z)]
    if not zips:
        raise SystemExit("[멈춤] results/sessions/ 에 session_*.zip 이 없습니다.")
    work = os.path.join(RES, "_unpacked")
    sessions, envs = unpack(zips, work)
    print(f"[세션] {sessions}")

    jobs = {j["run_id"]: j for j in load_json(os.path.join(ROOT, "jobs", "all_jobs.json"))}
    found = {}  # run_id -> list of (session, dir, finished)
    for sess in sessions:
        for ep in glob.glob(os.path.join(work, sess, "runs", "*", "EVAL_DONE")):
            d = os.path.dirname(ep)
            rid = os.path.basename(d)
            e = load_json(os.path.join(d, "eval_summary.json"))
            found.setdefault(rid, []).append((sess, d, e.get("finished", "")))

    rows, items, dups = [], [], {}
    for rid, hits in sorted(found.items()):
        hits.sort(key=lambda h: h[2])
        if len(hits) > 1:
            dups[rid] = [h[0] for h in hits]
        sess, d, _ = hits[0]
        e = load_json(os.path.join(d, "eval_summary.json"))
        s = load_json(os.path.join(d, "summary.json")) if os.path.exists(os.path.join(d, "summary.json")) else {}
        rc = load_json(os.path.join(d, "run_config.json")) if os.path.exists(os.path.join(d, "run_config.json")) else {}
        j = jobs.get(rid, {})
        row = {"run_id": rid, "session": sess, "kind": j.get("kind"), "cond": j.get("cond"), "N": j.get("N"),
               "seed": j.get("seed"), "planned_session": j.get("session"), **flatten(e, s, rc)}
        rows.append(row)
        if rid in jobs:
            it = item_rows(d, rid)
            if it is not None:
                items.append(it)
    allr = pd.DataFrame(rows)
    main_runs = allr[allr.run_id.isin(jobs)] if len(allr) else allr

    missing = [r for r in jobs if r not in set(main_runs.run_id)]
    diverged = main_runs[main_runs.diverged == True].run_id.tolist() if "diverged" in main_runs else []  # noqa: E712
    torch_set = sorted({str(v.get("torch")) for v in envs.values()})
    run_torch = sorted(main_runs.torch.dropna().unique().tolist()) if "torch" in main_runs else []
    ai2d_base = sorted(main_runs.ai2d_answer_base.dropna().unique().tolist()) if "ai2d_answer_base" in main_runs else []

    os.makedirs(RES, exist_ok=True)
    main_runs.to_csv(os.path.join(RES, "all_runs.csv"), index=False)
    if items:
        pd.concat(items, ignore_index=True).to_parquet(os.path.join(RES, "item_level.parquet"), index=False)
    if len(main_runs):
        m = main_runs[main_runs.kind == "main"]
        if len(m):
            vals = [c for c in ("test_acc", "test_acc_perm", "test_acc_K3plus", "ai2d_acc", "pope_random_f1",
                                "pope_adversarial_f1", "test_ece", "best_val_acc") if c in m]
            mean_sd(m, ["cond", "N"], vals).to_csv(os.path.join(RES, "summary_main.csv"), index=False)
            print("\n[주 실험] 조건 × N별 test 정확도 평균 (시드 수)")
            print(m.pivot_table(index="cond", columns="N", values="test_acc", aggfunc="mean").round(4).to_string())
        a = main_runs[main_runs.kind == "aux"]
        if len(a):
            mean_sd(a, ["cond", "N"], ["test_acc", "best_val_acc"]).to_csv(os.path.join(RES, "summary_aux.csv"),
                                                                             index=False)
            print("\n[보조 실험] 조건 × N별 test 정확도 평균")
            print(a.pivot_table(index="cond", columns="N", values="test_acc", aggfunc="mean").round(4).to_string())
        z = main_runs[main_runs.kind == "zs"]
        for _, r in z.iterrows():
            print(f"\n[무학습] test {r.test_acc:.4f}  순열 {r.test_acc_perm}  AI2D {r.ai2d_acc}  "
                  f"POPE F1 random {r.get('pope_random_f1')} / adversarial {r.get('pope_adversarial_f1')}")

    report = {"sessions": sessions, "n_planned": len(jobs), "n_done": int(len(main_runs)),
              "missing": missing, "diverged": diverged, "duplicates": dups,
              "env_torch_by_session": {k: v.get("torch") for k, v in envs.items()},
              "env_same_torch": len(torch_set) <= 1 and len(run_torch) <= 1, "run_torch": run_torch,
              "ai2d_answer_base": ai2d_base}
    save_json(report, os.path.join(RES, "collect_report.json"))
    save_json({"session": "ALL", "jobs": missing}, os.path.join(RES, "leftover_ALL.json"))
    print(f"\n[완료] {len(main_runs)}/{len(jobs)}개 실행. 빠진 실행 {len(missing)}개, 발산 {len(diverged)}개, "
          f"중복 {len(dups)}개")
    if missing:
        print("  빠진 실행:", missing[:20], "..." if len(missing) > 20 else "")
    if not report["env_same_torch"]:
        print(f"  ⚠ 세션마다 torch가 다릅니다: {report['env_torch_by_session']} / 실행 {run_torch}")


if __name__ == "__main__":
    main()
