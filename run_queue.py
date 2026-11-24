"""여러 실행을 GPU 개수만큼 동시에 돌리는 실행기 (Kaggle T4 x2용)

두 가지 방식
  python scripts/run_queue.py --pilot             # 파일럿 P3K, P4~P7 (학습만, 예전 방식 그대로)
  python scripts/run_queue.py --session M1        # 본실험 세션 (학습 + 평가)

본실험 세션에서 하는 일
  1) 환경 검사: GPU·torch·transformers·peft가 config experiment.expected_env와 같은지 (스모크는 경고만)
  2) 이전 결과 가져오기 (/kaggle/input 아래를 자동으로 찾음)
     - 재사용: M1의 smolvlm_main-Rand_N400_s0 ← 파일럿 P3K LoRA (torch가 파일럿과 같을 때만)
     - 남은 실행: 다른 세션 Output의 leftover_*.json에 적힌 실행을 먼저 처리 (학습까지 끝난 것은 평가만)
     - 이미 다른 곳에서 EVAL_DONE까지 끝난 실행은 건너뜀
  3) 실행 1개 = 04_train.py(학습 DONE 없을 때) → 05_evaluate.py, 같은 GPU에서 이어서
  4) 시간 예산: 남은 시간 >= 예상 시간 x 1.2 일 때만 시작. 끝난 실행의 실제 시간으로 예상을 보정
  5) 실행이 끝날 때마다 session_results_{세션}.csv, leftover_{세션}.json, session_{세션}.zip(LoRA 제외) 갱신
  6) 실패한 실행은 로그 끝부분을 보여 주고, 시간이 남으면 맨 뒤에서 1번 더 시도
"""
import argparse
import glob
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_config, load_json, save_json  # noqa: E402


def runs_dir(cfg):
    """train_common.runs_dir와 같음 (실행기는 torch를 불러오지 않음)."""
    d = os.environ.get("HN_RUNS_DIR") or cfg["paths"]["runs_dir"]
    parent = os.path.dirname(d.rstrip("/"))
    if not os.path.isdir(parent):
        raise SystemExit(f"[멈춤] {parent} 폴더가 없습니다. 저장 위치(HN_RUNS_DIR)를 확인하세요.")
    os.makedirs(d, exist_ok=True)
    return d

PILOT_ORDER = ["P4", "P5", "P6", "P7", "P3K"]  # 2개씩: (P4,P5) → (P6,P7) → P3K
INPUT_DIR = os.environ.get("HN_INPUT_DIR", "/kaggle/input")


def gpu_names():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
        return [l.split(":", 1)[1].split("(")[0].strip() for l in out.splitlines() if l.startswith("GPU ")]
    except FileNotFoundError:
        return []


def gpu_count():
    return max(1, len(gpu_names()))


def tail(path, n=25):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except FileNotFoundError:
        return "(로그 없음)"


def last_progress(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = [l.strip() for l in f if l.strip().startswith(("step", "[", "====="))]
        return lines[-1] if lines else "(준비 중)"
    except FileNotFoundError:
        return "(준비 중)"


# ================================================================== 파일럿 (예전 방식)
def pilot_jobs(cfg, rdir):
    from train_common import pilot_run_id
    p = os.path.join(rdir, "pilot_lr.json")
    lr = load_json(p)["chosen_lr"] if os.path.exists(p) else cfg["train"]["lr"]
    runs = cfg["pilot"]["runs"]
    return [{"run_id": pilot_run_id(k, runs[k], lr), "trainset": runs[k]["trainset"], "lr": lr, "seed": 0}
            for k in PILOT_ORDER]


def run_pilot(args, cfg, rdir, logdir):
    jobs = pilot_jobs(cfg, rdir) if args.pilot else json.load(open(args.jobs, encoding="utf-8"))
    todo = [j for j in jobs if not os.path.exists(os.path.join(rdir, j["run_id"], "DONE"))]
    n_gpu = args.gpus or gpu_count()
    print(f"[대기열] 남은 학습 {len(todo)}개, GPU {n_gpu}개", flush=True)
    free, running, failed = list(range(n_gpu)), {}, []
    while todo or running:
        while todo and free:
            g, j = free.pop(0), todo.pop(0)
            logp = os.path.join(logdir, f"{j['run_id']}.log")
            cmd = [sys.executable, os.path.join(ROOT, "scripts", "04_train.py"), "--trainset", j["trainset"],
                   "--run_id", j["run_id"], "--lr", str(j["lr"]), "--seed", str(j["seed"])]
            logf = open(logp, "w", encoding="utf-8")
            proc = subprocess.Popen(cmd, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g)),
                                    stdout=logf, stderr=subprocess.STDOUT)
            running[g] = (proc, j, logf, logp)
            print(f"[시작] GPU{g}  {j['run_id']}", flush=True)
        time.sleep(20)
        for g in list(running):
            proc, j, logf, logp = running[g]
            if proc.poll() is None:
                continue
            logf.close()
            if proc.returncode != 0:
                failed.append(j["run_id"])
                print(f"[실패] {j['run_id']}\n{tail(logp)}", flush=True)
            else:
                print(f"[끝]  {j['run_id']}", flush=True)
            del running[g]
            free.append(g)
    if failed:
        sys.exit(1)


# ================================================================== 본실험 세션
def check_env(cfg, smoke):
    import importlib.metadata as md
    exp = cfg["experiment"]["expected_env"]
    names = gpu_names()
    got = {"gpu": names[0] if names else None, "n_gpu": len(names), "python": sys.version.split()[0]}
    for pkg in ("torch", "transformers", "peft", "accelerate", "datasets"):
        try:
            got[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            got[pkg] = None
    bad = {k: (got.get(k), v) for k, v in exp.items() if str(got.get(k)) != str(v)}
    print("[환경] " + "  ".join(f"{k}={got[k]}" for k in ("gpu", "n_gpu", "torch", "transformers", "peft", "python")),
          flush=True)
    if bad:
        msg = "  ".join(f"{k}: 지금 {a} / 기대 {b}" for k, (a, b) in bad.items())
        if smoke:
            print(f"[경고] 환경이 expected_env와 다릅니다 → {msg}\n"
                  f"       두 계정의 스모크 결과가 서로 같은지 비교하세요 (설계서 8장 규칙).", flush=True)
        else:
            raise SystemExit(f"[멈춤] 환경이 expected_env와 다릅니다 → {msg}\n"
                             f"       Environment: Pin to original environment 설정을 확인하세요.")
    else:
        print("[환경] expected_env와 모두 같습니다.", flush=True)
    got["matches_expected"] = not bad
    got["reuse_ok"] = got["torch"] == cfg["experiment"]["pilot_env_torch"]
    got["pilot_torch"] = cfg["experiment"]["pilot_env_torch"]
    return got


def find_dirs(name):
    """Input 아래에서 이름이 name인 폴더들 (얕은 곳부터)."""
    hits = glob.glob(os.path.join(INPUT_DIR, "**", name), recursive=True)
    return sorted([h for h in hits if os.path.isdir(h)], key=lambda p: p.count(os.sep))


def copy_run(src, dst, note):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("EVAL_DONE", "eval_summary.json"))
    rc = os.path.join(dst, "run_config.json")
    if os.path.exists(rc):
        c = load_json(rc)
        c["copied_from"] = {"path": src, "note": note, "when": datetime.now().isoformat(timespec="seconds")}
        save_json(c, rc)


def prepare_inputs(jobs, env, rdir, all_jobs, session):
    """재사용 복사, 다른 세션의 남은 실행 추가, 이미 끝난 실행 표시."""
    extra, notes = [], []
    if os.path.isdir(INPUT_DIR):
        for lp in sorted(glob.glob(os.path.join(INPUT_DIR, "**", "leftover_*.json"), recursive=True)):
            left = load_json(lp)
            if left.get("session") == session:
                continue
            for rid in left.get("jobs", []):
                if rid in all_jobs and rid not in [j["run_id"] for j in jobs + extra]:
                    extra.append(dict(all_jobs[rid]))
                    notes.append(f"[가져옴] {rid} ← {left.get('session')} 세션의 남은 실행")
    jobs = extra + jobs
    for j in jobs:
        rid, dst = j["run_id"], os.path.join(rdir, j["run_id"])
        if os.path.exists(os.path.join(dst, "EVAL_DONE")):
            continue
        done_elsewhere = [d for d in find_dirs(rid) if os.path.exists(os.path.join(d, "EVAL_DONE"))]
        if done_elsewhere:
            j["skip_reason"] = f"다른 Output에서 이미 끝남: {done_elsewhere[0]}"
            continue
        trained = [d for d in find_dirs(rid) if os.path.exists(os.path.join(d, "DONE"))
                   and os.path.isdir(os.path.join(d, "adapter"))]
        if trained and not os.path.exists(os.path.join(dst, "DONE")):
            copy_run(trained[0], dst, "이전 세션에서 학습까지 끝남 → 평가만")
            notes.append(f"[가져옴] {rid}: 학습 결과 복사 → 평가만")
        if j.get("reuse_from") and not os.path.exists(os.path.join(dst, "DONE")):
            src = [d for d in find_dirs(j["reuse_from"]) if os.path.exists(os.path.join(d, "DONE"))
                   and os.path.isdir(os.path.join(d, "adapter"))]
            if not env["reuse_ok"]:
                notes.append(f"[재사용 안 함] {rid}: torch {env['torch']} ≠ 파일럿 {env['pilot_torch']} → 새로 학습")
            elif not src:
                notes.append(f"[재사용 안 함] {rid}: Input에서 {j['reuse_from']}(LoRA 포함)를 못 찾음 → 새로 학습")
            else:
                copy_run(src[0], dst, f"파일럿 {j['reuse_from']} 재사용 (평가만)")
                c = load_json(os.path.join(dst, "run_config.json"))
                c["reused_from"] = j["reuse_from"]
                save_json(c, os.path.join(dst, "run_config.json"))
                s = load_json(os.path.join(dst, "summary.json"))
                s["reused_from"], s["run_id"] = j["reuse_from"], rid
                save_json(s, os.path.join(dst, "summary.json"))
                notes.append(f"[재사용] {rid} ← {src[0]}")
    for n in notes:
        print(n, flush=True)
    return jobs


def est_minutes(j, rdir):
    trained = os.path.exists(os.path.join(rdir, j["run_id"], "DONE"))
    return (0 if trained or j["kind"] == "zs" else j["est_train_min"]) + j["est_eval_min"]


def job_command(j, rdir, smoke, limit):
    py, sd = shlex.quote(sys.executable), os.path.join(ROOT, "scripts")
    parts = []
    if j["kind"] != "zs" and not os.path.exists(os.path.join(rdir, j["run_id"], "DONE")):
        parts.append(f"{py} {sd}/04_train.py --trainset {j['trainset']} --run_id {j['run_id']} "
                     f"--lr {j['lr']} --seed {j['seed']}" + (" --smoke" if smoke else ""))
    ev = f"{py} {sd}/05_evaluate.py --run_id {j['run_id']} --sets {','.join(j['eval_sets'])}"
    ev += (" --perm" if j["perm"] else "") + (" --base" if j["kind"] == "zs" else "")
    ev += f" --limit {limit}" if smoke else ""
    parts.append(ev)
    return " && ".join(parts)


def result_row(j, rdir, status, minutes=None):
    d = os.path.join(rdir, j["run_id"])
    row = {"run_id": j["run_id"], "kind": j["kind"], "cond": j["cond"], "N": j["N"], "seed": j["seed"],
           "status": status, "minutes": round(minutes, 1) if minutes else None}
    sp, ep = os.path.join(d, "summary.json"), os.path.join(d, "eval_summary.json")
    if os.path.exists(sp):
        s = load_json(sp)
        for k in ("best_val_acc", "best_step", "final_step", "stopped_early", "diverged", "attempt",
                  "letter_rate", "train_min", "reused_from"):
            row[k] = s.get(k)
    if os.path.exists(ep):
        e = load_json(ep)
        for k in ("val_acc", "test_acc", "test_acc_K3plus", "test_ece", "test_acc_perm", "perm_gap", "ai2d_acc"):
            row[k] = e.get(k)
        for k, v in (e.get("test_acc_by_K") or {}).items():
            row[f"test_acc_K{k}"] = v
        for k, v in (e.get("test_acc_by_subject") or {}).items():
            row[f"test_acc_{k.split()[0]}"] = v
        for cat in ("random", "adversarial"):
            for m in ("acc", "f1", "yes_ratio"):
                row[f"pope_{cat}_{m}"] = (e.get(f"pope_{cat}") or {}).get(m)
        row["eval_min"] = e.get("minutes", {}).get("total")
    return row


def write_outputs(session, jobs, status, rdir, work, env, minutes):
    import csv
    rows = [result_row(j, rdir, status.get(j["run_id"], "todo"), minutes.get(j["run_id"])) for j in jobs]
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    rp = os.path.join(work, f"session_results_{session}.csv")
    with open(rp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    left = [j["run_id"] for j in jobs if status.get(j["run_id"]) not in ("done", "skipped")]
    lp = os.path.join(work, f"leftover_{session}.json")
    save_json({"session": session, "updated": datetime.now().isoformat(timespec="seconds"), "jobs": left,
               "reasons": {r: status.get(r, "todo") for r in left}}, lp)
    zp = os.path.join(work, f"session_{session}.zip")
    tmp = zp + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(rp, os.path.basename(rp))
        z.write(lp, os.path.basename(lp))
        z.writestr(f"env_{session}.json", json.dumps(env, ensure_ascii=False, indent=2))
        for j in jobs:
            d = os.path.join(rdir, j["run_id"])
            for sub in [d] + glob.glob(d + "_diverged_try*"):
                for root, dirs, files in os.walk(sub):
                    dirs[:] = [x for x in dirs if x != "adapter"]  # LoRA 가중치는 빼고
                    for fn in files:
                        p = os.path.join(root, fn)
                        z.write(p, os.path.join("runs", os.path.relpath(p, rdir)))
            lg = os.path.join(rdir, "_logs", f"{j['run_id']}.log")
            if os.path.exists(lg):
                z.write(lg, os.path.join("runs", "_logs", os.path.basename(lg)))
    os.replace(tmp, zp)
    return rows, left


def print_table(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    if "test_acc" not in df or df.test_acc.isna().all():
        return
    t = df[df.kind != "zs"].dropna(subset=["test_acc"])
    if len(t):
        piv = t.pivot_table(index=["kind", "cond"], columns="N", values="test_acc", aggfunc="mean")
        print("\n[조건 × N별 test 정확도 (이 세션에 있는 시드 평균)]")
        print(piv.round(4).to_string(), flush=True)
    z = df[df.kind == "zs"].dropna(subset=["test_acc"])
    for _, r in z.iterrows():
        print(f"[무학습] test {r.test_acc:.4f}  순열 {r.get('test_acc_perm')}  AI2D {r.get('ai2d_acc')}  "
              f"POPE-random F1 {r.get('pope_random_f1')}  POPE-adv F1 {r.get('pope_adversarial_f1')}", flush=True)


def run_session(args, cfg, rdir, logdir):
    ex = cfg["experiment"]
    sess = load_json(os.path.join(ROOT, "jobs", "sessions", f"{args.session}.json"))
    smoke = sess.get("smoke", False)
    all_jobs = {j["run_id"]: j for j in load_json(os.path.join(ROOT, "jobs", "all_jobs.json"))}
    jobs = [dict(j) for j in sess["job_defs"]] if smoke else [dict(all_jobs[r]) for r in sess["jobs"]]
    work = os.path.dirname(rdir.rstrip("/"))
    t0 = float(os.environ.get("HN_T0", time.time()))
    budget = (args.budget_h or ex["time_budget_h"]) * 60
    margin = ex["start_margin"]
    limit = ex["smoke"]["eval_limit"]
    print(f"===== 세션 {args.session}: {sess['desc']} (실행 {len(jobs)}개, 예산 {budget / 60:.1f}시간) =====",
          flush=True)

    env = check_env(cfg, smoke)
    env.update({"session": args.session, "checked": datetime.now().isoformat(timespec="seconds")})
    try:
        env["git"] = subprocess.run(["git", "describe", "--tags", "--always"], cwd=ROOT,
                                    capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        env["git"] = None
    save_json(env, os.path.join(rdir, f"_session_{args.session}", "env.json"))

    jobs = prepare_inputs(jobs, env, rdir, all_jobs, args.session)
    status, minutes, attempts, ratios = {}, {}, {}, []
    todo = []
    for j in jobs:
        if os.path.exists(os.path.join(rdir, j["run_id"], "EVAL_DONE")):
            status[j["run_id"]] = "done"
            print(f"[건너뜀] {j['run_id']} (EVAL_DONE 있음)", flush=True)
        elif j.get("skip_reason"):
            status[j["run_id"]] = "skipped"
            print(f"[건너뜀] {j['run_id']}: {j['skip_reason']}", flush=True)
        else:
            todo.append(j)
    n_gpu = args.gpus or gpu_count()
    est_total = sum(est_minutes(j, rdir) for j in todo)
    print(f"[대기열] 실행 {len(todo)}개, 예상 GPU {est_total / 60:.1f}시간 → GPU {n_gpu}개로 약 "
          f"{est_total / 60 / n_gpu:.1f}시간", flush=True)
    write_outputs(args.session, jobs, status, rdir, work, env, minutes)

    def ratio():
        if not ratios:
            return 1.0
        r = sorted(ratios)[len(ratios) // 2]
        return min(max(r, 0.6), 2.5)

    free, running, last_beat = list(range(n_gpu)), {}, time.time()
    while todo or running:
        while free and todo:
            left_min = budget - (time.time() - t0) / 60
            pick = next((j for j in todo if est_minutes(j, rdir) * ratio() * margin <= left_min), None)
            if pick is None:
                break
            todo.remove(pick)
            g = free.pop(0)
            rid = pick["run_id"]
            logp = os.path.join(logdir, f"{rid}.log")
            logf = open(logp, "a", encoding="utf-8")
            est = est_minutes(pick, rdir)
            cmd = job_command(pick, rdir, smoke, limit)
            proc = subprocess.Popen(["bash", "-c", cmd], cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED="1"))
            running[g] = (proc, pick, time.time(), logf, logp, est)
            status[rid] = "running"
            print(f"[시작] GPU{g}  {rid}  (예상 {est * ratio():.0f}분, 남은 예산 {left_min:.0f}분)", flush=True)
        if not running:
            break  # 남은 실행은 예산 안에 끝낼 수 없음
        time.sleep(float(os.environ.get("HN_POLL_SEC", 5 if smoke else 20)))
        for g in list(running):
            proc, j, ts, logf, logp, est = running[g]
            if proc.poll() is None:
                continue
            logf.close()
            rid, mins = j["run_id"], (time.time() - ts) / 60
            del running[g]
            free.append(g)
            if proc.returncode == 0 and os.path.exists(os.path.join(rdir, rid, "EVAL_DONE")):
                status[rid], minutes[rid] = "done", mins
                if est > 0:
                    ratios.append(mins / est)
                e = load_json(os.path.join(rdir, rid, "eval_summary.json"))
                s = load_json(os.path.join(rdir, rid, "summary.json")) if os.path.exists(
                    os.path.join(rdir, rid, "summary.json")) else {}
                flag = "  ⚠ 발산" if s.get("diverged") else ""
                val = s.get("best_val_acc", e.get("val_acc"))
                print(f"[끝]  GPU{g}  {rid}  val {val}  test {e.get('test_acc', float('nan')):.4f}"
                      + (f"  순열 {e['test_acc_perm']:.4f}" if e.get("test_acc_perm") is not None else "")
                      + (f"  AI2D {e['ai2d_acc']:.4f}" if e.get("ai2d_acc") is not None else "")
                      + f"  ({mins:.0f}분, 예상 대비 {mins / max(est, 1):.2f}배){flag}", flush=True)
            else:
                attempts[rid] = attempts.get(rid, 0) + 1
                print(f"[실패] GPU{g}  {rid}  (종료 코드 {proc.returncode}, {mins:.0f}분). 로그 끝부분:\n{tail(logp)}",
                      flush=True)
                if attempts[rid] < 2:
                    todo.append(j)
                    status[rid] = "retry"
                    print(f"       → 맨 뒤에서 1번 더 시도합니다.", flush=True)
                else:
                    status[rid] = "failed"
            write_outputs(args.session, jobs, status, rdir, work, env, minutes)
        if running and time.time() - last_beat > args.heartbeat_min * 60:
            last_beat = time.time()
            used = (time.time() - t0) / 60
            print(f"  … 경과 {used:.0f}분 / 예산 {budget:.0f}분, 남은 대기 {len(todo)}개", flush=True)
            for g, (_, j, ts, _, logp, _) in running.items():
                print(f"    GPU{g} {j['run_id']} {(time.time() - ts) / 60:.0f}분 | {last_progress(logp)}", flush=True)

    for j in todo:
        status[j["run_id"]] = "time" if status.get(j["run_id"]) != "retry" else "failed"
    rows, left = write_outputs(args.session, jobs, status, rdir, work, env, minutes)
    print_table(rows)
    n_done = sum(1 for r in rows if r["status"] in ("done", "skipped"))
    print(f"\n===== 세션 {args.session} 끝: 완료 {n_done}/{len(rows)}, 실패 "
          f"{[r for r, s in status.items() if s == 'failed']}, 남은 실행 {len(left)}개, "
          f"경과 {(time.time() - t0) / 60:.0f}분 =====", flush=True)
    if left:
        print(f"  남은 실행 → leftover_{args.session}.json (SPILL 노트북 Input으로 이 Output을 추가)", flush=True)
    print(f"  내려받기: {os.path.join(work, f'session_{args.session}.zip')}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="파일럿 P3K, P4~P7")
    ap.add_argument("--jobs", help="(파일럿 방식) 학습 목록 json [{run_id, trainset, lr, seed}, ...]")
    ap.add_argument("--session", help="본실험 세션 이름 (SMOKE, M1~M4, X1, X2, SPILL)")
    ap.add_argument("--gpus", type=int, default=None, help="동시에 쓸 GPU 수 (기본: 전부)")
    ap.add_argument("--budget_h", type=float, default=None, help="시간 예산 (기본: config)")
    ap.add_argument("--heartbeat_min", type=float, default=10)
    args = ap.parse_args()
    cfg = load_config()
    rdir = runs_dir(cfg)
    logdir = os.path.join(rdir, "_logs")
    os.makedirs(logdir, exist_ok=True)
    if args.session:
        run_session(args, cfg, rdir, logdir)
    else:
        run_pilot(args, cfg, rdir, logdir)


if __name__ == "__main__":
    main()
