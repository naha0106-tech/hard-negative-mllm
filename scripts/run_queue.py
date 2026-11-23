"""여러 학습을 GPU 개수만큼 동시에 돌리는 실행기 (Kaggle T4 x2용)

- 학습 1개 = 04_train.py 1번. GPU마다 하나씩 붙여 동시에 돌리고, 끝나면 다음 것을 시작합니다.
- 이미 DONE이 있는 학습은 건너뜁니다. 실패한 학습은 로그 끝부분을 보여 주고 다음으로 갑니다.
- 각 학습의 화면 출력은 runs/_logs/{run_id}.log 에 저장됩니다.

실행
  python scripts/run_queue.py --pilot            # 파일럿 P3K, P4~P7
  python scripts/run_queue.py --pilot --gpus 1   # GPU 1개로 차례대로
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_config, load_json  # noqa: E402
from train_common import pilot_run_id, runs_dir  # noqa: E402

PILOT_ORDER = ["P4", "P5", "P6", "P7", "P3K"]  # 2개씩: (P4,P5) → (P6,P7) → P3K


def gpu_count():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
        return max(1, sum(1 for line in out.splitlines() if line.startswith("GPU ")))
    except FileNotFoundError:
        return 1


def pilot_jobs(cfg, rdir):
    p = os.path.join(rdir, "pilot_lr.json")
    lr = load_json(p)["chosen_lr"] if os.path.exists(p) else cfg["train"]["lr"]
    if lr is None:
        raise SystemExit("[멈춤] 학습률이 없습니다. pilot_lr.json 또는 config train.lr을 확인하세요.")
    runs = cfg["pilot"]["runs"]
    return [{"run_id": pilot_run_id(k, runs[k], lr), "trainset": runs[k]["trainset"], "lr": lr, "seed": 0}
            for k in PILOT_ORDER]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="파일럿 P3K, P4~P7")
    ap.add_argument("--jobs", help="학습 목록 json 파일 [{run_id, trainset, lr, seed}, ...]")
    ap.add_argument("--gpus", type=int, default=None, help="동시에 쓸 GPU 수 (기본: 전부)")
    ap.add_argument("--heartbeat_min", type=float, default=10)
    args = ap.parse_args()
    cfg = load_config()
    rdir = runs_dir(cfg)
    logdir = os.path.join(rdir, "_logs")
    os.makedirs(logdir, exist_ok=True)

    jobs = pilot_jobs(cfg, rdir) if args.pilot else json.load(open(args.jobs, encoding="utf-8"))
    todo = [j for j in jobs if not os.path.exists(os.path.join(rdir, j["run_id"], "DONE"))]
    for j in jobs:
        if j not in todo:
            print(f"[건너뜀] {j['run_id']} (DONE 있음)")
    n_gpu = args.gpus or gpu_count()
    print(f"[대기열] 남은 학습 {len(todo)}개, GPU {n_gpu}개로 동시에 실행", flush=True)

    free = list(range(n_gpu))
    running = {}  # gpu -> (proc, job, start, logf)
    t_start, last_beat, failed = time.time(), time.time(), []
    while todo or running:
        while todo and free:
            g, j = free.pop(0), todo.pop(0)
            logp = os.path.join(logdir, f"{j['run_id']}.log")
            logf = open(logp, "w", encoding="utf-8")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED="1")
            cmd = [sys.executable, os.path.join(ROOT, "scripts", "04_train.py"), "--trainset", j["trainset"],
                   "--run_id", j["run_id"], "--lr", str(j["lr"]), "--seed", str(j["seed"])]
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT)
            running[g] = (proc, j, time.time(), logf, logp)
            print(f"[시작] GPU{g}  {j['run_id']}  (lr={j['lr']}, seed={j['seed']})", flush=True)
        time.sleep(20)
        for g in list(running):
            proc, j, t0, logf, logp = running[g]
            if proc.poll() is None:
                continue
            logf.close()
            mins = (time.time() - t0) / 60
            sp = os.path.join(rdir, j["run_id"], "summary.json")
            if proc.returncode == 0 and os.path.exists(os.path.join(rdir, j["run_id"], "DONE")):
                s = load_json(sp)
                flag = "  ⚠ 발산" if s.get("diverged") else ""
                print(f"[끝]  GPU{g}  {j['run_id']}  best val {s['best_val_acc']:.4f} @ step {s['best_step']}"
                      f"  ({mins:.0f}분){flag}", flush=True)
            else:
                failed.append(j["run_id"])
                print(f"[실패] GPU{g}  {j['run_id']}  (종료 코드 {proc.returncode}). 로그 끝부분:\n{tail(logp)}", flush=True)
            del running[g]
            free.append(g)
        if running and time.time() - last_beat > args.heartbeat_min * 60:
            last_beat = time.time()
            for g, (_, j, t0, _, logp) in running.items():
                print(f"  … GPU{g} {j['run_id']} {(time.time() - t0) / 60:.0f}분 경과 | {last_progress(logp)}", flush=True)
    print(f"[완료] 전체 {(time.time() - t_start) / 60:.0f}분" + (f", 실패 {failed}" if failed else ""), flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
