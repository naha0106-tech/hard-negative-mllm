"""본실험 실행 목록과 세션별 Kaggle 노트북을 한 번에 만듭니다.

만드는 것
  jobs/all_jobs.json                  103개 실행 (무학습 1 + 주 실험 60 + 보조 실험 42)
  jobs/sessions/{SMOKE,M1..M4,X1,X2,SPILL}.json
  notebooks/sessions/run_{세션}.ipynb   팀원이 Kaggle에 Import해서 Save & Run All만 누르는 파일

실행 (코드가 바뀌거나 세션 구성을 바꿀 때만)
  python scripts/make_jobs.py
  python scripts/make_jobs.py --code_ref prereg-v1.1     # 사전 등록 뒤 코드를 고쳤을 때
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_config, save_json  # noqa: E402


def max_steps_for(n, tcfg):
    """train_common.max_steps_for와 같은 식 (torch 없이 돌도록 따로 둠)."""
    eff = tcfg["micro_batch"] * tcfg["grad_accum"]
    return max(tcfg["min_steps"], math.ceil(tcfg["epochs_equiv"] * n / eff))

# 세션 구성 (본실험 실행 설계서 5.1). 계정 A: M1·M3·X1, 계정 B: M2·M4·X2
SESSIONS = {
    "M1": {"account": "A", "desc": "무학습 + 주 실험 시드 0, N 100·200·400·800"},
    "M2": {"account": "B", "desc": "주 실험 시드 1, N 100·200·400·800"},
    "M3": {"account": "A", "desc": "주 실험 시드 2, N 100·200·400·800"},
    "M4": {"account": "B", "desc": "주 실험 N 1,200 × 시드 0·1·2"},
    "X1": {"account": "A", "desc": "보조 시드 0 전부 + 보조 시드 2 N 1,200 (Full 제외)"},
    "X2": {"account": "B", "desc": "보조 시드 1 전부 + 보조 시드 2 N 100 + 보조 Full_N1200_s2"},
}


def session_of(j):
    if j["kind"] == "zs":
        return "M1"
    if j["kind"] == "main":
        if j["N"] == 1200:
            return "M4"
        return {0: "M1", 1: "M2", 2: "M3"}[j["seed"]]
    if j["seed"] == 0:
        return "X1"
    if j["seed"] == 1:
        return "X2"
    if j["N"] == 100 or j["cond"] == "Full":
        return "X2"
    return "X1"


def build_jobs(cfg):
    ex, tcfg = cfg["experiment"], cfg["train"]
    est = ex["est_min"]
    lr = tcfg["lr"]
    jobs = [{"run_id": "smolvlm_ZS", "kind": "zs", "cond": "ZS", "N": 0, "seed": None, "trainset": None,
             "lr": None, "eval_sets": ex["zs_eval_sets"], "perm": True, "reuse_from": None,
             "est_train_min": 0, "est_eval_min": est["eval_zs_val"] + est["eval_full"] + est["eval_perm_extra"]}]
    for kind in ("main", "aux"):
        g = ex[kind]
        for seed in cfg["seeds"]["train"]:
            for n in g["sizes"]:
                for cond in g["conditions"]:
                    rid = f"smolvlm_{kind}-{cond}_N{n}_s{seed}"
                    ts = f"{g['trainset_dir']}/{cond}_N{n}_s{seed}.jsonl"
                    assert os.path.exists(os.path.join(ROOT, ts)), f"학습 세트 없음: {ts}"
                    perm = kind == "main" and n in g.get("perm_sizes", [])
                    ev = est["eval_full"] if kind == "main" else est["eval_test_only"]
                    ev += est["eval_perm_extra"] if perm else 0
                    jobs.append({"run_id": rid, "kind": kind, "cond": cond, "N": n, "seed": seed,
                                 "trainset": ts, "lr": lr, "eval_sets": g["eval_sets"], "perm": perm,
                                 "reuse_from": ex.get("reuse", {}).get(rid),
                                 "est_train_min": est["train"][max_steps_for(n, tcfg)], "est_eval_min": ev})
    for j in jobs:
        j["session"] = session_of(j)
    return jobs


def order(jobs, cfg):
    """무학습 먼저 → N 오름차순 → (시드) → 같은 N·시드 안에서 조건 순서는 시드로 섞음."""
    base = cfg["experiment"]["order_seed"]

    def key(j):
        if j["kind"] == "zs":
            return (-1, -1, 0, 0)
        g = cfg["experiment"][j["kind"]]["conditions"]
        perm = list(np.random.default_rng([base, j["N"], j["seed"], len(g)]).permutation(len(g)))
        return (j["N"], j["seed"], 0 if j["kind"] == "main" else 1, perm.index(g.index(j["cond"])))
    return sorted(jobs, key=key)


def smoke_jobs(cfg):
    lr = cfg["train"]["lr"]
    return [
        {"run_id": "smoke_ZS", "kind": "zs", "cond": "ZS", "N": 0, "seed": None, "trainset": None, "lr": None,
         "eval_sets": cfg["experiment"]["zs_eval_sets"], "perm": True, "reuse_from": None,
         "est_train_min": 0, "est_eval_min": 8},
        {"run_id": "smoke_main-Hard_N100_s0", "kind": "main", "cond": "Hard", "N": 100, "seed": 0,
         "trainset": "trainsets/smolvlm_matched/Hard_N100_s0.jsonl", "lr": lr,
         "eval_sets": cfg["experiment"]["main"]["eval_sets"], "perm": True, "reuse_from": None,
         "est_train_min": 6, "est_eval_min": 6},
        {"run_id": "smoke_aux-D3_N100_s0", "kind": "aux", "cond": "D3", "N": 100, "seed": 0,
         "trainset": "trainsets/smolvlm/D3_N100_s0.jsonl", "lr": lr,
         "eval_sets": cfg["experiment"]["aux"]["eval_sets"], "perm": False, "reuse_from": None,
         "est_train_min": 6, "est_eval_min": 3},
    ]


# ------------------------------------------------------------------ 노트북
def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def notebook(session, info, n_jobs, est_h, ref, cfg):
    ex = cfg["experiment"]
    if session == "SMOKE":
        inputs = "없음"
        when = "본실험 전 한 번 (계정 A·B 둘 다). 약 30~40분"
    elif session == "M1":
        inputs = ("**파일럿 결과** (pilot_P3K_Rand_N400_lr2e-4 폴더가 들어 있는 데이터셋 또는 파일럿 노트북 Output). "
                  "없으면 Rand_N400_s0를 새로 학습합니다(약 30분 더 걸림)")
        when = f"계정 {info['account']}"
    elif session == "SPILL":
        inputs = "**남은 실행이 있는 세션의 Output** (leftover.json이 들어 있는 노트북 Output을 + Add Input으로 추가)"
        when = "남은 실행을 처리할 때"
    else:
        inputs = "없음 (이전 세션 Output을 추가하면 그 세션의 남은 실행도 먼저 처리합니다)"
        when = f"계정 {info['account']}"
    est = f"예상 약 {est_h:.1f}시간 (T4 x2)" if est_h else "남은 실행 수에 따라 다름"
    cells = [
        md(f"""
# 본실험 세션 {session}: {info['desc']}

- 누가: {when}
- 실행 {n_jobs}개, {est}. 시간 예산 {ex['time_budget_h']}시간 안에 끝날 실행만 시작하고, 못 한 실행은 `leftover_{session}.json`에 남깁니다.
- 코드: GitHub `{ref}` (수정할 것 없음)

**시작 전 설정 (오른쪽 패널 Session options)**
1. Accelerator: **GPU T4 x2**
2. Internet: **On**
3. Environment: **Pin to original environment**
4. Input: {inputs}

**실행**: 오른쪽 위 **Save Version → Save & Run All (Commit)**. 브라우저를 닫아도 끝까지 돕니다.

**끝나면**: 노트북 → Output 탭에서 **`session_{session}.zip`** 을 내려받아 전달합니다.
맨 아래 셀의 요약(완료·실패·남은 실행)도 캡처해서 함께 보냅니다.
"""),
        md("## 1. 설정 + 코드 불러오기 + GPU 확인"),
        code(f"""
SESSION = '{session}'
CODE_REF = '{ref}'
REPO_URL = '{ex['repo_url']}'

import os, time, subprocess
os.environ['HN_T0'] = str(time.time())          # 시간 예산 계산의 시작점
REPO = '/tmp/hard-negative-mllm'                # 코드는 Output에 섞이지 않게 /tmp에
RUNS = '/kaggle/working/runs'                   # 결과는 Output으로 저장되는 곳
os.environ.update({{'HN_RUNS_DIR': RUNS, 'USE_TF': '0', 'TF_CPP_MIN_LOG_LEVEL': '3',
                   'TOKENIZERS_PARALLELISM': 'false', 'HF_HUB_DISABLE_PROGRESS_BARS': '1'}})
os.makedirs(RUNS, exist_ok=True)

!rm -rf {{REPO}} && git clone -q {{REPO_URL}} {{REPO}}
%cd {{REPO}}
!git checkout -q {{CODE_REF}}
!git log --oneline -1
gpus = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout
print(gpus)
assert gpus.count('Tesla T4') >= 1, 'T4 GPU가 아닙니다. Accelerator를 GPU T4 x2로 바꾸세요.'
"""),
        md("## 2. 설치 + 모델·데이터셋 미리 내려받기 (약 10~15분)"),
        code("""
!pip install -q transformers==4.51.3 peft==0.15.2 accelerate datasets pyarrow pyyaml 2>&1 | grep -v "dependency resolver\\|requires\\|incompatible" || true
!pip freeze | grep -E "^(torch|transformers|peft|accelerate|datasets)=="
!python scripts/05_evaluate.py --prefetch
"""),
        md(f"## 3. 세션 실행 (학습 → 평가, GPU 2개 동시)\n환경 검사 → 이전 결과 가져오기 → 실행 → 실행이 끝날 때마다 `session_{session}.zip` 갱신."),
        code("!python scripts/run_queue.py --session {SESSION}"),
        md("## 4. 결과 요약"),
        code("""
import json, pandas as pd
p = f'/kaggle/working/session_results_{SESSION}.csv'
if os.path.exists(p):
    df = pd.read_csv(p)
    cols = [c for c in ['run_id', 'status', 'best_val_acc', 'test_acc', 'test_acc_perm', 'ai2d_acc',
                        'pope_random_f1', 'pope_adversarial_f1', 'diverged', 'minutes'] if c in df.columns]
    pd.set_option('display.width', 250)
    display(df[cols])
    print('완료', (df.status == 'done').sum(), '/', len(df))
lp = f'/kaggle/working/leftover_{SESSION}.json'
if os.path.exists(lp):
    left = json.load(open(lp))['jobs']
    print('남은 실행:', len(left), left)
print('내려받을 파일:', [f for f in os.listdir('/kaggle/working') if f.endswith('.zip')])
"""),
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                        "name": "python3"},
                                         "language_info": {"name": "python"},
                                         "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True}},
            "nbformat": 4, "nbformat_minor": 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code_ref", default=None, help="본 세션 노트북이 불러올 태그 (기본: config experiment.code_ref)")
    args = ap.parse_args()
    cfg = load_config()
    ex = cfg["experiment"]
    ref = args.code_ref or ex["code_ref"]
    jobs = order(build_jobs(cfg), cfg)
    assert len(jobs) == 103, len(jobs)
    save_json(jobs, os.path.join(ROOT, "jobs", "all_jobs.json"))
    nb_dir = os.path.join(ROOT, "notebooks", "sessions")
    os.makedirs(nb_dir, exist_ok=True)
    print(f"전체 {len(jobs)}개 실행 (n_gpu=2 기준 벽시계 추정)")
    total = 0
    for s, info in SESSIONS.items():
        sj = [j for j in jobs if j["session"] == s]
        mins = sum((0 if j["reuse_from"] else j["est_train_min"]) + j["est_eval_min"] for j in sj)
        total += mins
        save_json({"session": s, **info, "smoke": False, "jobs": [j["run_id"] for j in sj],
                   "est_gpu_h": round(mins / 60, 1), "est_wall_h": round(mins / 120, 1)},
                  os.path.join(ROOT, "jobs", "sessions", f"{s}.json"))
        with open(os.path.join(nb_dir, f"run_{s}.ipynb"), "w", encoding="utf-8") as f:
            json.dump(notebook(s, info, len(sj), mins / 120, ref, cfg), f, ensure_ascii=False, indent=1)
        print(f"  {s} (계정 {info['account']}): {len(sj):2d}개  GPU {mins / 60:5.1f}h  벽시계 {mins / 120:4.1f}h  | {info['desc']}")
    print(f"  합계 GPU {total / 60:.1f}h, 벽시계 {total / 120:.1f}h")
    sm = smoke_jobs(cfg)
    save_json({"session": "SMOKE", "account": "A·B", "desc": "스모크 테스트 (10 step, 세트별 50문항)",
               "smoke": True, "job_defs": sm, "jobs": [j["run_id"] for j in sm]},
              os.path.join(ROOT, "jobs", "sessions", "SMOKE.json"))
    with open(os.path.join(nb_dir, "run_SMOKE.ipynb"), "w", encoding="utf-8") as f:
        json.dump(notebook("SMOKE", {"account": "A·B", "desc": "스모크 테스트"}, len(sm), 0, ex["branch"], cfg),
                  f, ensure_ascii=False, indent=1)
    save_json({"session": "SPILL", "account": "A·B", "desc": "남은 실행 처리", "smoke": False, "jobs": []},
              os.path.join(ROOT, "jobs", "sessions", "SPILL.json"))
    with open(os.path.join(nb_dir, "run_SPILL.ipynb"), "w", encoding="utf-8") as f:
        json.dump(notebook("SPILL", {"account": "A·B", "desc": "남은 실행 처리 (leftover)"}, 0, 0, ref, cfg),
                  f, ensure_ascii=False, indent=1)
    print("노트북:", sorted(os.listdir(nb_dir)))


if __name__ == "__main__":
    main()
