"""4단계(파일럿·본실험) 공통 도구 상자.

- 모델 불러오기: 2단계 난이도 측정과 같은 방식(이미지 longest_edge, 같은 프롬프트)
- 학습 배치: 오른쪽 패딩 + 정답 글자와 종료 토큰만 손실에 넣는 라벨 마스킹
- 채점 배치: 왼쪽 패딩 + 마지막 위치에서 " A"~" E" 로그확률 비교 (2단계와 같은 채점)
- 이미지 미리 올리기, 환경 정보 기록

test split 이미지는 평가 코드(05_evaluate.py)가 allow_test=True로 부를 때만 불러옵니다.
학습 코드(04_train.py)는 test를 절대 불러오지 않습니다.
"""
import json
import math
import os
import platform
import time
import warnings

# TensorFlow 로그·경고 끄기 (transformers가 TF를 쓰지 않게)
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step\\(\\)` before")

import numpy as np
import pandas as pd
import torch

from common import DATA, LETTERS, ROOT, build_prompt, load_config, load_json, load_scienceqa

EOU = "<end_of_utterance>"


# ------------------------------------------------------------------ 기본
def set_all_seeds(seed):
    from transformers import set_seed
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def runs_dir(cfg):
    d = os.environ.get("HN_RUNS_DIR") or cfg["paths"]["runs_dir"]  # 환경변수로 바꿀 수 있음
    parent = os.path.dirname(d.rstrip("/"))
    if not os.path.isdir(parent):
        raise SystemExit(f"[멈춤] {parent} 폴더가 없습니다. 저장 위치(HN_RUNS_DIR)를 확인하세요.")
    os.makedirs(d, exist_ok=True)
    return d


def max_steps_for(n, tcfg):
    """최대 step = max(150, ceil(4N/16))."""
    eff = tcfg["micro_batch"] * tcfg["grad_accum"]
    return max(tcfg["min_steps"], math.ceil(tcfg["epochs_equiv"] * n / eff))


def env_info(model_id, revision):
    import transformers
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "model_id": model_id,
            "model_revision": revision,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda": torch.version.cuda}
    try:
        import peft
        info["peft"] = peft.__version__
    except ImportError:
        info["peft"] = None
    return info


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def lr_tag(lr):
    return f"{lr:.0e}".replace("e-0", "e-")  # 5e-05 -> 5e-5


def pilot_run_id(key, spec, lr):
    """P1 + Rand_N400_s0 + 5e-5 -> pilot_P1_Rand_N400_lr5e-5 (파일럿 실행 가이드 5장 이름)."""
    cond, n = os.path.basename(spec["trainset"]).replace(".jsonl", "").split("_")[:2]
    rid = f"pilot_{key}_{cond}_{n}"
    if key in ("P1", "P2", "P3", "P3K"):
        rid += f"_lr{lr_tag(lr)}"
    if key == "P7":
        rid += "_rep"
    return rid


# ------------------------------------------------------------------ 데이터
def load_items():
    items = pd.read_parquet(f"{DATA}/items.parquet")
    items["choices"] = items["choices"].apply(list)
    return items.set_index("item_id")


def load_trainset(path):
    path = path if os.path.isabs(path) else os.path.join(ROOT, path)
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def val500_records(items):
    """검증 500문항: 원래 보기 K개 그대로."""
    ids = load_json(f"{DATA}/val500_ids.json")
    recs = []
    for iid in ids:
        r = items.loc[iid]
        assert r.split == "validation", f"{iid}는 validation 문항이 아닙니다"
        recs.append({"item_id": iid, "question": r.question, "hint": r.hint,
                     "options": list(r.choices), "answer_letter": LETTERS[int(r.answer)],
                     "K": int(r.K)})
    return recs


def load_images(cfg, items, item_ids, allow_test=False):
    """필요한 이미지만 메모리에 올립니다. 학습은 train / validation만, test는 평가에서만."""
    ok = ("train", "validation", "test") if allow_test else ("train", "validation")
    need = {}
    for iid in item_ids:
        split = items.loc[iid].split
        if split not in ok:
            raise SystemExit(f"[멈춤] {iid}: {split} split은 4단계에서 쓰지 않습니다.")
        need.setdefault(split, []).append(iid)
    out = {}
    for split, ids in need.items():
        ds = load_scienceqa(cfg, split)
        for iid in ids:
            out[iid] = ds[int(items.loc[iid].row_idx)]["image"].convert("RGB")
    return out


# ------------------------------------------------------------------ 모델
def resolve_revision(cfg, name="smolvlm"):
    rev = cfg["models"][name].get("revision")
    if not rev:
        raise SystemExit("[멈춤] config.yaml의 models.smolvlm.revision이 비어 있습니다. 노트북 2번 셀을 먼저 실행하세요.")
    return rev


def load_model_and_processor(cfg, name="smolvlm"):
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoM
    m = cfg["models"][name]
    rev = resolve_revision(cfg, name)
    processor = AutoProcessor.from_pretrained(m["id"], revision=rev)
    if m.get("longest_edge"):
        processor.image_processor.size = {"longest_edge": m["longest_edge"]}
    dtype = torch.float16 if cfg["train"]["precision"] == "fp16" else torch.bfloat16
    model = AutoM.from_pretrained(m["id"], revision=rev, torch_dtype=dtype).to("cuda")
    return model, processor


def letter_ids(processor):
    """" A"~" E"(공백+글자) 토큰 id와 종료 토큰 id. 2단계 letter_format_smolvlm.json과 대조합니다."""
    tok = processor.tokenizer
    ids = {}
    for L in LETTERS:
        t = tok.encode(" " + L, add_special_tokens=False)
        assert len(t) == 1, f"' {L}'가 토큰 1개가 아닙니다: {t}"
        ids[L] = t[0]
    ref = load_json(f"{DATA}/letter_format_smolvlm.json")["token_ids"]["space"]
    for L in LETTERS:
        assert ids[L] == ref[L], f"' {L}' 토큰 id {ids[L]} != 2단계 기록 {ref[L]}"
    eou = tok.convert_tokens_to_ids(EOU)
    assert eou is not None and eou != tok.unk_token_id, "종료 토큰을 찾지 못했습니다"
    return ids, eou


def attach_lora(model, cfg, seed):
    from peft import LoraConfig, get_peft_model
    lc = cfg["train"]["lora"]
    set_all_seeds(seed)  # LoRA 초기값도 시드로 고정
    conf = LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"], bias="none",
                      task_type="CAUSAL_LM", target_modules=lc["target_regex"])
    pm = get_peft_model(model, conf)  # 기본값 autocast_adapter_dtype=True → LoRA 가중치는 fp32
    return pm


def lora_module_report(pm, model_config):
    names = sorted({n.rsplit(".lora_A", 1)[0] for n, _ in pm.named_modules() if ".lora_A" in n})
    bad = [n for n in names if "vision" in n or "connector" in n]
    tc = getattr(model_config, "text_config", model_config)
    n_layers = tc.num_hidden_layers
    trainable = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    total = sum(p.numel() for p in pm.parameters())
    return {"n_modules": len(names), "expected": n_layers * 7, "text_layers": n_layers,
            "bad_modules": bad, "trainable_params": trainable, "total_params": total,
            "lora_dtype": str(next(p for p in pm.parameters() if p.requires_grad).dtype),
            "examples": names[:3]}


# ------------------------------------------------------------------ 입력 만들기
def chat_text(processor, rec):
    msgs = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": build_prompt(rec["question"], rec["hint"], rec["options"])}]}]
    return processor.apply_chat_template(msgs, add_generation_prompt=True)


def target_text(letter):
    return " " + letter + EOU


def encode(processor, texts, images, side):
    processor.tokenizer.padding_side = side
    return processor(text=texts, images=[[im] for im in images], return_tensors="pt", padding=True)


def train_batch(processor, recs, images, lids, eou, max_length):
    """오른쪽 패딩. 손실은 마지막 2토큰(" X", <end_of_utterance>)만. 맞는지 매번 assert로 확인합니다."""
    texts = [chat_text(processor, r) + target_text(r["answer_letter"]) for r in recs]
    enc = encode(processor, texts, [images[r["item_id"]] for r in recs], "right")
    labels = torch.full_like(enc["input_ids"], -100)
    lengths = enc["attention_mask"].sum(1).tolist()
    for i, (r, L) in enumerate(zip(recs, lengths)):
        if L > max_length:
            raise SystemExit(f"[멈춤] {r['item_id']} 길이 {L} > max_length {max_length}")
        a, b = enc["input_ids"][i, L - 2].item(), enc["input_ids"][i, L - 1].item()
        assert a == lids[r["answer_letter"]] and b == eou, \
            f"[마스킹 오류] {r['item_id']}: 끝 토큰 {a},{b} != {lids[r['answer_letter']]},{eou}"
        labels[i, L - 2:L] = enc["input_ids"][i, L - 2:L]
    enc["labels"] = labels
    return enc


def to_cuda(enc):
    return {k: v.to("cuda") for k, v in enc.items()}


# ------------------------------------------------------------------ 채점
@torch.no_grad()
def score_records(model, processor, recs, images, lids, batch_size, autocast_dtype=torch.float16):
    """2단계와 같은 채점: 마지막 위치 log-softmax에서 유효 글자 K개 중 최댓값."""
    model.eval()
    rows = []
    for i in range(0, len(recs), batch_size):
        b = recs[i:i + batch_size]
        enc = to_cuda(encode(processor, [chat_text(processor, r) for r in b],
                             [images[r["item_id"]] for r in b], "left"))
        with torch.autocast("cuda", dtype=autocast_dtype):
            logits = model(**enc).logits[:, -1, :]
        lp = torch.log_softmax(logits.float(), dim=-1).cpu()
        for row, r in zip(lp, b):
            sc = [row[lids[L]].item() for L in LETTERS[:r["K"]]]
            pred = LETTERS[int(np.argmax(sc))]
            rows.append({"item_id": r["item_id"], "K": r["K"],
                         **{f"lp_{L}": (sc[j] if j < r["K"] else None) for j, L in enumerate(LETTERS)},
                         "pred": pred, "answer": r["answer_letter"], "correct": pred == r["answer_letter"]})
    return pd.DataFrame(rows)


@torch.no_grad()
def generation_check(model, processor, recs, images, lids, batch_size, n_examples=20, max_new_tokens=3):
    """학습 후 실제 생성에서 첫 토큰이 ' A'~' E' 중 유효 글자인 비율 (B6)."""
    model.eval()
    hit, examples = 0, []
    for i in range(0, len(recs), batch_size):
        b = recs[i:i + batch_size]
        enc = to_cuda(encode(processor, [chat_text(processor, r) for r in b],
                             [images[r["item_id"]] for r in b], "left"))
        with torch.autocast("cuda", dtype=torch.float16):
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        new = gen[:, enc["input_ids"].shape[1]:].cpu()
        for row, r in zip(new, b):
            ok = row[0].item() in {lids[L] for L in LETTERS[:r["K"]]}
            hit += ok
            if len(examples) < n_examples:
                examples.append({"item_id": r["item_id"], "answer": r["answer_letter"],
                                 "output": processor.tokenizer.decode(row, skip_special_tokens=False),
                                 "is_letter": ok})
    return {"letter_rate": hit / len(recs), "n": len(recs), "examples": examples}


# ------------------------------------------------------------------ 본실험 평가용
def test_records(items):
    """ScienceQA test 2,017문항: 원래 보기 K개, 원래 순서."""
    t = items[items.split == "test"].sort_index()
    return [{"item_id": iid, "question": r.question, "hint": r.hint, "options": list(r.choices),
             "answer_letter": LETTERS[int(r.answer)], "K": int(r.K), "subject": r.subject,
             "grade_band": r.grade_band} for iid, r in t.iterrows()]


def chat_text_prompt(processor, prompt):
    """이미지 1장 + 임의의 텍스트 프롬프트 (POPE용)."""
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    return processor.apply_chat_template(msgs, add_generation_prompt=True)


def single_token_id(tok, text):
    t = tok.encode(text, add_special_tokens=False)
    return t[0], len(t)


def yes_no_ids(processor):
    """POPE 채점용 토큰: " Yes"/"Yes", " No"/"No". 여러 토큰이면 첫 토큰을 씀(기록)."""
    tok = processor.tokenizer
    out, notes = {}, {}
    for k in (" Yes", "Yes", " No", "No"):
        out[k], n = single_token_id(tok, k)
        notes[k] = {"id": out[k], "n_tokens": n}
    assert len({out[" Yes"], out["Yes"]} & {out[" No"], out["No"]}) == 0, "Yes/No 토큰이 겹칩니다"
    return out, notes


@torch.no_grad()
def score_tokens(model, processor, texts, images, token_ids, batch_size, autocast_dtype=torch.float16):
    """마지막 위치 log-softmax에서 token_ids의 값만 뽑아 [문항 수, len(token_ids)] 배열로."""
    model.eval()
    idx = torch.tensor(token_ids)
    out = []
    for i in range(0, len(texts), batch_size):
        enc = to_cuda(encode(processor, texts[i:i + batch_size], images[i:i + batch_size], "left"))
        with torch.autocast("cuda", dtype=autocast_dtype):
            logits = model(**enc).logits[:, -1, :]
        lp = torch.log_softmax(logits.float(), dim=-1).cpu()
        out.append(lp[:, idx].numpy())
    return np.concatenate(out, 0) if out else np.zeros((0, len(token_ids)))


def git_info():
    import subprocess
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip() or None
        except FileNotFoundError:
            return None
    return {"commit": run("rev-parse", "HEAD"), "describe": run("describe", "--tags", "--always")}


def lora_state(pm):
    return {k: v.detach().cpu().clone() for k, v in pm.state_dict().items() if "lora_" in k}


def timer():
    t0 = time.time()
    return lambda: time.time() - t0
