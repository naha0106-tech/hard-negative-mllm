# Hard Negative Difficulty × Data Scale in LoRA-tuned MLLMs

제한된 데이터에서 오답(Hard Negative) 난이도가 LoRA 기반 MLLM(SmolVLM-500M, Qwen2.5-VL-3B)의 데이터 효율성에 미치는 영향을 분석하는 연구 코드입니다.
이 저장소에는 **데이터셋이 만들어지는 전 과정**이 들어 있어, 아래 순서대로 실행하면 똑같은 학습 세트가 다시 만들어집니다.

## 실행 순서 (Google Colab, GPU 필요)

`notebooks/run_in_colab.ipynb`를 코랩에서 열고 위에서부터 실행하면 됩니다. 직접 실행하려면:

```bash
pip install -r requirements.txt
python scripts/01_prepare_data.py                    # 원본 정리 → data/
python scripts/02_score_difficulty.py --model smolvlm  # 오답 난이도 + 관문 A (GPU)
python scripts/03_build_trainsets.py --model smolvlm   # 학습 세트 105개 + 자동 검증
```

| 단계 | 스크립트 | 결과물 | 확인할 파일 |
|---|---|---|---|
| 1 | `01_prepare_data.py` | 문항 표, 오답 쌍 표, 평가 ID 목록 | `logs/01_prepare.log` |
| 2 | `02_score_difficulty.py` | 오답 난이도(m), 5구간(D1~D5) | `data/difficulty_report_smolvlm.json` |
| 2-검수 | (수작업) | D1 50쌍 라벨 | `data/d1_audit.csv` |
| 3 | `03_build_trainsets.py` | `trainsets/smolvlm/*.jsonl` | `logs/03_checks_smolvlm.json` |

## 원칙
- 평가셋(ScienceQA-IMG test 2,017문항)은 본실험 평가 전까지 열어 보지 않습니다.
- 모든 시드와 설정은 `config.yaml`에만 있습니다.
- 데이터 버전은 git 태그(`data-v1.0` 등)로 표시합니다.

## 데이터 출처
- ScienceQA-IMG (`lmms-lab/ScienceQA`), AI2D (`lmms-lab/ai2d`), POPE (`lmms-lab/POPE`)
