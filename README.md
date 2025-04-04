# tree_canopy_seg
Tree Canopy Semantic Segmentation



"""
TREE_CANOPY_SEG/
├── models/
│   └── blocks ...
    ....
├── dataset/
│   └── custom_dataset.py       ← ✅ 데이터로더 정의
├── train.py                    ← ✅ 학습 루프만
├── validate.py                 ← ✅ 검증 루프만
├── predict.py                  ← ✅ 추론용 스크립트
├── main.py                     ← 🧠 실험 실행 entry point
├── config.py                   ← 하이퍼파라미터, 경로 등 관리
├── utils/
│   └── metrics.py              ← IoU, Dice 등
│   └── losses.py               ← BCE 등 래핑
│   └── visualizer.py           ← 결과 시각화
"""