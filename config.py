from enum import Enum, auto
import torch

class Mode(Enum):
    TRAIN = auto()
    PREDICT = auto()

class ModelType(Enum):
    DUAL = auto()
    BASE = auto()

CONFIG = {
    "mode": Mode.TRAIN,  # or Mode.PREDICT

    # 모델 설정
    "model_type": ModelType.DUAL,
    "in_channels_rgb": 3,
    "in_channels_fc": 3,
    "num_classes": 1,

    # 학습 설정
    "batch_size": 8,
    "lr": 1e-4,
    "num_epochs": 100,
    "do_validation": True,

    # Transformer 구조
    "patch_size": 16,
    "embed_dim": 256,
    "transformer_depth": 4,
    "num_heads": 4,

    # 경로 설정
    "rgb_dir": "./data/rgb/",
    "fake_dir": "./data/fake/",
    "label_dir": "./data/label/",
    "save_dir": "./checkpoints",
    "resume_path": "./checkpoints/latest.pt",
    "log_dir": "./logs",
    "result_dir": "./results",

    # 시스템
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "exp_name": "exp01_treecanopy"
}
