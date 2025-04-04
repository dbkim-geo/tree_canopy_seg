import os
import torch

def save_checkpoint(model, optimizer, epoch, loss, save_path, model_tag=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "model_tag": model_tag  # ✅ 모델 구조 정보 저장
    }, save_path)

    print(f"[Checkpoint] Saved at {save_path}")

def load_checkpoint(model, optimizer=None, ckpt_path=None, device="cpu", model_tag=None):
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"[Checkpoint] ❌ Checkpoint not found at: {ckpt_path}")
        return model, optimizer, 0, None  # 새 학습

    checkpoint = torch.load(ckpt_path, map_location=device)

    # ✅ 구조 일치 여부 확인
    if "model_tag" in checkpoint and model_tag and checkpoint["model_tag"] != model_tag:
        print(f"⚠️ 모델 구조 불일치: {checkpoint['model_tag']} (saved) ≠ {model_tag} (current)")
        print("➡️ 새 학습으로 시작합니다.")
        return model, optimizer, 0, None

    model.load_state_dict(checkpoint["model_state"])

    if optimizer and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", None)

    print(f"[Checkpoint] ✅ Loaded from {ckpt_path} (epoch {epoch})")
    return model, optimizer, epoch, loss
