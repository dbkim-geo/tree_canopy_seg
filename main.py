import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np
from PIL import Image

from config import CONFIG, Mode, ModelType
from utils.logger import setup_logger
from utils.checkpoint import save_checkpoint, load_checkpoint
from dataset.custom_dataset import TreeCanopyDataset
from train import train_one_epoch
from val import validate
from models.dual_transformer_unet import DualStreamTransformerUNet
from models.base_unet import BaseUNet


def build_model():
    if CONFIG["model_type"] == ModelType.DUAL:
        return DualStreamTransformerUNet(
            in_channels_rgb=CONFIG["in_channels_rgb"],
            in_channels_fc=CONFIG["in_channels_fc"],
            out_channels=CONFIG["num_classes"],
            patch_size=CONFIG["patch_size"],
            embed_dim=CONFIG["embed_dim"],
            transformer_depth=CONFIG["transformer_depth"],
            num_heads=CONFIG["num_heads"]
        )
    elif CONFIG["model_type"] == ModelType.BASE:
        return BaseUNet(
            in_channels=CONFIG["in_channels_rgb"],
            out_channels=CONFIG["num_classes"]
        )
    else:
        raise ValueError(f"Unknown model_type: {CONFIG['model_type']}")


def get_dataloader(shuffle=True):
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3)
    ])
    dataset = TreeCanopyDataset(
        rgb_dir=CONFIG["rgb_dir"],
        fake_dir=CONFIG["fake_dir"],
        label_dir=CONFIG["label_dir"],
        transform=transform
    )
    return DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=shuffle)


@torch.no_grad()
def predict(model, dataloader, device, model_type: ModelType, save_dir: str, threshold=0.5):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    for i, batch in enumerate(dataloader):
        rgb = batch["rgb"].to(device)
        fake = batch["fake"].to(device)

        if model_type == ModelType.DUAL:
            preds = model(rgb, fake)
        elif model_type == ModelType.BASE:
            preds = model(rgb)
        else:
            raise ValueError(f"[Predict] Unknown model_type: {model_type}")

        preds_bin = (preds > threshold).float()

        for b in range(preds_bin.shape[0]):
            mask = preds_bin[b, 0].cpu().numpy() * 255
            Image.fromarray(mask.astype(np.uint8)).save(
                os.path.join(save_dir, f"pred_{i * preds_bin.shape[0] + b:04}.png")
            )


def main():
    device = CONFIG["device"]
    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    logger = setup_logger(CONFIG["log_dir"], CONFIG["exp_name"])
    logger.info("🚀 Start Experiment: " + CONFIG["exp_name"])
    logger.info(f"🖥️ Device: {device}")
    logger.info("📋 Config:")
    for k, v in CONFIG.items():
        logger.info(f"    {k}: {v}")

    model = build_model().to(device)
    dataloader = get_dataloader(shuffle=(CONFIG["mode"] == Mode.TRAIN))
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    loss_fn = nn.BCELoss()

    model_tag = (
        f"{CONFIG['model_type'].name}_{CONFIG['embed_dim']}"
        if CONFIG["model_type"] == ModelType.DUAL else f"{CONFIG['model_type'].name}_base"
    )

    # ───────────────────────────────────────
    if CONFIG["mode"] == Mode.PREDICT:
        logger.info("🔍 Inference Mode")
        model, _, _, _ = load_checkpoint(model, None, CONFIG["resume_path"], device, model_tag)
        predict(model, dataloader, device, CONFIG["model_type"], CONFIG["result_dir"])
        logger.info(f"🖼️ Prediction saved to {CONFIG['result_dir']}")
        return

    # ───────────────────────────────────────
    elif CONFIG["mode"] == Mode.TRAIN:
        logger.info("🧪 Training Mode")
        start_epoch = 0
        if os.path.exists(CONFIG["resume_path"]):
            model, optimizer, start_epoch, _ = load_checkpoint(
                model, optimizer, CONFIG["resume_path"], device, model_tag
            )
            logger.info(f"🔁 Resumed from {CONFIG['resume_path']}")

        for epoch in range(start_epoch, CONFIG["num_epochs"]):
            train_stats = train_one_epoch(model, dataloader, optimizer, loss_fn, device, CONFIG["model_type"])
            logger.info(f"[Epoch {epoch+1}] 🔧 Train Loss={train_stats['loss']:.4f} | Dice={train_stats['dice']:.4f} | IoU={train_stats['iou']:.4f}")

            if CONFIG["do_validation"]:
                val_stats = validate(model, dataloader, loss_fn, device, CONFIG["model_type"])
                logger.info(f"[Epoch {epoch+1}] ✅ Val   Loss={val_stats['loss']:.4f} | Dice={val_stats['dice']:.4f} | IoU={val_stats['iou']:.4f}")

            ckpt_path = os.path.join(CONFIG["save_dir"], f"epoch_{epoch+1:03}.pt")
            save_checkpoint(model, optimizer, epoch+1, train_stats["loss"], ckpt_path, model_tag=model_tag)
            logger.info(f"💾 Checkpoint saved: {ckpt_path}")

        logger.info("🏁 Training Completed.")

    else:
        raise ValueError(f"Unknown mode: {CONFIG['mode']}")


if __name__ == "__main__":
    main()
