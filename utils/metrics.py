def dice_coef(pred, target, threshold=0.5, eps=1e-7):
    pred_bin = (pred > threshold).float()
    target = target.float()

    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum()

    dice = (2 * intersection + eps) / (union + eps)
    return dice.item()

def iou_score(pred, target, threshold=0.5, eps=1e-7):
    pred_bin = (pred > threshold).float()
    target = target.float()

    intersection = (pred_bin * target).sum()
    union = (pred_bin + target).clamp(0, 1).sum()

    iou = (intersection + eps) / (union + eps)
    return iou.item()

def precision(pred, target, threshold=0.5, eps=1e-7):
    pred_bin = (pred > threshold).float()
    target = target.float()

    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1 - target)).sum()

    return (tp + eps) / (tp + fp + eps)

def recall(pred, target, threshold=0.5, eps=1e-7):
    pred_bin = (pred > threshold).float()
    target = target.float()

    tp = (pred_bin * target).sum()
    fn = ((1 - pred_bin) * target).sum()

    return (tp + eps) / (tp + fn + eps)
