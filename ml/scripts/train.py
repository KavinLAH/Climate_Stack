"""
PestCast — EfficientNet-Lite0 Training Script

Fine-tunes a pretrained EfficientNet-Lite0 on the curated IP102 subset
for on-device pest classification.
"""

import os
import json
import time
import argparse
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import numpy as np
from sklearn.metrics import classification_report, top_k_accuracy_score
from tqdm import tqdm


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_transforms(config: dict, split: str) -> transforms.Compose:
    """Build data transforms for train/val/test."""
    input_size = config["model"]["input_size"]
    aug = config["augmentation"]

    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(
                input_size,
                scale=tuple(aug["random_resized_crop"]["scale"]),
            ),
            transforms.RandomHorizontalFlip() if aug["random_horizontal_flip"] else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(aug["random_rotation"]),
            transforms.ColorJitter(
                brightness=aug["color_jitter"]["brightness"],
                contrast=aug["color_jitter"]["contrast"],
                saturation=aug["color_jitter"]["saturation"],
                hue=aug["color_jitter"]["hue"],
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(input_size * 1.14)),  # 256 for 224 input
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def build_model(config: dict) -> nn.Module:
    """Load pretrained EfficientNet-Lite0 and replace classifier head."""
    arch = config["model"]["architecture"]
    num_classes = config["model"]["num_classes"]
    dropout = config["model"]["dropout"]

    # timm uses 'efficientnet_lite0' name
    model = timm.create_model(arch, pretrained=config["model"]["pretrained"])

    # Replace classifier head
    if hasattr(model, "classifier"):
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
    elif hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    return model


def freeze_backbone(model: nn.Module):
    """Freeze all layers except the classifier head."""
    for name, param in model.named_parameters():
        if "classifier" not in name and "fc" not in name:
            param.requires_grad = False


def unfreeze_backbone(model: nn.Module):
    """Unfreeze all layers."""
    for param in model.parameters():
        param.requires_grad = True


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict:
    model.train()
    running_loss = 0.0
    correct_top1 = 0
    correct_top3 = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

        # Top-1 accuracy
        _, pred = outputs.topk(1, dim=1)
        correct_top1 += pred.squeeze().eq(labels).sum().item()

        # Top-3 accuracy
        _, pred3 = outputs.topk(min(3, outputs.size(1)), dim=1)
        correct_top3 += sum(labels[i] in pred3[i] for i in range(labels.size(0)))

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct_top1/total:.3f}")

    return {
        "loss": running_loss / total,
        "top1_acc": correct_top1 / total,
        "top3_acc": correct_top3 / total,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split: str = "val",
) -> dict:
    model.eval()
    running_loss = 0.0
    correct_top1 = 0
    correct_top3 = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    pbar = tqdm(loader, desc=f"[{split}]", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

        probs = torch.softmax(outputs, dim=1)
        _, pred = outputs.topk(1, dim=1)
        correct_top1 += pred.squeeze().eq(labels).sum().item()

        _, pred3 = outputs.topk(min(3, outputs.size(1)), dim=1)
        correct_top3 += sum(labels[i] in pred3[i] for i in range(labels.size(0)))

        all_preds.extend(pred.squeeze().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return {
        "loss": running_loss / total,
        "top1_acc": correct_top1 / total,
        "top3_acc": correct_top3 / total,
        "predictions": np.array(all_preds),
        "labels": np.array(all_labels),
        "probabilities": np.array(all_probs),
    }


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    """Cosine annealing with linear warmup."""
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser(description="Train PestCast pest classifier")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--data-dir", type=str, default="data/ip102_curated",
                        help="Curated dataset directory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu/mps)")
    args = parser.parse_args()

    config = load_config(args.config)
    tc = config["training"]
    data_dir = Path(args.data_dir)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Paths
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    logs_dir = Path(config["paths"]["logs_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Load labels
    labels_path = data_dir / "labels.json"
    with open(labels_path) as f:
        labels_map = json.load(f)
    class_names = [labels_map[str(i)] for i in range(len(labels_map))]
    print(f"Classes ({len(class_names)}): {class_names}")

    # Datasets
    train_transform = get_transforms(config, "train")
    val_transform = get_transforms(config, "val")

    train_dataset = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(data_dir / "val", transform=val_transform)
    test_dataset = datasets.ImageFolder(data_dir / "test", transform=val_transform)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=tc["batch_size"], shuffle=True,
        num_workers=config["dataset"]["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=tc["batch_size"], shuffle=False,
        num_workers=config["dataset"]["num_workers"], pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=tc["batch_size"], shuffle=False,
        num_workers=config["dataset"]["num_workers"], pin_memory=True,
    )

    # Model
    model = build_model(config)
    model = model.to(device)
    print(f"Model: {config['model']['architecture']}, "
          f"params: {sum(p.numel() for p in model.parameters()):,}")

    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=tc["label_smoothing"])

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=tc["learning_rate"],
        weight_decay=tc["weight_decay"],
    )

    # Scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, tc["warmup_epochs"], tc["epochs"], len(train_loader)
    )

    # Resume
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.4f}")

    # Training log
    history = {"train_loss": [], "val_loss": [], "train_top1": [], "val_top1": [],
               "train_top3": [], "val_top3": [], "lr": []}

    print(f"\n{'='*60}")
    print(f"Starting training for {tc['epochs']} epochs")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, tc["epochs"]):
        epoch_start = time.time()

        # Backbone freeze/unfreeze
        if epoch < tc["freeze_backbone_epochs"]:
            freeze_backbone(model)
            if epoch == 0:
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"Backbone frozen. Trainable params: {trainable:,}")
        elif epoch == tc["freeze_backbone_epochs"]:
            unfreeze_backbone(model)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Backbone unfrozen. Trainable params: {trainable:,}")

        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        scheduler.step()

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, "val")

        # Log
        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_top1"].append(train_metrics["top1_acc"])
        history["val_top1"].append(val_metrics["top1_acc"])
        history["train_top3"].append(train_metrics["top3_acc"])
        history["val_top3"].append(val_metrics["top3_acc"])
        history["lr"].append(current_lr)

        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch:2d}/{tc['epochs']-1} ({elapsed:.0f}s) | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Train Top1: {train_metrics['top1_acc']:.3f} | "
              f"Val Top1: {val_metrics['top1_acc']:.3f} | "
              f"Val Top3: {val_metrics['top3_acc']:.3f} | "
              f"LR: {current_lr:.6f}")

        # Save best model
        if val_metrics["top3_acc"] > best_val_acc:
            best_val_acc = val_metrics["top3_acc"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "config": config,
                "class_names": class_names,
            }, checkpoint_dir / "best_model.pth")
            print(f"  -> New best model saved (top3: {best_val_acc:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "config": config,
                "class_names": class_names,
            }, checkpoint_dir / f"checkpoint_epoch_{epoch}.pth")

    # Final test evaluation
    print(f"\n{'='*60}")
    print("Final Test Evaluation")
    print(f"{'='*60}")

    # Load best model for test
    best_ckpt = torch.load(checkpoint_dir / "best_model.pth", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_metrics = evaluate(model, test_loader, criterion, device, "test")
    print(f"\nTest Top-1 Accuracy: {test_metrics['top1_acc']:.4f}")
    print(f"Test Top-3 Accuracy: {test_metrics['top3_acc']:.4f}")
    print(f"\nPer-class report:")
    print(classification_report(
        test_metrics["labels"],
        test_metrics["predictions"],
        target_names=class_names,
    ))

    # Save history
    with open(logs_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save test results
    with open(logs_dir / "test_results.json", "w") as f:
        json.dump({
            "top1_accuracy": test_metrics["top1_acc"],
            "top3_accuracy": test_metrics["top3_acc"],
            "best_epoch": best_ckpt["epoch"],
            "class_names": class_names,
        }, f, indent=2)

    print(f"\nTraining complete. Best model: {checkpoint_dir / 'best_model.pth'}")
    print(f"History: {logs_dir / 'training_history.json'}")


if __name__ == "__main__":
    main()
