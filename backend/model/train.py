"""
Ghana Currency Recognition — CNN Training Script  (PyTorch)
============================================================
Uses MobileNetV2 transfer learning + fine-tuning via torchvision.
Switched from TensorFlow to PyTorch for Python 3.14 compatibility.

Dataset layout expected at  ../../dataset/
  dataset/
    1_GHS/    <- place note images here
    2_GHS/
    5_GHS/
    10_GHS/
    20_GHS/
    50_GHS/
    100_GHS/
    200_GHS/

Run from the project root:
    python backend/model/train.py
"""

import os
import json
import copy
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Limit CPU threads to prevent overheating on laptops/low-end PCs
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report

# ── Configuration ─────────────────────────────────────────────────────────────
IMG_SIZE        = 224
BATCH_SIZE      = 8
INITIAL_EPOCHS  = 25   # more epochs — small dataset needs longer training
FINETUNE_EPOCHS = 15
LEARNING_RATE   = 1e-3
FINETUNE_LR     = 5e-5
WEIGHT_DECAY    = 1e-3  # L2 regularisation to reduce overfitting
EARLY_STOP_PAT  = 12   # generous patience for small datasets

HERE         = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.normpath(os.path.join(HERE, "..", "..", "dataset"))
MODEL_SAVE   = os.path.join(HERE, "ghana_currency_model.pt")
CLASS_NAMES  = os.path.join(HERE, "class_names.json")
PLOTS_DIR    = os.path.join(HERE, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet normalisation constants used by pretrained MobileNetV2
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Data loaders ──────────────────────────────────────────────────────────────
def get_transforms():
    """
    Aggressive augmentation for small datasets.
    Val transform is deterministic (no augmentation).
    """
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.3, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                                shear=10, scale=(0.85, 1.15)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def get_data_loaders():
    """
    Stratified 80/20 split — guarantees every class appears in both
    train and val sets, which is critical with small datasets.
    """
    train_tf, val_tf = get_transforms()

    # Load full dataset once to get labels; transform assigned per-subset below
    full = datasets.ImageFolder(DATASET_PATH)
    targets = np.array(full.targets)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(targets, targets))

    # Separate datasets with correct transforms
    train_ds = copy.deepcopy(full)
    train_ds.transform = train_tf
    val_ds   = copy.deepcopy(full)
    val_ds.transform   = val_tf

    train_subset = Subset(train_ds, train_idx)
    val_subset   = Subset(val_ds,   val_idx)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    print(f"  Train samples : {len(train_idx)}")
    print(f"  Val   samples : {len(val_idx)}")

    # Show per-class val counts so we can verify balance
    val_labels = targets[val_idx]
    for i, name in enumerate(full.classes):
        print(f"    {name}: {(val_labels == i).sum()} val images")

    return train_loader, val_loader, full.classes


# ── Model ──────────────────────────────────────────────────────────────────────
def build_model(num_classes: int) -> nn.Module:
    """
    Phase-1: frozen MobileNetV2 backbone + lightweight head.
    Simpler head (one hidden layer) generalises better on small data.
    """
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    model   = models.mobilenet_v2(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.6),              # stronger dropout vs original 0.5
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
    )
    return model.to(DEVICE)


def unfreeze_top_layers(model: nn.Module, n_layers: int = 10):
    """Phase-2: unfreeze last n feature layers for fine-tuning."""
    feature_layers = list(model.features.children())
    for layer in feature_layers[-n_layers:]:
        for param in layer.parameters():
            param.requires_grad = True


# ── Training helpers ───────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None):
    """Single train or eval pass. optimizer=None means eval mode."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            if is_train:
                optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            _, preds    = torch.max(outputs, 1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

    return total_loss / total, correct / total


def train_phase(model, train_loader, val_loader, epochs, lr, phase_name):
    # Label smoothing reduces overconfidence — helps generalisation
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=WEIGHT_DECAY
    )
    # Cosine annealing decays LR smoothly — better than ReduceLROnPlateau
    # for small datasets where val loss can be noisy
    scheduler  = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    best_acc   = 0.0
    best_state = None
    no_improve = 0
    history    = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(f"[{phase_name}] Epoch {epoch:>3}/{epochs} | "
              f"Train loss={tr_loss:.4f} acc={tr_acc*100:.2f}% | "
              f"Val   loss={va_loss:.4f} acc={va_acc*100:.2f}%")

        if va_acc > best_acc:
            best_acc   = va_acc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            print(f"           -> New best val accuracy: {va_acc*100:.2f}%")
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PAT:
                print(f"           Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    print(f"[{phase_name}] Best val accuracy: {best_acc*100:.2f}%")
    return model, history


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_history(h1: dict, h2: dict = None):
    def join(key):
        return h1[key] + (h2[key] if h2 else [])

    acc, vacc   = join("train_acc"),  join("val_acc")
    loss, vloss = join("train_loss"), join("val_loss")
    ep = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(ep, acc,  color="#006B3F", label="Train Accuracy")
    ax1.plot(ep, vacc, color="#FCD116", ls="--", label="Val Accuracy")
    ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(ep, loss,  color="#CE1126", label="Train Loss")
    ax2.plot(ep, vloss, color="#000000", ls="--", label="Val Loss")
    ax2.set_title("Loss"); ax2.set_xlabel("Epoch")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle("Ghana Currency CNN — Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "training_history.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[Plot] Training history -> {path}")


def plot_confusion_matrix(model, val_loader, names):
    print("Generating confusion matrix ...")
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs.to(DEVICE))
            _, preds = torch.max(outputs, 1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.numpy())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=names, yticklabels=names,
                linewidths=0.5, linecolor="gray")
    plt.title("Confusion Matrix — Ghana Currency Recognition",
              fontsize=14, fontweight="bold")
    plt.ylabel("Actual Denomination"); plt.xlabel("Predicted Denomination")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[Plot] Confusion matrix -> {path}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=names))


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Ghana Currency Recognition — CNN Training  (PyTorch)")
    print("=" * 60)
    print(f"  Device : {DEVICE}")
    print(f"  Dataset: {DATASET_PATH}")

    if not os.path.exists(DATASET_PATH):
        print(f"\n[ERROR] Dataset not found at: {DATASET_PATH}")
        print("\nCreate the following folder structure:")
        for d in ["1_GHS", "2_GHS", "5_GHS", "10_GHS",
                  "20_GHS", "50_GHS", "100_GHS", "200_GHS"]:
            print(f"  dataset/{d}/   <- put your note images here")
        return

    print("\nLoading dataset ...")
    train_loader, val_loader, class_names = get_data_loaders()
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Training   batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    with open(CLASS_NAMES, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"Class names saved -> {CLASS_NAMES}")

    # Phase 1 — backbone frozen, train only the new head
    print("\n--- Phase 1: Feature Extraction ---")
    model = build_model(num_classes)
    model, h1 = train_phase(model, train_loader, val_loader,
                            INITIAL_EPOCHS, LEARNING_RATE, "Phase1")

    # Phase 2 — unfreeze last 10 backbone layers for fine-tuning
    print("\n--- Phase 2: Fine-Tuning ---")
    unfreeze_top_layers(model, n_layers=10)
    model, h2 = train_phase(model, train_loader, val_loader,
                            FINETUNE_EPOCHS, FINETUNE_LR, "Phase2")

    # Save as TorchScript (.pt) — portable, no class definition needed at load
    model.eval()
    example  = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    scripted = torch.jit.trace(model, example)
    torch.jit.save(scripted, MODEL_SAVE)
    print(f"\nModel saved -> {MODEL_SAVE}")

    plot_history(h1, h2)
    plot_confusion_matrix(model, val_loader, class_names)
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
