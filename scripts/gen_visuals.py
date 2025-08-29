#!/usr/bin/env python3
import json
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_val_curve(log_glob: str, out_png: str) -> bool:
    logs = sorted(Path('assets').glob(log_glob), key=lambda p: p.stat().st_mtime)
    if not logs:
        return False
    text = logs[-1].read_text(encoding='utf-8', errors='ignore')
    epochs, accs = [], []
    for line in text.splitlines():
        m = re.search(r"Epoch (\d+)/(\d+) \| loss=([0-9.]+) \| val_acc=([0-9.]+)", line)
        if m:
            epochs.append(int(m.group(1)))
            accs.append(float(m.group(4)))
    if not epochs:
        return False
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, accs, marker='o', lw=1.8)
    plt.xlabel('Epoch')
    plt.ylabel('Val Acc')
    plt.title('V6 attn_pool Validation Accuracy')
    plt.grid(True, alpha=0.3)
    plt.ylim(0.0, 1.05)
    plt.tight_layout()
    Path('docs').mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()
    return True


def plot_cm(json_path: str, out_png: str, title: str) -> bool:
    p = Path(json_path)
    if not p.exists():
        return False
    obj = json.loads(p.read_text(encoding='utf-8'))
    cm = np.array(obj.get('matrix', []), dtype=np.int64)
    if cm.size == 0:
        return False
    plt.figure(figsize=(5.6, 5.0))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    h, w = cm.shape
    for i in range(h):
        for j in range(w):
            plt.text(j, i, str(int(cm[i, j])), ha='center', va='center', color='black', fontsize=8)
    plt.xticks(range(w))
    plt.yticks(range(h))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    Path('docs').mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()
    return True


if __name__ == '__main__':
    ok1 = plot_val_curve('v6_training_*.log', 'docs/v6_val_acc_curve.png')
    ok2 = plot_cm('assets/confusion_matrix.json', 'docs/v6_confusion_matrix.png', 'V6 Confusion Matrix (Val)')
    ok3 = plot_cm('assets/test_confusion_matrix.json', 'docs/v6_test_confusion_matrix.png', 'V6 Confusion Matrix (Test)')
    print({'val_curve': ok1, 'val_cm': ok2, 'test_cm': ok3})


