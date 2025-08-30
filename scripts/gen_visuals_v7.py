#!/usr/bin/env python3
import json
import re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'assets_v7'
DOCS = ROOT / 'docs'
DOCS.mkdir(parents=True, exist_ok=True)


def plot_val_curve_from_logs(out_png: str) -> bool:
    logs = list(ASSETS.glob('v7_training_epoch_*.log'))
    # 同时并入 training_v7/logs（无周期导出时）
    fallback_dir = ROOT.parent / 'training_v7' / 'logs'
    if fallback_dir.exists():
        logs += list(fallback_dir.glob('v7_training_*.log'))
    # 去重并按修改时间排序
    logs = sorted(set(logs), key=lambda p: p.stat().st_mtime)
    if not logs:
        return False
    epochs, accs = [], []
    # 合并读取全部可用日志（跳过过小/空日志）
    for log in logs:
        try:
            if log.stat().st_size < 200:
                continue
        except Exception:
            continue
        text = log.read_text(encoding='utf-8', errors='ignore')
        for line in text.splitlines():
            m = re.search(r"Epoch (\d+)/(\d+) \| loss=([0-9.\-]+) \| val_acc=([0-9.]+)", line)
            if m:
                e = int(m.group(1))
                a = float(m.group(4))
                if not epochs or e > epochs[-1]:
                    epochs.append(e)
                    accs.append(a)
    if not epochs:
        return False
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, accs, marker='o', lw=1.8)
    plt.xlabel('Epoch')
    plt.ylabel('Val Acc')
    plt.title('V7 attn_pool Validation Accuracy')
    plt.grid(True, alpha=0.3)
    plt.ylim(0.0, 1.05)
    plt.tight_layout()
    plt.savefig(DOCS / out_png, dpi=150)
    plt.close()
    return True


def plot_cm(json_path: Path, out_png: str, title: str) -> bool:
    if not json_path.exists():
        return False
    obj = json.loads(json_path.read_text(encoding='utf-8'))
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
    plt.savefig(DOCS / out_png, dpi=150)
    plt.close()
    return True


if __name__ == '__main__':
    ok_curve = plot_val_curve_from_logs('v7_val_acc_curve.png')
    # 混淆矩阵（验证与测试）
    ok_cm_val = False
    ok_cm_test = False
    # 取最新的混淆矩阵快照
    snaps = sorted(ASSETS.glob('v7_confusion_matrix_epoch_*.json'), key=lambda p: p.stat().st_mtime)
    if snaps:
        ok_cm_val = plot_cm(snaps[-1], 'v7_confusion_matrix.png', 'V7 Confusion Matrix (Val)')
    # 若有测试混淆矩阵（人工/脚本复制到 assets_v7/test_confusion_matrix.json）
    test_cm = ASSETS / 'test_confusion_matrix.json'
    if test_cm.exists():
        ok_cm_test = plot_cm(test_cm, 'v7_test_confusion_matrix.png', 'V7 Confusion Matrix (Test)')
    print({'v7_val_curve': ok_curve, 'v7_val_cm': ok_cm_val, 'v7_test_cm': ok_cm_test})


