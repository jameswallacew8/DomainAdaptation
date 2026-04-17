#!/usr/bin/env python3
"""
================================
Fine‑tunes scgpt and saves model after each epoch
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad

import scipy.sparse as sp
import torch
from torch.nn.modules import loss
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from helical.models.scgpt import scGPTConfig, scGPTFineTuningModel

LABELS = {"normal": 0, "tumour": 1}


def prepare_data(train_data_path: Path, celltype_filter):
    adata = sc.read_h5ad(train_data_path)
    print(f"[prepare_data] Loaded {adata.n_obs} cells × {adata.n_vars} genes")

    adata.obs["label"] = adata.obs.apply(celltype_filter, axis=1)
    adata = adata[adata.obs["label"].notna()].copy()

    
    unique, counts = np.unique(adata.obs["label"], return_counts=True)
    class_counts = dict(zip(unique, counts))

    downsample_majority = True   
    if downsample_majority:
        # 1. Identify counts
        counts = adata.obs["label"].value_counts()
        maj_label, min_label = counts.idxmax(), counts.idxmin()

        # 2. Split data
        adata_maj = adata[adata.obs["label"] == maj_label].copy()
        adata_rest = adata[adata.obs["label"] != maj_label].copy()

        # 3. Downsample Majority to match Minority size (1:1 ratio)
        target_n = counts[min_label] 
        if len(adata_maj) > target_n:
            sc.pp.subsample(adata_maj, n_obs=target_n, random_state=42)

        # 4. Merge back
        adata = ad.concat([adata_maj, adata_rest], merge="same")


    print(adata.obs['label'].value_counts())

    # Ensure integer counts
    if sp.issparse(adata.X):
        adata.X.data = np.rint(adata.X.data).astype(np.int32)
        adata.X = adata.X.astype(np.int32)
    else:
        adata.X = np.rint(adata.X).astype(np.int32)

    print(f"[prepare_data] Before filtering: {adata.n_vars} genes")
    sc.pp.filter_genes(adata, min_counts=1)  # Keep genes expressed at least once
    print(f"[prepare_data] After filtering: {adata.n_vars} genes")
    
    adata.var["gene_names"] = adata.var.index  # <- will cause issues if gene ids are not in index
    
    return adata.copy()


def build_model(batch_size: int, n_classes: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[build_model] Using device: {device}")
    config = scGPTConfig(batch_size=batch_size, device=device)
    model = scGPTFineTuningModel(config, fine_tuning_head="classification", output_size=2)
    model.model.to(device).eval()

    return model


# ───────────────────────────────────────────────────────────────────────────────
# Metric evaluation helper (loss, accuracy, precision)
# ───────────────────────────────────────────────────────────────────────────────
def evaluateSCGPT(model, ds, labels, n_classes, batch_size=64, device="cuda"):
    """Evaluate model on a dataset and return loss, accuracy, precision, recall, and f1."""
    
    outputs = model.get_outputs(ds)
    preds = outputs.argmax(axis=1)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="macro", zero_division=0)
    rec = recall_score(labels, preds, average="macro", zero_division=0)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    print(classification_report(labels, preds, target_names=["Non-Tumour", "Tumour"]))
    return 0.0, acc, prec, rec, f1

def outputFMResults(model, tr_ds_fold, y_train_fold, val_ds_fold, y_val_fold, fold, epoch, n_classes, batch_size, device, all_results, out_dir):
    
    # Evaluate after each epoch
    tr_loss, tr_acc, tr_prec, tr_rec, tr_f1 = evaluateSCGPT(model, tr_ds_fold, y_train_fold, n_classes, batch_size, device)
    val_loss, val_acc, val_prec, val_rec, val_f1 = evaluateSCGPT(model, val_ds_fold, y_val_fold, n_classes, batch_size, device)
    
    print(f"  Train: loss {tr_loss:.4f} | acc {tr_acc:.4f} | prec {tr_prec:.4f} | rec {tr_rec:.4f} | f1 {tr_f1:.4f}")
    print(f"  Val:   loss {val_loss:.4f} | acc {val_acc:.4f} | prec {val_prec:.4f} | rec {val_rec:.4f} | f1 {val_f1:.4f}")
    
    # Store results for this epoch
    epoch_results = {
        'fold': fold,
        'epoch': epoch,
        'train_loss': float(tr_loss),
        'train_acc': float(tr_acc),
        'train_prec': float(tr_prec),
        'train_rec': float(tr_rec),
        'train_f1': float(tr_f1),
        'val_loss': float(val_loss),
        'val_acc': float(val_acc),
        'val_prec': float(val_prec),
        'val_rec': float(val_rec),
        'val_f1': float(val_f1)
    }
    all_results.append(epoch_results)
        
    with open(out_dir / "cv_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    return all_results, float(val_f1)

# ───────────────────────────────────────────────────────────────────────────────
# Main training routine
# ───────────────────────────────────────────────────────────────────────────────

# ---------- Dataset filters which should be chosen below ----------

# --- Renal ccRCC label assignment ---
def _assignCCRCCNormalVsTumour(row):
    if row['tissue'] in ("Normal kidney"):
        return LABELS["normal"]
    elif row['scienceIDType'] == "RCC" and row['tissue'] == "Tumour":
        return LABELS["tumour"]
    else:
        return np.nan

# --- Lung LUAD label assignment ---
LUAD_patient_ids = ('P12', 'P5', 'P39', 'P2', 'P35', 'P32', 'P21', 'P13', 'P33', 'P9', 'P20', 'P38', 'P28', 'P24', 'P16', 'P8', 'P29', 'P34')
def _assignLungLUADNormalVsTumour(row):
    if row['Patient'] in LUAD_patient_ids:
        if row['Celltype (major-lineage)'] == ("Malignant"):
            return LABELS["tumour"]
        else:
            return LABELS['normal']
    else:
        return np.nan

# --- Breast ER+ label assignment ---
def _assignBreastERNormalVsTumour(row):
    if row['subtype'] == 'ER+':
        if row['celltype_major'] == "Cancer Epithelial":
            return LABELS["tumour"]
        elif row['celltype_major'] != 'CAFs':
            return LABELS["normal"]
        else:
            return np.nan
    else:
            return np.nan
    
def main():
    out_dir = Path(r"path_to\LungCancerType\scGPT\Train")
    out_dir.mkdir(parents=True, exist_ok=True)

    TRAIN_PATH = Path(r"path_to\LungCancerType.h5ad")
    dataset_cell_filter = _assignLungLUADNormalVsTumour  # example using lung

    N_HVGS = 10000
    batch_size = 12
    epochs = 10
    lr = 5e-5

    #  3. Load & Prepare Full Data
    print("⏳ Loading and preparing data...")
    ad_full = prepare_data(TRAIN_PATH, dataset_cell_filter)

    y_full = ad_full.obs["label"].astype(np.int64).values

    # 4. Create 10% Validation Split
    print("✂️ Splitting data (90% Train / 10% Validation)...")
    ad_train, ad_val, y_train, y_val = train_test_split(
        ad_full, 
        y_full, 
        test_size=0.1, 
        stratify=y_full, 
        random_state=42
    )
    model = build_model(batch_size, 2)
  
    tr_ds = model.process_data(ad_train, gene_names="gene_names", fine_tuning=True, n_top_genes=N_HVGS)
    val_ds = model.process_data(ad_val, gene_names="gene_names")

    steps_per_epoch = math.ceil(len(tr_ds) / batch_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Count occurrences of each class in your labels
    unique, counts = np.unique(y_train, return_counts=True)
    class_counts = dict(zip(unique, counts))

    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(y_train), 
        y=y_train
    )
    # Doesn't have any impact as class is already balanced, useful in unbalanced experiments
    weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    all_results = [] # To store metrics

    # ─────────── Epoch loop with metric + checkpoint per epoch ───────────────
    for epoch in range(1, epochs + 1):
        print(f"\n🟢 Epoch {epoch}/{epochs}")

        model.train(
            train_input_data=tr_ds,
            train_labels=y_train,
            validation_input_data=val_ds,   # manual eval below
            validation_labels=y_val,
            epochs=1,                  # one epoch per loop
            loss_function=loss.CrossEntropyLoss(weight=weights),
            optimizer_params={"lr": lr},
        )
    
        all_results,_ = outputFMResults(
            model=model, 
            tr_ds_fold=tr_ds, 
            y_train_fold=y_train, 
            val_ds_fold=val_ds, 
            y_val_fold=y_val, 
            fold=1, 
            epoch=epoch, 
            n_classes=2, 
            batch_size=batch_size, 
            device=device, 
            all_results=all_results, 
            out_dir=out_dir
        )

        # 10. Save Checkpoint
        torch.save(model.state_dict(), out_dir / f"pytorch_model_epoch{epoch}.bin")
        print(f"💾 Checkpoint saved: epoch{epoch}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[ERROR]", e)
        
