import scanpy as sc
import numpy as np
import pandas as pd
import torch
import typing
import os
import json
import scipy.sparse as sp # Used for type checking adata.X
from anndata import AnnData
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from helical.models.geneformer import GeneformerConfig, GeneformerFineTuningModel
from helical.models.scgpt import scGPTConfig, scGPTFineTuningModel

LABELS = {"normal": 0, "tumour": 1}

def chooseDataset(dataset) -> AnnData:
    if dataset == "US":
        adata = sc.read_loom(r"path_to\RawDatasets\UnitedStates\output.loom")
        def _assignFineGrained(row):
            if row["CellType"] in ("PT-A", "PT-B", "tAL", "TAL", "IC-A", "IC-B", "CNT", "IC-PC", "PC", "DCT", "DL") and row["SampleClass"] == "Benign":
                return LABELS["normal"]
            if row["CellType"] in ("Tumor") and row["SampleClass"] == "ccRCC":
                return LABELS["tumour"]
            return np.nan
        
        adata.obs["label"] = adata.obs.apply(_assignFineGrained, axis=1)
        downsample_tumour = True
        if downsample_tumour:
            unique, counts = np.unique(adata.obs["label"], return_counts=True)
            class_counts = dict(zip(unique, counts))

            adata_ep = adata[adata.obs['label'] == 0.0]
            adata_tum = adata[adata.obs['label'] == 1.0]

            # 2. Downsample Primary to 3x the size of Invasive
            target_n = len(adata_ep) * 1  # approx 3387 cells
            sc.pp.subsample(adata_tum, n_obs=target_n, random_state=42)

            # 3. Concatenate back
            adata = adata_ep.concatenate(adata_tum)
        # Remove duplicate genes
        adata = adata[:, ~adata.var["gene_names"].duplicated(keep='first')]
        return adata
    
    if dataset == "US-CH":
        adata_ch = sc.read_loom(r"path_to\RawDatasets\UnitedStates\chromophobeRCC.loom")
        adata_epi = sc.read_loom(r"path_to\RawDatasets\UnitedStates\output.loom")
        adata_combined = ad.concat(
            [adata_ch, adata_epi], 
            join='outer', 
            label='batch', 
            keys=['chromophobe', 'epithelial'],
            merge='same'
        )
        def _assign2(row):
            # TODO - Does it require Epithelial cells to
            if row["CellType"] == "Tumor" and row['SampleClass'] == 'chRCC':
                return LABELS["tumour"]
            elif row["SampleClass"] == "Benign" and row["CellType"] in ("PT-A", "PT-B", "tAL", "TAL", "IC-A", "IC-B", "CNT", "IC-PC", "PC", "DCT", "DL") :
                return LABELS["normal"]
        
        adata_combined.obs["label"] = adata_combined.obs.apply(_assign2, axis=1)
        # Remove duplicate genes
        adata_combined = adata_combined[adata_combined.obs["label"].notna()].copy()
        downsample_tumour = True
        if downsample_tumour:
            unique, counts = np.unique(adata_combined.obs["label"], return_counts=True)
            class_counts = dict(zip(unique, counts))

            adata_ep = adata_combined[adata_combined.obs['label'] == 0.0]
            adata_tum = adata_combined[adata_combined.obs['label'] == 1.0]

            # 2. Downsample Primary to 3x the size of Invasive
            target_n = len(adata_ep) * 1  # approx 3387 cells
            sc.pp.subsample(adata_tum, n_obs=target_n, random_state=42)

            # 3. Concatenate back
            adata_combined = adata_ep.concatenate(adata_tum)
        adata_combined = adata_combined[:, ~adata_combined.var["gene_names"].duplicated(keep='first')]
        return adata_combined

    if dataset == "LITHUANIA":
        adata = sc.read(r"path_to\RawDatasets\Lithuania\GSE242299_all_cells_50236_33538.h5ad\GSE242299_all_cells_50236_33538.h5ad")
        def _assign2(row):
            if row["broad_cell_type"] == "Epithelial": return LABELS["normal"]
            elif row["broad_cell_type"] == "Tumor": return LABELS["tumour"]
            else:
                return np.nan
        
        adata.obs["label"] = adata.obs.apply(_assign2, axis=1)
        adata = adata[adata.obs["label"].notna()].copy()
        adata.obs["n_counts"] = pd.to_numeric(adata.obs["n_counts"], errors='coerce')
        adata.var["gene_names"] = adata.var.index

        normal_count = (adata.obs["label"] == LABELS["normal"]).sum()
        tumour_count = (adata.obs["label"] == LABELS["tumour"]).sum()
        
        print(f"[prepare_data] Class distribution before downsampling:")
        print(f"  Normal: {normal_count}, Tumour: {tumour_count}")
        
        # Get indices for each class
        normal_idx = adata.obs["label"] == LABELS["normal"]
        tumour_idx = adata.obs["label"] == LABELS["tumour"]
        
        # Randomly sample tumour cells to match normal count
        # tumour_cells = adata[tumour_idx].obs.index
        # np.random.seed(42)  # For reproducibility
        # sampled_tumour = np.random.choice(tumour_cells, size=normal_count, replace=False)
        normal_cells = adata[normal_idx].obs.index
        np.random.seed(42)  # For reproducibility
        sampled_normal = np.random.choice(normal_cells, size=tumour_count, replace=False)
        
        # Combine normal cells with downsampled tumour cells
        keep_idx = adata.obs.index.isin(sampled_normal) | tumour_idx
        adata = adata[keep_idx].copy()
        
        print(f"[prepare_data] Class distribution after downsampling:")
        print(f"  Normal: {(adata.obs['label'] == LABELS['normal']).sum()}")
        print(f"  Tumour: {(adata.obs['label'] == LABELS['tumour']).sum()}")

        return adata
    
    if dataset == "CHINA":
        adata = sc.read(r"path_to\RawDatasets\China\GSE156632_RAW\all_cells_all_genes.loom")
        def _assign2(row):
            if row["CellType"] == "Epithelial1": return LABELS["normal"]
            elif row["CellType"] == "Tumor1": return LABELS["tumour"]
            return np.nan
        
        adata.obs["label"] = adata.obs.apply(_assign2, axis=1)
        adata = adata[adata.obs["label"].notna()].copy()

        adata.var["gene_names"] = adata.var.index

        downsample_tumour = True
        if downsample_tumour:
            unique, counts = np.unique(adata.obs["label"], return_counts=True)
            class_counts = dict(zip(unique, counts))

            adata_ep = adata[adata.obs['label'] == 0.0]
            adata_tum = adata[adata.obs['label'] == 1.0]

            # 2. Downsample Primary to 3x the size of Invasive
            target_n = len(adata_tum) * 1  # approx 3387 cells
            sc.pp.subsample(adata_ep, n_obs=target_n, random_state=42)

            # 3. Concatenate back
            adata = adata_tum.concatenate(adata_ep)

        return adata

    
    else:
        # Bone metastasis
        adata = sc.read(r"path_to\GSE202813_RAW\GSE202813_merged_Benign_Tumor_Only.h5ad")

        def _assign(row):
            if row['condition'] == 'Tumor' and row['cell_type']== 'Tumor1':
                return LABELS["tumour"]
            elif row['condition'] == 'Normal' and row['cell_type']== 'MSCs1':
                return LABELS["normal"]
            else:
                return np.nan
        
        adata.obs["label"] = adata.obs.apply(_assign, axis=1)
        adata = adata[adata.obs["label"].notna()].copy()
        
        downsample_normal = True   
        if downsample_normal:
            unique, counts = np.unique(adata.obs["label"], return_counts=True)
            class_counts = dict(zip(unique, counts))

            adata_tum = adata[adata.obs['label'] == 1.0]
            adata_normal = adata[adata.obs['label'] == 0.0]

            # 2. Downsample Primary to 3x the size of Invasive
            target_n = len(adata_tum) * 1  # approx 3387 cells
            sc.pp.subsample(adata_normal, n_obs=target_n, random_state=42)

            # 3. Concatenate back
            adata = adata_tum.concatenate(adata_normal)

        adata.var["gene_names"] = adata.var.index
        return adata
    
    

def chooseModel(modelName):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if modelName == "GF1":
        config = GeneformerConfig(model_name="gf-6L-30M-i2048", batch_size=12, device=device)
        model = GeneformerFineTuningModel(config, fine_tuning_head="classification", output_size=2)
        # Balanced train
        model.load_state_dict(torch.load(r"path_to\Checkpoints\TischLung\geneformer\LuscAllVsTumour\Train\pytorch_model_epoch2.bin", map_location=device))

        model.model.to(device).eval()
        return model

    else:
        config = scGPTConfig(batch_size=12, device=device)
        model = scGPTFineTuningModel(config, fine_tuning_head="classification", output_size=2)
        # Balanced Train
        model.load_state_dict(torch.load(r"path_to\Checkpoints\NormalStromaIncEpVsTumourCore\scGPT\10k-HVG_Norm_5e5lr_5Fold_10Epoch\Train\pytorch_model_epoch1.bin", map_location=device))
        model.model.to(device).eval()
        return model
    
def outputFMResults(model, val_ds, y_val, out_dir):
    """
    Generates ROC/PR curves if mixed data is present, otherwise plots confidence density.
    """
    
    # --- 1. Get Model Predictions ---
    print("Generating outputs...")
    outputs = model.get_outputs(val_ds)
    logits_tensor = torch.tensor(outputs)
    probs_tensor = F.softmax(logits_tensor, dim=-1)
    
    # We specifically need the probability of being TUMOR (Class 1) for ROC/PR
    # Assuming Class 1 = Tumor/Metastasis
    y_scores = probs_tensor[:, 1].numpy()
    save_path = f"{out_dir}/results.npz"
    
    # We save 'y_true' and 'y_scores' into a single compressed file
    np.savez(save_path, y_true=y_val, y_scores=y_scores)
    
    print(f"💾 Results saved for future plotting: {save_path}")

    # --- 2. Diagnostic Text Report ---
    print("\n" + "="*50)
    print("MODEL PERFORMANCE DIAGNOSTICS")
    print("="*50)
    print(f"Total Cells Evaluated: {len(y_scores)}")
    print(f"Mean Tumor Probability: {y_scores.mean():.4f}")
    
    # --- 3. Determine Plot Type based on Labels ---
    # Check if we have both classes (0 and 1) in the ground truth
    unique_classes = np.unique(y_val)
    
    if len(unique_classes) > 1:
        # === SCENARIO A: MIXED DATA (Plot ROC + PR) ===
        print("-> Mixed classes detected. Generating ROC & PR Curves.")
        
        # Calculate Metrics
        fpr, tpr, _ = roc_curve(y_val, y_scores)
        roc_auc = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(y_val, y_scores)
        pr_auc = average_precision_score(y_val, y_scores)
        
        print(f"ROC AUC:               {roc_auc:.4f}")
        print(f"PR AUC (AUPRC):        {pr_auc:.4f}")
        
        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot Left: ROC Curve
        ax = axes[0]
        ax.plot(fpr, tpr, color='#E74C3C', lw=2, label=f'Model (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Chance')
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

        # Plot Right: Precision-Recall Curve
        ax = axes[1]
        ax.plot(recall, precision, color='#2ECC71', lw=2, label=f'Model (AUPRC = {pr_auc:.3f})')
        
        # Add baseline (ratio of positives)
        baseline = np.sum(y_val) / len(y_val)
        ax.axhline(y=baseline, color='gray', linestyle='--', label=f'Baseline ({baseline:.2f})')
        
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
        ax.set_ylabel('Precision (Purity)', fontsize=12)
        ax.legend(loc="lower left")
        ax.grid(alpha=0.3)
        
        save_filename = f"{out_dir}/model_performance_roc_pr.png"
    else:
        # === SCENARIO B: SINGLE CLASS (Plot Density Histogram) ===
        # This runs if you pass ONLY NK cells or ONLY Tumor cells
        class_name = "Normal/Negative" if unique_classes[0] == 0 else "Tumor/Positive"
        print(f"-> Single class ({class_name}) detected. ROC undefined. Generating Density Plot.")
        
        fig = plt.figure(figsize=(10, 6))
        
        sns.histplot(
            y_scores, 
            kde=True, 
            stat="percent", 
            bins=30, 
            color="#3498DB" if unique_classes[0] == 0 else "#E74C3C", 
            element="step", 
            alpha=0.4,
            label=f"Predicted Probabilities"
        )
        
        plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Boundary (0.5)')
        plt.title(f"Prediction Distribution on Pure '{class_name}' Set", fontsize=14, fontweight='bold')
        plt.xlabel("Predicted Probability of Tumor (Class 1)", fontsize=12)
        plt.ylabel("Percentage of Cells (%)", fontsize=12)
        plt.xlim(0, 1.0)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        save_filename = f"{out_dir}/model_density_single_class.png"

    # --- 4. Save and Close ---
    plt.tight_layout()
    plt.savefig(save_filename, dpi=300)
    print(f"✅ Plot saved to: {save_filename}")
    plt.close()
    print("="*50 + "\n")

    val_preds = outputs.argmax(axis=1)

   
    print(classification_report(y_val, val_preds, target_names=["Non-Tumour", "Tumour"]))


    
adata = chooseDataset("US")
model = chooseModel("scgpt")

print("📋 adata.obs columns:", adata.obs.columns.tolist())
print("🔎 Sample obs values:\n", adata.obs.head())

# --- Step 1: Load and Prepare Data ---
print("\n--- Step 1: Loading and Preparing Data ---")
print(f"Loaded AnnData: {adata.n_obs} cells, {adata.n_vars} genes initially.")


adata = adata[adata.obs["label"].notna()].copy()
print(f"Filtered AnnData: {adata.n_obs} cells with valid labels.")

if sp.issparse(adata.X): # Ensure integer counts
    adata.X.data = np.rint(adata.X.data).astype(np.int32)
    adata.X = adata.X.astype(np.int32)
else:
    adata.X = np.rint(adata.X).astype(np.int32)


labels_for_split = adata.obs["label"].astype(int).tolist()

adata_test = adata.copy()
y_val = adata_test.obs["label"].astype(int).tolist()
print(f"Test set created with {adata_test.n_obs} cells.")

print("Class distribution:")
# print(f"Training: {np.bincount(labels_train)}")
print(f"Testing: {np.bincount(y_val)}")

adata_test.var["filter_pass"] = True

adata_test.obs['labels'] = y_val
adata_test.X = sp.csr_matrix(adata_test.X)
val_ds = model.process_data(adata_test , gene_names='gene_names')

# out_dir = Path(r"path_to\Checkpoints\TischLung\geneformer\LuscAllVsTumour\SCLC")

out_dir = Path(r"path_to\Checkpoints\NormalStromaIncEpVsTumourCore\scGPT\10k-HVG_Norm_5e5lr_5Fold_10Epoch\PAPILLARY")
out_dir.mkdir(parents=True, exist_ok=True)


outputFMResults(model, val_ds, y_val, out_dir)


