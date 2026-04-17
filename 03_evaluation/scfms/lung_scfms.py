# Requires manual switching of foundation model evaluated

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


LABELS = {"normal": 0, "tumour": 1,}

def use_alveolar_epithelial(dataset_id, isLusc, splitByPatients):
    dataset_fp = Path(rf"path_to\RawDatasets\Lung_Tisch\{dataset_id}_annotated.h5ad")
    adata_nsclc = sc.read(dataset_fp)
    hlca_path = r"path_to\RawDatasets\093ff880-81ef-490b-9b33-7b4f2f1b2162\droplet_normal_lung_blood_scanpy.20200205.RC4.h5ad\droplet_normal_lung_blood_scanpy.20200205.RC4.h5ad"
    adata_alveolar = sc.read_h5ad(hlca_path)
    adata_combined = ad.concat(
        [adata_nsclc, adata_alveolar], 
        join='outer', 
        label='batch', 
        keys=['nsclc', 'alveolar'],
        merge='same'
    )
    def get_patient_ids(dataset_id): 
        if dataset_id == 'NSCLC_GSE127465':
            if isLusc:
                return ('p1', 'p2')  
            else: 
                return ('p3', 'p4', 'p5', 'p6', 'p7')
        elif dataset_id == 'NSCLC_GSE117570':
            if isLusc:
                return ('P2') 
            else: 
                return ('P1', 'P3', 'P4')
        elif dataset_id == 'NSCLC_GSE148071':
            if isLusc:
                return ("P22", "P31", "P1", "P41", "P14", "P23", "P36", "P7", "P17", "P10", "P4", "P25", "P3", "P19", "P15", "P40", "P37", "P18")
            else: 
                return ('P12', 'P5', 'P39', 'P2', 'P35', 'P32', 'P21', 'P13', 'P33', 'P9', 'P20', 'P38', 'P28', 'P24', 'P16', 'P8', 'P29', 'P34')
            
    normal_class = ("Basal_P1", 'Basal_P2', 'Basal_P3', 'Differentiating Basal_P1', 'Differentiating Basal_P2', 'Differentiating Basal_P3') if isLusc else ("Alveolar Epithelial Type 2_P3", 'Alveolar Epithelial Type 2_P2', 'Alveolar Epithelial Type 2_P1')
    
    def _assign2(row):
        # TODO - Does it require Epithelial cells to
        if row['Celltype (major-lineage)'] == ("Malignant"):
            if splitByPatients:
                if row['Patient'] in get_patient_ids(dataset_id):
                    return LABELS["tumour"]
            else: return LABELS["tumour"]
        elif row["free_annotation"] in normal_class:
            # and row["CellType"] in ("PT-A", "PT-B", "tAL", "TAL", "IC-A", "IC-B", "CNT", "IC-PC", "PC", "DCT", "DL") :
            return LABELS["normal"]
    
    adata_combined.obs["label"] = adata_combined.obs.apply(_assign2, axis=1)
    # Remove duplicate genes
    adata = adata_combined[adata_combined.obs["label"].notna()].copy()
    downsample_alveolar = True
    print(f"[prepare_data] Loaded {adata.n_obs} cells × {adata.n_vars} genes")
    
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

        # Check result
        print(adata.obs["label"].value_counts())

    adata.var["gene_names"] = adata.var.index
    print(adata.obs['label'].value_counts())
    
    return adata

def chooseDataset(dataset) -> AnnData:
    dataset_fp = Path(rf"path_to\RawDatasets\Lung_Tisch\{dataset}_annotated.h5ad")
    adata = sc.read(dataset_fp)
    def _assign(row):
        if row["Celltype (major-lineage)"] == "Malignant":
            return LABELS["tumour"]
        else:
            return LABELS["normal"]
    
    
    adata.obs["label"] = adata.obs.apply(_assign, axis=1)
    adata = adata[adata.obs["label"].notna()].copy()
    
    downsample_tumour = False   
    if downsample_tumour:
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

        # Check result
        print(adata.obs["label"].value_counts())

    adata.var["gene_names"] = adata.var.index
    print(adata.obs['label'].value_counts())
    return adata
    

def chooseModel(modelName):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if modelName == "GF1":
        config = GeneformerConfig(model_name="gf-6L-30M-i2048", batch_size=12, device=device)
        model = GeneformerFineTuningModel(config, fine_tuning_head="classification", output_size=2)
        # Balanced train
        model.load_state_dict(torch.load(r"path_to\Checkpoints\LUAD_Train\Geneformer\Train\pytorch_model_epoch4.bin", map_location=device))

        model.model.to(device).eval()
        return model

    else:
        config = scGPTConfig(batch_size=12, device=device)
        model = scGPTFineTuningModel(config, fine_tuning_head="classification", output_size=2)
        # Balanced Train
        model.load_state_dict(torch.load(r"path_to\Checkpoints\LUAD_Train\scGPT\Train\pytorch_model_epoch3.bin", map_location=device))
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


model = chooseModel("GF1")

dataset_list = ['NSCLC_GSE150660', 'NSCLC_GSE117570', 'NSCLC_GSE117570', 'NSCLC_GSE127465', 'NSCLC_GSE127465', 'NSCLC_GSE143423', 'NSCLC_GSE148071', 'SCLC_GSE150766']
isLusc =       [False,              False,              True,               False,          True,               False,             True,                 False]

for dataset, isLuscCheck in zip(dataset_list, isLusc):
    splitByPatients = dataset in ['NSCLC_GSE117570', 'NSCLC_GSE127465', 'NSCLC_GSE148071']
    ad_test = use_alveolar_epithelial(dataset, isLusc=isLuscCheck, splitByPatients=splitByPatients)
    if sp.issparse(ad_test.X): # Ensure integer counts
        ad_test.X.data = np.rint(ad_test.X.data).astype(np.int32)
        ad_test.X = ad_test.X.astype(np.int32)
    else:
        ad_test.X = np.rint(ad_test.X).astype(np.int32)

    y_val = ad_test.obs["label"].astype(int).tolist()

    unique, counts = np.unique(y_val, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")

    datasetName = f'{dataset}_LUSC' if isLuscCheck else f'{dataset}_LUAD'
    out_dir = Path(rf"path_to\Checkpoints\LUAD_Train\Geneformer\Lung_MultipleExternal\{datasetName}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    val_ds = model.process_data(ad_test , gene_names='gene_names')

    outputFMResults(model, val_ds, y_val, out_dir)


