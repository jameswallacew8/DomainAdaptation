# TODO - messy script, datasets requite different pre-processing functions and models must be changed manually

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

def prepare_breast_primary_atlas(patient_ids):
    # Uses normal epithelial from GSE161529 and merges with 'pan caner blueprint' dataset i.e. EMTAB8107
    dataset_fp = Path(r"path_to\Breast_GSE161529\BRCA_GSE161529_annotated.h5ad")
    priamry_tum_atlas = Path(r"path_to\Breast_PanCancerBlueprint\Breast_PanCancer_Annotated.h5ad")

    adata_tum = sc.read_h5ad(priamry_tum_atlas)


    def _assignPrimaryTumour(row):
        # Atlas includes datasets already evaluated, so exclude
        if str(row['PatientNumber']) in ('53', '54') and row['CellType'] == 'Cancer':
            return LABELS["tumour"]
            # elif row['Cell_Type_Annotation'] == 'Epithelial Cells':
            #     return LABELS["normal"]
          
        else:
                return np.nan
        
    adata_tum.obs["label"] = adata_tum.obs.apply(_assignPrimaryTumour, axis=1)
    adata_tum = adata_tum[adata_tum.obs["label"].notna()].copy()
    print(adata_tum.obs["label"].value_counts())

    def _assignEpi(row):
        if row['Patient'] in patient_ids and row['Celltype (major-lineage)'] == 'Epithelial': return LABELS['normal']
        else: return np.nan
    
    adata_epi_normal = sc.read_h5ad(dataset_fp)
    adata_epi_normal.obs["label"] = adata_epi_normal.obs.apply(_assignEpi, axis=1)
    adata_epi_normal = adata_epi_normal[adata_epi_normal.obs["label"].notna()].copy()
    print(adata_epi_normal.obs["label"].value_counts())

    adata_combined = ad.concat(
        [adata_epi_normal, adata_tum], 
        join='outer', 
        label='batch', 
        keys=['epi', 'erTum'],
        merge='same'
    )
    
    adata = adata_combined.copy()
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

def prepare_breast_train_data(isTNBC):
    LOOM_PATH = Path(r"path_to\Breast_GSE176078_train\breast_processed.h5ad")

    adata = sc.read_h5ad(LOOM_PATH)
    print(f"[prepare_data] Loaded {adata.n_obs} cells × {adata.n_vars} genes")

    subtype = 'TNBC' if isTNBC else 'HER2+'
    def _assignbreast(row):
        if row['subtype'] == subtype:
            if row['celltype_major'] == "Cancer Epithelial":
                return LABELS["tumour"]
            elif row['celltype_major'] == 'Normal Epithelial':
                return LABELS["normal"]
            else:
                return np.nan
        else:
                return np.nan
        
        
    adata.obs["label"] = adata.obs.apply(_assignbreast, axis=1)
    adata = adata[adata.obs["label"].notna()].copy()

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

def create_breast_subtype_dataset(patient_ids, isErPosExternal):
    dataset_fp = Path(r"path_to\Breast_GSE161529\BRCA_GSE161529_annotated.h5ad")
    dataset_er_pos = Path(r"path_to\Breast_PRJNA1140267\Integrated_Dataset_Annotated.h5ad")
    
    adata = sc.read_h5ad(dataset_fp)
    if isErPosExternal:
        def _assignEpi(row):
            if row['Patient'] in patient_ids and row['Celltype (major-lineage)'] == 'Epithelial': return LABELS['normal']
            else: return np.nan
        
        adata.obs["label"] = adata.obs.apply(_assignEpi, axis=1)
        adata_epi = adata[adata.obs["label"].notna()].copy()
        print(adata_epi.obs["label"].value_counts())

        # external ER+
        disease = 'Primary'
        def _assignbreastER(row):
            if row['Disease'] == disease and row['major_celltype'] == "Malignant": return LABELS['tumour']
            else: return np.nan

        adata_er = sc.read_h5ad(dataset_er_pos)
        adata_er.obs["label"] = adata_er.obs.apply(_assignbreastER, axis=1)
        adata_er = adata_er[adata_er.obs["label"].notna()].copy()
        print(adata_er.obs["label"].value_counts())

        adata_combined = ad.concat(
            [adata_epi, adata_er], 
            join='outer', 
            label='batch', 
            keys=['epi', 'er'],
            merge='same'
        )
        
        adata = adata_combined.copy()
    else:
        def _assignbreast(row):
            if row['Patient'] in patient_ids:
                if row['Celltype (major-lineage)'] == "Malignant":
                    return LABELS['tumour']
                elif row['Celltype (major-lineage)'] == 'Epithelial':
                    return LABELS['normal']
                else:
                    return np.nan
            else:
                return np.nan
            
        
        adata.obs["label"] = adata.obs.apply(_assignbreast, axis=1)
        # Remove duplicate genes
        adata = adata[adata.obs["label"].notna()].copy()
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


def chooseModel(modelName):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if modelName == "GF1":
        config = GeneformerConfig(model_name="gf-6L-30M-i2048", batch_size=12, device=device)
        model = GeneformerFineTuningModel(config, fine_tuning_head="classification", output_size=2)
        # Balanced train
        model.load_state_dict(torch.load(r"path_to\Checkpoints\BreastERPositive\Geneformer\Train\pytorch_model_epoch1.bin", map_location=device))

        model.model.to(device).eval()
        return model
    else:
        config = scGPTConfig(batch_size=12, device=device)
        model = scGPTFineTuningModel(config, fine_tuning_head="classification", output_size=2)
        # Balanced Train
        model.load_state_dict(torch.load(r"path_toCheckpoints\BreastERPositive\scGPT\Train\pytorch_model_epoch1.bin", map_location=device))
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

# Patient IDs
# removed 'patient: 0114' which is duplidcated in er_plus and tnbc
er_plus = ['patient: 0001', 'patient: 0125', 'patient: 0360', 
        'patient: 0032', 'patient: 0042', 'patient: 0025', 'patient: 0151',
        'patient: 0163', 'patient: 0029-7C', 'patient: 0029-9C', 'patient: 0040',
        'patient: 0043', 'patient: 0056', 'patient: 0064', 'patient: 0167',
        'patient: 0173', 'patient: 0178', 'patient: 0068']
# Triple Negative Tumour (Non-BRCA1)
tnbc_patients = [
    "patient: 0126", "patient: 0135", "patient: 0106", "patient: 0114"
]

# Triple Negative BRCA1 Tumour
tnbc_brca1_patients = [
    "patient: 4031", "patient: 0131", "patient: 0554", "patient: 0177"
]

# HER2+ Tumour
her2_positive_patients = [
    "patient: 0308", "patient: 0337", "patient: 0031", "patient: 0069", "patient: 0161",
    "patient: 0176"
]

model = chooseModel("scgpt")


dataset_ids = ['GSE161529_ERPlus','GSE161529_TNBC_NonBRCA1', 'GSE161529_TNBC_BRCA1', 'GSE161529_HER2Plus']
patient_ids = [er_plus,            tnbc_patients,             tnbc_brca1_patients, her2_positive_patients]

dataset_ids = ['PRJNA1140267_ERPlus_Primary']
isErPosExternal = [None]

dataset_ids = ['Train_TNBC','Train_HER2']
patient_ids = [er_plus,              er_plus] # need er_plus patients to obtain relevant epithelial cells

dataset_ids = ['PanCancerLuminal']
patient_ids = [er_plus] # need er_plus patients to obtain relevant epithelial cells
isErPosExternal = [None]

for dataset_id, patient_id, erPos in zip(dataset_ids, patient_ids, isErPosExternal):
    ad_test = prepare_breast_primary_atlas(patient_id)
    if sp.issparse(ad_test.X): # Ensure integer counts
        ad_test.X.data = np.rint(ad_test.X.data).astype(np.int32)
        ad_test.X = ad_test.X.astype(np.int32)
    else:
        ad_test.X = np.rint(ad_test.X).astype(np.int32)

    y_val = ad_test.obs["label"].astype(int).tolist()

    unique, counts = np.unique(y_val, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")
    # out_dir = Path(rf"path-to\scRNA\Checkpoints\BreastERPositive\Geneformer\BRCA_GSE161529\{dataset_id}")
    out_dir = Path(rf"path-to\scRNA\Checkpoints\BreastERPositive\scGPT\BRCA_GSE161529\{dataset_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    val_ds = model.process_data(ad_test , gene_names='gene_names')

    outputFMResults(model, val_ds, y_val, out_dir)


