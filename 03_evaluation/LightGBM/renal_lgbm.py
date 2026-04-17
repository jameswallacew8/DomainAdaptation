import scanpy as sc
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
import torch
import typing
import os
import pickle
import json
import scipy.sparse as sp # Used for type checking adata.X
from anndata import AnnData
import anndata as ad

from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import resample
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

import lightgbm as lgb
from lightgbm import LGBMClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import shap

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


def preprocess_adaptive_balanced(X, y, current_genes, saved_genes):
    """
    Adaptive Preprocessing with BALANCED FITTING.
    
    1. Fits scaler on a balanced subset (50/50) of the new data 
       to avoid class-imbalance bias.
    2. Applies that scaler to the full dataset.
    """
    if isinstance(X, AnnData):
        X = X.X
        
    print(f"\n{'='*60}")
    print(f"Adaptive Validation (Balanced Batch Scaling)")
    print(f"{'='*60}")
    
    # --- STEP 0: Log-Normalization ---
    print(f"\n0. Log-Normalization (Target Sum: 10,000)")
    
    # 1. Library Size Normalization
    counts_per_cell = np.array(X.sum(axis=1)).reshape(-1, 1)
    counts_per_cell[counts_per_cell == 0] = 1 

    scale_factor = 1e4
    
    if sp.issparse(X):
        print("   (Converting sparse matrix to dense for scaling)")
        X = X.toarray()

    X_norm = (X / counts_per_cell) * scale_factor
    X_norm = np.log1p(X_norm)
    print("   Done. Data is now in log1p(CPM) space.")

    # --- STEP 1: Strict Alignment to Training Genes ---
    print(f"\n1. Aligning to {len(saved_genes)} training genes...")
    
    n_samples = X_norm.shape[0]
    n_features = len(saved_genes)
    
    # Create the "Template" matrix (Initialized to 0.0)
    X_aligned = np.zeros((n_samples, n_features), dtype=np.float32)
    
    # Map current genes to their indices in the X_norm matrix
    current_gene_to_idx = {gene: i for i, gene in enumerate(current_genes)}
    
    missing_genes = []
    present_count = 0
    
    # Fill the template
    # We loop through SAVED_GENES to ensure the order is exactly what the model expects
    for i, target_gene in enumerate(saved_genes):
        if target_gene in current_gene_to_idx:
            source_idx = current_gene_to_idx[target_gene]
            X_aligned[:, i] = X_norm[:, source_idx]
            present_count += 1
        else:
            missing_genes.append(target_gene)
            # It remains 0.0 as initialized
            
    if missing_genes:
        print(f"   WARNING: {len(missing_genes)} training genes are missing in this dataset.")
        print(f"   (Filled with 0.0, e.g., {missing_genes[:3]}...)")
    else:
        print("   All training genes found.")
    
    # --- 2. Create Balanced Subset for Fitting ---
    print(f"\n2. Creating Balanced Subset for Scaler Fitting...")
    
    # Find indices of each class
    idx_0 = np.where(np.array(y) == 0)[0] # Epithelial
    idx_1 = np.where(np.array(y) == 1)[0] # Tumor
    
    n_samples = min(len(idx_0), len(idx_1))
    print(f"   Downsampling to {n_samples} per class for scaler fitting.")
    
    # Sample equal numbers
    idx_0_bal = resample(idx_0, n_samples=n_samples, random_state=42)
    idx_1_bal = resample(idx_1, n_samples=n_samples, random_state=42)
    
    balanced_indices = np.concatenate([idx_0_bal, idx_1_bal])
    X_subset_balanced = X_aligned[balanced_indices]
    
    # --- 3. Fit Scaler on Balanced Subset ---
    # print(f"   Fitting scaler on balanced subset ({len(balanced_indices)} cells)...")
    scaler = StandardScaler()
    scaler.fit(X_subset_balanced) # <--- The Magic Step
    
    # --- 4. Transform EVERYTHING ---
    print(f"   Applying scaler to full dataset ({X_aligned.shape[0]} cells)...")
    X_scaled = scaler.transform(X_aligned)

    # --- 5. Clip ---
    X_scaled = np.clip(X_scaled, -10, 10)
    
    return X_scaled

def interpret_LGBM(lgbm_model, X_val, genes_final, shap_save_dir):
    
    shap_save_dir.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(lgbm_model)
    shap_values = explainer.shap_values(X_val)

    train_genes = genes_final
    # Check format
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use tumor class

    print(f"SHAP values shape: {shap_values.shape}")  # Should be (9052, 6939)

    # Calculate mean absolute SHAP value per gene
    mean_shap = np.abs(shap_values).mean(axis=0)  # Shape: (6939,)

    print(f"Mean SHAP shape: {mean_shap.shape}")
    print(f"Common genes length: {len(train_genes)}")

    # Create importance dataframe
    shap_importance = pd.DataFrame({
        'Gene': train_genes,
        'SHAP_Importance': mean_shap
    }).sort_values('SHAP_Importance', ascending=False)

    print("\nTop 15 Genes by SHAP Importance:")
    print(shap_importance.head(15))

    # Summary plot (beeswarm) - shows distribution of impacts
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_val, feature_names=train_genes,
                    max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(shap_save_dir / 'shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Bar plot - shows average importance
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_val, feature_names=train_genes,
                    plot_type="bar", max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(shap_save_dir / 'shap_bar_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Dependence plot for top gene
    top_gene = shap_importance.iloc[0]['Gene']
    top_gene_idx = train_genes.index(top_gene)

    plt.figure(figsize=(10, 6))
    shap.dependence_plot(top_gene_idx, shap_values, X_val,
                        feature_names=train_genes, show=False)
    plt.title(f'SHAP Dependence Plot: {top_gene}')
    plt.tight_layout()
    plt.savefig(shap_save_dir / f'shap_dependence_{top_gene}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Compare SHAP importance vs LightGBM feature importance
    comparison = pd.DataFrame({
        'Gene': train_genes,
        'LightGBM_Importance': lgbm_model.feature_importances_,
        'SHAP_Importance': mean_shap
    }).sort_values('SHAP_Importance', ascending=False)

    print(f"\nCorrelation between LightGBM and SHAP importance: {comparison['LightGBM_Importance'].corr(comparison['SHAP_Importance']):.3f}")

    comparison.to_csv(shap_save_dir / 'importance_comparison.csv', index=False)

    # Save SHAP values for later use
    # np.save(shap_save_dir / 'shap_values.npy', shap_values)
    shap_importance.to_csv(shap_save_dir / 'shap_importance.csv', index=False)

    print(f"\nSHAP analysis complete! Results saved to {shap_save_dir}")

def evaluate_lightgbm(ad_val, y_val, out_dir, interpret):
    """Load pre-trained and evaluate LightGBM model on external datasets."""
    
    save_dir = Path(r"path_to\Checkpoints\TischLung\LGBM\LuscAllVsTumour")

    with open(save_dir / 'model_metadata.json', 'r') as f:
        metadata = json.load(f)

    train_genes = metadata['common_genes'] 

    ad_val_processed = preprocess_adaptive_balanced(
        ad_val.X, 
        y_val,
        ad_val.var["gene_names"], 
        saved_genes=train_genes,     # From training
    )
    # Load model
    with open(save_dir / 'lgbm_model.pkl', 'rb') as f:
        lgbm_model = pickle.load(f)
    # --- NEW: Generate Diagnostic Plot ---
    # This creates the 'nice plot' starting at zero
    if interpret:
        shap_save_dir = out_dir / "SHAP"
        interpret_LGBM(lgbm_model, ad_val_processed,train_genes, shap_save_dir)

    plot_lgbm_performance(lgbm_model,y_val, ad_val_processed, out_dir)
    val_preds = lgbm_model.predict(ad_val_processed)
   
    print(classification_report(y_val, val_preds, target_names=["Non-Tumour", "Tumour"]))

   
def plot_lgbm_performance(model, y_val, X_processed, out_dir):
    """
    Generates a density plot for LightGBM predictions to diagnose 
    metastasis detection confidence.
    """
    # 1. Get Probabilities (Class 1 = Metastasis/Tumor)
    # Check if model is sklearn API (LGBMClassifier) or raw Booster
    y_scores = model.predict_proba(X_processed)[:, 1]
 
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

    

ad_test = chooseDataset("US")
if sp.issparse(ad_test.X): # Ensure integer counts
    ad_test.X.data = np.rint(ad_test.X.data).astype(np.int32)
    ad_test.X = ad_test.X.astype(np.int32)
else:
    ad_test.X = np.rint(ad_test.X).astype(np.int32)

y_test = ad_test.obs["label"].astype(int).tolist()
unique, counts = np.unique(y_test, return_counts=True)
print(f"Class distribution: {dict(zip(unique, counts))}")

out_dir = Path(r"path_to\Checkpoints\TischLung\LGBM\LuscAllVsTumour\SCLC")
out_dir.mkdir(parents=True, exist_ok=True)

print(f"\n🟡 Evaluate LightGBM Baseline...")
lgbm_results = evaluate_lightgbm(
    ad_test, y_test, out_dir, interpret=True
)


with open(out_dir / "Val_results.json", "w") as f:
    json.dump(lgbm_results, f, indent=2)

