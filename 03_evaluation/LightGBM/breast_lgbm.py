# TODO - messy script, datasets requite different pre-processing functions
import scanpy as sc
import numpy as np
import pandas as pd
import seaborn as sns
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

def prepare_breast_pancancer_atlas(patient_ids):
    # Uses normal epithelial from GSE161529 and merges with 'pan caner blueprint' dataset i.e. EMTAB8107
    dataset_fp = Path(r"path_to\RawDatasets\Breast_GSE161529\BRCA_GSE161529_annotated.h5ad")
    priamry_tum_atlas = Path(r"path_to\RawDatasets\Breast_PanCancerBlueprint\Breast_PanCancer_Annotated.h5ad")

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
    # uses TNBC and HER2 from train dataset
    LOOM_PATH = Path(r"path_to\RawDatasets\Breast_GSE176078_train\breast_processed.h5ad")

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

def create_breast_subtype_dataset(patient_ids, isErPosExternal, isErPosMetastasis):
    dataset_fp = Path(r"path_to\RawDatasets\Breast_GSE161529\BRCA_GSE161529_annotated.h5ad")
    dataset_er_pos = Path(r"path_to\RawDatasets\Breast_PRJNA1140267\Integrated_Dataset_Annotated.h5ad")
    
    adata = sc.read_h5ad(dataset_fp)
    if isErPosExternal:
        def _assignEpi(row):
            if row['Patient'] in patient_ids and row['Celltype (major-lineage)'] == 'Epithelial': return LABELS['normal']
            else: return np.nan
        
        adata.obs["label"] = adata.obs.apply(_assignEpi, axis=1)
        adata_epi = adata[adata.obs["label"].notna()].copy()
        print(adata_epi.obs["label"].value_counts())

        # external ER+
        disease = 'Metastasis' if isErPosMetastasis else 'Primary'
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

def evaluate_lightgbm(ad_val, y_val, out_dir, interpret=True):
    """Load pre-trained and evaluate LightGBM model on external datasets."""
    
    save_dir = Path(r"path_to\Checkpoints\BreastERPositive\LGBM\Train")

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

    if interpret:
        shap_save_dir = out_dir / "SHAP"
        interpret_LGBM(lgbm_model, ad_val_processed,train_genes, shap_save_dir)
    # --- NEW: Generate Diagnostic Plot ---
    # This creates the 'nice plot' starting at zero
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

    # NSCLC_GSE148071


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

# function needed: create_breast_subtype_dataset()
dataset_ids = ['GSE161529_ERPlus','GSE161529_TNBC_NonBRCA1', 'GSE161529_TNBC_BRCA1', 'GSE161529_HER2Plus']
patient_ids = [er_plus,            tnbc_patients,             tnbc_brca1_patients, her2_positive_patients]
isErPosExternal = [False, False, False, False]

# function needed: create_breast_subtype_dataset()
dataset_ids = ['PRJNA1140267_ERPlus_Primary','PRJNA1140267_ERPlus_Met']
isErPosExternal = [True, True]
dataset_ids = ['Train_TNBC','Train_HER2']

isErPosMetastasis = [True, False]
patient_ids = [er_plus,              er_plus]

# function needed: prepare_breast_pancancer_atlas()
dataset_ids = ['PanCancerLuminal']
isErPosMetastasis = [None]
patient_ids = [er_plus] # need er_plus patients to obtain relevant epithelial cells
isErPosExternal = [None]

for dataset_id, patient_id, erPos, erPosMet in zip(dataset_ids, patient_ids, isErPosExternal, isErPosMetastasis):
    ad_test = prepare_breast_pancancer_atlas(patient_id)
    if sp.issparse(ad_test.X): # Ensure integer counts
        ad_test.X.data = np.rint(ad_test.X.data).astype(np.int32)
        ad_test.X = ad_test.X.astype(np.int32)
    else:
        ad_test.X = np.rint(ad_test.X).astype(np.int32)

    y_test = ad_test.obs["label"].astype(int).tolist()
    unique, counts = np.unique(y_test, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")

    out_dir = Path(rf"path_to\Checkpoints\BreastERPositive\LGBM\BRCA_GSE161529\{dataset_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🟡 Evaluate LightGBM Baseline...")
    lgbm_results = evaluate_lightgbm(
        ad_test, y_test,out_dir
    )

    with open(out_dir / "Val_results.json", "w") as f:
        json.dump(lgbm_results, f, indent=2)

