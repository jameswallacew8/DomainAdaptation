# TODO - messy script, datasets requite different pre-processing functions

import pickle
from pathlib import Path
import torch
import numpy as np
import scanpy as sc
import scipy.sparse as sp # Used for type checking adata.X
from sklearn.model_selection import train_test_split
from helical.models.geneformer import GeneformerConfig, GeneformerFineTuningModel
from captum.attr import IntegratedGradients
from collections import defaultdict
from tqdm.auto import tqdm
import pandas as pd
import anndata as ad

import json
from pathlib import Path
from functools import lru_cache

# --- Configuration Section ---
print("--- Starting Configuration ---")

# --- Subset Configuration ---
CELL_SUBSET = 500
RANDOM_SEED = 42

LABELS = {"normal": 0, "tumour": 1}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE_MODEL_CONFIG = 12
MODEL_NAME = "gf-6L-30M-i2048" # Ensure this matches your model details

CHECKPOINT_PATH = Path(r"path_to\Checkpoints\BreastERPositive\Geneformer\Train\pytorch_model_epoch1.bin")
GENE_TOKEN_DICT_PATH = Path(r"additional_data\geneformer_gene_dictionaries_30m\token_dictionary_gc30M.pkl")
MAP_GENES_DICT_PATH = Path(r"additional_data\geneformer_gene_dictionaries_30m\gene_name_id_dict_gc30M.pkl")


# --- 1. Robust Mapper Loader (Cached) ---
@lru_cache(maxsize=1)
def get_ensembl_to_name_mapper(dict_path: str):
    """
    Loads the pickle dictionary and returns an inverted {EnsemblID: GeneName} map.
    Cached so it only loads from disk once.
    """
    path = Path(dict_path)
    if not path.exists():
        raise FileNotFoundError(f"❌ Dictionary not found at: {path}")

    print(f"📖 Loading gene dictionary from: {path.name}...")
    with open(path, "rb") as f:
        # Assumes dict is {GeneName: EnsemblID} or {Key: Value}
        original_dict = pickle.load(f)
    
    # Invert to {Value: Key} -> {EnsemblID: GeneName}
    mapper = {v: k for k, v in original_dict.items()}
    print(f"✅ Dictionary loaded. Mapped {len(mapper)} IDs.")
    return mapper

# --- 2. Main Processing Function ---
def save_mapped_attribution_results(
    output_dir: Path,
    gene_counts: dict,
    aggregated_sums: dict,
    mean_attributions: dict,
    mapper_path: str
):
    """
    Saves gene counts, sums, and means to CSVs with mapped Gene Names.
    """
    # Load mapper (will use cache if already loaded)
    id_to_name = get_ensembl_to_name_mapper(str(mapper_path))
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Helper function to process a single dictionary
    def _process_and_save(data_dict, value_col_name, filename):
        # 1. Create DataFrame
        df = pd.DataFrame(list(data_dict.items()), columns=['EnsemblID', value_col_name])
        
        # 2. Map Gene Names
        df['GeneSymbol'] = df['EnsemblID'].map(id_to_name)
        
        # 3. Robustness: Fill missing names with the original EnsemblID
        missing_count = df['GeneSymbol'].isna().sum()
        if missing_count > 0:
            print(f"⚠️ Warning: {missing_count} IDs in {filename} could not be mapped to names.")
            df['GeneSymbol'] = df['GeneSymbol'].fillna(df['EnsemblID'])
            
        # 4. Reorder columns: GeneSymbol first, then ID, then Value
        df = df[['GeneSymbol', 'EnsemblID', value_col_name]]
        
        # 5. Sort descending
        df = df.sort_values(by=value_col_name, ascending=False)
        
        # 6. Save
        save_file = out_path / filename
        df.to_csv(save_file, index=False)
        return save_file

    # --- Process All 3 Files ---
    _process_and_save(gene_counts, 'Count', "gene_counts.csv")
    _process_and_save(aggregated_sums, 'TotalAttributionSum', "aggregated_sum_attributions.csv")
    _process_and_save(mean_attributions, 'MeanAttribution', "final_mean_attributions.csv")

    print(f"✅ Saved 3 mapped files to: {out_path}")


def create_breast_subtype_dataset(patient_ids, isErPosExternal):
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

def prepare_breast_train_data(isTNBC):
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

def prepare_breast_pancancer_atlas(patient_ids):
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

def model_forward_for_captum(token_embeddings, attention_mask, target_class_idx):
    input_ids_for_mask_creation = torch.zeros(token_embeddings.shape[0], token_embeddings.shape[1], dtype=torch.long, device=DEVICE)
    extended_attention_mask = model.model.bert.get_extended_attention_mask(
        attention_mask, input_ids_for_mask_creation.shape, DEVICE)
    encoder_outputs = model.model.bert.encoder(
        hidden_states=token_embeddings, attention_mask=extended_attention_mask,
        output_attentions=False, output_hidden_states=False, return_dict=True)
    sequence_output = encoder_outputs.last_hidden_state
    transformed_output = model.model.cls.predictions.transform(sequence_output)
    logits_per_token = model.fine_tuning_head(transformed_output)
    num_output_classes = model.fine_tuning_head.linear.out_features
    if num_output_classes > 1: cell_logits = torch.mean(logits_per_token, dim=1)
    elif num_output_classes == 1:
        cell_logits = logits_per_token.squeeze(-1)
        if cell_logits.ndim == 2: cell_logits = torch.mean(cell_logits, dim=1)
    else: raise ValueError(f"Num output classes not positive: {num_output_classes}")
    return cell_logits[:, target_class_idx]

print("model_forward_for_captum defined.")
    
 # --- Load Gene Token Dictionary ---
print("\n--- Loading Gene Token Dictionary ---")
with open(GENE_TOKEN_DICT_PATH, "rb") as f:
    gene_token_dict = pickle.load(f)
id_to_gene = {v: k for k, v in gene_token_dict.items()} # ensembl_id: token_id
pad_token_id = gene_token_dict.get("<pad>", 0)
print(f"Gene token dictionary loaded. Pad token ID: {pad_token_id}")

# --- Step 1: Load and Prepare Data ---
print("\n--- Step 1: Loading and Preparing Data ---")


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

# --- Step 1: Load and Prepare Data ---
dataset_ids = ['GSE161529_ERPlus','GSE161529_TNBC_NonBRCA1', 'GSE161529_TNBC_BRCA1', 'GSE161529_HER2Plus']
patient_ids = [er_plus,            tnbc_patients,             tnbc_brca1_patients, her2_positive_patients]
# dataset_ids = ['PRJNA1140267_ERPlus_Primary']
# isErPosExternal = [True,]
# patient_ids = [er_plus]

# dataset_ids = ['Train_TNBC','Train_HER2']
# isErPosExternal = [True, False]

dataset_ids = ['PanCancerLuminal']
patient_ids = [er_plus] # need er_plus patients to obtain relevant epithelial cells
isErPosExternal = [None]


dataset_ids = ['Train_ER']
patient_ids = [er_plus] # need er_plus patients to obtain relevant epithelial cells
isErPosExternal = [None]

for dataset_id, patient_id, erPos in zip(dataset_ids, patient_ids, isErPosExternal):
    # adata = create_breast_subtype_dataset(patient_id, erPos, erPosMet)
    adata = prepare_breast_train_data(erPos)

   # Ensure integer counts
    if sp.issparse(adata.X):
        adata.X.data = np.rint(adata.X.data).astype(np.int32)
        adata.X = adata.X.astype(np.int32)
    else:
        adata.X = np.rint(adata.X).astype(np.int32)

    y_val = adata.obs["label"].astype(int).tolist()

    unique, counts = np.unique(y_val, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")

    print(f"[prepare_data] Before filtering: {adata.n_vars} genes")
    sc.pp.filter_genes(adata, min_counts=1)  # Keep genes expressed at least once
    print(f"[prepare_data] After filtering: {adata.n_vars} genes")

    # --- Step 7: Create Test Set ---
    labels_for_split = adata.obs["label"].astype(int).tolist()
    adata_test = adata.copy()
    y_test = adata_test.obs["label"].astype(int).tolist()
    print(f"Test set created with {adata_test.n_obs} cells, {adata_test.n_vars} genes.")

    # --- Model Loading and Dataset Creation ---
    print("\n--- Loading Model and Creating Test Dataset ---")
    cfg = GeneformerConfig(model_name=MODEL_NAME, batch_size=BATCH_SIZE_MODEL_CONFIG, device=DEVICE)
    model = GeneformerFineTuningModel(cfg, fine_tuning_head="classification", output_size=len(LABELS))
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.model.to(DEVICE).eval() # Base HuggingFace model
    model.to(DEVICE)      # Wrapper model
    print("GeneformerFineTuningModel loaded and in evaluation mode.")

    adata_test.X = sp.csr_matrix(adata_test.X)
    dataset_test = model.process_data(adata_test, gene_names="gene_names")
    dataset_test = dataset_test.add_column("label", y_test)
    print(f"HuggingFace test dataset created with {len(dataset_test)} samples.")

    # --- Determine Target Sequence Length ---
    try:
        TARGET_SEQ_LEN = model.model.config.max_position_embeddings
    except AttributeError:
        print("Warning: Could not get TARGET_SEQ_LEN from model.model.config.max_position_embeddings.")
        if "i2048" in MODEL_NAME: TARGET_SEQ_LEN = 2048
        elif "i512" in MODEL_NAME: TARGET_SEQ_LEN = 512
        else: TARGET_SEQ_LEN = 2048; print(f"Using fallback TARGET_SEQ_LEN: {TARGET_SEQ_LEN}. PLEASE VERIFY.")
    print(f"Using TARGET_SEQ_LEN: {TARGET_SEQ_LEN}")

    word_embedding_layer = model.model.bert.embeddings.word_embeddings


    print("\n--- Setting up Integrated Gradients (FIXED FOR CUSTOM TRANSFORMER) ---")
   
    # --- Re-initialize Integrated Gradients ---
    ig_tumour = IntegratedGradients(lambda emb, attn_m: model_forward_for_captum(emb, attn_m, LABELS["tumour"]))

    print("Integrated Gradients instances initialized.")

    baseline_input_ids = torch.full((1, TARGET_SEQ_LEN), fill_value=pad_token_id, dtype=torch.long, device=DEVICE)
    baseline_embeddings = word_embedding_layer(baseline_input_ids).to(DEVICE)
    print(f"Baseline embeddings shape: {baseline_embeddings.shape}")

    aggregated_attributions_tumour = defaultdict(float)
    gene_counts_tumour = defaultdict(int)
    
    # --- Step 4: Attribution Calculation ---
    print("\n--- Step 4: Starting Attribution Calculation ---")
    def pad_sequence_helper(ids_list, max_len, pad_value): # Renamed from pad_sequence to avoid potential conflicts
        if len(ids_list) > max_len: return ids_list[:max_len]
        return ids_list + [pad_value] * (max_len - len(ids_list))


    def get_mean_attributions(aggregated_attrs, gene_counts_dict):
        mean_attrs = defaultdict(float)
        for gene, total_attr in aggregated_attrs.items():
            if gene_counts_dict.get(gene, 0) > 0: 
                mean_attrs[gene] = total_attr / gene_counts_dict[gene]
            else: 
                mean_attrs[gene] = 0.0
        return mean_attrs

    def prepare_df_data_with_symbols(mean_attributions_dict, sort_descending=True):
        df_data = []
        sorted_genes = sorted(mean_attributions_dict.items(), key=lambda item: item[1], reverse=sort_descending)
        for gene_id, score in sorted_genes:
            df_data.append({'GeneSymbol': gene_id, 'MeanAttribution': score})
        return df_data

    all_tumour_cell_indices = [i for i, label_val in enumerate(y_test) if label_val == LABELS["tumour"]]

    print(f"Total available tumour cells in test dataset: {len(all_tumour_cell_indices)}")

    # 2. Select the pool for the folds (Total 2500 cells)
    total_cells_needed = CELL_SUBSET

    # Set Seed for Reproducibility
    np.random.seed(RANDOM_SEED)

    # Randomly select 2500 cells without replacement (if possible)
    if len(all_tumour_cell_indices) >= total_cells_needed:
        indices_pool = np.random.choice(all_tumour_cell_indices, total_cells_needed, replace=False)
    else:
        print(f"⚠️ Warning: Not enough cells. Using all {len(all_tumour_cell_indices)} available cells and reshuffling/recycling.")
        # If not enough, we take what we have, shuffle, and might have smaller or overlapping folds depending on logic
        # Here we just take all, shuffle them, and split as much as we can
        indices_pool = np.array(all_tumour_cell_indices)
        np.random.shuffle(indices_pool)

    print(f"Processing pool size: {len(indices_pool)} cells")


    aggregated_attributions_tumour = defaultdict(float)
    gene_counts_tumour = defaultdict(int)   

    printed_debug_first_cell_tumour = False

    for i_loop_idx, cell_dataset_idx in enumerate(tqdm(indices_pool, desc=f"IG Progress")):
        sample = dataset_test[int(cell_dataset_idx)]
        original_input_ids_list = sample["input_ids"]
        padded_input_ids_list = pad_sequence_helper(original_input_ids_list, TARGET_SEQ_LEN, pad_token_id)
        
        input_ids = torch.tensor(padded_input_ids_list, device=DEVICE).unsqueeze(0)
        attention_mask = (input_ids != pad_token_id).long()
        input_embeddings = word_embedding_layer(input_ids).to(DEVICE)

        if not printed_debug_first_cell_tumour:
            # print(f" [Debug] First cell input shape: {input_embeddings.shape}")
            printed_debug_first_cell_tumour = True

        try:
            # --- RUN IG ---
            # Using your existing ig_tumour object and baseline_embeddings
            attrs = ig_tumour.attribute(inputs=input_embeddings, baselines=baseline_embeddings,
                                            additional_forward_args=(attention_mask,), n_steps=50, internal_batch_size=1)
            
            # --- AGGREGATE ---
            token_attributions = attrs.sum(dim=2).squeeze(0)
            active_tokens_mask_bool = attention_mask.squeeze(0).bool()
            actual_ids = input_ids.squeeze(0)[active_tokens_mask_bool].tolist()
            actual_attrs = token_attributions[active_tokens_mask_bool].tolist()
            
            for token_id, attr_score in zip(actual_ids, actual_attrs):
                gene_ensg = id_to_gene.get(token_id)
                if gene_ensg and gene_ensg not in ["<pad>", "<cls>", "<eos>", "<unk>", "<mask>"] and token_id != pad_token_id:
                    aggregated_attributions_tumour[gene_ensg] += attr_score
                    gene_counts_tumour[gene_ensg] += 1
                    
        except RuntimeError as e:
            print(f"Error in cell {cell_dataset_idx}: {e}")
            continue

    # --- SAVE OUTPUT FOR THIS FOLD ---
    # 1. Calculate Means using your helper function
    mean_attributions_tumour = get_mean_attributions(aggregated_attributions_tumour, gene_counts_tumour)

    # 2. Setup Output Directory
    base_path = Path(r"path_to\scRNArena\interpret\notebooks\geneformer\InDomainBreast")

    output_dir = base_path / dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_mapped_attribution_results(
        output_dir=output_dir,
        gene_counts=gene_counts_tumour,         # Your dict
        aggregated_sums=aggregated_attributions_tumour, # Your dict
        mean_attributions=mean_attributions_tumour,   # Your dict
        mapper_path=MAP_GENES_DICT_PATH
    )

    print(f"✅  Saved: 3 files (Counts, Sums, Means) to {output_dir}")

