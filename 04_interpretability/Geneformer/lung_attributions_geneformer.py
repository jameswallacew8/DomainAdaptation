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

print("--- Starting Configuration ---")

# --- Fold Configuration ---
CELL_SUBSET = 500
RANDOM_SEED = 42

LABELS = {"normal": 0, "tumour": 1}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE_MODEL_CONFIG = 12
MODEL_NAME = "gf-6L-30M-i2048" # Ensure this matches your model details

CHECKPOINT_PATH = Path(r"path_to\Checkpoints\LUAD_Train\Geneformer\Train\pytorch_model_epoch4.bin")
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

dataset_list = ['NSCLC_GSE150660', 'NSCLC_GSE117570', 'NSCLC_GSE117570', 'NSCLC_GSE127465', 'NSCLC_GSE127465', 'NSCLC_GSE143423', 'NSCLC_GSE148071', 'SCLC_GSE150766']
isLusc =       [False,              False,              True,               False,          True,               False,             True,                 False]

for dataset, isLuscCheck in zip(dataset_list, isLusc):
    splitByPatients = dataset in ['NSCLC_GSE117570', 'NSCLC_GSE127465', 'NSCLC_GSE148071']
    adata = use_alveolar_epithelial(dataset, isLusc=isLuscCheck, splitByPatients=splitByPatients)
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
    base_path = Path(r"path_to\scRNArena\interpret\notebooks\geneformer\InDomainLung")
    datasetName = f'LUAD_Train_{dataset}_LUSC' if isLuscCheck else f'LUAD_Train_{dataset}_LUAD'

    output_dir = base_path / datasetName
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_mapped_attribution_results(
        output_dir=output_dir,
        gene_counts=gene_counts_tumour,         # Your dict
        aggregated_sums=aggregated_attributions_tumour, # Your dict
        mean_attributions=mean_attributions_tumour,   # Your dict
        mapper_path=MAP_GENES_DICT_PATH
    )

    print(f"✅  Saved: 3 files (Counts, Sums, Means) to {output_dir}")

