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
from functools import lru_cache
import anndata as ad
from anndata import AnnData

# --- Configuration Section ---
print("--- Starting Configuration ---")

LABELS = {"normal": 0, "tumour": 1}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# BATCH_SIZE_MODEL_CONFIG is for GeneformerConfig initialization.
# IG internal_batch_size will be set to 1 later for stability.
BATCH_SIZE_MODEL_CONFIG = 12
MODEL_NAME = "gf-6L-30M-i2048" # Ensure this matches your model details

# --- END USER VERIFICATION FOR PATHS ---

CHECKPOINT_PATH = Path(r"path_to\Checkpoints\NormalStromaIncEpVsTumourCore\Geneformer\Train\pytorch_model_epoch4.bin")
GENE_TOKEN_DICT_PATH = Path(r"additional_data\geneformer_gene_dictionaries_30m\token_dictionary_gc30M.pkl")
MAP_GENES_DICT_PATH = Path(r"additional_data\geneformer_gene_dictionaries_30m\gene_name_id_dict_gc30M.pkl")

CELL_SUBSET = 500
RANDOM_SEED = 42

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
    

# --- Load Gene Token Dictionary ---
print("\n--- Loading Gene Token Dictionary ---")
with open(GENE_TOKEN_DICT_PATH, "rb") as f:
    gene_token_dict = pickle.load(f)
id_to_gene = {v: k for k, v in gene_token_dict.items()} # ensembl_id: token_id
pad_token_id = gene_token_dict.get("<pad>", 0)
print(f"Gene token dictionary loaded. Pad token ID: {pad_token_id}")

# --- Step 1: Load and Prepare Data ---
print("\n--- Step 1: Loading and Preparing Data ---")

adata = chooseDataset("US")

print(adata.obs['label'].value_counts())

if sp.issparse(adata.X): # Ensure integer counts
    adata.X.data = np.rint(adata.X.data).astype(np.int32)
    adata.X = adata.X.astype(np.int32)
else:
    adata.X = np.rint(adata.X).astype(np.int32)


labels_for_split = adata.obs["label"].astype(int).tolist()
adata_test = adata.copy()
y_test = adata_test.obs["label"].astype(int).tolist()
print(f"Test set created with {adata_test.n_obs} cells.")

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

# --- Integrated Gradients Setup ---
print("\n--- Setting up Integrated Gradients ---")
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

total_cells_needed = CELL_SUBSET
np.random.seed(RANDOM_SEED)

if len(all_tumour_cell_indices) >= total_cells_needed:
    indices_pool = np.random.choice(all_tumour_cell_indices, total_cells_needed, replace=False)
else:
    print(f"⚠️ Warning: Not enough cells. Using all {len(all_tumour_cell_indices)} available cells and reshuffling/recycling.")
    # If not enough, we take what we have, shuffle, and might have smaller or overlapping folds depending on logic
    # Here we just take all, shuffle them, and split as much as we can
    indices_pool = np.array(all_tumour_cell_indices)
    np.random.shuffle(indices_pool)

print(f"Processing pool size: {len(indices_pool)} cells")

print(f"All inputs padded/truncated to TARGET_SEQ_LEN: {TARGET_SEQ_LEN}")

# Process TUMOUR cells
print("\nCalculating attributions for TUMOUR cells...")

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
        print(f"\nDEBUG INFO FOR FIRST PROCESSED TUMOUR CELL (loop idx {i_loop_idx}, dataset idx {cell_dataset_idx}):")
        print(f"  Original length: {len(original_input_ids_list)}, Padded input_ids shape: {input_ids.shape}")
        print(f"  Attention_mask shape: {attention_mask.shape}, Sum: {attention_mask.sum().item()}")
        print(f"  Input_embeddings shape: {input_embeddings.shape}, Baseline_embeddings shape: {baseline_embeddings.shape}")
        printed_debug_first_cell_tumour = True
    try:
        attrs = ig_tumour.attribute(inputs=input_embeddings, baselines=baseline_embeddings,
                                        additional_forward_args=(attention_mask,), n_steps=50, internal_batch_size=1)
    except RuntimeError as e:
        print(f"\nRuntimeError for TUMOUR cell (loop idx {i_loop_idx}, dataset idx {cell_dataset_idx}): {e}"); raise
    token_attributions = attrs.sum(dim=2).squeeze(0)
    
    active_tokens_mask_bool = attention_mask.squeeze(0).bool()
    actual_ids = input_ids.squeeze(0)[active_tokens_mask_bool].tolist()
    actual_attrs = token_attributions[active_tokens_mask_bool].tolist()
    
    for token_id, attr_score in zip(actual_ids, actual_attrs):
        gene_ensg = id_to_gene.get(token_id)
        if gene_ensg and gene_ensg not in ["<pad>", "<cls>", "<eos>", "<unk>", "<mask>"] and token_id != pad_token_id:
            aggregated_attributions_tumour[gene_ensg] += attr_score
            gene_counts_tumour[gene_ensg] += 1

# --- SAVE OUTPUT ---
# 1. Calculate Means using your helper function
mean_attributions_tumour = get_mean_attributions(aggregated_attributions_tumour, gene_counts_tumour)

# 2. Setup Output Directory
base_path = Path(r"path_to\geneformer\InDomainCCRCC")
output_dir = base_path / "ccRCCTrain_USTumour"
output_dir.mkdir(parents=True, exist_ok=True)

save_mapped_attribution_results(
    output_dir=output_dir,
    gene_counts=gene_counts_tumour,         # Your dict
    aggregated_sums=aggregated_attributions_tumour, # Your dict
    mean_attributions=mean_attributions_tumour,   # Your dict
    mapper_path=MAP_GENES_DICT_PATH
)

