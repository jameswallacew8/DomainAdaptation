import json
from pathlib import Path
import pickle
from pathlib import Path
import torch
import numpy as np
import scanpy as sc
import scipy.sparse as sp # Used for type checking adata.X
from sklearn.model_selection import train_test_split
from helical.models.scgpt import scGPTConfig, scGPTFineTuningModel
from captum.attr import IntegratedGradients
from collections import defaultdict
from tqdm.auto import tqdm
import pandas as pd
import anndata as ad
from anndata import AnnData

# --- Configuration Section ---
print("--- Starting Configuration ---")

# --- Configuration ---
CELL_SUBSET = 500
RANDOM_SEED = 42

LABELS = {"normal": 0, "tumour": 1}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# IG internal_batch_size will be set to 1 later for stability.
BATCH_SIZE_MODEL_CONFIG = 12

# --- USER MUST VERIFY THESE PATHS ---
CHECKPOINT_PATH = Path(r"path_to\Checkpoints\NormalStromaIncEpVsTumourCore\scGPT\10k-HVG_Norm_5e5lr_5Fold_10Epoch\Train\pytorch_model_epoch1.bin")
GENE_TOKEN_DICT_PATH = Path(r"additional_data\scgpt_tokens_to_genes\vocab.json")
# --- END USER VERIFICATION FOR PATHS ---


# --- Load Gene Token Dictionary ---
print("\n--- Loading Gene Token Dictionary ---")

with open(GENE_TOKEN_DICT_PATH, 'r') as f:
        gene_token_dict = json.load(f)
id_to_gene = {v: k for k, v in gene_token_dict.items()} # ensembl_id: token_id
pad_token_id = gene_token_dict.get("<pad>", 0)

print(f"Gene token dictionary loaded. Pad token ID: {pad_token_id}")

# --- Step 1: Load and Prepare Data ---
print("\n--- Step 1: Loading and Preparing Data ---")

print(f"[prepare_data] Class distribution after downsampling:")


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
    
adata = chooseDataset("US")

# Ensure integer counts
if sp.issparse(adata.X):
    adata.X.data = np.rint(adata.X.data).astype(np.int32)
    adata.X = adata.X.astype(np.int32)
else:
    adata.X = np.rint(adata.X).astype(np.int32)

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
config = scGPTConfig(batch_size=BATCH_SIZE_MODEL_CONFIG, device=DEVICE)
model = scGPTFineTuningModel(config, fine_tuning_head="classification", output_size=2)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.model.to(DEVICE).eval() # Base HuggingFace model
model.to(DEVICE)
# .eval()      # Wrapper model
print("scGPTFineTuningModel loaded and in evaluation mode.")

#  fine_tuning = True normalises & log p transforms the data
dataset_test = model.process_data(adata_test, gene_names="gene_names")
dataset_test_label = y_test 

# --- Determine Target Sequence Length ---
try:
    TARGET_SEQ_LEN = model.model.config.max_position_embeddings
except AttributeError:
    print("Warning: Could not get TARGET_SEQ_LEN from model.model.config.max_position_embeddings.")
    TARGET_SEQ_LEN = 2048; print(f"Using fallback TARGET_SEQ_LEN: {TARGET_SEQ_LEN}. PLEASE VERIFY.")
print(f"Using TARGET_SEQ_LEN: {TARGET_SEQ_LEN}")

word_embedding_layer = model.model.encoder.embedding


# --- Updated model forward function for scGPT (CUSTOM TRANSFORMER) ---
print("\n--- Setting up Integrated Gradients (FIXED FOR CUSTOM TRANSFORMER) ---")

def model_forward_for_captum(token_embeddings, attention_mask, target_class_idx):
    """
    Forward pass using scGPT's TransformerModel architecture
    token_embeddings: [batch, seq_len, embed_dim] 
    attention_mask: [batch, seq_len]
    """
    
    # TransformerEncoder in scGPT expects:
    # - input: [seq_len, batch, embed_dim]
    # - src_key_padding_mask: [seq_len, batch] (DIFFERENT from standard PyTorch!)
    
    # Transpose embeddings to [seq_len, batch, embed_dim]
    token_embeddings_transposed = token_embeddings.transpose(0, 1)
    
    # Create padding mask and transpose to [seq_len, batch]
    if attention_mask is not None:
        # True where we should mask (where attention_mask == 0)
        src_key_padding_mask = (attention_mask == 0).transpose(0, 1)  # [seq_len, batch]
    else:
        src_key_padding_mask = None
    
    # Pass through transformer encoder
    transformer_output = model.model.transformer_encoder(
        token_embeddings_transposed,
        src_key_padding_mask=src_key_padding_mask
    )
    
    # Transpose back to [batch, seq_len, embed_dim]
    transformer_output = transformer_output.transpose(0, 1)
    
    # Apply the fine-tuning head
    logits_per_token = model.fine_tuning_head(transformer_output)
    
    # Get dimensions
    num_output_classes = model.fine_tuning_head.linear.out_features
    
    # Aggregate logits across sequence
    if num_output_classes > 1:
        # Multi-class: average over sequence length
        cell_logits = torch.mean(logits_per_token, dim=1)  # [batch, num_classes]
    elif num_output_classes == 1:
        # Single output: squeeze and average
        cell_logits = logits_per_token.squeeze(-1)  # [batch, seq_len]
        if cell_logits.ndim == 2:
            cell_logits = torch.mean(cell_logits, dim=1)  # [batch]
    else:
        raise ValueError(f"Num output classes not positive: {num_output_classes}")
    
    # Return the logit for the target class
    if num_output_classes > 1:
        return cell_logits[:, target_class_idx]
    else:
        return cell_logits

print("model_forward_for_captum defined.")

# Test the forward function with a dummy input to ensure shapes are correct
print("\n--- Testing forward function ---")
dummy_input_ids = torch.randint(0, word_embedding_layer.num_embeddings, (2, TARGET_SEQ_LEN)).to(DEVICE)
dummy_embeddings = word_embedding_layer(dummy_input_ids)
dummy_attention_mask = torch.ones(2, TARGET_SEQ_LEN, dtype=torch.long, device=DEVICE)

try:
    test_output = model_forward_for_captum(dummy_embeddings, dummy_attention_mask, 0)
    print(f"✓ Forward function test passed! Output shape: {test_output.shape}")
except Exception as e:
    print(f"✗ Forward function test failed: {e}")
    # Let's debug the transformer encoder to understand its expectations
    print("\n--- Debugging transformer encoder ---")
    print(f"Dummy embeddings shape: {dummy_embeddings.shape}")
    print(f"After transpose: {dummy_embeddings.transpose(0, 1).shape}")
    print(f"Attention mask shape: {dummy_attention_mask.shape}")
    print(f"After transpose: {dummy_attention_mask.transpose(0, 1).shape}")
    raise

# --- Re-initialize Integrated Gradients ---
ig_tumour = IntegratedGradients(lambda emb, attn_m: model_forward_for_captum(emb, attn_m, LABELS["tumour"]))

print("Integrated Gradients instances re-initialized.")


# Re-initialize aggregation dictionaries
aggregated_attributions_tumour = defaultdict(float)
gene_counts_tumour = defaultdict(int)

print("\n--- Setup complete! Ready to process cells. ---")


def pad_sequence_helper(ids_list, max_len, pad_value): # Renamed from pad_sequence to avoid potential conflicts
    if len(ids_list) > max_len: return ids_list[:max_len]
    return ids_list + [pad_value] * (max_len - len(ids_list))

# baseline_embeddings = medoid()
# --- Recreate baseline embeddings ---
baseline_input_ids = torch.full((1, TARGET_SEQ_LEN), fill_value=pad_token_id, dtype=torch.long, device=DEVICE)
baseline_embeddings = word_embedding_layer(baseline_input_ids).to(DEVICE)
print(f"Baseline embeddings shape: {baseline_embeddings.shape}")

print(f"Baseline embeddings shape: {baseline_embeddings.shape}")

print("\n--- Step 4: Starting Attribution Calculation ---")


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


all_tumour_cell_indices = [i for i, label_val in enumerate(dataset_test_label) if label_val == LABELS["tumour"]]
print(f"Total available tumour cells in test dataset: {len(all_tumour_cell_indices)}")


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
    
    # Inner Loop: Process 250 cells
for i_loop_idx, cell_dataset_idx in enumerate(tqdm(indices_pool, desc=f"IG Progress")):
    sample = dataset_test[cell_dataset_idx]
    original_input_ids_list = sample["genes"].tolist()
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
base_path = Path(r"path_to\interpret\notebooks\scgpt\InDomainCCRCC")
output_dir = base_path / "ccRCCTrain_USTumour"
output_dir.mkdir(parents=True, exist_ok=True)

# --- FILE 1: Gene Counts (gene_counts_dict) ---
# Create DataFrame directly from dict items
df_counts = pd.DataFrame(list(gene_counts_tumour.items()), columns=['GeneSymbol', 'Count'])
# Optional: Sort by count descending
df_counts = df_counts.sort_values(by='Count', ascending=False)
df_counts.to_csv(output_dir / "gene_counts.csv", index=False)

# --- FILE 2: Aggregated Raw Sums (aggregated_attrs) ---
df_sums = pd.DataFrame(list(aggregated_attributions_tumour.items()), columns=['GeneSymbol', 'TotalAttributionSum'])
df_sums = df_sums.sort_values(by='TotalAttributionSum', ascending=False)
df_sums.to_csv(output_dir / "aggregated_sum_attributions.csv", index=False)

# --- FILE 3: Final Mean Attributions ---
df_means = pd.DataFrame(list(mean_attributions_tumour.items()), columns=['GeneSymbol', 'MeanAttribution'])
df_means = df_means.sort_values(by='MeanAttribution', ascending=False)
df_means.to_csv(output_dir / "final_mean_attributions.csv", index=False)

