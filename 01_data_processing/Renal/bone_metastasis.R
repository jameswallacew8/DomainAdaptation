# Script taken from: https://github.com/Sr933/rcc/tree/main

# Specify the library path
.libPaths(c("~/R/libs", .libPaths()))
lib_path <- "~/R/libs"

if (!requireNamespace("cpp11", quietly = TRUE)) install.packages("cpp11", lib = lib_path)

# Install the 'readr' package to the specified library path
if (!requireNamespace("readr", quietly = TRUE)) install.packages("readr", lib = lib_path)


if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr", lib = lib_path)
# Load Seurat library
library(Seurat)

# Load libraries
library(readr)
library(Seurat)
library(dplyr)
Sys.setenv("VROOM_CONNECTION_SIZE" = 500000)
# Replace with your file path
dir_path <- "./data/GSE202813_RAW"

# List all files in the directory with the .csv.gz extension and excluding "annotation" in the filename
files <- list.files(path = dir_path, pattern = "\\.csv\\.gz$", full.names = TRUE)
files <- files[!grepl("annotation", files)]  # Exclude files with 'annotation' in the name

# Loop over the filtered files
for (file in files) {
  # Read count matrix for each file
  count_matrix <- read_csv(file)
  print(paste("Processing file:", file))

  # Convert to data frame
  count_matrix <- as.data.frame(count_matrix)

  # Rename unnamed first column
  colnames(count_matrix)[1] <- "Gene_ID"

  # Remove duplicates based on 'Gene_ID'
  count_matrix <- count_matrix[!duplicated(count_matrix$Gene_ID), ]

  # Set rownames to 'Gene_ID'
  rownames(count_matrix) <- count_matrix$Gene_ID

  # Remove the 'Gene_ID' column
  count_matrix <- count_matrix[, -which(colnames(count_matrix) == "Gene_ID")]

  # Print the first few rows
  print(head(count_matrix))

  # Create a Seurat object
  seurat_obj <- CreateSeuratObject(counts = count_matrix)
  seurat_obj <- NormalizeData(seurat_obj)

  # Define marker genes for each cell type
  markers <- list(
    Erythroid = c("GYPA", "GATA1", "TFRC", "HBG1", "HBB", "HBA1", "HBA2", "HBD", "CA1", "SMIM1"),
    Endothelial = c("RAMP2", "TM4SF1", "RNASE1", "EGFL7", "RAMP3", "PLVAP", "AQP1", "ECSCR", "FKBP1A", "EMP1", "DARC", "VWF", "EMCN"),
    Pericytes = c("RGS5", "ACTA2", "MYH11", "MT1M", "FRZB", "MT1A", "PPP1R14A", "PHLDA1", "NDUFA4L2"),
    Fibroblasts = c("DCN", "LUM", "PTN", "IGF1", "APOD", "COL1A2", "FBLN1", "MEG3", "CXCL12"),
    PDC = c("IRF7", "IRF4", "LILRA4", "PPP1R14B", "SOX4", "TSPAN13", "KIAA0226", "PTCRA", "RAB11FIP1", "IL3RA"),
    B_cells = c("MS4A1", "CD79B", "BTG1", "VPREB3", "BANK1", "CD79A", "IGLL5"),
    Macrophage = c("C1QA", "C1QC", "C1QB", "CSF1R", "SDF2L1", "NEAT1", "THBS1", "VCAN"),
    Monocytes = c("S100A9", "FCN1", "S100A8", "EREG", "CSTA", "C15orf48", "CSTA"),
    mDC = c("CD1C", "PKIB", "INSIG1", "CLEC10A", "PPA1"),
    NK = c("NKG7", "GNLY", "KLRD1", "KLRB1", "PRF1", "FGFBP2"),
    CTL = c("CD8A", "CD8B", "GZMH", "GZMA", "PTPRC"),
    Naive_Th = c("CCR7", "SELL", "CD4", "CD3D", "CD3E"),
    Thelper = c("IL2", "TNF", "IL17A", "IL17F", "RORC", "CCR6", "RORA"),
    Treg = c("TIGIT", "CTLA4", "SOD1", "TNFRSF4", "TNFRSF18", "FOXP3"),
    Progenitors = c("SPINK2", "IGLL1", "SSBP2", "CD34", "STMN1", "SOX4", "PRSS57"),
    NKT = c("KLRC1", "CD44", "COTL1", "TBX21", "EOMES"),
    Osteoblasts = c("BGLAP", "SPP1", "MMP9", "IBSP", "COL", "CEPP1"),
    Osteoclasts = c("SPP1", "IBSP", "TCIRG1", "ACP5"),
    MSCs = c("CXCL12", "PDGFRB", "LEPR", "NGFR", "VCAM1"),
    Pericytes_Monocytes_Pro = c("ACTA2", "RGSS", "NDUFA4L2", "STMN1", "S100A8"),
    Other = c("KIAA0101", "CALD1", "COL3A1", "COL1A2")
  )

  # Check if Tumor markers should be included based on the file name
  if (grepl("Tumor", file)) {
    markers$Tumor <- c("CA9", "KRT8", "KRT18", "PDK4", "VEGFA", "NNMT")
  }

  # Loop through the cell types and add module scores
for (cell_type in names(markers)) {
  # Check which genes are present in the Seurat object
  present_genes <- markers[[cell_type]][markers[[cell_type]] %in% rownames(seurat_obj)]
  
  # Check if there are enough genes for module scoring
  if (length(present_genes) < 3) {  # Adjust this threshold as needed
    warning(paste("Not enough features for", cell_type, ":", length(present_genes), "found in the Seurat object"))
  } else {
    # Add module score for the cell type if enough genes are present
    seurat_obj <- AddModuleScore(seurat_obj, features = list(present_genes), name = cell_type)
  }
}

# Extract scores
scores <- seurat_obj@meta.data[, grep("^[a-zA-Z]+1$", colnames(seurat_obj@meta.data))]

# Assign the cell type with the highest score
seurat_obj$cell_type <- apply(scores, 1, function(row) names(row)[which.max(row)])

# Export the annotated cell metadata for each file
annotated_metadata <- seurat_obj@meta.data
write.csv(annotated_metadata, paste0(file, "_cell_annotation.csv"), row.names = TRUE)}