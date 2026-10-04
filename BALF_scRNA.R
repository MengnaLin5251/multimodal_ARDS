############################################################
## 1. Packages
############################################################
library(Seurat)
library(dplyr)
library(Matrix)
library(DoubletFinder)
library(SingleCellExperiment)
library(decontX)
library(harmony)

library(edgeR)
library(clusterProfiler)
library(org.Hs.eg.db)
library(ReactomePA)

############################################################
## 2. Single-sample QC
############################################################

# seu: one Seurat object
DefaultAssay(seu) <- "RNA"

seu[["percent.mt"]] <- PercentageFeatureSet(
  seu,
  pattern = "^MT-"
)

rb_genes <- c(
  "HBA1","HBA2","HBB","HBD","HBE1",
  "HBG1","HBG2","HBM","HBQ1","HBZ"
)

rb_genes <- intersect(
  rb_genes,
  rownames(seu)
)

seu[["percent.rb"]] <- PercentageFeatureSet(
  seu,
  features = rb_genes
)

# QC thresholds used in the analysis
seu <- subset(
  seu,
  subset =
    nFeature_RNA >= 500 &
    nFeature_RNA <= 9000 &
    nCount_RNA >= 1000 &
    nCount_RNA <= 50000 &
    percent.mt <= 15 &
    percent.rb <= 5
)

############################################################
## 3. Doublet removal
############################################################

seu <- NormalizeData(seu)
seu <- FindVariableFeatures(
  seu,
  nfeatures = 3000
)
seu <- ScaleData(seu)
seu <- RunPCA(
  seu,
  npcs = 30
)

seu <- FindNeighbors(
  seu,
  dims = 1:20
)

seu <- FindClusters(
  seu,
  resolution = 0.5
)

# identify optimal pK
sweep.res <- paramSweep(
  seu,
  PCs = 1:20,
  sct = FALSE
)

sweep.stats <- summarizeSweep(
  sweep.res,
  GT = FALSE
)

bcmvn <- find.pK(sweep.stats)

pK <- as.numeric(
  as.character(
    bcmvn$pK[
      which.max(bcmvn$BCmetric)
    ]
  )
)

# expected doublet number
nExp <- round(
  0.075 * ncol(seu)
)

homotypic.prop <- modelHomotypic(
  seu$seurat_clusters
)

nExp.adj <- round(
  nExp * (1 - homotypic.prop)
)

seu <- doubletFinder(
  seu,
  PCs = 1:20,
  pN = 0.25,
  pK = pK,
  nExp = nExp.adj,
  reuse.pANN = FALSE,
  sct = FALSE
)

DF_col <- grep(
  "^DF.classifications",
  colnames(seu@meta.data),
  value = TRUE
)

seu <- subset(
  seu,
  subset = get(DF_col) == "Singlet"
)
############################################################
## 4. Ambient RNA contamination: decontX
############################################################

DefaultAssay(sc.all) <- "RNA"

counts_mat <- LayerData(
  sc.all,
  assay = "RNA",
  layer = "counts"
)

sce <- SingleCellExperiment(
  assays = list(
    counts = counts_mat
  )
)

sce <- decontX(
  sce,
  verbose = FALSE
)

sc.all$Contamination <-
  colData(sce)$decontX_contamination

# threshold used in your workflow
sc.all <- subset(
  sc.all,
  subset = Contamination <= 0.30
)
############################################################
## 5. Normalization and Harmony integration
############################################################

DefaultAssay(sc.all) <- "RNA"

sc.all <- NormalizeData(
  sc.all,
  normalization.method = "LogNormalize",
  scale.factor = 10000
)

sc.all <- FindVariableFeatures(
  sc.all,
  selection.method = "vst",
  nfeatures = 3000
)

hvg <- VariableFeatures(sc.all)

sc.all <- ScaleData(
  sc.all,
  features = hvg
)

sc.all <- RunPCA(
  sc.all,
  npcs = 30
)

# sample-level integration
sc.all <- RunHarmony(
  object = sc.all,
  group.by.vars = "orig.ident",
  reduction.use = "pca",
  dims.use = 1:30
)

sc.all <- FindNeighbors(
  sc.all,
  reduction = "harmony",
  dims = 1:30
)

sc.all <- FindClusters(
  sc.all,
  resolution = 0.5
)

sc.all <- RunUMAP(
  sc.all,
  reduction = "harmony",
  dims = 1:30
)

DimPlot(
  sc.all,
  reduction = "umap",
  group.by = "celltype"
)
############################################################
## 6. Observed-to-expected ratio (Ro/e)
############################################################

meta <- scRNA_final_clean@meta.data

meta$group <- factor(
  meta$group,
  levels = c(
    "low risk",
    "intermediate risk",
    "high risk"
  )
)

# cell type × risk group
tab <- table(
  meta$celltype_final,
  meta$group
)

# expected counts
row_sum <- rowSums(tab)
col_sum <- colSums(tab)
total   <- sum(tab)

expected <- outer(
  row_sum,
  col_sum,
  FUN = "*"
) / total

# observed / expected
roe <- tab / expected

roe
library(ComplexHeatmap)
library(circlize)

Heatmap(
  roe,
  name = "Ro/e",
  cluster_rows = TRUE,
  cluster_columns = FALSE
)
############################################################
## 7. High-risk vs low-risk pseudobulk edgeR
############################################################

obj_hl <- subset(
  scRNA_final_clean,
  subset = group %in%
    c("low risk", "high risk")
)

obj_hl$group <- factor(
  obj_hl$group,
  levels = c(
    "low risk",
    "high risk"
  )
)

major_celltypes <- c(
  "Neutrophils",
  "Macrophages",
  "CD4+ T cells",
  "CD8+ T cells",
  "NK cells",
  "Epithelial cells"
)

run_pseudobulk_edger <- function(
    seu,
    celltype,
    min_cells = 20
){
  
  ## select cell type
  obj <- subset(
    seu,
    subset =
      celltype_major_refined == celltype
  )
  
  ## number of cells per patient
  sample_count <- obj@meta.data %>%
    count(
      orig.ident,
      group,
      name = "n_cells"
    )
  
  keep_samples <- sample_count %>%
    filter(
      n_cells >= min_cells
    ) %>%
    pull(orig.ident)
  
  obj <- subset(
    obj,
    subset =
      orig.ident %in% keep_samples
  )
  
  ## patient metadata
  sample_meta <- obj@meta.data %>%
    transmute(
      sample = as.character(orig.ident),
      group  = as.character(group)
    ) %>%
    distinct()
  
  sample_meta$group <- factor(
    sample_meta$group,
    levels = c(
      "low risk",
      "high risk"
    )
  )
  
  ## raw counts
  counts <- LayerData(
    obj,
    assay = "RNA",
    layer = "counts"
  )
  
  ## aggregate cells into patients
  cell_sample <-
    as.character(obj$orig.ident)
  
  sample_order <-
    unique(cell_sample)
  
  design_pb <-
    Matrix::sparse.model.matrix(
      ~0 + factor(
        cell_sample,
        levels = sample_order
      )
    )
  
  pb <- counts %*% design_pb
  colnames(pb) <- sample_order
  
  ## match metadata
  sample_meta <-
    sample_meta[
      match(
        colnames(pb),
        sample_meta$sample
      ),
    ]
  
  ## edgeR
  y <- DGEList(
    counts = pb
  )
  
  keep <- filterByExpr(
    y,
    group = sample_meta$group
  )
  
  y <- y[
    keep,
    ,
    keep.lib.sizes = FALSE
  ]
  
  y <- calcNormFactors(y)
  
  design <- model.matrix(
    ~ group,
    data = sample_meta
  )
  
  y <- estimateDisp(
    y,
    design,
    robust = TRUE
  )
  
  fit <- glmQLFit(
    y,
    design,
    robust = TRUE
  )
  
  qlf <- glmQLFTest(
    fit,
    coef = "grouphigh risk"
  )
  
  de <- topTags(
    qlf,
    n = Inf
  )$table
  
  de$gene <- rownames(de)
  
  return(de)
}
DE_list <- lapply(
  major_celltypes,
  function(ct){
    
    run_pseudobulk_edger(
      obj_hl,
      celltype = ct,
      min_cells = 20
    )
    
  }
)

names(DE_list) <- major_celltypes
levels = c(
  "low risk",
  "high risk"
)
############################################################
## 8. Prepare ranked gene list
############################################################

prepare_rank <- function(de){
  
  rank_df <- de %>%
    filter(
      !is.na(gene),
      !is.na(logFC),
      !is.na(F)
    ) %>%
    mutate(
      rank_stat =
        sign(logFC) * sqrt(F)
    )
  
  gene_map <- bitr(
    unique(rank_df$gene),
    fromType = "SYMBOL",
    toType   = "ENTREZID",
    OrgDb    = org.Hs.eg.db
  )
  
  rank_df <- rank_df %>%
    inner_join(
      gene_map,
      by = c(
        "gene" = "SYMBOL"
      )
    ) %>%
    arrange(
      desc(abs(rank_stat))
    ) %>%
    distinct(
      ENTREZID,
      .keep_all = TRUE
    ) %>%
    arrange(
      desc(rank_stat)
    )
  
  geneList <- rank_df$rank_stat
  names(geneList) <- rank_df$ENTREZID
  
  sort(
    geneList,
    decreasing = TRUE
  )
}
gsea_kegg <- gseKEGG(
  geneList = geneList,
  organism = "hsa",
  keyType = "ncbi-geneid",
  minGSSize = 15,
  maxGSSize = 500,
  pvalueCutoff = 1,
  pAdjustMethod = "BH",
  eps = 0,
  verbose = FALSE
)
GO_sig <- as.data.frame(gsea_go) %>%
  filter(p.adjust < 0.05)

KEGG_sig <- as.data.frame(gsea_kegg) %>%
  filter(p.adjust < 0.05)

Reactome_sig <- as.data.frame(gsea_reactome) %>%
  filter(p.adjust < 0.05)
