s.genes = c('Mcm5', 'Pcna', 'Tyms', 'Fen1', 'Mcm2', 'Mcm4', 'Rrm1', 'Ung', 'Gins2',
            'Mcm6', 'Cdca7', 'Dtl', 'Prim1', 'Uhrf1', 'CENPU', 'Hells', 'Rfc2',
            'Rpa2', 'Nasp', 'Rad51ap1', 'Gmnn', 'Wdr76', 'Slbp', 'Ccne2', 'Ubr7',
            'Pold3', 'Msh2', 'Atad2', 'Rad51', 'Rrm2', 'Cdc45', 'Cdc6', 'Exo1', 'Tipin',
            'Dscc1', 'Blm', 'Casp8ap2', 'Usp1', 'Clspn', 'Pola1', 'Chaf1b', 'Brip1', 'E2f8')
s.genes = unlist(lapply(s.genes, toupper))
g2m.genes = c('Hmgb2', 'Cdk1', 'Nusap1', 'Ube2c', 'Birc5', 'Tpx2', 'Top2a', 'Ndc80',
              'Cks2', 'Nuf2', 'Cks1b', 'Mki67', 'Tmpo', 'Cenpf', 'Tacc3', 'PIMREG',
              'Smc4', 'Ccnb2', 'Ckap2l', 'Ckap2', 'Aurkb', 'Bub1', 'Kif11', 'Anp32e',
              'Tubb4b', 'Gtse1', 'Kif20b', 'Hjurp', 'Cdca3', 'JPT1', 'Cdc20', 'Ttk',
              'Cdc25c', 'Kif2c', 'Rangap1', 'Ncapd2', 'Dlgap5', 'Cdca2', 'Cdca8',
              'Ect2', 'Kif23', 'Hmmr', 'Aurka', 'Psrc1', 'Anln', 'Lbr', 'Ckap5',
              'Cenpe', 'Ctcf', 'Nek2', 'G2e3', 'Gas2l3', 'Cbx5', 'Cenpa')
g2m.genes = unlist(lapply(g2m.genes, toupper))

library(Seurat)
library(Signac)
library(tidyverse)
barcodes = read.delim("../filtered_cells.txt", header = F, stringsAsFactors = F)$V1
leiden = read.delim("../leiden.txt", header = F, stringsAsFactors = F)$V1
leiden = as.factor(leiden)
input.data = Read10X(data.dir = "../filtered_feature_bc_matrix/")
hspc = CreateSeuratObject(counts = input.data$`Gene Expression`[,barcodes])
hspc = SCTransform(hspc, assay = 'RNA', new.assay.name = 'SCT', verbose = F)
hspc = CellCycleScoring(hspc, s.features = s.genes, g2m.features = g2m.genes, assay = 'SCT', set.ident = TRUE)
hspc = SCTransform(hspc, assay = 'RNA', new.assay.name = 'SCT', vars.to.regress = c('S.Score', 'G2M.Score'), verbose = F)
hspc = RunPCA(hspc, verbose = FALSE)
ElbowPlot(hspc, ndims = 50)
hspc = RunUMAP(hspc, dims = 1:30, reduction.name = 'umap.rna', reduction.key = 'rnaUMAP_')
hspc[["ATAC"]] <- CreateAssayObject(counts = input.data$`Peaks`[,barcodes], min.cells = 1)
rm(input.data); gc()
DefaultAssay(hspc) <- "ATAC"
hspc <- FindTopFeatures(hspc, min.cutoff = 5)
hspc <- RunTFIDF(hspc)
hspc <- RunSVD(hspc)
DepthCor(hspc)
hspc <- RunUMAP(hspc, reduction = 'lsi', dims = 2:30, reduction.name = "umap.atac", reduction.key = "atacUMAP_")
hspc <- FindMultiModalNeighbors(hspc, reduction.list = list("pca", "lsi"), dims.list = list(1:30, 2:30), k.nn = 50)
hspc <- RunUMAP(hspc, nn.name = "weighted.nn", reduction.name = "wnn.umap", reduction.key = "wnnUMAP_")
Idents(hspc) = leiden
p1 <- DimPlot(hspc, reduction = "umap.rna", label = TRUE, label.size = 2.5, repel = TRUE) + NoLegend()
p2 <- DimPlot(hspc, reduction = "umap.atac", label = TRUE, label.size = 2.5, repel = TRUE) + NoLegend()
p3 <- DimPlot(hspc, reduction = "wnn.umap", label = TRUE, label.size = 2.5, repel = TRUE) + NoLegend()
p1 + p2 + p3
write.table(rownames(hspc[['SCT']]$scale.data), "sct_genes.txt", sep = ',', row.names = F, col.names = F, quote = F)
write.table(hspc@neighbors$weighted.nn@nn.idx, "nn_idx.txt", sep = ',', row.names = F, col.names = F, quote = F)
write.table(hspc@neighbors$weighted.nn@nn.dist, "nn_dist.txt", sep = ',', row.names = F, col.names = F, quote = F)
write.table(hspc@neighbors$weighted.nn@cell.names, "nn_cell_names.txt", sep = ',', row.names = F, col.names = F, quote = F)
