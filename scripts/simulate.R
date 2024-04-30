# Warning! R 3.4 and Bioconductor 3.5 are required for splatter!

# install.packages("BiocInstaller", repos="https://bioconductor.org/packages/3.6/bioc")
library(BiocInstaller)
biocLite('splatter')

library(scater)
library(splatter) # requires splatter >= 1.2.0
library(SingleCellExperiment)
setwd("/Users/yufeideng/Documents/GitHub/bis687-project/")

# sc_example_counts <- read.csv("endoderm/endoderm.csv", header = TRUE, row.names = 1)
# sc_example <- SingleCellExperiment(assays = list(counts = as.matrix(sc_example_counts)))
# params <- splatEstimate(sc_example)

save.sim <- function(sim, dir) {
  counts     <- counts(sim)
  truecounts <- assays(sim)$TrueCounts
  drp <- 'Dropout' %in% names(assays(sim))
  if (drp) {
    dropout    <- assays(sim)$Dropout
    mode(dropout) <- 'integer'
  }
  cellinfo   <- colData(sim)
  geneinfo   <- rowData(sim)

  # save count matrices
  write.table(counts, paste0(dir, '/counts.tsv'),
              sep='\t', row.names=T, col.names=T, quote=F)
  write.table(truecounts, paste0(dir, '/info_truecounts.tsv'),
              sep='\t', row.names=T, col.names=T, quote=F)

  if (drp) {
    # save ground truth dropout labels
    write.table(dropout, paste0(dir, '/info_dropout.tsv'),
                sep='\t', row.names=T, col.names=T, quote=F)
  }

  # save metadata
  write.table(cellinfo, paste0(dir, '/info_cellinfo.tsv'), sep='\t',
              row.names=F, quote=F)
  write.table(geneinfo, paste0(dir, '/info_geneinfo.tsv'), sep='\t',
              row.names=F, quote=F)

  saveRDS(sim, paste0(dir, '/sce.rds'))
}


############# version in python notebook ############
simulate <- function(nGroups=2, nGenes=200, batchCells=2000, dropout=5)
{
  if (nGroups > 1) method <- 'groups'
  else             method <- 'single'
  
  group.prob <- rep(1, nGroups) / nGroups
  
  # new splatter requires dropout.type
  if ('dropout.type' %in% slotNames(newSplatParams())) {
    if (dropout)
      dropout.type <- 'experiment'
    else
      dropout.type <- 'none'
    
    sim <- splatSimulate(group.prob=group.prob, nGenes=nGenes, batchCells=batchCells,
                         dropout.type=dropout.type, method=method,
                         seed=42, dropout.shape=-1, dropout.mid=dropout)
    
  } else {
    sim <- splatSimulate(group.prob=group.prob, nGenes=nGenes, batchCells=batchCells,
                         dropout.present=!dropout, method=method,
                         seed=42, dropout.shape=-1, dropout.mid=dropout)        
  }
  
  dirname <- paste0('group', nGroups, '_dropout', dropout, ifelse(swap, '/swap', ''))
  if (!dir.exists(dirname))
    dir.create(dirname, showWarnings=F, recursive=T)
  save.sim(sim, dirname)
  # counts     <- as.data.frame(t(counts(sim)))
  # truecounts <- as.data.frame(t(assays(sim)$TrueCounts))
  # 
  # dropout    <- as.matrix(assays(sim)$Dropout)
  # mode(dropout) <- 'integer'
  # dropout    <- as.data.frame(t(dropout))
  # 
  # cellinfo   <- as.data.frame(colData(sim))
  # geneinfo   <- as.data.frame(rowData(sim))
  
  # list(counts=counts,
  #      cellinfo=cellinfo,
  #      geneinfo=geneinfo,
  #      truecounts=truecounts,
  #      dropout=dropout)
}



############# previous version ##############
for (dropout in c(0.5)) {
  for (ngroup in c(2)) {
    for(swap in c(F)) {

      nGenes <- 5000
      batchCells <- 2000

      if (swap) {
        tmp <- nGenes
        nGenes <- batchCells
        batchCells <- tmp
      }

      # split nCells into roughly ngroup groups
      if(ngroup==1) {
        group.prob <- 1
      } else {
        group.prob <- rep(1, ngroup)/ngroup
      }
      method <- ifelse(ngroup == 1, 'single', 'groups')

      # dirname <- paste0('group', ngroup, '_dropout', dropout, ifelse(swap, '/swap', ''))
      # if (!dir.exists(dirname))
      #   dir.create(dirname, showWarnings=F, recursive=T)
      # 
      # #### Estimate parameters from the real dataset
      # # data(sc_example_counts)
      # # params <- splatEstimate(sc_example_counts)
      # 
      # # simulate scRNA data
      # sim <- splatSimulate(params, group.prob=group.prob, nGenes=nGenes,
      #                      # dropout.present=(dropout!=0), 
      #                      dropout.shape=-1,
      #                      dropout.mid=dropout, seed=42, method=method,
      #                      bcv.common=1) # limit disp to get fewer true zeros
      # save.sim(sim, dirname)

      ### Simulate data without using real data
      dirname <- paste0('data/sim/group', ngroup, 'dropout', dropout, ifelse(swap, '/swap', ''))
      if (!dir.exists(dirname))
        dir.create(dirname, showWarnings=F, recursive=T)

      sim <- splatSimulate(group.prob=group.prob, nGenes=nGenes, batchCells=batchCells,
                           # dropout.present=(dropout!=0),
                           method=method,
                           seed=42, dropout.shape=-1, dropout.mid=dropout)
      save.sim(sim, dirname)
    }
  }
}



} 