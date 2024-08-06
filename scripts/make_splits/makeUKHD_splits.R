library(data.table)
library(TCGAutils)
library(TCGAbiolinks)
library(GenomicDataCommons)
library(caret)

new_slidelist <- fread("~/histopathology/metadata/labels_with_new_batch.csv", header=T)

cases() %>% 
  filter(submitter_id%in%cnv_status$TCGA_Case) %>% 
  select(c("project.project_id", "submitter_id")) %>% 
  results_all() -> project_map

project_map.dt <- data.table(data.frame("case_uuid" = project_map$case_id, case_id = project_map$submitter_id, project = project_map$project$project_id))

cnv_status <- merge(cnv_status, project_map.dt, by.x="TCGA_Case", by.y="case_id")


cnv_status <- cnv_status[CT_Status %in% c("Chromothripsis", "No Chromothripsis")]

## Need to figure out how to disambiguiate these cnv profiles
cnv_status[TCGA_Case %in% cnv_status[duplicated(cnv_status$TCGA_Case),TCGA_Case]]

# some of them actually also disagree
cnv_status[TCGA_Case %in% cnv_status[duplicated(cnv_status$TCGA_Case),TCGA_Case]][,length(unique(CT_Status)), TCGA_Case][V1>1]

## For simplicity, we remove these 7 that disagree for now, we will disambiguiate which SNP array corresponds to which image later. 

cnv_status <- cnv_status[!TCGA_Case %in% cnv_status[TCGA_Case %in% cnv_status[duplicated(cnv_status$TCGA_Case),TCGA_Case]][,length(unique(CT_Status)), TCGA_Case][V1>1, TCGA_Case]]

## Now we split controlling for class imbalance and tissue type

cnv_status[,table(project, CT_Status)]


## First, we try a naive split using just CT status as a blocking variable, into 60, 20, 20. 

cnv_status_uniq_ct <- unique(cnv_status[,.(TCGA_Case, CT_Status)])

set.seed(42)

train_samples <- createDataPartition(cnv_status_uniq_ct$CT_Status, p=0.6)[[1]]
other_samples <- seq_len(nrow(cnv_status_uniq_ct))[!seq_len(nrow(cnv_status_uniq_ct))%in%train_samples]
length(train_samples) + length(other_samples) == nrow(cnv_status_uniq_ct)

train_table <- cnv_status_uniq_ct[train_samples]
other_table <- cnv_status_uniq_ct[other_samples]


valid_samples <- createDataPartition(other_table$CT_Status, p=0.5)[[1]]
test_samples <- seq_len(nrow(other_table))[!seq_len(nrow(other_table))%in%valid_samples]


valid_table <- other_table[valid_samples]
test_table <- other_table[test_samples]

## Load in the slide mapping

primary_slides <- rbindlist(lapply(list.files("~/histopathology/metadata/TCGA/slides_list/", pattern="primary", full.names = TRUE), fread, header=FALSE))

primary_slides[,tstrsplit(V1, split="\\.")]
primary_slides[,slide_name := primary_slides[,tstrsplit(V1, split="\\.")][,V1]]
primary_slides[,TCGA_Case := TCGAutils::TCGAbarcode(slide_name)]



## Record how many patches were extracted per slide file

feature_files <- list.files("/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/TCGA/ffpe/HIPT_256/", 
                            patt=".zarr",
                            full.names=TRUE)

feature_files_short <- list.files("/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/TCGA/ffpe/HIPT_256/", 
                                  patt=".zarr",
                                  full.names=FALSE)
feature_files_short <- gsub(feature_files_short, pat="\\.zarr", rep="")

getNumPatches <- function(fl){
  Rarr::zarr_overview(fl, as_data_frame = T)$dim[[1]][1]
}

library(doParallel)
registerDoParallel(16)


numPatches <- foreach(fl=feature_files, .combine = c) %dopar%{getNumPatches(fl)}

patchesPerPatient <- data.frame(slide_name = feature_files_short, num_patches = numPatches)

primary_slides <- merge(primary_slides, patchesPerPatient, by="slide_name")





train_table <- merge(train_table, primary_slides, all.x=F, all.y=F)

valid_table <- merge(valid_table, primary_slides, all.x=F, all.y=F)

test_table <- merge(test_table, primary_slides, all.x=F, all.y=F)


fwrite(train_table, "~/histopathology/metadata/TCGA_no_dup_SNP6_CT__11_1MB_10_train.csv")
fwrite(valid_table, "~/histopathology/metadata/TCGA_no_dup_SNP6_CT__11_1MB_10_valid.csv")
fwrite(test_table, "~/histopathology/metadata/TCGA_no_dup_SNP6_CT__11_1MB_10_test.csv")


