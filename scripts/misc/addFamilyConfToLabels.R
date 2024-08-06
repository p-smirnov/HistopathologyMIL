
library(RSQLite)
library(data.table)
library(dplyr)

path2db <- "/b06x-isi/g703/g-i/IlmnArrayDB"


con <- dbConnect(RSQLite::SQLite(), dbname = file.path(path2db,"methylarrayDB.sqlite"))
family_preds <- as.data.table(tbl(con, "mnp_v12.8_epic_v2_families"))

slide_labels <- fread("~/histopathology/metadata/labels_with_new_batch.csv")

out.x <- merge(slide_labels, family_preds, by="idat", all.x=T)

purity <- as.data.table(tbl(con, "purity"))

out.x <- merge(out.x, purity, by="idat", all.x=T)


out.x[,table(prediction_superfamily == max_super_family_class)]

write.csv(out.x, "~/histopathology/metadata/labels_with_new_batch_v12.8.csv")
