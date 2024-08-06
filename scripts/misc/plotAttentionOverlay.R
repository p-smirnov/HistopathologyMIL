library(data.table)
library(Matrix)

## hack for current slide, need to get this out of python with the attentionmap csv:

slideDim <- c(61752,78471)

test <- fread("~/histopathology/metadata/attention_maps/63454350-699E-4498-B842-096D9A55E236.csv")
coords <- test[,tstrsplit(gsub(V1, pat="\\.jpg", rep=""), split="_")][,.(V2,V3)]
coords <- cbind(coords, "Attention"=(test$V2-min(test$V2))/(max(test$V2) - min(test$V2)))

coords[,V2:=as.numeric(V2)]
coords[,V3:=as.numeric(V3)]

m <- matrix(nrow = 618, ncol = 785, data = NA_real_)#, sparse = TRUE)


for(i in seq(nrow(coords))){
  xcoords <- unique(round((coords[i][[1]]):((coords[i][[1]]+383))/100))
  ycoords <- unique(round((coords[i][[2]]):((coords[i][[2]]+383))/100))
  m[xcoords,ycoords] <- coords[i][[3]]
}

library(ComplexHeatmap)

# Heatmap(m, cluster_rows = FALSE, cluster_columns = FALSE)

library(ggplot2)
library(cowplot)

m.d <- reshape2::melt(m)

colnames(m.d) <- c("x", "y", "Attention") 

png("~/test_attention_heatmap.png", bg='transparent')
ggplot(m.d, aes(x,y,fill=Attention)) + geom_raster(alpha=0.6) + theme_nothing() + scale_fill_viridis_c(na.value="transparent", option = "A")
dev.off()


## repeat with medullo slide


"013A1CFA-1CA9-4B36-BAC8-5A88842FE723"


slideDim <- c(111652,93570)


test <- fread("~/histopathology/metadata/attention_maps/013A1CFA-1CA9-4B36-BAC8-5A88842FE723.csv")
coords <- test[,tstrsplit(gsub(V1, pat="\\.jpg", rep=""), split="_")][,.(V2,V3)]
coords <- cbind(coords, "Attention"=(test$V2-min(test$V2))/(max(test$V2) - min(test$V2)))

coords[,V2:=as.numeric(V2)]
coords[,V3:=as.numeric(V3)]

m <- matrix(nrow = 1117, ncol = 936, data = NA_real_)#, sparse = TRUE)


for(i in seq(nrow(coords))){
  xcoords <- unique(round((coords[i][[1]]):((coords[i][[1]]+383))/100))
  ycoords <- unique(round((coords[i][[2]]):((coords[i][[2]]+383))/100))
  m[xcoords,ycoords] <- coords[i][[3]]
}

library(ComplexHeatmap)

# Heatmap(m, cluster_rows = FALSE, cluster_columns = FALSE)

library(ggplot2)
library(cowplot)

m.d <- reshape2::melt(m)

colnames(m.d) <- c("x", "y", "Attention") 

png("~/test_attention_heatmap2.png", bg='transparent')
ggplot(m.d, aes(x,y,fill=Attention)) + geom_raster(alpha=0.6) + theme_nothing() + scale_fill_viridis_c(na.value="transparent", option = "A")
dev.off()

