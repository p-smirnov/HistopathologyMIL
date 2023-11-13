---
title: "Baseline Tumour Class Only Model"
output: html_notebook
---



```{r}
library(rhdf5)
library(data.table)

path_to_extracted_features = '/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/UKHD_Neuro/RetCLL_Features/1024'

slide_meta = fread("/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology//metadata/slides_FS_anno.csv")
ct_scoring = fread("/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology//metadata/CT_3_Class_Draft.csv", header = T)

tiles_per_pat = fread("~/histopathology/metadata/TilesPerPat_299.csv")

ntilespp = 250

```




```{r}


merged_meta <- merge(slide_meta, ct_scoring, by.x = 'txt_idat', by.y = 'idat', all = FALSE)

merged_meta <- merged_meta[CT_class %in% c("Chromothripsis", "No Chromothripsis")]

files = paste0(merged_meta$uuid, '.h5')
prop.table(table(merged_meta$CT_class))

```



```{r}

myx = file.exists(file.path(path_to_extracted_features, files))
sum(myx)

files <- files[myx]

TilesPerPat = fread("/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/metadata/TilesPerPat_1024.csv")
filestokeep = TilesPerPat[`Tiles Per Slide`>=ntilespp, File]
files <- intersect(files,filestokeep)
length(files)
```



```{r}

set.seed(42)

myfiles <- c(sample(merged_meta[CT_class=="Chromothripsis" & paste0(uuid, ".h5") %in% files, paste0(uuid, ".h5")], 100),sample(merged_meta[CT_class=="No Chromothripsis" & paste0(uuid, ".h5") %in% files, paste0(uuid, ".h5")], 100))

mytilenum <- TilesPerPat[match(myfiles, File), `Tiles Per Slide`]

test <- lapply(seq_along(myfiles),function(i) rhdf5::h5read(file.path(path_to_extracted_features, myfiles[i]), index=list(NULL, sort(sample(mytilenum[i], ntilespp))), 'feats'))


names(test) <- myfiles
fulldata <- do.call(cbind, test)

```


```{r}
ct.status.per.file <- merged_meta[match(gsub(names(test), pat='\\.h5', rep=''), uuid), CT_class]

ct.status.vec <- rep(ct.status.per.file, times=ntilespp)



```

```{r}

library(uwot)


my.umap <- umap(t(fulldata))

library(ggplot2)
toPlot <- data.frame(x=my.umap[,1], y=my.umap[,2], patient=rep(myfiles, each=ntilespp))

ggplot(toPlot, aes(x,y,col=patient)) + geom_point(size=0.1, alpha=0.5)+ theme_classic() + theme(legend.position = 'none') 

```



```{r}

pvals <- sapply(seq_len(nrow(fulldata)),function(i)t.test(fulldata[i,]~ct.status.vec)$p.val)

boxplot(fulldata[37,]~ct.status.vec)

```

```{r}

toPlot <- data.frame(x=my.umap[,1], y=my.umap[,2], CT=ct.status.vec)

ggplot(toPlot, aes(x,y,col=CT)) + geom_point(size=0.1, alpha=0.8) + theme_classic() +  theme(legend.position = 'none')


```



```{r}

isglio <- merged_meta[match(gsub(myfiles, pat='\\.h5', rep=''), uuid), family]=='glioblastoma'

isglio.vec <- rep(isglio, times=sapply(test, ncol))


toPlot <- data.frame(x=my.umap[,1], y=my.umap[,2], glio=isglio.vec)

ggplot(toPlot, aes(x,y,col=glio)) + geom_point(size=0.1, alpha=0.1) + theme(legend.position = 'none')


```



```{r}

tumour_type <- merged_meta[match(gsub(myfiles, pat='\\.h5', rep=''), uuid), family]

tumour_type <- rep(tumour_type, times=ntilespp)


toPlot <- data.frame(x=my.umap[,1], y=my.umap[,2], tumour=tumour_type)

ggplot(toPlot, aes(x,y,col=tumour)) + geom_point(size=0.1, alpha=0.8) + theme(legend.position = 'none')


```


```{r}

pca <- prcomp(t(fulldata))

toPlot <- data.frame(x=pca$x[,1], y=pca$x[,2], CT=ct.status.vec)

ggplot(toPlot, aes(x,y,col=CT)) + geom_point(alpha=0.9, size=0.9) + theme(legend.position = 'none')


```

```{r}


toPlot <- data.frame(x=pca$x[,1], y=pca$x[,2], glio=isglio.vec)

ggplot(toPlot, aes(x,y,col=glio)) + geom_point() + theme(legend.position = 'none')


```

```{r}
pdf(height=30, width = 30)
pairs(pca$x[,1:5], pch='.',col=c('red', 'blue')[as.numeric(factor(ct.status.vec))])
dev.off()
```


```{r}

fulldata_train <- fulldata[,c(1:(80*ntilespp), (100*ntilespp):(180*ntilespp))] ## todo: fix off by 1 error

pca_train <- prcomp(t(fulldata_train))

train_x <- pca_train$x[,1:100]

ct.status.vec_train <- ct.status.vec[c(1:(80*ntilespp), (100*ntilespp):(180*ntilespp))]

y <- as.numeric(factor(ct.status.vec_train)) -1 

topredict <- data.frame(y=y,train_x)

test.glm <- glm(y~.,family="binomial", data=topredict)
# summary(test.glm)
```


```{r}

sum(predict(test.glm, type='response')<0.5)

recall = sum(predict(test.glm, type='response')<0.5& ct.status.vec_train=='Chromothripsis')/sum(ct.status.vec_train=='Chromothripsis')

precision = sum(predict(test.glm, type='response')<0.5& ct.status.vec_train=='Chromothripsis')/sum(predict(test.glm, type='response')<0.5)

sum(predict(test.glm, type='response')>0.5& ct.status.vec_train=='No Chromothripsis')/sum(ct.status.vec_train=='No Chromothripsis')

2*(precision*recall)/(precision+recall)

```


```{r}

ct.status.vec_test <- ct.status.vec[c((80*ntilespp):(100*ntilespp), (180*ntilespp):(200*ntilespp))]

test_x <- predict(pca_train,t(fulldata[,c((80*ntilespp):(100*ntilespp), (180*ntilespp):(200*ntilespp))]))[,1:100]

test_preds <- predict(test.glm, newdata=data.frame(test_x), type='response')

sum(test_preds<0.5)

recall = sum(test_preds<0.5& ct.status.vec_test=='Chromothripsis')/sum(ct.status.vec_test=='Chromothripsis')

precision = sum(test_preds<0.5& ct.status.vec_test=='Chromothripsis')/sum(test_preds<0.5)


2*(precision*recall)/(precision+recall)

```

```{r}
library(PRROC)
y_test = as.numeric(factor(ct.status.vec_test))-1


plot(roc.curve(test_preds, weights.class0=y_test, curve=TRUE))

plot(pr.curve(test_preds, weights.class0=y_test, curve=TRUE))


```


Lets calculate the performance on a per-patient basis:


```{r}

pred_ct <- (test_preds<0.5)

sum(sapply(1:20, function(i) {
  mean(test_preds[(((i-1)*500)+1):((i)*500)])<0.5
}))


sum(sapply(21:40, function(i) {
  mean(test_preds[(((i-1)*500)+1):((i)*500)])>0.5
}))


sum(sapply(1:40, function(i) {
  mean(test_preds[(((i-1)*500)+1):((i)*500)])<0.5
}))

```


### Corrected for tumour type




```{r}

fulldata_train <- fulldata[,c(1:(80*500), (100*500):(180*500))] ## todo: fix off by 1 error

pca_train <- prcomp(t(fulldata_train))

train_x <- pca_train$x[,1:100]

ct.status.vec_train <- ct.status.vec[c(1:(80*500), (100*500):(180*500))]

y <- as.numeric(factor(ct.status.vec_train)) -1 

topredict <- data.frame(y=y, tumour = tumour_type[c(1:(80*500), (100*500):(180*500))], train_x)

test.glm <- glm(y~.,family="binomial", data=topredict)
test.glmnull <- glm(y~tumour,family="binomial", data=topredict)
anova(test.glmnull, test.glm,  test="LRT")
```


```{r}

sum(predict(test.glm, type='response')<0.5)

recall = sum(predict(test.glm, type='response')<0.5& ct.status.vec_train=='Chromothripsis')/sum(ct.status.vec_train=='Chromothripsis')

precision = sum(predict(test.glm, type='response')<0.5& ct.status.vec_train=='Chromothripsis')/sum(predict(test.glm, type='response')<0.5)

sum(predict(test.glm, type='response')>0.5& ct.status.vec_train=='No Chromothripsis')/sum(ct.status.vec_train=='No Chromothripsis')

2*(precision*recall)/(precision+recall)

```


```{r}

ct.status.vec_test <- ct.status.vec[c((80*500):(100*500), (180*500):(200*500))]

test_x <- predict(pca_train,t(fulldata[,c((80*500):(100*500), (180*500):(200*500))]))[,1:100]

test_preds <- predict(test.glm, newdata=data.frame(test_x), type='response')

sum(test_preds<0.5)

recall = sum(test_preds<0.5& ct.status.vec_test=='Chromothripsis')/sum(ct.status.vec_test=='Chromothripsis')

precision = sum(test_preds<0.5& ct.status.vec_test=='Chromothripsis')/sum(test_preds<0.5)


2*(precision*recall)/(precision+recall)

```

Lets calculate the performance on a per-patient basis:


```{r}

pred_ct <- (test_preds<0.5)

sum(sapply(1:20, function(i) {
  mean(test_preds[(((i-1)*500)+1):((i)*500)])<0.5
}))


sum(sapply(21:40, function(i) {
  mean(test_preds[(((i-1)*500)+1):((i)*500)])>0.5
}))

```



# GLMNET on features

Lets try a GLMNET on the feature space directly. 


```{r}


library(glmnet)

load("workspace.RData")


y <- as.numeric(factor(ct.status.vec_train)) - 1

glmnet_x_train <- t(fulldata_train)

glmnet.cv.fit <- cv.glmnet(x = glmnet_x_train, y = y, family = "binomial", alpha = 1)


```



```{r}

table(predict(glmnet.fit, glmnet_x_train, type = "class"), y)

glmnet.train.preds <- predict(glmnet.fit, glmnet_x_train, type = "response")

recall = sum(glmnet.train.preds<0.5& ct.status.vec_train=='Chromothripsis')/sum(ct.status.vec_train=='Chromothripsis')

precision = sum(glmnet.train.preds<0.5& ct.status.vec_train=='Chromothripsis')/sum(glmnet.train.preds<0.5)

sum(glmnet.train.preds>0.5& ct.status.vec_train=='No Chromothripsis')/sum(ct.status.vec_train=='No Chromothripsis')

2*(precision*recall)/(precision+recall)


```









```{r}
# 
# fulldata_train <- pca$x[1:(80*500),1:100]
# 
# isglio.vec_train <- isglio.vec[1:(80*500)]
# 
# y <- as.numeric(factor(isglio.vec_train)) -1 
# 
# topredict <- data.frame(y=y, fulldata_train)
# 
# test.glm2 <- glm(y~.,family="binomial", data=topredict)
# summary(test.glm2)
```


```{r}
# 
# isglio.vec_test <- isglio.vec[(80*500):(100*500)]
# fulldata_test <- pca$x[(80*500):(100*500),1:100]
# 
# test_preds <- predict(test.glm, newdata=data.frame(fulldata_test), type='response')
# 
# sum(test_preds<0.5)
# 
# recall = sum(test_preds<0.5& !isglio.vec_test)/sum(!isglio.vec_test)
# 
# precision = sum(test_preds<0.5& !isglio.vec_test)/sum(test_preds<0.5)
# 
# 
# 2*(precision*recall)/(precision+recall)
# 
# 
# recall = sum(test_preds>0.5& isglio.vec_test)/sum(isglio.vec_test)
# 
# precision = sum(test_preds>0.5& isglio.vec_test)/sum(test_preds>0.5)
# 
# 
# 2*(precision*recall)/(precision+recall)

```

