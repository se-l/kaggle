#Visualize parameter sweep
library(rgl)
library(data.table) #Faster reading
library(ggvis)
library(dplyr)
path <- "C:\\0Sebs_other_data\\Machine_Learning\\Kaggle\\BNP_Claims\\"
AllTestResult <- fread(paste(path,"AllTestResult.csv",sep=""), stringsAsFactors=TRUE)
idx <- which(AllTestResult$nfold==1)
#idx <- which(AllTestResult$tree_depth==7)
plot3d(x = AllTestResult$subsample[idx]
       ,y = AllTestResult$colsample_bytree[idx]
       ,z = AllTestResult$bestMean[idx]
       , type="p"
       , col="red"
       , site=5
       , lwd=15
       , xlab = "Subsample"
       , ylab = "ColSubsample_bytree"
       , zlab = "Best Mean")

#write.csv(nfoldsweep, paste(path,"nfold_param_sweep.csv",sep=""), row.names=F, quote=F)
ggvisplot <- AllTestResult$subsample %>% ggvis(~bestIter)
