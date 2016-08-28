### Examine NAs using mice and VIM packages ###
path <- "C:\\0Sebs_other_data\\Machine_Learning\\Kaggle\\BNP_Claims\\"
# read file and convert char to int
library(readr)
train <- read_csv(paste(path,"train.csv",sep=""))
for (f in names(train)) {
  if (class(train[[f]])=="character") {
    levels <- unique(train[[f]])
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
  }
}

# make a table of missing values
# library(mice)
# missers <- md.pattern(train[, -c(1:2)])
# head(missers)
# write_csv(as.data.frame(missers),"NAsTable.csv")

# make plots of missing values
library(VIM)

png(filename="NAsPatternEq.png",
    type="cairo",
    units="in",
    width=12,
    height=6.5,
    pointsize=10,
    res=300)

miceplot1 <- aggr(train[, -c(1:2)], col=c("dodgerblue","dimgray"),
                 numbers=TRUE, combined=TRUE, varheight=FALSE, border="gray50",
                 sortVars=TRUE, sortCombs=FALSE, ylabs=c("Missing Data Pattern"),
                 labels=names(train[-c(1:2)]), cex.axis=.7)
dev.off()

png(filename="NAsPatternAdj.png",
    type="cairo",
    units="in",
    width=12,
    height=6.5,
    pointsize=10,
    res=300)

miceplot2 <- aggr(train[, -c(1:2)], col=c("dodgerblue","dimgray"),
                 numbers=TRUE, combined=TRUE, varheight=TRUE, border=NA,
                 sortVars=TRUE, sortCombs=FALSE, ylabs=c("Missing Data Pattern w/ Height Adjustment"),
                 labels=names(train[-c(1:2)]), cex.axis=.7)
dev.off()

