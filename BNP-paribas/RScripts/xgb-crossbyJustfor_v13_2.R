#credit for original script goes to justfor:
#https://www.kaggle.com/justfor/bnp-paribas-cardif-claims-management/xgb-cross-val-and-feat-select

library(data.table) #Faster reading
library(xgboost)

path <- "C:\\0Sebs_other_data\\Machine_Learning\\Kaggle\\BNP_Claims\\"

# Start the clock!
start_time <- Sys.time()

na.roughfix2 <- function (object, ...) {
    res <- lapply(object, roughfix)
    structure(res, class = "data.frame", row.names = seq_len(nrow(object)))
}

roughfix <- function(x) {
    missing <- is.na(x)
    if (!any(missing)) return(x)
    
    if (is.numeric(x)) {
        x[missing] <- median.default(x[!missing])
    } else if (is.factor(x)) {
        freq <- table(x)
        x[missing] <- names(freq)[which.max(freq)]
    } else {
        stop("na.roughfix only works for numeric or factor")
    }
    x
}

# Set a seed for reproducibility
set.seed(6337)

cat("reading the train and test data\n")
# Read train and test
train_raw <- fread(paste(path,"train.csv",sep=""), stringsAsFactors=TRUE) 
print(dim(train_raw))
print(sapply(train_raw, class))

y <- train_raw$target
train_raw$target <- NULL
train_raw$ID <- NULL
n <- nrow(train_raw)

test_raw <- fread(paste(path,"test.csv",sep=""), stringsAsFactors=TRUE) 
test_id <- test_raw$ID
test_raw$ID <- NULL
print(dim(test_raw))
print(sapply(test_raw, class))
cat("Data read ")
print(difftime( Sys.time(), start_time, units = 'sec'))

# Preprocess data
# Find factor variables and translate to numeric
cat("Preprocess data\n")
all_data <- rbind(train_raw,test_raw)
all_data <- as.data.frame(all_data) # Convert data table to data frame

#  Result of Boruta, thanks to Florian
# https://www.kaggle.com/jimthompson/bnp-paribas-cardif-claims-management/using-the-boruta-package-to-determine-fe/discussion/comment/109207#post109207

#cat("Drop rejected vars - not important as found by Boruta\n")
#all_data$v72 <- NULL
#all_data$v62 <- NULL
#all_data$v112 <- NULL
#all_data$v107 <- NULL
#all_data$v125 <- NULL
#all_data$v75 <- NULL
#all_data$v71 <- NULL
#all_data$v91 <- NULL
#all_data$v74 <- NULL
#all_data$v52 <- NULL
#all_data$v22 <- NULL
#all_data$v3 <- NULL

N <- ncol(all_data)
all_data$NACount_Init_N <- rowSums(is.na(all_data)) / N 
all_data$NACount_Init <- rowSums(is.na(all_data)) 

# Idea from https://www.kaggle.com/sinaasappel/bnp-paribas-cardif-claims-management/exploring-paribas-data
levels(all_data$v3)[1] <- NA #to remove the "" level and replace by NA
levels(all_data$v22)[1] <- NA
levels(all_data$v30)[1] <- NA
levels(all_data$v31)[1] <- NA
levels(all_data$v52)[1] <- NA
levels(all_data$v56)[1] <- NA
levels(all_data$v91)[1] <- NA
levels(all_data$v107)[1] <- NA
levels(all_data$v112)[1] <- NA
levels(all_data$v113)[1] <- NA
levels(all_data$v125)[1] <- NA

# Small feature addition - Count NA percentage
N <- ncol(all_data)
all_data$NACount_N <- rowSums(is.na(all_data)) / N 
all_data$NACount <- rowSums(is.na(all_data)) 


feature.names <- names(all_data)

#all_data$d115_69 <- all_data$v115 / all_data$v69
#all_data$d26_46 <- all_data$v26  / all_data$v46

#cat("Remove highly correlated features\n")
#highCorrRemovals <- c("v8","v23","v25","v36","v37","v46",
#                      "v51","v53","v54","v63","v73","v81",
#                      "v82","v89","v92","v95","v105","v107",
#                      "v108","v109","v116","v117","v118",
#                      "v119","v123","v124","v128")

# Some noFeatures:
# highCorrRemovals  <- c('v3','v22','v24','v30','v31','v47',
#                        'v52','v56','v66','v71','v74','v75',
#                        'v79','v91','v107','v110','v112','v113',
#                        'v125')
#from Artem
highCorrRemovals  <- c('v8','v23','v25','v31','v36','v37',
                       'v46','v51','v53','v54','v63','v73',
                       'v75','v79','v81','v82','v89','v92',
                       'v95','v105','v107','v108','v109','v110',
                       'v116','v117','v118','v119','v123','v124',
                       'v128')
            
all_data <- all_data[,-which(names(all_data) %in% highCorrRemovals)]

cat("assuming text variables are categorical & replacing them with numeric ids\n")
cat("re-factor categorical vars & replacing them with numeric ids\n")
for (f in feature.names) {
    if (class(all_data[[f]])=="character" || class(all_data[[f]])=="factor") {
        all_data[[f]] <- as.integer(factor(all_data[[f]]))
    }
}

feature.names <- names(all_data)
# make feature of counts of zeros factor
all_data$ZeroCount <- rowSums(all_data[,feature.names]== 0) / N
all_data$Below0Count <- rowSums(all_data[,feature.names] < 0) / N


train <- all_data[1:n,]
test <- all_data[(n+1):nrow(all_data),] 

print(dim(train))
#summary(train)tr
print(dim(test))
#summary(test)

#rm(all_data)
#gc()

if (FALSE) {
# Boruta for feature selection used instead of ks
#Feature selection using KS test with 0.004 as cutoff.
tmpJ = 1:ncol(test)
ksMat = NULL
for (j in tmpJ) {
    cat(j," ")
    ksMat = rbind(ksMat, cbind(j, ks.test(train[,j],test[,j])$statistic))
}

ksMat2 = ksMat[ksMat[,2]<0.004,]
feats = as.numeric(ksMat2[,1]) 
cat(length(feats),"\n")
cat(names(train)[feats],"\n")
var_to_drop <- setdiff(names(all_data), names(train)[feats])
cat("\nVars to drop:", var_to_drop, "\n")
# Input missing data & convert to xgb-data structure
#train[is.na(train)] <- -1
#test[is.na(test)] <- -1

#xgtrain = xgb.DMatrix(as.matrix(train[,feats]), label = y, missing = -1)
#xgtest = xgb.DMatrix(as.matrix(test[,feats]), missing=-1)

all_data <- rbind(train[,feats],test[,feats])
}


#all_data <- na.roughfix2(all_data)

train <- all_data[1:n,]
test <- all_data[(n+1):nrow(all_data),] 

xgtrain = xgb.DMatrix(as.matrix(train), label = y, missing=NA)
xgtest = xgb.DMatrix(as.matrix(test), missing=NA)

# Do cross-validation with xgboost - xgb.cv
docv <- function(param0, iter) {
    model_cv = xgb.cv(
            params = param0
            , nrounds = iter
            , nfold = 2
            , data = xgtrain
            , early.stop.round = 10
            , maximize = FALSE
            , nthread = 8
            , verbose = 1
            )
    gc()
    best <- min(model_cv$test.logloss.mean)
    bestIter <- which(model_cv$test.logloss.mean==best)
    
    cat("\n",best, bestIter,"\n")
    print(model_cv[bestIter])
    
    bestIter-1
}

doTest <- function(param0, iter) {
    watchlist <- list('train' = xgtrain)
    model = xgb.train(
            nrounds = iter,
            params = param0, 
            data = xgtrain, 
            watchlist = watchlist, 
            print.every.n = 20, 
            nthread = 8
            )
    p <- c(model, xgtest)
    rm(model)
    gc()
    p
}

param0 <- list(
        # some generic, non specific params
        "objective"  = "binary:logistic",
        "eval_metric" = "logloss",
        "eta" = 0.01,
        "subsample" = 0.8,
        "colsample_bytree" = 0.9,
        "colsample_bylevel" = 0.6,
        "min_child_weight" = 0,
        "max_depth" = 10
        )

#print(proc.time() - start_time)
print( difftime( Sys.time(), start_time, units = 'sec'))
cat("Training a XGBoost classifier with cross-validation\n")
set.seed(6338)
#cv <- docv(param0, 500)
# Show the clock
print( difftime( Sys.time(), start_time, units = 'sec'))

# sample submission total analysis
submission <- read.csv(paste(path,"sample_submission.csv",sep=""))
ensemble <- rep(0, nrow(test))

cv <- 3
cat("Calculated rounds:", cv, " Starting ensemble\n")

# Bagging of single xgboost for ensembling
# change to e.g. 1:10 to get quite good results
for (i in 1:3) {
    print(i)
    set.seed(i + 2017)
    p <- doTest(param0, cv) 
    # use 40% to 50% more than the best iter rounds from your cross-fold number.
    # as you have another 50% training data now, which gives longer optimal training time
    ensemble <- ensemble + p
}

# Finalise prediction of the ensemble
cat("Making predictions\n")
submission$PredictedProb <- ensemble/i

# Prepare submission
write.csv(submission, "BNPCardif-Rxgb-O-13.csv", row.names=F, quote=F)
summary(submission$PredictedProb)

# Stop the clock
#print(proc.time() - start_time)
print( difftime( Sys.time(), start_time, units = 'min'))