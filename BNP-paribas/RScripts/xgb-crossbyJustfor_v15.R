#credit for original script goes to justfor:
#https://www.kaggle.com/justfor/bnp-paribas-cardif-claims-management/xgb-cross-val-and-feat-select

#Parameter Sweeping with xgb.cv on 100% of training set. n-fold=4.
#set eta at o.5 at first for speedy sweeping.
#Parameters to sweep:
#min_child, max_depth, 
#Rough 2D sampling. 3 max_depth -> 9 3 min-child: settings.

library(data.table) #Faster reading
library(xgboost)
library(caret)

path <- "C:\\0Sebs_other_data\\Machine_Learning\\Kaggle\\BNP_Claims\\"
bestMean <- numeric()
bestIter <- numeric()
nfold <- integer()
tree_depth <- integer()
min_child_weight <- integer()
subsample <- numeric()
colsample_bytree <- numeric()

# Start the clock!
start_time <- Sys.time()
set.seed(3456)

cat("reading the train and test data\n")
# Read train and test
train_raw <- fread(paste(path,"train.csv",sep=""), stringsAsFactors=TRUE)

#Data Preparation
n <- nrow(train_raw)
cat("Preprocess data\n")
train_raw <- as.data.frame(train_raw) # Convert data table to data frame

N <- ncol(train_raw)
train_raw$NACount_Init_N <- rowSums(is.na(train_raw)) / N 
train_raw$NACount_Init <- rowSums(is.na(train_raw))

# Idea from https://www.kaggle.com/sinaasappel/bnp-paribas-cardif-claims-management/exploring-paribas-data
levels(train_raw$v3)[1] <- NA #to remove the "" level and replace by NA
levels(train_raw$v22)[1] <- NA
levels(train_raw$v30)[1] <- NA
levels(train_raw$v31)[1] <- NA
levels(train_raw$v52)[1] <- NA
levels(train_raw$v56)[1] <- NA
levels(train_raw$v91)[1] <- NA
levels(train_raw$v107)[1] <- NA
levels(train_raw$v112)[1] <- NA
levels(train_raw$v113)[1] <- NA
levels(train_raw$v125)[1] <- NA

# Small feature addition - Count NA percentage
N <- ncol(train_raw)
train_raw$NACount_N <- rowSums(is.na(train_raw)) / N 
train_raw$NACount <- rowSums(is.na(train_raw))

feature.names <- names(train_raw)

#from Artem
highCorrRemovals  <- c('v8','v23','v25','v31','v36','v37',
                       'v46','v51','v53','v54','v63','v73',
                       'v75','v79','v81','v82','v89','v92',
                       'v95','v105','v107','v108','v109','v110',
                       'v116','v117','v118','v119','v123','v124',
                       'v128')

train_raw <- train_raw[,-which(names(train_raw) %in% highCorrRemovals)]

cat("assuming text variables are categorical & replacing them with numeric ids\n")
cat("re-factor categorical vars & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(train_raw[[f]])=="character" || class(train_raw[[f]])=="factor") {
    train_raw[[f]] <- as.integer(factor(train_raw[[f]]))
  }
}

feature.names <- names(train_raw)
# make feature of counts of zeros factor
train_raw$ZeroCount <- rowSums(train_raw[,feature.names]== 0) / N
train_raw$Below0Count <- rowSums(train_raw[,feature.names] < 0) / N

#Splitting DataSets
trainIndex <- createDataPartition(train_raw$ID, p = .8,
                                  list = FALSE,
                                  times = 1)

trainSet <- train_raw[ trainIndex,]
testSet  <- train_raw[-trainIndex,]
testSet$ID <- NULL
trainSet$ID <- NULL

#Labels
train.y <- trainSet$target
trainSet$target <- NULL
test.y <- testSet$target
testSet$target <- NULL

#converting sets
xgtrain = xgb.DMatrix(as.matrix(trainSet), label = train.y, missing=NA)
xgtest = xgb.DMatrix(as.matrix(testSet), label = test.y, missing=NA)

#Initialize analytics
bestMean <- numeric()
bestIter <- numeric()
nfold <- integer()
tree_depth <- integer()
min_child_weight <- integer()
subsample <- numeric()
colsample_bytree <- numeric()

for (k in c(9)) {
  for (m in c(1.1)) {
    for (l in c(0.9)) {
      for (o in c(0.7)) {
param0 <- list(
  # some generic, non specific params
  "objective"  = "binary:logistic",
  "eval_metric" = "logloss"
  ,"eta" = 0.01
  ,"max_depth" = k
  ,"min_child_weight" = m
  ,"subsample" = l
  ,"colsample_bytree" = o
  #,"colsample_bylevel" = 0.6
)
set.seed(3456)
model = xgb.train(params = param0
                  , watchlist = list('test' = xgtest)
                  , nrounds = 5000
                  #, nfold = 2
                  , data = xgtrain
                  , early.stop.round = 20
                  #, maximize = FALSE
                  , nthread = 8
                  , verbose = 1
                  #, prediction = TRUE
)

bestMean <- c(bestMean, model$bestScore)
bestIter <- c(bestIter, model$bestInd)
tree_depth <- c(tree_depth, k)
min_child_weight <- c(min_child_weight, m)
subsample <- c(subsample, l)
colsample_bytree <- c(colsample_bytree, o)

#cat("\ntree_depth:",k," min_child",m)
cat("\nBestMean:",tail(bestMean,n=1), "Iteration:", tail(bestIter,n=1),"\n")
print(model[tail(bestIter,n=1)])
#rm(model)
gc()
  }
  }
  }
}

nfold <- c(rep(1,length(bestMean)))
eta <- c(rep(0.01,length(bestMean)))

testResult <- data.frame(bestIter, bestMean, tree_depth, min_child_weight,nfold,eta,subsample,colsample_bytree)
#show(testResult)
AllTestResult <- rbind(AllTestResult, testResult)
write.csv(AllTestResult, paste(path,"AllTestResult.csv",sep=""), row.names=F, quote=F)
write.csv(testResult, paste(path,"TestResult12_04_2.csv",sep=""), row.names=F, quote=F)
