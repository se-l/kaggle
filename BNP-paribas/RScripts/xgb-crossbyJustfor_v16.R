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
pred_raw <- fread(paste(path,"test.csv",sep=""), stringsAsFactors=TRUE) 

#Saving Label
train.y_raw <- train_raw$target
train_raw$target <- NULL

#Data Preparation
all_data <- rbind(train_raw,pred_raw)
n <- nrow(train_raw)
cat("Preprocess data\n")
all_data <- as.data.frame(all_data) # Convert data table to data frame

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

#Splitting Pred and Train Set
train <- all_data[1:n,]
pred <- all_data[(n+1):nrow(all_data),] 

#Splitting DataSets
trainIndex <- createDataPartition(train$ID, p = .5,
                                  list = FALSE,
                                  times = 1)

test  <- train[-trainIndex,]
train <- train[ trainIndex,]

#Labels
test.y <- train.y_raw[-trainIndex]
train.y <- train.y_raw[ trainIndex]

#Removing ID
train$ID <- NULL
test$ID <- NULL
pred_id <- pred$ID
pred$ID <- NULL

#converting sets
xgtrain = xgb.DMatrix(as.matrix(train), label = train.y, missing=NA)
xgtest = xgb.DMatrix(as.matrix(test), label = test.y, missing=NA)
xgpred = xgb.DMatrix(as.matrix(pred), missing=NA)

ensemble <- rep(0, nrow(pred))

#Initialize analytics
bestMean <- numeric()
bestIter <- numeric()
nfold <- integer()
tree_depth <- integer()
min_child_weight <- integer()
subsample <- numeric()
colsample_bytree <- numeric()

modelPerSet <- 1

for (set in seq(1,2)) {
  for (modelPerSet in seq(1,6)){
for (k in c(7)) {
  for (m in c(1.5)) {
    for (l in c(0.9)) {
      for (o in c(0.9)) {
        
        if (set==1){
          watchlist = list('test' = xgtest)
          data = xgtrain
        }
        else {
          watchlist = list('test2' = xgtrain)
          data = xgtest
        }
param0 <- list(
  # some generic, non specific params
  "objective"  = "binary:logistic",
  "eval_metric" = "logloss"
  ,"eta" = 0.05
  ,"max_depth" = k
  ,"min_child_weight" = m
  ,"subsample" = l
  ,"colsample_bytree" = o
  #,"colsample_bylevel" = 0.6
)
set.seed(3456)
model = xgb.train(params = param0
                  , watchlist = watchlist
                  , nrounds = 1000
                  #, nfold = 2
                  , data = data
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

cat("\nSet:",set," modelPerSet",modelPerSet)
cat("\ntree_depth:",k," min_child",m)
cat("\nBestMean:",tail(bestMean,n=1), "Iteration:", tail(bestIter,n=1),"\n")
print(tail(bestIter,n=1))
#make prediction
p <- predict(model, xgpred)
ensemble <- ensemble + p
#rm(model)
gc()
  }
  }
  }
}
}
}

nfold <- c(rep(2,length(bestMean)))
eta <- c(rep(0.05,length(bestMean)))

# Prepare submission

submission <- read.csv(paste(path,"sample_submission.csv",sep=""))
submission$PredictedProb <- ensemble/(set*modelPerSet)
write.csv(submission, "BNPCardif-Rxgb-O-16.csv", row.names=F, quote=F)
summary(submission$PredictedProb)

testResult <- data.frame(bestIter, bestMean, tree_depth, min_child_weight,nfold,eta,subsample,colsample_bytree)
#show(testResult)
#AllTestResult <- rbind(AllTestResult, testResult)
print(testResult)
#write.csv(AllTestResult, paste(path,"AllTestResultTest.csv",sep=""), row.names=F, quote=F)
write.csv(testResult, paste(path,"TestResult12_04_3.csv",sep=""), row.names=F, quote=F)
