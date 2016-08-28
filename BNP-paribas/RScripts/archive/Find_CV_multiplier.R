library(caret)
library(data.table) #Faster reading
library(xgboost)

path <- "C:\\0Sebs_other_data\\Machine_Learning\\Kaggle\\BNP_Claims\\"

# Start the clock!
start_time <- Sys.time()
set.seed(3456)

cat("reading the train and test data\n")
# Read train and test
train_raw <- fread(paste(path,"train.csv",sep=""), stringsAsFactors=TRUE)

#Data Preparation
y <- train_raw$target
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

#Splitting sets
trainIndex <- createDataPartition(train_raw$ID, p = .75,
                                  list = FALSE,
                                  times = 1)

Train_RSplit <- train_raw[ trainIndex,]
Test_RSplit  <- train_raw[-trainIndex,]
Test_RSplit$ID <- NULL

val.y <- y[-trainIndex]
train.y <- y[trainIndex]
xgtest <- xgb.DMatrix(as.matrix(Test_RSplit), label = val.y, missing=NA)

#Functions:
doTest <- function(param0, iter) {
  watchlist <- list('val' = xgtest, 'train' = xgtrain)
  model = xgb.train(
    nrounds = iter,
    params = param0, 
    data = xgtrain, 
    watchlist = watchlist, 
    print.every.n = 20, 
    nthread = 8,
    verbose = 1,
    early.stop.round = 20
  )
  show(model$bestInd)
  show(model$bestScore)
  p <- predict(model, xgtest)
  rm(model)
  gc()
  p
}
param0 <- list(
  # some generic, non specific params
  "objective"  = "binary:logistic",
  "eval_metric" = "logloss",
  "eta" = 0.03,
  "subsample" = 0.8,
  "colsample_bytree" = 0.9,
  #"colsample_bylevel" = 0.6,
  "min_child_weight" = 1,
  "max_depth" = 10
)
#Looping over increasing training sets
for (i in 1:4){
  trainIndex2 <- createDataPartition(Train_RSplit$ID, p = (i*0.25),
                                    list = FALSE,
                                    times = 1)
  
  xgtrain = xgb.DMatrix(as.matrix(Train_RSplit[trainIndex2,-Train_RSplit$ID]), label = train.y[trainIndex2], missing=NA)
  #xgtrain$ID <- NULL
  watchlist <- list('val' = xgtest, 'train' = xgtrain)
  model = xgb.train(
    nrounds = 100
    , params = param0
    , data = xgtrain
    , watchlist = watchlist
    , print.every.n = 20
    , nthread = 8
    #, verbose = 1
    , early.stop.round = 20
  )
  gc()
  
  best <- min(model$test.logloss.mean)
  bestIter <- which(model$test.logloss.mean==best)
  cat("\nNRow(Train) set",i ,":", length(trainIndex2),"\n")
  cat("Best:",best, bestIter,"\n")
  print(model[bestIter])
}