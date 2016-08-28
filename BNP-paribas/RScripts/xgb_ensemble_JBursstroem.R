doAnalysis <- function(train.n, val.n, y, train, param0, iter=100000, early.stop=T) { 
  
  xgtrain <- xgb.DMatrix(as.matrix(train[train.n, ]), label = y[train.n])
  xgval = xgb.DMatrix(as.matrix(train[val.n, ]), label = y[val.n]) 
  watchlist <- list('val' = xgval, 'train' = xgtrain) 
  
  if (early.stop) 
    { model = xgb.train( 
        nrounds = iter , 
        params = param0 , 
        data = xgtrain, 
        watchlist = watchlist , 
        early.stop.round = 200 , 
        print.every.n = 125 , 
        nthread = 8 ) 
    iter <- model$bestInd 
    show(model$bestScore) } 
  else { 
    model = xgb.train( 
      nrounds = iter , 
      params = param0 , 
      data = xgtrain , 
      watchlist = watchlist , 
      print.every.n = 125 , 
      nthread = 8 ) } 
  p <- predict(model, xgval, ntreelimit=iter) 
  rm(model) 
  gc() 
  p } 
param0 <- list( 
  # general , non specific params - just guessing
  "objective" = "binary:logistic" ,
  "eval_metric" = "logloss" ,
  "eta" = 0.01 ,
  "subsample" = 0.8 ,
  "colsample_bytree" = 0.8 ,
  "colsample_bylevel" = 0.6 ,
  "min_child_weight" = 0 ,
  "max_depth" = 10 )

# train.n is train set, val.n is validation set, ttrain is cbind(train, test) only need train really here

preds.ens <- c() 
cv <- 0 
for (i in 1:5) { 
  show(i) 
  val.n <- rand.set[folds==i] 
  train.n <- setdiff(1:n, val.n) 
  p <- doAnalysis(train.n, 
                  val.n, 
                  y, 
                  ttrain[1:n, ], 
                  param0) 
  preds.ens[val.n] <- p 
  show(e <- mean(ll(y[val.n], p))) 
  cv <- cv + e 
  show(cv/i) 
  #write.csv(preds.ens, "BNP_20x5_preds_1.csv") 
}