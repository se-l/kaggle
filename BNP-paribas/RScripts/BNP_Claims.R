library(dplyr) #manipulation
library(ggplot2) #visualization
library(ggthemes) # visualization
library(scales) # visualization
library(mice) # imputation
library(randomForest) # classification algorithm
library(extraTrees)
library(Metrics)
library(readr)

path <- "C:\\0Sebs_other_data\\Machine_Learning\\Kaggle\\BNP_Claims\\"

#sample_sub <- read.csv(paste(path,"sample_submission.csv", sep=""))
train <- read_csv(paste(path,"train.csv", sep=""))
test <- read_csv(paste(path,"test.csv", sep=""))
#full  <- bind_rows(train, test) # bind training & test data

#Count and proportions of claims eligible for accelerated approval
table(train$target)
prop.table(table(train$target))

##CLEANING
#train_ct <- train[-c('ID','target','v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128')]
# drop <- c("ID")
# train <-train[ ,!(names(train) %in% drop)]
# test <-test[ ,!(names(test) %in% drop)]

#small set for test
# train <- train[1:1000, ]
# test <- test[1:1000, ]

#Removing NAs
train[is.na(train)]   <- -999
test[is.na(test)]   <- -999
feature.names <- names(train)
nofeature=c('ID','target','v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128')
j=which(feature.names %in%  nofeature)
feature.names=feature.names[-j]

for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

tra<-train[ ,feature.names]
tes<-test[ ,feature.names]


x=tra
y=train$target

# Set a random seed
set.seed(754)

# Build the model (note: all possible variables are used)

clf_et=extraTrees(tra, y, ntree=500,numRandomCuts=1)
q_et <- predict(clf_et, tra)
#rf_model <- randomForest(x = feature.names, y=y, data=tra)

q_et[q_et>=1]=0.99999999
q_et[q_et<=0]=0.00000001

cat('\n',logLoss(train$target, q_et),'\n')


p_et <- predict(clf_et, tes)

# Show model error
plot(rf_model, ylim=c(0,0.36))
legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)

# Get importance
importance    <- importance(rf_model)
varImportance <- data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))

# Create a rank variable based on importance
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

# Use ggplot2 to visualize the relative importance of variables
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() + 
  theme_few()

##PREDICTION
# Predict using the test set
prediction <- predict(rf_model, test)

# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
solution <- data.frame(ID = test$ID, PredictedProb = prediction)

# Write the solution to file
write.csv(solution, file = 'rf_mod_Solution.csv', row.names = F)