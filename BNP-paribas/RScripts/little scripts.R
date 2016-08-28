train_raw <- as.data.frame((train_raw))
feature.names <- names(train_raw)
for (f in feature.names) {
  if (class(train_raw[[f]])=="integer") {
    print(f)
    #all_data[[f]] <- as.integer(factor(all_data[[f]]))
  }
}
all_data <- all_data_t
col_num_rounded <- NULL
#col_num <- NULL
for (f in feature.names) {
  if (class(all_data[[f]]) == "numeric") {
    #all_data[[f]] <- as.integer(factor(all_data[[f]]))
    #print(summary(all_data[[f]]))
    print(f)
    #print(summary(all_data[[f]]))
    all_data[[f]]<- round(all_data[[f]],2)
    
    n_vals <- length(unique(all_data[[f]]))
    #print(nlevels(all_data[[f]]))
    col_num_rounded <- c(col_num_rounded,f)
    col_num_rounded <- c(col_num_rounded, n_vals)
  }
}
col_num_rounded_2 <- col_num_rounded

str(all_data$v78)

as.data.frame(table(all_data$v93))
summary(all_data$v93)
hist(all_data$v35)

library(outliers)
hist(all_data$v1)
chisq.out.test(all_data$v1, variance = var(all_data$v1), opposite=TRUE)
chisq.out.test(all_data$v1, variance = var(all_data$v1), opposite=FALSE)

