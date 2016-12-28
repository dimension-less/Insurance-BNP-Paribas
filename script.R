library(xgboost)
library(methods)
library(data.table)
library(gtools)

# https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
nrounds <- 260
params <- list("eta"=0.1,
               "max_depth"=6,
               "colsample_bytree"=0.45,
               "objective"="binary:logistic",
               "eval_metric"="logloss")

data.train <- fread("../input/train.csv")
data.test <- fread("../input/test.csv")
  
data.train[is.na(data.train)] <- -1
data.test[is.na(data.test)] <- -1

x.train <- data.train[, -1, with=FALSE]
y.train <- data.train$target
x.test <- data.test[, -1, with=FALSE]

omit.var <- c(1:3,4:9,11,13,15:20,23,25:29,32:33,35:37,39,41:46,48:49,51,53:55,57:61,63:65,67:71,73:74,76:78,80:90,92:107,108:111,115:128,130:131)
x.train <- x.train[, -(omit.var+1), with=FALSE]
x.test <- x.test[, -omit.var, with=FALSE]

char.vars <- colnames(x.train)[sapply(x.train, is.character)]

cmb <- combinations(n=length(char.vars), r=2, v=char.vars)
for(i in 1:nrow(cmb)) {
    x.train[[paste0(cmb[i,1], cmb[i,2])]] <- paste(x.train[[cmb[i,1]]], x.train[[cmb[i,2]]])
    x.test[[paste0(cmb[i,1], cmb[i,2])]] <- paste(x.test[[cmb[i,1]]], x.test[[cmb[i,2]]])
}

cmb <- combinations(n=length(char.vars)-1, r=2, v=char.vars[-match("v22",char.vars)])
for(i in 1:nrow(cmb)) {
    x.train[[paste0("v22", cmb[i,1], cmb[i,2])]] <- paste(x.train[["v22"]], x.train[[cmb[i,1]]], x.train[[cmb[i,2]]])
    x.test[[paste0("v22", cmb[i,1], cmb[i,2])]] <- paste(x.test[["v22"]], x.test[[cmb[i,1]]], x.test[[cmb[i,2]]])
}

cmb <- combinations(n=length(char.vars)-1, r=length(char.vars)-3, v=char.vars[-match("v22",char.vars)])
for(i in 1:nrow(cmb)) {
    new.var.train <- x.train[["v22"]]
    new.var.test <- x.test[["v22"]]
    new.var.name <- "v22"
    for(v in 1:ncol(cmb)) {
        new.var.train <- paste(new.var.train , x.train[[cmb[i,v]]])
        new.var.test <- paste(new.var.test, x.test[[cmb[i,v]]])
        new.var.name <- paste0(new.var.name, cmb[i,v])
    }
    x.train[[new.var.name]] <- new.var.train
    x.test[[new.var.name]] <- new.var.test
}

# replace with target mean
for(var in colnames(x.test)) {
    if(is.character(x.test[[var]])) {
        target.mean <- x.train[, list(pr=mean(target)), by=eval(var)]
        x.test[[var]] <- target.mean$pr[match(x.test[[var]], target.mean[[var]])]
        temp <- rep(NA, nrow(x.train))
        for(i in 1:4) {
            ids.1 <- -seq(i, nrow(x.train), by=4)
            ids.2 <- seq(i, nrow(x.train), by=4)
            target.mean <- x.train[ids.1, list(pr=mean(target)), by=eval(var)]
            temp[ids.2] <- target.mean$pr[match(x.train[[var]][ids.2], target.mean[[var]])]
        }
        x.train[[var]] <- temp
    }
}

x.train[, target:=NULL]
exp.var <- as.data.frame(colnames(x.train))

x.train <- as.matrix(x.train)
x.test <- as.matrix(x.test)
x.train <- matrix(as.numeric(x.train), nrow(x.train), ncol(x.train))
x.test <- matrix(as.numeric(x.test), nrow(x.test), ncol(x.test))

xgb.train <- xgb.DMatrix(x.train, label=y.train)
model.xgb <- xgb.train(param=params, data=xgb.train, nrounds=nrounds, watchlist=list(train=xgb.train), print.every.n=50)
predict <- predict(model.xgb, x.test)
predict <- cbind(ID=data.test$ID, PredictedProb=predict)
write.csv(predict, paste0("Submission.csv"), row.names=FALSE)

exp.var