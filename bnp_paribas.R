train<-read.csv("train.csv")
test<-read.csv("test.csv")
summary(train)
str(train)
sapply(train,function(x){100*sum(is.na(x))/nrow(train)})
missing<-md.pattern(train)
train<-na.omit(train)
train$v22<-NULL
#Splitting into mod_train and mod_test
library(caTools)
spl<-sample.split(train$target,0.7)
mod_train<-subset(train,spl==T)
mod_test<-subset(train,spl==F)
table(train$target)
table(train$v3)
# Apply Boosting
library(xgboost)
library(Matrix)
sparse_train<-sparse.model.matrix(target~.-1-ID,data=mod_train)
sparse_train@Dim
dtrain<-xgb.DMatrix(data=sparse_train,label=mod_train$target)

sparse_mod_test<-sparse.model.matrix(target~.-1-ID,data=mod_test)
sparse_mod_test@Dim
dmod_test<-xgb.DMatrix(data=sparse_mod_test,label=mod_test$target)


watchlist<-list(test=dmod_test)
model_xgb<-xgb.train(data=dtrain,eta=0.001,objective="binary:logistic",nrounds = 10000,max_depth=6,watchlist = watchlist,eval_metric="logloss",early.stop.round = 100)

xgb.importance(feature_names = colnames(mod_train[,3:132]),model = model_xgb,data = dtrain,label = mod_train$target)



#Apply logistic regression
model_glm<-glm(target~.-ID,data=mod_train,family = "binomial")
summary(model_glm)
pred_glm<-predict(model_glm,newdata=mod_test,type="response")


# Developing test data in the same way
str(test)
test$v22<-NULL
options(na.action = "na.pass")
sparse_test<-sparse.model.matrix(~.-1-ID,data=test)
sparse_test@Dim
dtest<-xgb.DMatrix(data=sparse_test)
watchlist<-list(train=dtrain)
model_xgb<-xgb.train(data=dtrain,eta=0.01,objective="binary:logistic",nrounds = 1000,verbose = 2,max_depth=1,watchlist = watchlist,eval_metric="logloss")
pred_xgb<-predict(model_xgb,dtest)
pred_xgb
final<-as.data.frame(cbind(test$ID,pred_xgb))
rownames(final)<-NULL
colnames(final)<-c("ID","PredictedProb")
write.csv(final,file = "final.csv",quote = F,row.names = FALSE)

# Trying by scaling the vectors
