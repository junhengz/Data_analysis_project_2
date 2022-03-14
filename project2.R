library(ggplot2)
library(dplyr)
library(MASS)
library(glmnet)
library(pls)
library(mgcv)
library(kableExtra)
library(nnet)
library(rpart)
library(rpart.plot)
library(randomForest)
library(gbm)

###############
###read data###
###############

#please set working dirctory before reading the data
df <- read.csv("Data2021_final.csv") 
summary(df)


#############
###setting###
#############

#set random seed = 1 for reproduction
set.seed(1)


######################
###Helper functions###
######################

#calculate for MSPE
get.MSPE = function(Y, Y.hat){
  return(mean((Y - Y.hat)^2))
}

#get fold for cv
get.folds = function(n, K) {
  ### Get the appropriate number of fold labels
  n.fold = ceiling(n / K) # Number of observations per fold (rounded up)
  fold.ids.raw = rep(1:K, times = n.fold)
  fold.ids = fold.ids.raw[1:n]
  ### Shuffle the fold labels
  folds.rand = fold.ids[sample.int(n)]
  return(folds.rand)
}

#Rescale x1 so that columns of x2 range from 0 to 1
rescale <- function(x1,x2){
  for(col in 1:ncol(x1)){
    a <- min(x2[,col])
    b <- max(x2[,col])
    x1[,col] <- (x1[,col]-a)/(b-a)
  }
  x1
}



####################
###model training###
####################

###model 1
###Least square regression
mod.fit1 = lm(Y~., data = df)
summary(mod.fit1)

###model 2
###Stepwise regression
mod.fit2.start = lm(Y~., data = df)
mod.fit2 <- stepAIC(mod.fit2.start, direction = "both", trace = FALSE)
summary(mod.fit2)

###model 3
###Ridge regression
lambda.vals = seq(from = 0, to = 100, by = 0.05)
mod.fit3 = lm.ridge(Y~., lambda = lambda.vals,data = df)
#Get optimal lambda value
ind.min.GCV = which.min(mod.fit3$GCV)
lambda.min = lambda.vals[ind.min.GCV]
all.coefs.ridge = coef(mod.fit3)
all.coefs.ridge[ind.min.GCV,]

###model 4 and 5
###LASSO regression
data.mat.raw = model.matrix(Y ~ ., data = df)
data.mat = data.mat.raw[,-1]
mod.fit4_5 = cv.glmnet(data.mat, df$Y)
#Extract optimal lambda values
lambda.min = mod.fit4_5$lambda.min
lambda.1se = mod.fit4_5$lambda.1se
#Extract fitted coefficients
coef.min = predict(mod.fit4_5, s = lambda.min, type = "coef")
coef.1se = predict(mod.fit4_5, s = lambda.1se, type = "coef")
#Combine and print coefficient vectors
coef.lasso = cbind(coef.min, coef.1se)
coef.lasso = t(coef.lasso)
rownames(coef.lasso) = c("min", "1se")
print(signif(coef.lasso, 3))

###model 6
###Partial Least Squares
mod.fit6 = plsr(Y ~ ., data = df, validation = "CV", segments = 10)
#Get optimal number of folds
CV.pls = mod.fit6$validation
PRESS.pls = CV.pls$PRESS
which.min(PRESS.pls)
summary(mod.fit6)

###model 7
###Generalized additive models
mod.fit7 = gam(Y ~ s(X1) + s(X2) + s(X3) + s(X4) + s(X5) + s(X6) + s(X7) + s(X8) + s(X9) + s(X10) + s(X11) + s(X12) + s(X13) + s(X14) + s(X15), data = df)
summary(mod.fit7)


###Model 8
###Neural net
X.raw = dplyr::select(df, -Y)
X.nnet = rescale(X.raw, X.raw)
mod.fit8 = nnet(x=X.nnet,y=df$Y,size = 2,trace = FALSE)
summary(mod.fit8)


###Model 9
###Full tree
mod.fit9 = rpart(Y ~ ., data = df, cp = 0)
#Check tree graph
prp(mod.fit9, type = 1, extra = 1)


###Model 10
###Minimum CV error tree
#Get the CP table
info.tree = mod.fit9$cptable
#Get minimum CV error and corresponding CP value
ind.best = which.min(info.tree[,"xerror"])
CV.best = info.tree[ind.best, "xerror"]
CP.best = info.tree[ind.best, "CP"]
#Get the geometric mean of best CP with one above it
if(ind.best == 1){
  #If minimum CP is in row 1, store this value
  CP.GM = CP.best
} else{
  #If minimum CP is not in row 1, average this with the value from the
  #row above it.
  #Value from row above
  CP.above = info.tree[ind.best-1, "CP"]
  #(Geometric) average
  CP.GM = sqrt(CP.best * CP.above)
}
#Fit minimum CV error tree
mod.fit10 = prune(mod.fit9, cp = CP.best)
#Make plot
prp(mod.fit10, type = 1, extra = 1)


###Model 11
###1 SE rule CV tree
#Get 1se rule CP value
err.min = info.tree[ind.best, "xerror"]
se.min = info.tree[ind.best, "xstd"]
threshold = err.min + se.min
ind.1se = min(which(info.tree[1:ind.best,"xerror"] < threshold))
#Take geometric mean with superior row
CP.1se.raw = info.tree[ind.1se, "CP"]
if(ind.1se == 1){
  #If best CP is in row 1, store this value
  CP.1se = CP.1se.raw
} else{
  #If best CP is not in row 1, average this with the value from the
  #row above it.
  #Value from row above
  CP.above = info.tree[ind.1se-1, "CP"]
  #(Geometric) average
  CP.1se = sqrt(CP.1se.raw * CP.above)
}
#Prune the tree
mod.fit11 = prune(mod.fit9, cp = CP.1se)
#Make plot
prp(mod.fit11, type = 1, extra = 1)


###Model 12
###Random forest
mod.fit12 = randomForest(Y ~ ., data = df, importance = T)
#check graph and importance
plot(mod.fit12)
varImpPlot(mod.fit12)
#find best tunning variables
#Set parameter values
all.mtry = 1:15
all.nodesize = 1:8
all.pars = expand.grid(mtry = all.mtry, nodesize = all.nodesize)
n.pars = nrow(all.pars)
#Number of times to replicate process. OOB errors are based on bootstrapping
M = 5
#Create container for OOB MSPEs
OOB.MSPEs = array(0, dim = c(M, n.pars))
for(i in 1:n.pars){
  #Get current parameter values
  this.mtry = all.pars[i,"mtry"]
  this.nodesize = all.pars[i,"nodesize"]
  
  #Fit random forest models for each parameter combination
  for(j in 1:M){
    #Fit model using current parameter values
    fit.rf = randomForest(Y ~ ., data = df, importance = F,
                          mtry = this.mtry, nodesize = this.nodesize)
    #Get OOB predictions and MSPE, then store MSPE
    OOB.pred = predict(fit.rf)
    OOB.MSPE = get.MSPE(df$Y, OOB.pred)
    OOB.MSPEs[j, i] = OOB.MSPE 
  }
}
names.pars = paste0(all.pars$mtry,"-",
                    all.pars$nodesize)
colnames(OOB.MSPEs) = names.pars
OOB.RMSPEs = apply(OOB.MSPEs, 1, function(W) W/min(W))
OOB.RMSPEs = t(OOB.RMSPEs)
boxplot(OOB.RMSPEs, las = 2, main = "RMSPE Boxplot")
#Based on the RMSPE boxplot, the model with mtry=4 and nodesize=7 looks best to me
#fit best model
mod.fit12 = randomForest(Y ~ ., data = df, importance = T,mtry = 3, nodesize = 2)
#check graph and importance
plot(mod.fit12)
varImpPlot(mod.fit12)


###Model 13
###boosting
mod.fit13 <- gbm(data=df, Y~., distribution="gaussian")
summary(mod.fit13)
#use 2 reps of 5-fold CV to find best tunning variable
V=5
R=2 
n2 = nrow(df)
# Create the folds and save in a matrix
folds = matrix(NA, nrow=n2, ncol=R)
for(r in 1:R){
  folds[,r]=floor((sample.int(n2)-1)*V/n2) + 1
}
shr = c(.001,.005,.025,.125)
dep = 1:6
trees = 10000
NS = length(shr)
ND = length(dep)
gb.cv = matrix(NA, nrow=ND*NS, ncol=V*R)
opt.tree = matrix(NA, nrow=ND*NS, ncol=V*R)

qq = 1
for(r in 1:R){
  for(v in 1:V){
    pro.train = df[folds[,r]!=v,]
    pro.test = df[folds[,r]==v,]
    counter=1
    for(d in dep){
      for(s in shr){
        pro.gbm <- gbm(data=pro.train, Y~., distribution="gaussian", 
                       n.trees=trees, interaction.depth=d, shrinkage=s, 
                       bag.fraction=0.8)
        treenum = min(trees, 2*gbm.perf(pro.gbm, method="OOB", plot.it=FALSE))
        opt.tree[counter,qq] = treenum
        preds = predict(pro.gbm, newdata=pro.test, n.trees=treenum)
        gb.cv[counter,qq] = mean((preds - pro.test$Y)^2)
        counter=counter+1
      }
    }
    qq = qq+1
  }  
}
parms = expand.grid(shr,dep)
row.names(gb.cv) = paste(parms[,2], parms[,1], sep="|")
row.names(opt.tree) = paste(parms[,2], parms[,1], sep="|")
(mean.tree = apply(opt.tree, 1, mean))
(mean.cv = sqrt(apply(gb.cv, 1, mean)))
min.cv = apply(gb.cv, 2, min)
x11(h=7,w=10,pointsize=8)
boxplot(sqrt(t(gb.cv)/min.cv), use.cols=TRUE, las=2, 
        main="GBM Fine-Tuning Variables and Node Sizes")
#Based on the RMSPE boxplot, the model with dep = 6 and shr = 0.025 looks best to me
#fit best model
mod.fit13 <- gbm(data=data.train, Y~., distribution="gaussian", 
                 n.trees=10000, interaction.depth=6, shrinkage=0.025)
n.tree = gbm.perf(mod.fit13, plot.it = F)*2
summary(mod.fit13)


######################
###Model evaluation###
######################

#use 10 fold cv
K = 10 #Number of folds

#Container for CV MSPEs
all.models = c("LS", "Step", "Ridge", "LAS-Min", "LAS-1se", "PLS", "GAM","NNet", "Full-Tree", "Min-Tree", "1SE-Tree", "RanF", "Boosting")
CV.MSPEs = array(0, dim = c(length(all.models), K))
rownames(CV.MSPEs) = all.models
colnames(CV.MSPEs) = 1:K
#Get CV fold labels
n = nrow(df)
folds = get.folds(n, K)

#Define parameters for ridge model
lambda.vals = seq(from = 0, to = 100, by = 0.05)

#Define parameters for nnet model
nodes = c(1, 3, 5, 7, 9)
shrinkage = c(0.001, 0.1, 0.5, 1, 2)
all.pars = expand.grid(nodes = nodes, shrink = shrinkage)
n.pars = nrow(all.pars)
K.inner = 5 # Number of folds for inner CV
M = 10 # Number of times to re-fit neural net
best.pars = rep(0, times = K) #Container for tuning parameters for nnet


#Perform cross-validation
for (i in 1:K) {
  #Get training and validation sets
  data.train = df[folds != i, ]
  data.valid = df[folds == i, ]
  Y.train = data.train$Y
  Y.valid = data.valid$Y
  
  mat.train.int = model.matrix(Y ~ ., data = data.train)
  mat.train = mat.train.int[,-1]
  mat.valid.int = model.matrix(Y ~ ., data = data.valid)
  mat.valid = mat.valid.int[,-1]
  
  X.train.raw = dplyr::select(data.train, -Y)
  X.train.nnet = rescale(X.train.raw, X.train.raw)
  
  X.valid.raw = dplyr::select(data.valid, -Y)
  X.valid.nnet = rescale(X.valid.raw, X.train.raw)
  
  
  ##########
  ### LS ###
  ##########
  mod.fit1 = lm(Y~., data = data.train)
  pred.ls = predict(mod.fit1, data.valid)
  MSPE.ls = get.MSPE(Y.valid, pred.ls)
  CV.MSPEs["LS", i] = MSPE.ls
  
  ############
  ### Step ###
  ############
  mod.fit2.start = lm(Y~., data = data.train)
  mod.fit2 <- stepAIC(mod.fit2.start, direction = "both", trace = 0)
  pred.step = predict(mod.fit2, data.valid)
  MSPE.step = get.MSPE(Y.valid, pred.step)
  CV.MSPEs["Step", i] = MSPE.step
  
  #############
  ### Ridge ###
  #############
  ### Fit ridge regression
  ### We already definted lambda.vals. No need to re-invent the wheel
  mod.fit3 = lm.ridge(Y ~ ., lambda = lambda.vals,
                       data = data.train)
  ### Get optimal lambda value
  ind.min.GCV = which.min(mod.fit3$GCV)
  lambda.min = lambda.vals[ind.min.GCV]
  ### Get coefficients for optimal model
  all.coefs.ridge = coef(mod.fit3)
  coef.min.ridge = all.coefs.ridge[ind.min.GCV,]
  ### Get predictions and MSPE on validation set
  pred.ridge = mat.valid.int %*% coef.min.ridge
  pred.ridge = as.numeric(pred.ridge)
  MSPE.ridge = get.MSPE(Y.valid, pred.ridge)
  CV.MSPEs["Ridge", i] = MSPE.ridge
  
  #############
  ### LASSO ###
  #############
  ### Fit model
  mod.fit4_5 = cv.glmnet(mat.train, Y.train)
  ### Get optimal lambda values
  lambda.min = mod.fit4_5$lambda.min
  lambda.1se = mod.fit4_5$lambda.1se
  ### Get predictions
  pred.min = predict(mod.fit4_5, mat.valid, lambda.min)
  pred.1se = predict(mod.fit4_5, mat.valid, lambda.1se)
  ### Get and store MSPEs
  MSPE.min = get.MSPE(Y.valid, pred.min)
  MSPE.1se = get.MSPE(Y.valid, pred.1se)
  CV.MSPEs["LAS-Min", i] = MSPE.min
  CV.MSPEs["LAS-1se", i] = MSPE.1se
  
  #############################
  ### Partial Least Squares ###
  #############################
  mod.fit6 = plsr(Y ~ ., data = data.train, validation = "CV", segments = 10)
  #Get optimal number of folds
  CV.pls = mod.fit6$validation
  PRESS.pls = CV.pls$PRESS
  n.comps =which.min(PRESS.pls)
  #Get predictions and MSPE
  pred.pls = predict(mod.fit6, data.valid, ncomp = n.comps)
  MSPE.pls = get.MSPE(Y.valid, pred.pls)
  CV.MSPEs["PLS", i] = MSPE.pls
  
  ###########
  ### GAM ###
  ###########
  mod.fit7 = gam(Y ~ s(X1) + s(X2) + s(X3) + s(X4) + s(X5) + s(X6) + s(X7) + s(X8) + s(X9) + s(X10) + s(X11) + s(X12) + s(X13) + s(X14) + s(X15), data = data.train)
  #Get predictions and MSPE
  pred.gam = predict(mod.fit7, data.valid)
  MSPE.gam = get.MSPE(Y.valid, pred.gam)
  CV.MSPEs["GAM", i] = MSPE.gam
  
  
  ##########
  ###NNet###
  ##########
  #Container for tuning MSPEs
  tuning.MSPEs = array(0, dim = c(nrow(all.pars), K.inner))
  #Get CV fold labels for innner cv
  n.inner = nrow(data.train)
  folds.inner = get.folds(n.inner, K.inner)
  #Perform inner cross-validation
  for (j in 1:K.inner) {
    data.train.inner = data.train[folds.inner != j,]
    X.train.inner.raw = dplyr::select(data.train.inner, -Y)
    X.train.inner = rescale(X.train.inner.raw, X.train.inner.raw)
    Y.train.inner = data.train.inner$Y
    
    data.valid.inner = data.train[folds.inner == j,]
    X.valid.inner.raw = dplyr::select(data.valid.inner, -Y)
    X.valid.inner = rescale(X.valid.inner.raw, X.train.inner.raw)
    Y.valid.inner = data.valid.inner$Y
    
    #Fit nnets with each parameter combination
    for(l in 1:n.pars){
      #Get current parameter values
      this.n.hidden = all.pars[l, 1]
      this.shrink = all.pars[l, 2]
      #containers to store refit models and their errors.
      all.nnets = list(1:M)
      all.SSEs = rep(0, times = M)
      #Fit each model multiple times
      for (ii in 1:M) {
        #Fit model
        fit.nnet = nnet(
          x=X.train.inner,
          y=Y.train.inner,
          linout = TRUE,
          size = this.n.hidden,
          decay = this.shrink,
          maxit = 500,
          trace = FALSE
        )
        #Get model SSE
        SSE.nnet = fit.nnet$value
        #Store model and its SSE
        all.nnets[[ii]] = fit.nnet
        all.SSEs[ii] = SSE.nnet
      }
      #Get best fit using current parameter values
      ind.best = which.min(all.SSEs)
      fit.nnet.best = all.nnets[[ind.best]]
      #Get predictions and MSPE, then store MSPE
      pred.nnet = predict(fit.nnet.best, X.valid.inner)
      MSPE.nnet = get.MSPE(Y.valid.inner, pred.nnet)
      tuning.MSPEs[l, j] = MSPE.nnet # Be careful with indices
    }
  }
  #Get best tuning MSPEs by minimizing average
  ave.tune.MSPEs = apply(tuning.MSPEs, 1, mean)
  best.comb = which.min(ave.tune.MSPEs)
  best.pars[i] = best.comb
  
  #Get chosen tuning parameter values
  best.n.hidden = all.pars[best.comb, "nodes"]
  best.shrink = all.pars[best.comb, "shrink"]
  #containers to store refit models and their errors.
  all.nnets = list()
  all.SSEs = rep(0, times = M)
  #Fit each model multiple times
  for (ii in 1:M) {
    #Fit model
    fit.nnet = nnet(
      x=X.train.nnet,
      y=Y.train,
      linout = TRUE,
      size = best.n.hidden,
      decay = best.shrink,
      maxit = 500,
      trace = F
    )
    #Get model SSE
    SSE.nnet = fit.nnet$value
    #Store model and its SSE
    all.nnets[[ii]] = fit.nnet
    all.SSEs[ii] = SSE.nnet
  }
  #Get best fit using current parameter values
  ind.best = which.min(all.SSEs)
  fit.best = all.nnets[[ind.best]]
  #Get predictions and MSPE
  pred = predict(fit.best, X.valid.nnet)
  this.MSPE = get.MSPE(Y.valid, pred)
  CV.MSPEs["NNet", i] = this.MSPE
  
  
  #################
  ### Full Tree ###
  #################
  mod.fit9 = rpart(Y ~ ., data = data.train, cp = 0)
  #Get the CP table
  info.tree = mod.fit9$cptable
  #Get predictions
  pred.full = predict(mod.fit9, data.valid)
  MSPE.full = get.MSPE(Y.valid, pred.full)
  CV.MSPEs["Full-Tree", i] = MSPE.full
  
  ###################
  ### Min CV Tree ###
  ###################
  #Get minimum CV error and corresponding CP value
  ind.best = which.min(info.tree[, "xerror"])
  CV.best = info.tree[ind.best, "xerror"]
  CP.best = info.tree[ind.best, "CP"]
  #Get the geometric mean of best CP with one above it
  if (ind.best == 1) {
    #If minimum CP is in row 1, store this value
    CP.GM = CP.best
  } else{
    #If minimum CP is not in row 1, average this with the value from the
    #row above it.
    #Value from row above
    CP.above = info.tree[ind.best - 1, "CP"]
    #(Geometric) average
    CP.GM = sqrt(CP.best * CP.above)
  }
  #Fit minimum CV error tree
  mod.fit10 = prune(mod.fit9, cp = CP.best)
  #Get predictions and MSPE
  pred.min = predict(mod.fit10, data.valid)
  MSPE.min = get.MSPE(Y.valid, pred.min)
  CV.MSPEs["Min-Tree", i] = MSPE.min
  
  ########################
  ### 1SE Rule CV Tree ###
  ########################
  #Get 1se rule CP value
  err.min = info.tree[ind.best, "xerror"]
  se.min = info.tree[ind.best, "xstd"]
  threshold = err.min + se.min
  ind.1se = min(which(info.tree[1:ind.best, "xerror"] < threshold))
  #Take geometric mean with superior row
  CP.1se.raw = info.tree[ind.1se, "CP"]
  if (ind.1se == 1) {
    #If best CP is in row 1, store this value
    CP.1se = CP.1se.raw
  } else{
    #If best CP is not in row 1, average this with the value from the
    #row above it.
    #Value from row above
    CP.above = info.tree[ind.1se - 1, "CP"]
    ### (Geometric) average
    CP.1se = sqrt(CP.1se.raw * CP.above)
  }
  ### Prune the tree
  mod.fit11 = prune(mod.fit9, cp = CP.1se)
  ### Get predictions and MSPE
  pred.1se = predict(mod.fit11, data.valid)
  MSPE.1se = get.MSPE(Y.valid, pred.1se)
  CV.MSPEs["1SE-Tree", i] = MSPE.1se
  
  
  #####################
  ### Random forest ###
  #####################
  mod.fit12 = randomForest(Y ~ ., data = data.train, importance = F,mtry = 7, nodesize = 5)
  pred.rf = predict(mod.fit12, data.valid)
  MSPE.rf = get.MSPE(Y.valid, pred.rf)
  CV.MSPEs["RanF", i] = MSPE.rf
  
  
  ################
  ### Boosting ###
  ################
  mod.fit13 <- gbm(data=data.train, Y~., distribution="gaussian", 
                 n.trees=10000, interaction.depth=5, shrinkage=0.025)
  n.tree = gbm.perf(mod.fit13, plot.it = F)*2
  pred.bst = predict(mod.fit13, data.valid, n.tree)
  MSPE.bst = get.MSPE(Y.valid, pred.bst)
  CV.MSPEs["Boosting", i] = MSPE.bst
}

#Get full-data MSPEs
full.MSPEs = apply(CV.MSPEs, 1, mean)

#MSPE Boxplot
plot.MSPEs = t(CV.MSPEs)
boxplot(plot.MSPEs)

#Compute RMSPEs
plot.RMSPEs = apply(CV.MSPEs, 2, function(W){
  best = min(W)
  return(W/best)
})
plot.RMSPEs = t(plot.RMSPEs)
#RMSPE Boxplot
boxplot(plot.RMSPEs)

#Based on the RMSPE plot and full MSPE values, the Random Forest model is the best model to me
#fit best model
mod.best <- gbm(data=df, Y~., distribution="gaussian", 
                 n.trees=10000, interaction.depth=5, shrinkage=0.025)
#get importance for variables



importance(mod.best)
varImpPlot(mod.best)



##################
### Prediction ###
##################
#please set working dirctory before reading the data
df.train <- read.csv("Data2021_final.csv") 
df.predict <- read.csv("Data2021test_final_noY.csv")
set.seed(1)

#fit best model
mod.best <- gbm(data=df.train, Y~., distribution="gaussian", 
                n.trees=10000, interaction.depth=5, shrinkage=0.025)

#make prediction
n.tree = gbm.perf(mod.best, plot.it = F)*2
predictions = predict(mod.best, df.predict, n.tree)

#output result
write.table(predictions, "test.csv", sep = ",", row.names = F, col.names =F)



summary(mod.best)








