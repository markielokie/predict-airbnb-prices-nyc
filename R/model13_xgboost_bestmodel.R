# Using boosting with XGBoost (best model)
# 20 predictors used in this best model
# This resulted in lowest test RMSE of 60.47 (private) and 53.19 (public)

# Setting up inputs to tune model
trt <- designTreatmentsZ(dframe=train_reduced,
                         varlist=names(train_reduced)[2:21])

newvars <- trt$scoreFrame[trt$scoreFrame$code%in% c("clean", "lev"), "varName"]

train_input <- prepare(treatmentplan=trt,
                       dframe=train_reduced,
                       varRestriction=newvars)

test_input <- prepare(treatmentplan=trt,
                      dframe=test_reduced,
                      varRestriction=newvars)

# Use cross validation to find nrounds
tune_nrounds <- xgb.cv(data=as.matrix(train_input),
                       label=train_reduced$price,
                       nrounds=250,
                       nfold=5,
                       verbose=0)

# Visualize optimal nrounds
ggplot(data=tune_nrounds$evaluation_log, aes(x=iter, y=test_rmse_mean)) +
  geom_point(size=0.4, color="sienna") +
  geom_line(size=0.1, alpha=0.1) +
  theme_bw()

# Find nrounds with the lowest RMSE
which.min(tune_nrounds$evaluation_log$test_rmse_mean)

# Use XGBoost to fit train_reduced data with nrounds = 32
mod13 <- xgboost(data=as.matrix(train_input),
                 label=train_reduced$price,
                 nrounds=32,
                 verbose=0)

# Predict and show train RMSE
pred_mod13 <- predict(mod13, newdata=as.matrix(train_input))
rmse_mod13 <- sqrt(mean((pred_mod13-train_reduced$price)^2))
rmse_mod13

# Predict on test and save as .csv
predicted.prices13 <- predict(mod13, newdata=as.matrix(test_input))

submission <- data.frame(id=test$id, price=predicted.prices13)

write.csv(x=submission, file="Model 13 Predictions.csv", row.names=F)
