
# if (!require("BiocManager", quietly = TRUE))
#       install.packages("BiocManager")
# BiocManager::install("pcaMethods")

library(tidyverse)
library(missCompare)
library(missMethods)
library(caret)
library(blandr)

source("https://raw.githubusercontent.com/aysuvorov/medstats/master/R_scripts/medstats.R")

# +-----------------------------------------------------------------------------
# Constants

RS = 0

# +-----------------------------------------------------------------------------
# Load data

data(mtcars)

mtcars = FactorTransformer(mtcars, 4)

# Split test / train

set.seed(RS)
mtcars$id <- 1:nrow(mtcars)

#use 70% of dataset as training set and 30% as test set 
train <- mtcars |> dplyr::sample_frac(0.70)
test  <- dplyr::anti_join(mtcars, train, by = 'id')

train = train |> select(-id)
test = test |> select(-id)

dim(train) # 22 12
dim(test) # 10 12

# +-----------------------------------------------------------------------------
# Fit step model with `mpg` as Y on full data:

mod = MASS::stepAIC(
    lm(mpg ~ ., train)
)

model_full_data = lm(formula(mod), train) 
model_full_data |> summary() # Adjusted R-squared:  0.9001 

# Prediction quality
preds = predict(model_full_data, newdata = test)
MAE(preds, test$mpg) # 3.109758
RMSE(preds, test$mpg) # 3.60968
R2(preds, test$mpg, form = "traditional") # 0.2687944

plot(preds, test$mpg)

# +-----------------------------------------------------------------------------
# Create missings

set.seed(RS) 
train_na = delete_MCAR(train, 0.1)
train_na$mpg = train$mpg

metadata <- missCompare::get_data(
    clean(train_na),
    matrixplot_sort = T,
    plot_transform = T)

simulated <- missCompare::simulate(rownum = metadata$Rows,
                                   colnum = metadata$Columns,
                                   cormat = metadata$Corr_matrix,
                                   meanval = 0,
                                   sdval = 1)

missCompare::MCAR(simulated$Simulated_matrix,
             MD_pattern = metadata$MD_Pattern,
             NA_fraction = metadata$Fraction_missingness,
             min_PDM = 2)

# +-----------------------------------------------------------------------------
# Train / validate model / check the best method

method = c()
Rsq = c()
mae = c()
rmse = c()
F_la = c()

# METHODS OF IMPUTATION
# ---------------------

# 1 random replacement
# 2 median imputation
# 3 mean imputation
# 4 missMDA Regularized
# 5 missMDA EM
# 6 pcaMethods PPCA
# 7 pcaMethods svdImpute
# 8 pcaMethods BPCA
# 9 pcaMethods NIPALS
# 10 pcaMethods NLPCA
# 11 mice mixed
# 12 mi Bayesian
# 13 Amelia II
# 14 missForest
# 15 Hmisc aregImpute
# 16 VIM kNN


for (meth_num in seq(16)) {
    tryCatch(
        expr = {
            set.seed(42)
            imputed <- missCompare::impute_data(clean(train_na), 
                                    scale = F, 
                                    n.iter = 1, 
                                    sel_method = meth_num)

            train_imp = as.data.frame(imputed[meth_num][[1]])
            facts = test |> select_if(is.factor) |> names()
            train_imp = train_imp |> mutate_at(vars(all_of(facts)), round)|> 
                mutate_at(vars(all_of(facts)), factor)

            # Test imputed
            mod = MASS::stepAIC(
                lm(mpg ~ ., data = train_imp)
            )


            model_full_data = lm(formula(mod), data = train_imp) 

            test_num = test |> mutate_at(vars(all_of(facts)), as.numeric) |> 
                mutate_at(vars(all_of(facts)), factor)

            # Prediction quality
            preds = predict(model_full_data, newdata = test_num)
            
            method = c(method, meth_num)
            mae = c(mae, MAE(preds, test_num$mpg))
            rmse = c(rmse, RMSE(preds, test_num$mpg))
            Rsq = c(Rsq, R2(preds, test_num$mpg, form = "traditional"))
            F_la = c(F_la, formula(model_full_data))
            dimens = dim(train_imp)[1]
        },
        error = function(e){ 
                    method = c(method, meth_num)
                    mae = c(mae, 0)
                    rmse = c(rmse, 0)
                    Rsq = c(Rsq, 0)
                    F_la = c(F_la, 0)
                    dimens = 0
                }
    )
}

# +-----------------------------------------------------------------------------
# Summary of all methods

(quality_tab = cbind(method, Rsq, mae, rmse, F_la, dimens))

# +-----------------------------------------------------------------------------
# Select best method

set.seed(42)
imputed <- missCompare::impute_data(clean(train_na), 
                        scale = F, 
                        n.iter = 1, 
                        sel_method = 9)

train_imp = as.data.frame(imputed[9][[1]])
facts = test |> select_if(is.factor) |> names()
train_imp = train_imp |> mutate_at(vars(all_of(facts)), round)|> mutate_at(vars(all_of(facts)), factor)

# Test imputed
mod = MASS::stepAIC(
    lm(mpg ~ ., data = train_imp)
)


model_full_data = lm(formula(mod), data = train_imp) 
model_full_data |> summary()

test_num = test |> mutate_at(vars(all_of(facts)), as.numeric) |> mutate_at(vars(all_of(facts)), factor)

# Prediction quality
preds = predict(model_full_data, newdata = test_num)

R2(preds, test_num$mpg, form = "traditional")

blandr.draw(preds, test_num$mpg, ciDisplay = FALSE)

# +-----------------------------------------------------------------------------
# Work with categoricals? (1, 11, 14, 16)

imputed <- missCompare::impute_data(train_na, 
                        scale = F, 
                        n.iter = 1, 
                        sel_method = 14)

train_imp = as.data.frame(imputed[14][[1]])

# Test imputed
mod = MASS::stepAIC(
    lm(mpg ~ ., data = train_imp)
)

model_full_data = lm(formula(mod), data = train_imp) 
model_full_data |> summary()

# Prediction quality
preds = predict(model_full_data, newdata = test)

R2(preds, test$mpg, form = "traditional")

blandr.draw(preds, test$mpg, ciDisplay = FALSE)


# +-----------------------------------------------------------------------------
# +-----------------------------------------------------------------------------
