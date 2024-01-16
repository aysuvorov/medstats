# https://cran.r-project.org/web/views/MixedModels.html

# Power analysis and simulation

# Bootstrapping

# simr

# +-----------------------------------------------------------------------------
library(faux)
library(survival)
library(survminer)
library(gtsummary)

# +-----------------------------------------------------------------------------
# New functions



# +-----------------------------------------------------------------------------
source("https://raw.githubusercontent.com/aysuvorov/medstats/master/R_scripts/medstats.R")


data(mtcars)
mtcars = FactorTransformer(mtcars, 4)

set.seed(0)
faux_df = faux::sim_df(mtcars, 500)
mtcars |> head()
mtcars |> names()

rsmpld_df = data.frame()
set.seed(0)
for (i in rownames(mtcars) |> sample(500, replace = TRUE)) {
    rsmpld_df = rbind(rsmpld_df, mtcars[i, ])
}
rsmpld_df = rsmpld_df |> mutate_if(is.numeric, ContNoiser, z_scores = .5)

lm(mpg ~ hp + disp + hp + wt, mtcars) |> summary()
lm(mpg ~ hp + disp + hp + wt, faux_df) |> summary()
lm(mpg ~ hp + disp + hp + wt, rsmpld_df) |> summary()

plot(mpg ~ hp, rsmpld_df)
plot(mpg ~ hp, mtcars)


mtcars |> mutate_all(as.numeric) |> summarise_all(sd)
rsmpld_df |> mutate_all(as.numeric) |> summarise_all(sd)

# Survival
data(cancer, package="survival")
cancer <- 
  cancer %>% 
  mutate(
    status = recode(status, `1` = 0, `2` = 1)
  )

lung = data.frame()
set.seed(0)
for (i in rownames(cancer) |> sample(100, replace = TRUE)) {
    lung = rbind(lung, cancer[i, ])
}

rsmpld_df = data.frame()
set.seed(0)
for (i in rownames(lung) |> sample(500, replace = TRUE)) {
    rsmpld_df = rbind(rsmpld_df, lung[i, ])
}
rsmpld_df = rsmpld_df |> mutate_at(vars(inst, age, time, ph.karno, pat.karno, meal.cal, wt.loss), ContNoiser, 
    method = 'unif', z_scores = 1)

ggsurvplot(survfit(Surv(time, status) ~ sex, data = lung), data = lung, ggtheme = theme_bw(), 
    conf.int = TRUE, pval = TRUE)

ggsurvplot(survfit(Surv(time, status) ~ sex, data = rsmpld_df), data = rsmpld_df, ggtheme = theme_bw(), 
    conf.int = TRUE, pval = TRUE)

coxph(Surv(time, status) ~ sex + age + wt.loss + ph.ecog, data = lung) |> 
  tbl_regression(exp = TRUE) |> as_tibble()
coxph(Surv(time, status) ~ sex + age + wt.loss + ph.ecog, data = rsmpld_df) |> 
  tbl_regression(exp = TRUE) |> as_tibble()

lung |>
    ggplot(aes(age, wt.loss, col = factor(status))) +
    geom_point() +
    geom_smooth()

rsmpld_df |>
    ggplot(aes(age, wt.loss, col = factor(status))) +
    geom_point() +
    geom_smooth()


# +-----------------------------------------------------------------------------
# Bootstrap by hand

library(boot)
library(DescTools)
library(tidyverse)
library(broom)

set.seed(0)
vec1 = rnorm(20)
vec2 = vec1*3. + rnorm(length(vec1))

tidy(lm(vec1 ~ vec2),  conf.int = TRUE)

t.test(vec1, vec2)

MeanDiffCI(vec1, vec2)

func = mean
ind1 = seq(length(vec1))
ind2 = seq(length(vec2))
MD = c()

set.seed(0)
for (i in seq(1000)) {

  iddx = sample(ind1, length(vec1), replace = TRUE)
  iddx2 = sample(ind2, length(vec2), replace = TRUE)
  MD = c(MD, 
    func(vec1[iddx]) - 
    func(vec2[iddx2])
  )
}
quantile(MD, probs = c(0.025, 0.975))

hist(MD)

ind = seq(length(vec1))
MD = c()
for (i in seq(1000)) {
  iddx = sample(ind, length(vec1), replace = TRUE)
  MD = c(MD, 
    coef(summary(lm(vec1[iddx] ~ vec2[iddx])))[2,1]
  )
}
quantile(MD, probs = c(0.005, 0.995))

hist(MD)

median(MD)

hist(vec1)
hist(vec2)

# +-----------------------------------------------------------------------------
# Boot library

library(boot)
library(DescTools)

set.seed(0)
vec1 = rnorm(20)
vec2 = rnorm(20)

dft = data.frame(cbind(append(vec1, vec2), c(rep(1,80), rep(2, 80))))

meanDiff = function(dataFrame, indexVector) { 
    m1 = mean(subset(dataFrame[indexVector, 1], 
      dataFrame[indexVector, 2] == 1))
    m2 = mean(subset(dataFrame[indexVector, 1], 
      dataFrame[indexVector, 2] == 2))
    m = m1 - m2
    return(m)
}

totalBoot = boot(dft, meanDiff, R = 10000, strata = df[,2])
(totalBootCI = boot.ci(totalBoot, type = "basic"))



