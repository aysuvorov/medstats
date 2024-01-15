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
# Boot library

library(boot)
library(DescTools)

set.seed(0)
vec1 = seq(80)
vec2 = vec1 + rnorm(length(vec1))

MeanDiffCI(vec1, vec2)

df = data.frame(cbind(vec1, vec2))

mean.diff <- function(dataframe, indexVector) {
  return(mean(dataframe[indexVector,1] - dataframe[indexVector,2]))
}

mean(df[,1]) - mean(df[,2])

t.test(vec1, vec2)

(bts = boot(df, mean.diff, R = 1000) |> 
  boot.ci(conf = 0.95))

bts |> hist()
str(bts)

library(simpleboot)

b <- two.boot(vec1, vec2, mean, R = 1000)
boot.ci(b)  ## No studentized confidence intervals
hist(b)

M = cbind(vec1, vec2)

mean.diff <- function(mtx, indexVector) {
  return(mean(mtx[indexVector,1] - mtx[indexVector,2]))
}

mean.diff(M)

(bts = boot(M, mean.diff, R = 1000) |> 
  boot.ci(conf = 0.95))

0.0398973 + 2*0.1011916

ind = c(rep(1, length(vec1)), rep(2, length(vec2)))
as.numeric(table(ind))

mean.diff <- function(D, d) {
  E=D[d,]
  return(mean(E$vec1[d]) - mean(E$vec2))
}

boot(vec1, mean, R = 1000) |> boot.ci(type="basic")

two.boot(vec1, vec2, mean, R = 1000)
