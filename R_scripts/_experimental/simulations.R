https://cran.r-project.org/web/views/MixedModels.html

Power analysis and simulation

Bootstrapping

simr




# +-----------------------------------------------------------------------------
# faux: Simulation for Factorial Designs
library(faux)

source("https://raw.githubusercontent.com/aysuvorov/medstats/master/R_scripts/medstats.R")


data(mtcars)
mtcars = FactorTransformer(mtcars, 4)



set.seed(0)
faux_df = faux::sim_df(mtcars, 500)
mtcars |> head()
mtcars |> names()

rsmpld_df = data.frame()
set.seed(0)
for (i in rownames(mtcars) |> sample(1000, replace = TRUE)) {
    rsmpld_df = rbind(rsmpld_df, mtcars[i, ])
}
rsmpld_df = rsmpld_df |> mutate_if(is.numeric, ContNoiser, method = 'norm')

lm(mpg ~ hp, mtcars) |> summary()
lm(mpg ~ hp, faux_df) |> summary()
lm(mpg ~ hp, rsmpld_df) |> summary()

plot(mpg ~ hp, rsmpld_df)

mtcars |> mutate_all(as.numeric) |> summarise_all(sd)
rsmpld_df |> mutate_all(as.numeric) |> summarise_all(sd)
