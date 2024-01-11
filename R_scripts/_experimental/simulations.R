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
faux_df |> head()
mtcars |> names()
