
library(dplyr)

a = seq(10)
a

for (i in a) print(i)

i = 1
while (i <= 100) {
    
    if (i %% 2)
    print(i)
    
    i = i + 2

} 

data(mtcars)

y = "mpg"
X = names(mtcars)[-1]

vec_of_names = c()
r_sq_list = c()

for(variable in X) {
    vec_of_names = c(vec_of_names, variable)
    r_sq_list = c(r_sq_list, summary(
        lm(formula(paste(y, ' ~ ', variable)), 
            data = mtcars))$adj.r.squared)
}

tibble(vec_of_names, r_sq_list) |> summarize(Mean = mean(r_sq_list))



r_sq_list = sapply(
    X, function(x) summary(
        lm(formula(paste("mpg", ' ~ ', x)), data = mtcars)
            )$adj.r.squared
)

r_sq_list[r_sq_list > 0.5]

index = seq(8)

hem = sample(index, 20, replace = T) * 10

hem + rnorm(200, 0, sd(hem) * 1) |> hist()

runif(10, 0, sd(hem) * 1)|> hist()

hem |> hist()

mean(hem + runif(20, 0, sd(hem) * 1))

# ------

library(altair)
library("vegawidget")

data(mtcars)

library("altair")
library("vega_datasets")

vega_data <- import_vega_data()

chart <- 
  alt$Chart(vega_data$cars())$
  mark_point()$
  encode(
    x = "Horsepower:Q",
    y = "Miles_per_Gallon:Q",
    color = "Origin:N",
    tooltip = c("Name", "Horsepower", "Miles_per_Gallon", "Origin")
  )

vegawidget(chart)