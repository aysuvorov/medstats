
# Add p-values to typical plots in rstatix:

# BARPLOTS #####################################################################


# Load necessary libraries
library(ggplot2)
library(rstatix)
library(dplyr)
library(ggpubr)

set.seed(0)
data <- data.frame(
  Group = rep(c("Группа A", "Группа B"), each = 50),
  Outcome = sample(c("Жив", "Скончался"), 100, replace = TRUE)
)

# Perform Fisher's exact test
fisher_res <- data |> select(Group, Outcome) |> table() |> fisher_test()
fisher_res = fisher_res |> mutate(group1 = "Группа A", group2 = "Группа B", .y. = "Count")
fisher_res


summary_data <- data %>%
  group_by(Group, Outcome) %>%
  summarise(Count = n(), .groups = 'drop')

contingency_table <- table(data$Group, data$Outcome)

# Summarize the data for plotting
summary_data <- data %>%
  freq_table(Group, Outcome) |>
  mutate(
    `Доля,%` = prop,
    prop_txt = paste0(prop, ' %')
  )

ggbarplot(summary_data, 'Group', 'prop', fill = 'Outcome', 
  position = position_dodge( width = 0.5), label = summary_data$prop_txt) + labs(x = 'Группа', y = 'Доля, %') + 
    guides(fill = guide_legend(title = "Исходы")) + 
    stat_pvalue_manual(fisher_res, label = "p = {p}", y.position = max(summary_data$prop) + 5)
summary_data


# BOXPLOTS #####################################################################