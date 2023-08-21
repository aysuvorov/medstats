# Main functions file
#
# To load script run
# source("/home/guest/Yandex.Disk/GitHub/r-medstats/medstats.R")
# or
# source("/home/guest/Документы/medstats/R_scripts/medstats.R")
#
# env R_HOME=/usr/lib64/R radian  --NOT RUN
# radian --r-binary=/usr/bin/R    --NOT RUN
library(gtsummary)
library(rstatix)
library(stringr)
library(magrittr)
library(fastDummies)
library(DescTools)
library(pROC)
library(boot)

# TO READ
# Emmeans contrasts:
# https://stats.stackexchange.com/questions/451205/in-emmeans-package-how-to-exclude-certain-uninteresting-contrasts-from-pairwise



# +-----------------------------------------------------------------------------

columns_printer = function(column_names) {
  # Converts column_names to c('col1', 'col2', ...) format
  # 
  # Arguments
  # ---------
  # column_names: column names from colnames(df)/names(df) result
  # 
  # Returns
  # -------
  # String of c('col1', 'col2', ...) object as in python list(df.columns)
  
  return(dput(column_names))
}

columns = function(dataframe) {
  # Short version for medstats::columns_printer()
  # 
  # Arguments
  # ---------
  # dataframe: dataframe to print out column names
  # 
  # Returns
  # -------
  # String of c('col1', 'col2', ...) object as in python list(df.columns)
  
  return(dput(colnames(dataframe)))
}

# +-----------------------------------------------------------------------------

greppler = function(data, pattern) {
  
  return(columns_printer(colnames(data)[grepl(pattern, colnames(data))]))
  
}

# +-----------------------------------------------------------------------------
# Simple vars recorder

lex_coder = function(vec, olds, news) {
  
  if (class(vec) == 'factor') {
    
  }

  for (i in seq(length(olds))) {
    vec[vec == olds[i]] = news[i]
  }
  return(vec)
}

# +-----------------------------------------------------------------------------
# Dummification

dummification = function(data, var_lst) {
  return(dummy_cols(data, var_lst, ignore_na = TRUE, 
                    remove_selected_columns = TRUE))
}

# +-----------------------------------------------------------------------------

factorizer = function(data, fact_lst=NULL, numeric_lst=NULL) {
  # Creates factors from fact_lst, creates numerics from numeric_lst
  if (is.null(fact_lst) == FALSE) {
    data[fact_lst] = purrr::map(data[fact_lst], factor)  
  }
  
  if (is.null(numeric_lst) == FALSE) {
    data[numeric_lst] = purrr::map(data[numeric_lst], as.numeric)   # nolint
  }
  
  return(data)
}

# +-----------------------------------------------------------------------------

excel_numeric_grinder = function(vec) {

    for (i in seq(length(vec))) {
        vec[i] = as.numeric(str_replace(vec[i], ',', '.'))
    }
    vec = as.numeric(vec)
    return(vec)
}

# +-----------------------------------------------------------------------------

p_adjuster = function(data, p_var, n) {
  f = c()
  i = 1
  while (i <= n)  {
    f = cbind(f, as.numeric(data[[p_var]]))
    i = i + 1
  }
  d = c()
  
  for (i in seq(dim(f)[1])) {
    b = as.numeric(p.adjust(f[i,], n = n)[1])
    d = c(d, b)
  }
  return(d)
}

# +-----------------------------------------------------------------------------

naive_locf = function(df, list_to_locf) {
  
  # PRIMER
  # ------
  # uin = c(1,2,3,4,5,6,7,8,9,0)
  # x1 = rep("a", 10)
  # t1 = rep(c(1,"w"),5)
  # t2 = rep(c(NA,NA),5)
  # t3 = rep(c(NA,"e"),5)
  # 
  # df = cbind.data.frame(uin, x1, t1, t2, t3)
  # 
  # df
  # 
  # data = df[c('t1', 't2', 't3')]
  # 
  # for (i in seq(length(rownames(data)))) {
  #   data[i,] = t(LOCF(t(data[i,])))
  # }
  # 
  # df[c('t1', 't2', 't3')] = data
  # 
  # df  
  
  data = df[list_to_locf]
  for (i in seq(length(rownames(data)))) {
    data[i,] = t(LOCF(t(data[i,])))
  }
  df[list_to_locf] = data
  return(df)
}

# +-----------------------------------------------------------------------------

miss_counter = function(data) {
  
 # Counts missing values 
  
  columns = c()
  var_type = t(as.data.frame(flatten(list(lapply(data, class)))))
  uniques = c()
  missings = c()
  
  for (i in seq(length(colnames(data)))) {
    columns = append(columns, colnames(data[,i]))
    uniques = append(uniques, count(unique(na.omit(data[,i]))))
    missings = append(missings, 1 - count(na.omit(data[,i]))/count(data[,i]) )
  }
  
  uniques = unlist(uniques)
  missings = round(unlist(missings), 2)
  
  meta_table = cbind.data.frame(columns, var_type, uniques, missings)
  rownames(meta_table) <- NULL
  return(meta_table)
}

# +-----------------------------------------------------------------------------

# Filler
numeric_filler = function(vec) {
  vec[is.na(vec)] = median(vec, na.rm = T)
  return(vec)
}


factor_filler = function(vec, fill_value = '0') {
  levels(vec) = c(levels(vec), '0')
  vec = replace(vec, is.na(vec), '0')
  return(vec)
}


data_filler = function(data, fact_lst=NULL, numeric_lst=NULL) {
  if (is.null(fact_lst) == FALSE) {
    data[fact_lst] = purrr::map(data[fact_lst], factor_filler)  
  }
  
  if (is.null(numeric_lst) == FALSE) {
    data[numeric_lst] = purrr::map(data[numeric_lst], numeric_filler)  
  }
  
  return(data)
}

# +-----------------------------------------------------------------------------

# Shapiro test

shapiro_safe_p = function(x) {
  res = tryCatch(shapiro.test(x)$p.value, 
                 error = function(e) 0, 
                 warning = function(w) 0
  )
  return(res)
}

# --  --  --  --  --  --  --  --  --  --  

table_shapiro = function(data) {
  
  # df %>% select(where(is.numeric)) %>%
  #   summarise_all(.funs = funs(p.value = shapiro.test(.)$p.value)) %>% t()
  # 
  # sh_name = c()
  # sh_p = c()
  # 
  # for (col in seq(length(colnames(data)))) {
  #   if (unname(sapply(data[col], class)) == 'numeric') {
  #     sh_name = c(sh_name, colnames(data[col]))
  #     sh_p = c(sh_p, round(shapiro_safe_p(unlist(data[,col])), 3))
  #     
  #   } else {
  #     sh_name = c(sh_name, colnames(data[col]))
  #     sh_p = NA
  #   }
  # }
  # sh_df = cbind.data.frame(sh_name, sh_p)
  # colnames(sh_df) = c('Var', 'Shapiro_test_p')
  # return(sh_df)

  # UNSAFE

  # sh_df = data %>% select(where(is.numeric)) %>%
  #   summarise_all(.funs = funs(p.value = shapiro.test(.)$p.value)) %>% t() %>%
  #   as.data.frame()
  # #sh_df$Var = columns_printer(gsub('.{8}$', '', rownames(sh_df)))
  # sh_df$Var = columns_printer(colnames(data %>% select(where(is.numeric))))
  # sh_df$Shapiro_test_p = sh_df$V1
  # return(sh_df[c('Var', 'Shapiro_test_p')])

  # SAFE

    sh_df = data %>% select(where(is.numeric)) %>%
    summarise_all(.funs = funs(p.value = shapiro_safe_p(.))) %>% t() %>%
    as.data.frame()
    #sh_df$Var = columns_printer(gsub('.{8}$', '', rownames(sh_df)))
    sh_df$Var = columns_printer(colnames(data %>% select(where(is.numeric))))
    sh_df$Shapiro_test_p = sh_df$V1
    return(sh_df[c('Var', 'Shapiro_test_p')])

}

# +-----------------------------------------------------------------------------

# Summary statistics table of class tibble
# Produces table with Var, valid_N, mean+sd, median[iqr], shapiro_test_p value
# Can be saved via writexl

# summary_all = function(
#     data) {
#   
#   # calculates shapiro tests for numeric data
#   shap_df = table_shapiro(data)
#   
#   # Initialisation of resulting tibble
#   df = NULL
#   
#   # calculates table with means and medians for numeric data
#   df = data %>% 
#     tbl_summary(
#       missing="no", 
#       type = list(data %>% 
#                     select(where(is.numeric)) %>% 
#                     colnames() %>% columns_printer() ~ 'continuous2'),
#       digits = c(all_continuous() ~ c(1,1),
#                  all_categorical() ~ c(0,1)),
#       statistic = all_continuous() ~ c("{mean} ± {sd}", 
#                                        "{median} [{p25}; {p75}]")) %>% 
#     add_n() %>% as_tibble()
# 
#   colnames(df) = c('Var', 'Valid_N','Stat')
#   
#   # combines tables with shapiro test
#   df %<>% 
#     left_join(shap_df, by = 'Var')
#   
#   # creates index variable and puts variables into wright order
#   df$index = seq(length(rownames(df)))
#   df %<>% relocate(
#     index, Var, Valid_N, Stat, Shapiro_test_p
#   )
#   colnames(df) = c('Индекс', 'Показатель', 'Валидные N','Статистика',
#                    'Тест Ш-У, значимость')
#   
#   return(df)
# }

summary_all = function(
    data, digits = 1) {
  
  # calculates shapiro tests for numeric data
  shap_df = table_shapiro(data)
  
  # Initialisation of resulting tibble
  df = NULL
  
  # calculates table with means and medians for numeric data
  df = data %>% 
    tbl_summary(
      missing="no", 
      type = list(data %>% 
                    select(where(is.numeric)) %>% 
                    colnames() %>% columns_printer() ~ 'continuous'),
      digits = c(all_continuous() ~ c(0,digits,digits,digits,digits,
                                        digits,digits,digits),
                 all_categorical() ~ c(0,0,1)),
      statistic = list(all_continuous() ~ "{N_nonmiss}|-|{mean} ± {sd}|{median} [{p25}; {p75}]|{min}|{max}",
                    all_categorical() ~   "{N_nonmiss}|{n} ({p}%)|-|-|-|-")) %>% 
    as_tibble()
  
  colnames(df) = c('Var', 'Stat')
  
  # combines tables with shapiro test
  df %<>% 
    left_join(shap_df, by = 'Var')
  
  # creates index variable and puts variables into wright order
  df$index = seq(length(rownames(df)))
  df %<>% relocate(
    index, Var, Stat, Shapiro_test_p) %>%
    separate(
      col = Stat, 
      into = c("valid", "perc", "mn", 'mdn', 'min', 'max'), 
      sep = "\\|")
  
  colnames(df) = c('Индекс', 'Показатель', 'Валидные,N', 'Абс,доля,%',
                   'Среднее, ст.откл', 'Медиана и размахи', 'Мин', 'Макс', 
                   'Тест Ш-У, значимость')
  
    df$`Показатель` = lex_coder(df$`Показатель`, 
    c("Mean ± SD", "Median [25%; 75%]"), 
    c("Среднее ± Ст.откл.", "Медиана и [25%; 75%]"))
  
  return(df)
}

# +-----------------------------------------------------------------------------

# Comparative statistics table of class tibble
# Produces table with Var, valid_N, group-variable-levels, p_val, distribution
# Automaticaly chooses test for parametrics (t.test, UMW, ANOVA, Kruskal) 
# depending on number of group levels and distibutions
# 
# Can be saved via writexl

# compare_all = function(
#     data,
#     group_var
# ) {
#   # Arguments
#   # ---------
#   # data: data frame, tibble
#   # group_var: grouping variable as a character e.g. "group" 
#   # 
#   # Returns
#   # -------
#   # tibble
#   
#   # Estimate nueric variables distributions
#   shap_df = table_shapiro(data)
#   shap_df %<>% mutate(Distr = ifelse(Shapiro_test_p < 0.05, "Ненормальное", 
#                                      "Нормальное"))
#   
#   # creates vectors with normal and unnormal distributions
#   normal = shap_df$Var[shap_df$Shapiro_test_p >= 0.05]
#   unnormal = shap_df$Var[shap_df$Shapiro_test_p < 0.05]
#   
#   # Initialisation of resulting tibble
#   df = NULL
#   
#   # Calculating stats
#   data[[group_var]] = factor(data[[group_var]])
#   
#   if (length(levels(data[[group_var]])) < 3) {
#     
#     df = data %>% 
#       tbl_summary(
#         by = group_var,
#         missing="no", 
#         type = list(data %>% 
#                       select(where(is.numeric)) %>% 
#                       colnames() %>% columns_printer() ~ 'continuous2'),
#         digits = c(all_continuous() ~ c(1,1),
#                    all_categorical() ~ c(0,1)),
#         statistic = all_continuous() ~ c("{mean} ± {sd}", 
#                                          "{median} [{p25}; {p75}]")) %>% 
#       add_n() %>% add_p(
#         pvalue_fun = ~style_pvalue(.x, digits = 3),
#         list(columns_printer(normal) ~ "t.test",
#              columns_printer(unnormal) ~ "wilcox.test")) %>% as_tibble()
#     
#   } else {
#     df = data %>% 
#       tbl_summary(
#         by = group_var,
#         missing="no", 
#         type = list(data %>% 
#                       select(where(is.numeric)) %>% 
#                       colnames() %>% columns_printer() ~ 'continuous2'),
#         digits = c(all_continuous() ~ c(1,1),
#                    all_categorical() ~ c(0,1)),
#         statistic = all_continuous() ~ c("{mean} ± {sd}", 
#                                          "{median} [{p25}; {p75}]")) %>% 
#       add_n() %>% add_p(
#         pvalue_fun = ~style_pvalue(.x, digits = 3),
#         list(columns_printer(normal) ~ "aov",
#              columns_printer(unnormal) ~ "kruskal.test")) %>% as_tibble() 
#   }
#   
#   # Changing names, order, creating index variable, distribution variable
#   colnames(df) = c('Var', 'Valid_N', levels(data[[group_var]]), 'p_val')
#   df %<>% left_join(shap_df[c('Var', 'Distr')], by = 'Var')
#   df$index = seq(length(rownames(df)))
#   df %<>% relocate(
#     index, Var, Valid_N, levels(data[[group_var]]), p_val, Distr
#   )
#   
#   colnames(df) = c('Индекс', 'Показатель', 'Валидные N',
#                    levels(data[[group_var]]), 'Значимость, р', 'Распределение')
#   
#   return(df)
# }

compare_all = function(
    data,
    group_var, 
    digits = 1
) {
  # Arguments
  # ---------
  # data: data frame, tibble
  # group_var: grouping variable as a character e.g. "group" 
  # 
  # Returns
  # -------
  # tibble
  
  # Estimate nueric variables distributions
  shap_df = table_shapiro(data)
  shap_df %<>% mutate(Distr = ifelse(Shapiro_test_p < 0.05, "Ненормальное", 
                                     "Нормальное"))
  
  # creates vectors with normal and unnormal distributions
  normal = shap_df$Var[shap_df$Shapiro_test_p >= 0.05]
  unnormal = shap_df$Var[shap_df$Shapiro_test_p < 0.05]
  
  # Initialisation of resulting tibble
  df = NULL
  
  # Calculating stats
  data[[group_var]] = factor(data[[group_var]])
  
  if (length(levels(data[[group_var]])) < 3) {
    
    df = data %>% 
      tbl_summary(
        by = group_var,
        missing="no", 
        type = list(data %>% 
                      select(where(is.numeric)) %>% 
                      colnames() %>% columns_printer() ~ 'continuous2'),
        digits = c(all_continuous() ~ c(digits, digits),
                   all_categorical() ~ c(0,1)),
        statistic = all_continuous() ~ c("{mean} ± {sd}", 
                                         "{median} [{p25}; {p75}]")) %>% 
      add_p(
        pvalue_fun = ~style_pvalue(.x, digits = 3),
        list(columns_printer(normal) ~ "t.test",
             columns_printer(unnormal) ~ "wilcox.test")) %>% as_tibble()
    
  } else {
    df = data %>% 
      tbl_summary(
        by = group_var,
        missing="no", 
        type = list(data %>% 
                      select(where(is.numeric)) %>% 
                      colnames() %>% columns_printer() ~ 'continuous2'),
        digits = c(all_continuous() ~ c(digits, digits),
                   all_categorical() ~ c(0,1)),
        statistic = all_continuous() ~ c("{mean} ± {sd}", 
                                         "{median} [{p25}; {p75}]")) %>% 
      add_p(
        pvalue_fun = ~style_pvalue(.x, digits = 3),
        list(normal ~ "aov"
             # columns_printer(unnormal) ~ "kruskal.test"
            )) %>% as_tibble() 
  }
  
  # Changing names, order, creating index variable, distribution variable
  colnames(df) = c('Var', levels(data[[group_var]]), 'p_val')
  df %<>% left_join(shap_df[c('Var', 'Distr')], by = 'Var')
  df$index = seq(length(rownames(df)))
  df %<>% relocate(
    index, Var, levels(data[[group_var]]), p_val, Distr
  )
  
  # count valids
  na = data.frame(is.na(data))
  na = na + 1
  na = na %>% mutate_all(funs(recode(., `1` = 1L, `2` = 0L)))
  na[[group_var]] = data[[group_var]]
  
  na %<>% 
    group_by(na[[group_var]]) %>% select(-group_var) %>%
    summarise(across(everything(), sum)) %>% t() 
  
  na = na[-1,]
  
  na = data.frame(na)
  
  if (dim(na)[1] == length(colnames(data))) {
    na$Var = colnames(data)
  } else {
    na$Var = colnames(data)[-1]
  }
  #na$Var = colnames(data)[-1]
  
  #tryCatch({na$Var = colnames(data)}, 
  #                  function(e) {na$Var = colnames(data)[-1]})
  
  ncols = c()
  
  for (i in seq(length(columns_printer(colnames(na)[grepl('X', colnames(na))])))) {
    ncols = c(ncols, paste0('N', i))
  }
  
  colnames(na) = c(ncols, 'Var')
  
  df %<>% left_join(na, by = 'Var')
  df %<>% relocate(
    index, Var, levels(data[[group_var]]), p_val, Distr, ncols
  )
  
  colnames(df) = c('Индекс', 'Показатель', 
                   levels(data[[group_var]]), 'Значимость, р', 'Распределение',
                   ncols)
    df$`Показатель` = lex_coder(df$`Показатель`, 
      c("Mean ± SD", "Median [25%; 75%]"), 
      c("Среднее ± Ст.откл.", "Медиана и [25%; 75%]"))
  
  return(df)
}

# +-----------------------------------------------------------------------------
# +-----------------------------------------------------------------------------
# Models

# Friedman test for RM time subgroups with pairwise wilcox tests in markdown

friedman_mod = function(data, group_var, pattern) {

  timepoints = seq(1:length(greppler(data, pattern)))
  
  tab = data %>% select(c(group_var, greppler(data, pattern))) %>%
    mutate(uin = rownames(.))
  colnames(tab) = c('group', timepoints, 'uin')
  
  for (u in unique(tab$group)) {
    
    tabb = tab %>% subset(group == u) %>% select(-c(group, uin))
    mtx = as.matrix(tabb, nrow = length(rownames(tabb)), byrow = TRUE)
    print(paste0("Подгруппа: ", u))
    tidy(friedman.test(mtx)) %>% kable('pipe') %>% print()
    
    tabbb = tab %>% 
      pivot_longer(!c(group,uin), names_to = "time", values_to = "value") %>%
      filter(group == u) %>% # na.omit() %>% 
      mutate(time = as.factor(as.numeric(time)))
    
    tabbb %>% 
      wilcox_test(value ~ time, paired = TRUE, p.adjust.method = "holm") %>%
      select(-`.y.`) %>%
      kable('pipe') %>% print()
    
    cat('\n')
    
  }
}

# Lmer model

lmer_mod = function(data, group_var, pattern) {
  
  
  timepoints = seq(1:length(greppler(data, pattern)))
  
  tab = data %>% select(c(group_var, greppler(data, pattern))) %>%
    mutate(uin = rownames(.))
  colnames(tab) = c('group', timepoints, 'uin')
  tab %<>%
    pivot_longer(!c(group,uin), names_to = "time", values_to = "value") %>% 
    #  na.omit() %>% 
    mutate(time = as.factor(as.numeric(time)))
  
  #tab$value = tab$value + 0.01
  #tab$value = log(tab$value)
  
  model = lmer( value ~ group*time + (1|uin), data=tab)
  emm = emmeans(model, ~ group*time)
  
  cat('\n')
  print('Emmeans')
  #cat('\n')
  summary(emm) %>% as.data.frame() %>% kable('pipe') %>% print()
  #cat('\n')
  pairs(emm, by = 'group',reverse = TRUE)%>% as.data.frame() %>% kable('pipe') %>% print()
  #cat('\n')
  pairs(emm, by = 'time')%>% as.data.frame() %>% kable('pipe') %>% print()
  
}

# Univariate lm -models model

univariate_linear_regr = function(data, dep_var) {
  
  ddd = data.frame()
  
  for (col in columns_printer(colnames(data))) {
    tryCatch(
      {a = tidy(summary(lm(as.formula(paste(dep_var, '~.')), data[c(dep_var, col)])))[2,] %>%
        select(c(term, estimate, 
                 std.error, statistic)) |> mutate(term = col)}
      , error = function(e) {a <<- data.frame(term = col,
                                              estimate = NA)})
    
    
    tryCatch(
      {b = glance(lm(as.formula(paste(dep_var, '~.')), 
                     data[c(dep_var, col)]))[c('adj.r.squared','p.value')]}
      , error = function(e) {b <<- data.frame(std.error   = NA,
                                              statistic  = NA,
                                              adj.r.squared = NA,
                                              p.value = NA)})
    
    a = cbind(a, b)
    
    ddd = rbind(ddd,a)
    
  }
  colnames(ddd) = c('Показатель','Коэфф','SE','t.stat','Adj.R.squared', 'p-val')
  
  # Бинарные факторы в таблице автоматически получают суффикс "1", т.к. 1-я категория оценивается. 
  # Ниже убираем единицы из названий факторов
  
  var_names = ddd$`Показатель`
  for (i in seq(length(var_names))) {
      s = var_names[i]
      if (is.na(s) == T) {
          s = 'No-name' 
      }
      if (str_sub(s,-1) == "1") {
          s = substring(s,1, nchar(s)-1)
      }
      var_names[i] = s
  }
  ddd$`Показатель` = var_names
  ddd = ddd |> filter(`Показатель` != 'No-name')
  
  
  rownames(ddd) = seq(length(rownames(ddd)))
  ddd %<>% mutate(`Индекс` = as.numeric(rownames(ddd)),
                `Коэфф`= round(`Коэфф`, 3),
                  SE = round(SE, 3),
                  t.stat = round(t.stat, 3),
                  `Adj.R.squared` = round(`Adj.R.squared`, 3),
                  `p-val` = round(`p-val`, 3))
  
  
  return(ddd |> select(c("Индекс","Показатель", "Коэфф", "SE", "t.stat", "Adj.R.squared", "p-val")))
}

# Correlation models

univariate_cor_test = function(data, dep_var, method = 'spearman') {
  
  ddd = data.frame()
  var = c()
  
  for (col in columns_printer(colnames(data))) {
    var = c(var, col)
    tryCatch(
      {a =  
        tidy(cor.test(data[[col]], 
                      data[[dep_var]], method = method))[c('estimate', 
                                                           'p.value')]}
      
      , error = function(e) {a <<- cbind.data.frame(estimate = NA, 
                                                    p.value = NA)})
    print(colnames(a))
    
    ddd = rbind(ddd,a)
    
  }
  ddd = cbind(var, ddd)
  colnames(ddd) = c('Показатель','Коэфф','p-val')
  ddd %<>% mutate(`Коэфф`= round(`Коэфф`, 3),
                  `p-val` = round(`p-val`, 3))
  return(ddd)
  
}

## Плоская корреляционная матрица по всей таблице

correlation_flat_matrix = function(df, method = 'spearman') {

    params = t(combn(columns_printer(colnames(df)),2))
    len = dim(params)[1]
    cor_matrix = data.frame()
    for (i in seq(len)) {

        cor_test = cor.test(df[[params[i, c(1)]]], df[[params[i, c(2)]]], method = method)
        rho = round(cor_test$estimate, 3)
        p_val = round(cor_test$p.value, 3)
        cor_matrix = rbind(cor_matrix, c(params[i, c(1)], params[i, c(2)], rho, p_val))
    }

    colnames(cor_matrix) = c("Factor_1", "Factor_2", "Spearman_rho", "p_val")
    return(cor_matrix)
}


## Univariate logistic models

# data$point = recode(data$'KT(stepen)_1', 'Легкая' = 0L, 
#                     'Средне-тяжелая и тяжелая' = 1L)
# log_odds_uni(data, 'point')

log_odds_uni = function(data, dep_var) {
  
  tab = data.frame()
  
  for (col in columns_printer(colnames(data))) {
    if (col != dep_var) {
      
      tryCatch(
        {mod = glm(as.formula(paste(dep_var, '~.')), data[c(dep_var, col)], 
                   family = binomial())},
        error = function(e) {mod = list('g', 'h')})
      
      if (class(mod)[1] == "glm") {
        
        f =tryCatch({f = confint(mod)[2,]},
                    error = function(e) {f = c(NA, NA)})
        
        res = tryCatch(
          {res = cbind(tidy(mod)[2,], cbind(ci.l = f[1], 
                                            ci.h = f[2])) %>%
            mutate(OR = exp(estimate),
                   lower = exp(ci.l),
                   upper = exp(ci.h)) %>% 
            select(term, OR, lower, upper, p.value) %>%
            filter(term != '(Intercept)')}, 
          error = function(e) {res = cbind.data.frame(term=col, 
                                                      OR=NA, lower=NA, upper=NA, p.value=NA)})
        
      } else {
        res = cbind.data.frame(term=col, OR=NA, lower=NA, upper=NA, p.value=NA)
      }
      
      
      tab = rbind(tab, res)
    }
  }
  
  tab %<>% mutate(
    OR = round(OR, 2),
    lower = round(lower, 2),
    upper = round(upper, 2),
    p.value = round(p.value, 3)
    
  )
  
  rownames(tab) = seq(dim(tab)[1])
  colnames(tab) = c('Фактор', 'OR', 'lower_CI', 'upper_CI', 'p.value')
  tab$`Индекс` = as.numeric(rownames(tab))
  tab %>% mutate(OR = round(OR, 2),
                 lower_CI = round(lower_CI, 2),
                 upper_CI = round(upper_CI, 2))
  
  # Бинарные факторы в таблице автоматически получают суффикс "1", т.к. 1-я категория оценивается. 
  # Ниже убираем единицы из названий факторов
  
  var_names = tab$`Фактор`
  for (i in seq(length(var_names))) {
    s = var_names[i]
    if (str_sub(s,-1) == "1") {
        s = substring(s,1, nchar(s)-1)
    }
    var_names[i] = s
  }
  tab$`Фактор` = var_names
  
  return(tab[c('Индекс','Фактор', 'OR', 'lower_CI', 'upper_CI', 'p.value')])
}

# +-----------------------------------------------------------------------------
# Логистическая регрессия с ковариатами (одно уравнение, один проход)

log_model = \(glm_binomial_model) {

    result = glm_binomial_model |> 
        tidy(conf.int=T, exponentiate=T) |> 
        select(term, estimate, conf.low, conf.high, p.value) 

    colnames(result) =  c("Factor", "OR", "lowerCI", "upperCI", "p-val")
    return(result)
}

# Линейная регрессия с ковариатами (одно уравнение, один проход)

linear_model = \(lm_model) {
    result = tidy(lm_model, conf.int = T) |> 
        bind_cols(tibble(Adj.R.sq = summary(lm_model)$adj.r.squared)) |>
        select(term, estimate, conf.low, conf.high, std.error, 
            statistic, Adj.R.sq, p.value)
    colnames(result) = c('Factor', 'Coeff', 'lowerCI', 'upperCI', 'SE', 
            'Stat', 'Adj.R.sq', 'p-val')
    return(result)
}

# Линейная регрессия - анализ с поправками на ковариаты 
# (прогон по всей таблице)

linear_covariate = \(data, y_var, covariates, scale_data = T) {

    dataframe = data |> rename(y = y_var)
    
    col_lst = setdiff(colnames(dataframe), c('y', covariates))

    if (scale_data == T) {
        dataframe = dataframe |> 
            select(c('y', all_of(covariates), all_of(col_lst))) |>
            mutate(across(where(is.numeric), scale))
    }

    or.df = tibble()

    ID = 0

    for (col in col_lst) {
        
        ID = ID + 1

        tryCatch({
            model = dataframe |> 
                select(c(y, all_of(covariates), all_of(col))) |> tibble()
            model = lm(y ~ ., data = model)
            or.df = model |> tidy(conf.int=T) |> 
            bind_cols(tibble(Adj.R.sq = summary(model)$adj.r.squared)) |>
                select(term, estimate, conf.low, conf.high, std.error, 
            statistic, Adj.R.sq, p.value) |> 
                tail(1) |> bind_cols(tibble(id = ID)) |> 
                rbind(or.df)

            },
      
            error = function(e) {
                or.df = bind_rows(tibble(
                        id = ID,
                        term = col, 
                        estimate = NA, 
                        conf.low = NA, 
                        conf.high = NA, 
                        p.value = 1,
                        Adj.R.sq = NA,
                        statistic = NA,
                        std.error = NA))})

    }
    or.df = or.df |> 
        arrange(id) |>  
        select(id, term, estimate, conf.low, 
        conf.high, std.error, statistic, Adj.R.sq, p.value)
    colnames(or.df) = c('id', 'Factor', 'Coeff', 'lowerCI', 'upperCI', 'SE', 
            'Stat', 'Adj.R.sq', 'p-val')
  
    var_names = or.df$Factor
    for (i in seq(length(var_names))) {
        s = var_names[i]
        if (str_sub(s,-1) == "1") {
            if (str_sub(s,-2) != "_1") {
                s = substring(s,1, nchar(s)-1) 
            }
        }
        var_names[i] = s
    }
    or.df$Factor = var_names
  
  
    return(or.df)
}


# Логистичекая регрессия - анализ с поправками на ковариаты 
# (прогон по всей таблице)

log_covariate = \(dataframe, y_var, covariates) {

    dataframe = dataframe |> rename(y = y_var)
    col_lst = setdiff(colnames(dataframe), c('y', covariates))

    or.df = tibble()

    ID = 0

    for (col in col_lst) {
    # foreach(i = 1:length(col_lst)) %do% {
        
        ID = ID + 1

        tryCatch({
            model = dataframe |> select(c(y, all_of(covariates), all_of(col))) |> tibble()
            # model = dataframe |> select(c(y, all_of(covariates), all_of(col_lst[i]))) |> tibble()

            model = glm(y ~ ., data = model, family = binomial)
            or.df = model |> tidy(conf.int=T, exponentiate=T) |> 
                select(term, estimate, conf.low, conf.high, p.value) |> 
                tail(1) |> bind_cols(tibble(id = ID)) |> 
                rbind(or.df)
            },
      
            error = function(e) {
                or.df = bind_rows(tibble(
                        id = ID,
                        term = col,
                        # term = col_lst[i], 

                        estimate = NA, 
                        conf.low = NA, 
                        conf.high = NA, 
                        p.value = 1))})

    }

    or.df = or.df |> arrange(id) |> select(id, term, estimate, conf.low, 
        conf.high, p.value)
    colnames(or.df) = c('id', 'Factor', 'OR', 'lowerCI', 'upperCI', 'p-val')

    var_names = or.df$Factor
    for (i in seq(length(var_names))) {
        s = var_names[i]
        if (str_sub(s,-1) == "1") {
            if (str_sub(s,-2) != "_1") {
                s = substring(s,1, nchar(s)-1) 
            }
        }
        var_names[i] = s
    }
    or.df$Factor = var_names

    return(or.df)
}

# +-----------------------------------------------------------------------------
# Регрессия Кокса с ковариатами

cox_covariate = function(data, time, status, covariates) {
  
  time_local = data[[time]]
  status_local = as.numeric(data[[status]])
  
  or.df = data.frame()
  
  for (col in columns_printer(colnames(dplyr::select(data, 
                                                     -c(time, status, covariates))))) {
    tryCatch({
      mod = coxph(Surv(time_local, status_local) ~ ., 
                  data[c(covariates, col)])},
      error = function(e) {mod = list('g', 'h')})
    
    if (class(mod) == "coxph") {
      estimate = round(exp(rev(tidy(mod)$estimate)[1]),2)
      p_val = round(rev(tidy(mod)$p.value)[1],3)
      ci_25 = round(rev(exp(confint(mod))[,1])[1],2)
      ci_975 = round(rev(exp(confint(mod))[,2])[1],2)
    } else {
      estimate = p_val = ci_25 = ci_975 = NA
    }
    
    
    or.df = rbind(or.df, c(col, estimate, ci_25, ci_975, p_val))
    
  }
  colnames(or.df) = c('Factor', 'HR', 'CI_95%_low','CI_95%_high', 'p_val')
  return(or.df)
}

# +-----------------------------------------------------------------------------
# Определение порогов по ROC (pROC)


roc_thresholds = \(real, pred, x="best", input="threshold", full_coords = T) {

  rets = c('threshold', 'tn', 'tp', 'fn', 'fp', 
    'sensitivity', 'specificity', 'ppv', 'npv', 'youden')
  cols = c("threshold.50.", "tn.50.", "tp.50.", "fn.50.", 
    "fp.50.", "sensitivity.50.", "sensitivity.2.5.", "sensitivity.97.5.", 
    "specificity.50.","specificity.2.5.", "specificity.97.5.", "ppv.50.", 
    "ppv.2.5.", "ppv.97.5.", "npv.50.", "npv.2.5.", "npv.97.5.", "youden.50.")
  
  if (input != "threshold") {
      if (full_coords == TRUE) {
        coords(roc(real, pred), "local maximas", ret=rets) |> 
        kable('pipe') |> print()
      }

    rets = rets[-1]
    cols = cols[-1]

  }
  
  roc.tab = tryCatch(
      {roc.tab = ci.coords(roc(real, pred), x = x, input=input,
            ret = rets) |> data.frame()}
      , error = function(e) {roc.tab = 
        ci.coords(roc(real, pred), x = x, input=input, best.policy = "random",
            ret = rets) |> data.frame()})

  roc.tab = roc.tab |> select(all_of(cols))

  tab1 = roc.tab |> 
    pivot_longer(contains('.50'), names_to = "Factor", values_to = "Median") |>
    select(Factor, Median)

  tab2 = roc.tab |> 
    pivot_longer(contains('.2.5'), names_to = "Factor", values_to = "lower")|>
    select(Factor, lower)

  tab3 = roc.tab |> 
    pivot_longer(contains('.97.5'), names_to = "Factor", values_to = "upper") |>
    select(Factor, upper)
  
  tab2$Factor = tab3$Factor = 
    c("sensitivity.50.","specificity.50.","ppv.50.","npv.50." )

  roc.tab = left_join(left_join(tab1, tab2, by = 'Factor'), tab3, by = 'Factor')
  roc.tab$Factor = str_remove(roc.tab$Factor, pattern = '.50.')

  auc_val = auc(real, pred)[1]
  auc_ci = ci.auc(real, pred)

  colnames(roc.tab) = c("Статистика", "Значение","2.5% ДИ","97.5% ДИ")
  
  return(roc.tab |> rbind(c("AUC", auc_val, auc_ci[1], auc_ci[3])))
}

# +-----------------------------------------------------------------------------
# Определение AUC, Sens, Spec, NPV, PPV, Kappa с бустрепом 95% ДИ (Гогниева)

binary_classification_metrics = function(real, pred, nboot = 10) {
    df = data.frame(
        Real = real,
        Pred = pred
    ) |> na.omit()

    idx = seq(length(rownames(df)))

    classification_func = function(frame) {

        real_subset <- frame[, 1]
        predicted_subset <- frame[, 2]

        if (
            length(unique(real_subset)) == 2 & length(unique(predicted_subset)
            ) == 2) {

                # Stats:
                tp <- sum(real_subset == 1 & predicted_subset == 1)
                tn <- sum(real_subset == 0 & predicted_subset == 0)
                fp <- sum(real_subset == 0 & predicted_subset == 1)
                fn <- sum(real_subset == 1 & predicted_subset == 0)
                sensitivity <- tp / (tp + fn)
                specificity <- tn / (tn + fp)
                ppv <- tp / (tp + fp)
                npv <- tn / (tn + fn)
                # kappa <- Kappa(matrix(c(tp, fp, fn, tn), nrow = 2))
                # kappa_unw = kappa$Unweighted[1] |> as.numeric()
                # kappa_w = kappa$Weighted[1] |> as.numeric()
                p0 = (tp + tn) / (tp + fn + tn + fp)
                pe = ((tp + fn) * (tp + fp) + (fp + tn) * (fn + tn)) / (tp + tn + fp + fn)^2
                kappa = (p0 - pe) / (1- pe)
                auc_ci = auc(real_subset, predicted_subset)
                f1 = 2 * tp / (2 * tp + fp + fn)
                balanced_acc = (sensitivity + specificity)/2

                return(c(auc_ci, sensitivity, specificity, ppv, npv, f1, balanced_acc, 
                    kappa))        

        } else {
            # return(invisible(NULL))
            return(rep(NA, 8))
        }
    }

    res_matrix = matrix(ncol = 8)

    i = 0

    while (i < nboot) {

            new_idx = sample(idx, size = length(idx), replace = T)
            res_matrix = rbind(
                res_matrix, 
                classification_func(df[new_idx,])
            )
            res_matrix = na.omit(res_matrix)
            i = dim(res_matrix)[1]

        }

    res_matrix = t(sapply(1:8, function(i) {
        quantile(res_matrix[, i], probs = c(.025, .975))
    })) |> data.frame()
    colnames(res_matrix) = c('lower', 'upper')

    res_matrix = res_matrix |> mutate(Point_est = classification_func(df),
        Stats = c('AUC', 'Sens', 'Spec', 'NPV', 'PPV', 'f1', 'balanced_acc', 
                    'kappa')) |> dplyr::select(c(Stats, Point_est, lower, upper))

    if (res_matrix[1,2] < 0.5) {
        res_matrix[1,c(2,3,4)] = 1 - res_matrix[1,c(2,3,4)]
    }

    return(res_matrix)
}


# +-----------------------------------------------------------------------------
# +-----------------------------------------------------------------------------
# Plots

# Plot parallel groups repeated measures for numerics
# Accept various statistics and errorbars: 'MedianCI', 'Medrange', 'MeanCI', 
#   'MeanSD'


plot_numeric_dynamic = function(
    data,
    groupvar,
    wordpat,
    timepoint_as_numeric = 'FALSE',
    timepoint_manual = FALSE,
    timepoint_word = 'Визит ',
    stat = 'MedianCI', # 'MedianCI', 'Medrange', 'MeanCI', 'MeanSD'
    legend_title,
    group_names,
    xlab = '',
    ylab = '',
    plot_saved = TRUE,
    filename = 'plot',
    print_plot = TRUE) {
  
  if (timepoint_manual == TRUE) {
    timepoints = timepoint_manual
  } else {
    
    timepoints = c()
    
    for (i in seq(1:length(greppler(data, wordpat)))) {
      timepoints = c(timepoints, paste0(timepoint_word, i))
    }
  }
  
  tab = data %>% select(c(groupvar, greppler(data, wordpat))) 
  
  colnames(tab) = c('group', timepoints)
  group_cats = unique(tab$group)
  
  for (i in seq(length(group_cats))) {
    tab$group = case_when(tab$group == group_cats[i] ~ group_names[i],
                          TRUE ~ as.character(tab$group))
  }
  levels(tab$group) = group_names
  
  if (timepoint_as_numeric != 'FALSE') {
    tab %<>%
      pivot_longer(!group, names_to = "time", values_to = "value") %>%
      mutate(time = as.numeric(time))
  } else {
    tab %<>%
      pivot_longer(!group, names_to = "time", values_to = "value") %>%
      mutate(time = factor(time, levels = timepoints))
  }
  
  if (stat == 'MedianCI') {
    tab %<>%
      group_by(group, time) %>%
      summarize(med = median(value, na.rm = TRUE),
                lower = MedianCI(value, na.rm = TRUE)[2],
                upper = MedianCI(value, na.rm = TRUE)[3])
    
  } else if (stat == 'Medrange') {
    
    tab %<>%
      group_by(group, time) %>%
      summarize(med = median(value, na.rm = TRUE),
                lower = quantile(value, 0.25, na.rm = TRUE, names = FALSE),
                upper = quantile(value, 0.75,na.rm = TRUE, names = FALSE))
    
  } else if (stat == 'MeanCI')  {
    tab %<>%
      group_by(group, time) %>%
      summarize(med = mean(value, na.rm = TRUE),
                lower = MeanCI(value, na.rm = TRUE)[2],
                upper = MeanCI(value, na.rm = TRUE)[3])
  } else {
    tab %<>%
      group_by(group, time) %>%
      summarize(med = mean(value, na.rm = TRUE),
                sd = sd(value, na.rm = TRUE)) %>%
      mutate(lower = med - sd,
             upper = med + sd)
  }
  
  p = ggplot(tab, aes(x=time, y=med, group=group, color=group)) + 
    geom_line(position=position_dodge(0.05)) +
    geom_point(position=position_dodge(0.05)) +
    geom_errorbar(aes(ymin=lower, ymax=upper), width=0,
                  position=position_dodge(0.05)) + 
    xlab(xlab) +
    ylab(ylab)+ 
    labs(colour=legend_title)
  
  if (print_plot == TRUE) {
    print(p)
  } 

  if (plot_saved == TRUE) {
    ggsave(
      paste0(filename,'.png',sep = ''),
      device = 'png',
      width = 8,
      height = 6
    )
  }
}
