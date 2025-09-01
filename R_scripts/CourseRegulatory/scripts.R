
################################################################################
# Лекция 1
# Типы данных и исходы, конечные точки
################################################################################

library(dplyr)
library(readr)
library(stringr)
library(lubridate)

# Функция для определения типа данных
guess_data_type <- function(x) {
  if (all(is.na(x))) return("Неопределенный")
  
  # Проверка на идентификатор (уникальные числовые или текстовые значения)
  if (length(unique(x)) == length(x) && !any(is.na(x))) {
    if (is.numeric(x) && all(x == floor(x))) return("Идентификатор")
    if (is.character(x)) return("Идентификатор")
  }
  
  # Проверка на бинарные данные
  if (length(unique(na.omit(x))) == 2) return("Бинарный")
  
  # Проверка на целочисленные данные
  if (is.numeric(x) && all(x == floor(x), na.rm = TRUE)) return("Целочисленный")
  
  # Проверка на числа с плавающей точкой
  if (is.numeric(x)) return("Число с плавающей точкой")
  
  # Проверка на дату/время
  if (any(class(x) %in% c("Date", "POSIXct", "POSIXt"))) return("Дата / время")
  
  # Попытка распарсить дату
  if (is.character(x)) {
    date_attempt <- try(parse_date_time(x, orders = c("ymd", "dmy", "mdy", "ymd HMS", "dmy HMS", "mdy HMS")), silent = TRUE)
    if (!inherits(date_attempt, "try-error") && sum(!is.na(date_attempt)) > length(x) * 0.5) {
      return("Дата / время")
    }
  }
  
  # Проверка на категориальные данные (малое количество уникальных значений)
  if (is.character(x) || is.factor(x)) {
    if (length(unique(x)) <= 10 && length(unique(x)) > 2) return("Категориальный")
    if (length(unique(x)) > 10) return("Текст")
  }
  
  # Проверка на порядковые данные (есть естественный порядок)
  if (is.ordered(x)) return("Порядковый")
  
  # По умолчанию
  return("Иной")
}

# Основная функция для обработки dataframe
process_dataframe_types <- function(df) {
  cat("Текущие типы данных и предполагаемые типы:\n")
  cat("============================================\n")
  
  type_info <- data.frame(
    Column = names(df),
    Current_Type = sapply(df, function(x) class(x)[1]),
    Guessed_Type = sapply(df, guess_data_type),
    stringsAsFactors = FALSE
  )
  
  print(type_info)
  cat("\n")
  
  # Создаем копию dataframe для модификации
  df_modified <- df
  
  # Предлагаем пользователю изменить типы данных
  cat("Хотите изменить типы данных для определенных столбцов? (y/n): ")
  response <- readline()
  
  if (tolower(response) == "y") {
    for (i in 1:nrow(type_info)) {
      col_name <- type_info$Column[i]
      guessed_type <- type_info$Guessed_Type[i]
      
      cat(sprintf("\nСтолбец: %s\n", col_name))
      cat(sprintf("Текущий тип: %s\n", type_info$Current_Type[i]))
      cat(sprintf("Предполагаемый тип: %s\n", guessed_type))
      
      cat("Доступные типы:\n")
      cat("1 - Идентификатор\n")
      cat("2 - Целочисленный\n")
      cat("3 - Число с плавающей точкой\n")
      cat("4 - Текст\n")
      cat("5 - Категориальный\n")
      cat("6 - Бинарный\n")
      cat("7 - Порядковый\n")
      cat("8 - Дата / время\n")
      cat("9 - Иной\n")
      cat("0 - Пропустить\n")
      
      cat("Выберите тип (0-9) или нажмите Enter для использования предполагаемого типа: ")
      choice <- readline()
      
      if (choice == "") {
        target_type <- guessed_type
      } else if (choice == "0") {
        next
      } else {
        type_map <- c("1" = "Идентификатор", "2" = "Целочисленный", "3" = "Число с плавающей точкой",
                     "4" = "Текст", "5" = "Категориальный", "6" = "Бинарный",
                     "7" = "Порядковый", "8" = "Дата / время", "9" = "Иной")
        target_type <- type_map[choice]
      }
      
      # Преобразование типа данных
      df_modified <- convert_column_type(df_modified, col_name, target_type)
    }
  }
  
  # Показываем итоговые типы данных
  cat("\nИтоговые типы данных:\n")
  cat("=====================\n")
  final_types <- data.frame(
    Column = names(df_modified),
    Final_Type = sapply(df_modified, function(x) class(x)[1]),
    stringsAsFactors = FALSE
  )
  print(final_types)
  
  return(df_modified)
}

# Функция для преобразования типа данных столбца
convert_column_type <- function(df, col_name, target_type) {
  col_data <- df[[col_name]]
  
  switch(target_type,
         "Идентификатор" = {
           if (is.numeric(col_data)) {
             df[[col_name]] <- as.integer(col_data)
           } else {
             df[[col_name]] <- as.character(col_data)
           }
         },
         "Целочисленный" = {
           df[[col_name]] <- as.integer(col_data)
         },
         "Число с плавающей точкой" = {
           df[[col_name]] <- as.numeric(col_data)
         },
         "Текст" = {
           df[[col_name]] <- as.character(col_data)
         },
         "Категориальный" = {
           df[[col_name]] <- as.factor(col_data)
         },
         "Бинарный" = {
           df[[col_name]] <- as.factor(col_data)
         },
         "Порядковый" = {
           df[[col_name]] <- ordered(col_data)
         },
         "Дата / время" = {
           df[[col_name]] <- parse_date_time(col_data, orders = c("ymd", "dmy", "mdy", "ymd HMS", "dmy HMS", "mdy HMS"))
         },
         "Иной" = {
           # Оставляем как есть
         }
  )
  
  return(df)
}

# Пример использования
# Создаем тестовый dataframe
example_df <- data.frame(
  id = 1:10,
  age = c(25, 30, 35, 40, 45, 50, 55, 60, 65, 70),
  salary = c(50000.50, 60000.75, 70000.25, 80000.00, 90000.50, 100000.75, 110000.25, 120000.00, 130000.50, 140000.75),
  name = c("John", "Jane", "Bob", "Alice", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"),
  department = c("IT", "HR", "IT", "Finance", "HR", "IT", "Finance", "HR", "IT", "Finance"),
  active = c(TRUE, FALSE, TRUE, FALSE, TRUE, FALSE, TRUE, FALSE, TRUE, FALSE),
  rating = ordered(c("Low", "Medium", "High", "Low", "Medium", "High", "Low", "Medium", "High", "Low")),
  start_date = c("2020-01-01", "2019-06-15", "2021-03-10", "2018-12-01", "2022-05-20", 
                 "2017-08-15", "2020-11-30", "2019-02-14", "2021-07-04", "2018-04-01")
)

# Запускаем обработку
processed_df <- process_dataframe_types(example_df)

processed_df |> str()

################################################################################
# Лекция 1.2
# Принципы формирования выборок
################################################################################

# Различные выборочные смещения

library(ggplot2)
library(patchwork)

sampling_demo_ru <- function(global_mean,
                             seed        = 42,
                             avail_size  = 1e5,
                             sample_size = 50) {

  set.seed(seed)

  ## 1. Истинная популяция --------------------------------------------
  pop_N  <- 1e6
  sigma  <- 1
  mu_ln  <- log(global_mean) - 0.5 * sigma^2
  pop    <- rlnorm(pop_N, meanlog = mu_ln, sdlog = sigma)
  pop_mu <- mean(pop)

  ## 2. Доступная популяция (Bias-i) -----------------------------------
  thresh     <- quantile(pop, 0.60)               # верхние 40 %
  avail_pool <- pop[pop > thresh]
  avail      <- sample(avail_pool, avail_size)
  avail_mu   <- mean(avail)

  ## 3. Выборки (Bias-e / Bias-k) --------------------------------------
  nsamples    <- 20
  samples_lst <- replicate(nsamples,
                           sample(avail, sample_size, replace = TRUE),
                           simplify = FALSE)
  samples_df  <- data.frame(
    value     = unlist(samples_lst),
    sample_id = factor(rep(seq_len(nsamples), each = sample_size))
  )
  samp1     <- samples_lst[[1]]
  samp1_df  <- data.frame(value = samp1)
  samp1_mu  <- mean(samp1)

  #--------------------------------------------------------------------
  # 4. График 1  –  доступная популяция (+ глобальный μ)
  #--------------------------------------------------------------------
  p_avail_hist <- ggplot(data.frame(x = avail), aes(x)) +
    geom_histogram(bins = 60, fill = "skyblue4", alpha = .6,
                   colour = "white") +
    # две вертикальные линии-средние
    geom_vline(xintercept = pop_mu,   colour = "steelblue",
               linetype = "dashed", linewidth = 1) +
    geom_vline(xintercept = avail_mu, colour = "firebrick",
               linetype = "dashed",  linewidth = 1) +
    labs(title    = "Доступная популяция (смещение i)",
         subtitle = sprintf("Среднее = %.2f   |   N = %s",
                            avail_mu, format(avail_size, big.mark = " ")),
         x = "значение", y = "частота") +
    theme_minimal() +
    # подписи в правом-верхнем углу
    annotate("text", x = Inf, y = Inf,
             label = paste0("μ (true)  = ", round(pop_mu,  2)),
             hjust = 1.05, vjust = 1.2, colour = "steelblue") +
    annotate("text", x = Inf, y = Inf,
             label = paste0("μ (avail) = ", round(avail_mu, 2)),
             hjust = 1.05, vjust = 2.4, colour = "firebrick")

  p_avail_box <- ggplot(data.frame(y = avail), aes(x = 1, y = y)) +
    geom_jitter(width = .15, colour = "grey40", alpha = .3, size = .9) +
    geom_boxplot(width = .35, fill = "orange",
                 outlier.shape = NA, alpha = .7) +
    coord_flip() +
    theme_minimal() +
    theme(axis.title = element_blank(),
          axis.text.x = element_blank(),
          axis.ticks.x = element_blank())

  p1 <- p_avail_hist / p_avail_box

  #--------------------------------------------------------------------
  # 5. График 2  –  множество выборок
  #--------------------------------------------------------------------
  ymax <- max(samples_df$value)

  p2 <- ggplot(samples_df, aes(x = sample_id, y = value)) +
    geom_point(colour = "grey40", alpha = .3, size = 1,
               position = position_jitter(width = .15)) +
    geom_boxplot(fill = "grey90", colour = "grey20",
                 alpha = .4, outlier.shape = NA) +
    geom_hline(yintercept = pop_mu,   colour = "steelblue",
               linetype = "dashed", linewidth = 1) +
    geom_hline(yintercept = avail_mu, colour = "firebrick",
               linetype = "dashed", linewidth = 1) +
    labs(title    = "Выборки из доступной популяции (смещение e)",
         subtitle = "Синяя линия — истинное среднее, красная — среднее доступной популяции",
         x = "номер выборки", y = "значение") +
    theme_minimal() +
    coord_cartesian(clip = "off") +
    annotate("text", x = Inf, y = ymax,
             label = paste0("μ (истинная)  = ", round(pop_mu,  2)),
             hjust = 1.05, vjust = 1.2, colour = "steelblue") +
    annotate("text", x = Inf, y = ymax * 0.93,
             label = paste0("μ (доступная) = ", round(avail_mu, 2)),
             hjust = 1.05, vjust = 1.2, colour = "firebrick")

  #--------------------------------------------------------------------
  # 6. График 3  –  одна выборка
  #--------------------------------------------------------------------
  xmax <- max(samp1_df$value)

  p3 <- ggplot(samp1_df, aes(x = value, y = 1)) +
    geom_segment(aes(xend = samp1_mu, yend = 1), colour = "grey70") +
    geom_point(size = 2, alpha = .7, colour = "black",
               position = position_jitter(height = .1)) +
    geom_boxplot(width = .25, fill = "gold",
                 alpha = .4, outlier.shape = NA) +
    geom_vline(xintercept = samp1_mu, colour = "darkgreen",
               linetype = "dashed", linewidth = 1) +
    geom_vline(xintercept = avail_mu, colour = "firebrick",
               linetype = "dashed", linewidth = 1) +
    geom_vline(xintercept = pop_mu,   colour = "steelblue",
               linetype = "dashed", linewidth = 1) +
    scale_y_continuous(breaks = NULL) +
    labs(title    = "Одна случайная выборка (смещение k)",
         subtitle = "Зелёная линия — среднее этой выборки",
         x = "значение", y = NULL) +
    theme_minimal() +
    coord_cartesian(clip = "off") +
    annotate("text", x = xmax, y = 1.30,
             label = paste0("μ (истинная)   = ", round(pop_mu,  2)),
             hjust = 1.05, vjust = 1.2, colour = "steelblue") +
    annotate("text", x = xmax, y = 1.22,
             label = paste0("μ (доступная)  = ", round(avail_mu, 2)),
             hjust = 1.05, vjust = 1.2, colour = "firebrick") +
    annotate("text", x = xmax, y = 1.14,
             label = paste0("μ (выборочное) = ", round(samp1_mu, 2)),
             hjust = 1.05, vjust = 1.2, colour = "darkgreen")

  #--------------------------------------------------------------------
  # 7. Компоновка
  #--------------------------------------------------------------------
  (p1 | p2) / p3
}

## Пример
sampling_demo_ru(global_mean = 100,
                 seed        = 1,
                 avail_size  = 250,
                 sample_size = 50)


###############################################################################
# Вероятностный сэмплинг

library(tidyverse)

demonstrate_sampling_methods <- function(
    mean            = 100,      # среднее генеральной
    N               = 500,      # размер генеральной
    seed            = 0,
    sample_size     = 50,
    n_strata        = 3,
    strata_probs    = NULL,     # вектор длиной n_strata или NULL
    frame_pattern   = c(0, 1)   # короткий вектор 0/1; будет «плиткой» повторён
) {
  set.seed(seed)

  ## 1. Генеральная совокупность --------------------------------------
  population <- rnorm(N, mean = mean, sd = 15)

  population_df <- data.frame(
    id      = 1:N,
    value   = population,
    stratum = sample(1:n_strata, N, replace = TRUE,
                     prob = if (is.null(strata_probs))
                              rep(1 / n_strata, n_strata) else strata_probs)
  )

  ## 2. Простая случайная выборка -------------------------------------
  simple_random_sample <- population_df[sample(N, sample_size), ]
  simple_random_mean   <- mean(simple_random_sample$value)

  ## 3. Выборка по рамке (детерминированно) ---------------------------
  frame_full <- rep(frame_pattern, length.out = N)  # «плитка» до длины N
  in_frame   <- frame_full == 1
  frame_pool <- population_df[in_frame, ]

  if (nrow(frame_pool) < sample_size)
    stop("В рамке меньше единиц '1', чем требуется sample_size.")

  frame_sample <- frame_pool[seq_len(sample_size), ]   # первые sample_size
  frame_mean   <- mean(frame_sample$value)

  ## 4. Систематическая выборка ---------------------------------------
  k     <- ceiling(N / sample_size)
  start <- sample(1:k, 1)
  sys_i <- seq(start, N, by = k)
  systematic_sample <- population_df[sys_i, ]
  systematic_mean   <- mean(systematic_sample$value)

  ## 5. Стратифицированная выборка ------------------------------------
  str_sizes  <- table(population_df$stratum)
  str_samps  <- round(sample_size * str_sizes / N)
  if (sum(str_samps) != sample_size)
    str_samps[1] <- str_samps[1] + (sample_size - sum(str_samps))

  stratified_sample <- do.call(
    rbind,
    lapply(seq_len(n_strata), function(s) {
      rows <- population_df[population_df$stratum == s, ]
      if (str_samps[s] > 0)
        rows[sample(nrow(rows), str_samps[s]), ]
    })
  )
  stratified_mean <- mean(stratified_sample$value)
  weighted_stratified_mean <-
    sum(tapply(stratified_sample$value,
               stratified_sample$stratum, mean) * (str_sizes / N))

  ## 6. Сводка ---------------------------------------------------------
  comparison <- data.frame(
    Метод       = c("Генеральная", "Простая случайная", "Рамка",
                    "Систематическая", "Стратифицированная"),
    Среднее     = c(mean(population), simple_random_mean, frame_mean,
                    systematic_mean, stratified_mean),
    Размер      = c(N, sample_size, sample_size,
                    nrow(systematic_sample), nrow(stratified_sample)),
    Абс_ошибка  = c(0,
                    abs(simple_random_mean - mean(population)),
                    abs(frame_mean          - mean(population)),
                    abs(systematic_mean     - mean(population)),
                    abs(stratified_mean     - mean(population)))
  )

  ## 7. Возврат --------------------------------------------------------
  list(
    population            = population,
    population_df         = population_df,
    frame_full            = frame_full,
    simple_random_sample  = simple_random_sample,
    frame_sample          = frame_sample,
    systematic_sample     = systematic_sample,
    stratified_sample     = stratified_sample,
    results = list(
      population_mean     = mean(population),
      simple_random_mean  = simple_random_mean,
      frame_mean          = frame_mean,
      systematic_mean     = systematic_mean,
      stratified_mean     = stratified_mean,
      weighted_stratified_mean = weighted_stratified_mean,
      population_strata_pct = prop.table(str_sizes) * 100,
      sample_strata_pct     = prop.table(table(stratified_sample$stratum)) * 100
    ),
    comparison  = comparison,
    parameters = list(
      mean, N, seed, sample_size, n_strata,
      strata_probs = if (is.null(strata_probs))
                       rep(1 / n_strata, n_strata) else strata_probs,
      frame_pattern = frame_pattern
    )
  )
}

###############################################################################
#  Визуализация
###############################################################################
visualize_sampling_results <- function(res) {
  par(mfrow = c(2, 3), mar = c(4,4,3,1), oma = c(0,0,3,0))

  # 1. генеральная
  hist(res$population, main = "Генеральная совокупность",
       xlab = "Значение", col = "lightblue", border = "blue")
  abline(v = res$results$population_mean, col = "red", lwd = 2)

  # 2. простая случайная
  hist(res$simple_random_sample$value, main = "Простая случайная",
       xlab = "Значение", col = "lightgreen", border = "darkgreen")
  abline(v = res$results$simple_random_mean, col = "red", lwd = 2)
  abline(v = res$results$population_mean,   col = "blue", lwd = 2, lty = 2)

  # 3. рамка
  hist(res$frame_sample$value, main = "Рамка (первые 1-ки)",
       xlab = "Значение", col = "lightcoral", border = "red")
  abline(v = res$results$frame_mean,      col = "red", lwd = 2)
  abline(v = res$results$population_mean, col = "blue", lwd = 2, lty = 2)

  # 4. систематическая
  hist(res$systematic_sample$value, main = "Систематическая",
       xlab = "Значение", col = "lightyellow", border = "orange")
  abline(v = res$results$systematic_mean, col = "red", lwd = 2)
  abline(v = res$results$population_mean, col = "blue", lwd = 2, lty = 2)

  # 5. стратифицированная
  hist(res$stratified_sample$value, main = "Стратифицированная",
       xlab = "Значение", col = "lavender", border = "purple")
  abline(v = res$results$stratified_mean, col = "red", lwd = 2)
  abline(v = res$results$population_mean, col = "blue", lwd = 2, lty = 2)

  # 6. boxplot сравнения
  box_vals <- list(
    "Простая\nслуч." = res$simple_random_sample$value,
    "Рамка"          = res$frame_sample$value,
    "Систематич."    = res$systematic_sample$value,
    "Стратиф."       = res$stratified_sample$value
  )
  boxplot(box_vals, col = c("lightgreen","lightcoral",
                            "lightyellow","lavender"),
          border = c("darkgreen","red","orange","purple"),
          ylab = "Значение")
  means <- sapply(box_vals, mean)
  points(seq_along(means), means, pch = 18, cex = 2, col = "red")
  abline(h = res$results$population_mean, col = "blue", lwd = 2, lty = 2)
  legend("topright", legend = c("Среднее выборки", "Истинное среднее"),
         pch = c(18, NA), col = c("red","blue"), lty = c(NA,2), lwd = c(NA,2))

  mtext("Сравнение методов вероятностной выборки", outer = TRUE, cex = 1.3)
}

###############################################################################
#  ПРИМЕР ИСПОЛЬЗОВАНИЯ
###############################################################################
set.seed(123)
my_pattern <- c(0,0,1,0,1)   # задаём «шаблон» для рамки

res <- demonstrate_sampling_methods(
  mean          = 100,
  N             = 500,
  seed          = 42,
  sample_size   = 50,
  n_strata      = 4,
  strata_probs  = c(0.2, 0.3, 0.25, 0.25),
  frame_pattern = my_pattern          # <<< ключевой аргумент
)

# смотрим сводку
print(res$comparison)

res$stratified_sample %>% 
  group_by(stratum) %>% 
  summarise(
    Mean = mean(value),
    n    = n()
  ) %>% 
  mutate(`Доля, %` = n / sum(n) * 100)    # доля каждой страты, %

# визуализация
visualize_sampling_results(res)


