
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


