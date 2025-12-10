
library(tidyverse)

##############################################################################
#  0.  УНИВЕРСАЛЬНЫЕ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ                                ###
##############################################################################

# [Все предыдущие вспомогательные функции остаются без изменений]

# Добавим функцию для расчета d для одной выборки
one_sample_d <- function(mean, sd, mu0 = 0, n, conf.level = 0.95) {
  d <- (mean - mu0) / sd
  df <- n - 1
  se_d <- sqrt(1/n + d^2/(2*n))
  t_crit <- tcrit(df, conf.level)
  ci_d <- d + c(-1, 1) * t_crit * se_d
  
  list(d = d, ci_lo = ci_d[1], ci_hi = ci_d[2])
}

conv_or_to_d <- function(or) {
    log(or) * sqrt(3) / pi
  }

##############################################################################
#  1.  КОНВЕРТЕР РАЗМЕРОВ ЭФФЕКТА (ИСПРАВЛЕННАЯ ВЕРСИЯ)                    ###
##############################################################################
effect_convert <- function(
    # Основные параметры
  groups = c("two", "one"),
  paired = FALSE,
  control_group = 2,
  
  # Пропорции (бинарные данные)
  proportion = NULL,
  prop1 = NULL, prop2 = NULL,
  
  # Отношения шансов и рисков
  OR = NULL, OR_lo = NULL, OR_hi = NULL,
  RR = NULL, RR_lo = NULL, RR_hi = NULL,
  
  # Cohen's h (явно заданный)
  h = NULL,
  
  # Количественные данные
  d = NULL,
  
  # Сырые средние и стандартные отклонения
  mean = NULL, sd = NULL,           # для одной выборки
  mu0 = 0,                          # теоретическое среднее для одной выборки
  mean1 = NULL, mean2 = NULL,       # для двух выборок
  sd1 = NULL, sd2 = NULL,
  sd_control = NULL,
  
  # Корреляция и меры ассоциации
  r = NULL, r2 = NULL,
  eta2 = NULL, omega2 = NULL,
  
  # Ковариация для парных данных
  rho_paired = NULL,
  
  # Базовые частоты для преобразований
  base_rate = NULL,
  p0 = NULL, p1 = NULL,
  
  # Объемы выборок
  n = NULL,
  n1 = NULL, n2 = NULL,
  
  # Уровень доверия
  conf.level = 0.95) {
  
  # Проверка наличия dplyr
  if (!requireNamespace("dplyr", quietly = TRUE)) {
    stop("Требуется пакет dplyr. Установите его: install.packages('dplyr')")
  }
  
  # Определяем тип сравнения
  groups <- match.arg(groups)
  
  # Гармонизация параметров
  base_rate_final <- if (!is.null(base_rate)) base_rate else p0
  p1_final <- if (!is.null(p1)) p1 else prop1
  
  # Гармонизация объемов выборок
  if (groups == "one") {
    if (!is.null(n)) {
      n1 <- n
    }
    n2 <- NA
  } else {
    # Для двух групп
    if (is.null(n1) && !is.null(n)) {
      # Если задан только n, распределяем поровну
      n1 <- floor(n/2)
      n2 <- n - n1
    }
  }
  
  Ntot <- ifelse(is.na(n2), ifelse(is.na(n1), NA, n1), n1 + n2)
  ratio <- ifelse(is.na(n2), NA, n2 / n1)
  
  # Инициализация выходных данных
  out <- data.frame()
  
  # ────────────────────────────────────────────────────────────────────────
  # A. ОДНА ВЫБОРКА: СРЕДНЕЕ И СТАНДАРТНОЕ ОТКЛОНЕНИЕ
  # ────────────────────────────────────────────────────────────────────────
  if (!is.null(mean) && !is.null(sd) && groups == "one") {
    if (is.null(n1) && is.null(n)) {
      warning("Для расчета ДИ требуется объем выборки n")
    }
    
    # Вычисляем Cohen's d для одной выборки
    d_result <- one_sample_d(mean = mean, sd = sd, mu0 = mu0, 
                             n = if (!is.null(n1)) n1 else n, 
                             conf.level = conf.level)
    
    d_cohen <- d_result$d
    
    # Преобразования
    r_from_d <- conv_d_to_r(d_cohen)
    or_from_d <- conv_d_to_or(d_cohen)
    r2_from_d <- r_from_d^2
    f_from_d <- calc_f(r2_from_d)
    h_from_d <- conv_d_to_h(d_cohen)
    
    # Преобразование ДИ
    ci_d <- c(d_result$ci_lo, d_result$ci_hi)
    ci_r <- conv_d_to_r(ci_d)
    ci_or <- conv_d_to_or(ci_d)
    ci_r2 <- ci_r^2
    ci_h <- conv_d_to_h(ci_d)
    
    results <- data.frame(
      effect = c("Cohen's d (one sample)", "r (from d)", "OR (from d)",
                 "R² (from d)", "Cohen's f (from d)", "Cohen's h (from d)"),
      est = c(d_cohen, r_from_d, or_from_d, r2_from_d, f_from_d, h_from_d),
      ci_lo = c(ci_d[1], ci_r[1], ci_or[1], ci_r2[1], NA, ci_h[1]),
      ci_hi = c(ci_d[2], ci_r[2], ci_or[2], ci_r2[2], NA, ci_h[2]),
      stringsAsFactors = FALSE
    )
    
    out <- rbind(out, results)
  }
  
  # ────────────────────────────────────────────────────────────────────────
  # B. ОТНОШЕНИЕ ШАНСОВ (OR) как основной вход
  # ────────────────────────────────────────────────────────────────────────
  if (!is.null(OR)) {
    or_est <- OR
    
    # Преобразования OR → d → r → R² → f
    d_from_or <- conv_or_to_d(or_est)
    r_from_or <- conv_d_to_r(d_from_or)
    r2_from_or <- r_from_or^2
    f_from_or <- calc_f(r2_from_or)
    h_from_or <- conv_d_to_h(d_from_or)
    
    # Преобразования OR → RR (если задана базовая частота)
    rr_from_or <- NULL
    if (!is.null(base_rate_final)) {
      rr_from_or <- conv_or_to_rr(or_est, base_rate_final)
    }
    
    # Обработка доверительных интервалов
    ci_or <- c(lo = OR_lo, hi = OR_hi)
    
    if (!is.null(OR_lo) && !is.null(OR_hi)) {
      ci_d <- conv_or_to_d(c(OR_lo, OR_hi))
      ci_r <- conv_d_to_r(ci_d)
      ci_r2 <- ci_r^2
      ci_h <- conv_d_to_h(ci_d)
    } else {
      ci_d <- c(NA, NA)
      ci_r <- c(NA, NA)
      ci_r2 <- c(NA, NA)
      ci_h <- c(NA, NA)
    }
    
    # Создаем строки для результатов
    results <- data.frame(
      effect = c("OR", "d (from OR)", "r (from OR)", 
                 "R² (from OR)", "Cohen's f (from OR)", "Cohen's h (from OR)"),
      est = c(or_est, d_from_or, r_from_or, r2_from_or, f_from_or, h_from_or),
      ci_lo = c(ci_or["lo"], ci_d[1], ci_r[1], ci_r2[1], NA, ci_h[1]),
      ci_hi = c(ci_or["hi"], ci_d[2], ci_r[2], ci_r2[2], NA, ci_h[2]),
      stringsAsFactors = FALSE
    )
    
    # Добавляем RR если доступно
    if (!is.null(rr_from_or)) {
      results <- rbind(results,
                       data.frame(effect = "RR (from OR)",
                                  est = rr_from_or,
                                  ci_lo = NA,
                                  ci_hi = NA,
                                  stringsAsFactors = FALSE))
    }
    
    out <- rbind(out, results)
  }
  
  # ────────────────────────────────────────────────────────────────────────
  # C. COHEN'S H как основной вход
  # ────────────────────────────────────────────────────────────────────────
  if (!is.null(h)) {
    h_est <- h
    
    # Преобразования h → d → r → R² → OR
    d_from_h <- conv_h_to_d(h_est)
    r_from_h <- conv_h_to_r(h_est)
    r2_from_h <- r_from_h^2
    f_from_h <- calc_f(r2_from_h)
    or_from_h <- conv_d_to_or(d_from_h)
    
    # Преобразования h → RR
    rr_from_h <- exp(h_est)  # т.к. h = ln(RR)
    
    results <- data.frame(
      effect = c("Cohen's h", "d (from h)", "r (from h)", 
                 "R² (from h)", "Cohen's f (from h)", "OR (from h)", "RR (from h)"),
      est = c(h_est, d_from_h, r_from_h, r2_from_h, f_from_h, or_from_h, rr_from_h),
      ci_lo = NA,
      ci_hi = NA,
      stringsAsFactors = FALSE
    )
    
    out <- rbind(out, results)
  }
  
  # ────────────────────────────────────────────────────────────────────────
  # D. ОДНА ВЫБОРКА: ПРОПОРЦИЯ
  # ────────────────────────────────────────────────────────────────────────
  if (!is.null(proportion) && groups == "one") {
    p <- proportion
    
    # SE и ДИ для пропорции
    if (!is.null(n1) && n1 > 0) {
      se_p <- sqrt(p * (1 - p) / n1)
      zc <- qnorm(1 - (1 - conf.level) / 2)
      ci_p <- p + c(-1, 1) * zc * se_p
      ci_p <- pmax(0, pmin(1, ci_p))
    } else {
      ci_p <- c(NA, NA)
    }
    
    # Cohen's h (сравнение с 0.5)
    h_arcsine <- 2 * asin(sqrt(p)) - 2 * asin(sqrt(0.5))
    
    results <- data.frame(
      effect = c("Proportion", "Cohen's h (vs 0.5)"),
      est = c(p, h_arcsine),
      ci_lo = c(ci_p[1], NA),
      ci_hi = c(ci_p[2], NA),
      stringsAsFactors = FALSE
    )
    
    out <- rbind(out, results)
  }
  
  # ────────────────────────────────────────────────────────────────────────
  # E. ДВЕ ВЫБОРКИ: ПРОПОРЦИИ
  # ────────────────────────────────────────────────────────────────────────
  if (!is.null(prop1) && !is.null(prop2) && groups == "two") {
    p1 <- prop1
    p2 <- prop2
    
    # Основные метрики
    rd <- p1 - p2
    rr <- p1 / p2
    or <- (p1 / (1 - p1)) / (p2 / (1 - p2))
    
    # Cohen's h (арксинусная разность)
    h_arcsine <- cohens_h_prop(p1, p2)
    
    # Cohen's h (log RR)
    h_logrr <- cohens_h_logrr(rr)
    
    # Расчет стандартных ошибок и ДИ
    if (!is.na(n1) && !is.na(n2) && n1 > 0 && n2 > 0) {
      zc <- qnorm(1 - (1 - conf.level) / 2)
      
      # SE и ДИ для RD
      se_rd <- sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
      ci_rd <- rd + c(-1, 1) * zc * se_rd
      ci_rd <- pmax(-1, pmin(1, ci_rd))
      
      # SE и ДИ для RR
      se_logrr <- sqrt((1-p1)/(n1*p1) + (1-p2)/(n2*p2))
      ci_rr <- ratio_ci(rr, se_logrr, conf.level)
      
      # SE и ДИ для OR
      se_logor <- sqrt(1/(n1*p1) + 1/(n1*(1-p1)) + 1/(n2*p2) + 1/(n2*(1-p2)))
      ci_or <- ratio_ci(or, se_logor, conf.level)
      
      # SE и ДИ для h (log RR)
      ci_h_logrr <- h_logrr + c(-1, 1) * zc * se_logrr
    } else {
      ci_rd <- c(NA, NA)
      ci_rr <- c(NA, NA)
      ci_or <- c(NA, NA)
      ci_h_logrr <- c(NA, NA)
    }
    
    # Преобразования OR → другие метрики
    d_from_or <- conv_or_to_d(or)
    r_from_or <- conv_d_to_r(d_from_or)
    r2_from_or <- r_from_or^2
    f_from_or <- calc_f(r2_from_or)
    h_from_or <- conv_d_to_h(d_from_or)
    
    # Преобразование ДИ
    if (!is.na(ci_or[1]) && !is.na(ci_or[2])) {
      ci_d_from_or <- conv_or_to_d(ci_or)
      ci_r_from_or <- conv_d_to_r(ci_d_from_or)
      ci_r2_from_or <- ci_r_from_or^2
      ci_h_from_or <- conv_d_to_h(ci_d_from_or)
    } else {
      ci_d_from_or <- c(NA, NA)
      ci_r_from_or <- c(NA, NA)
      ci_r2_from_or <- c(NA, NA)
      ci_h_from_or <- c(NA, NA)
    }
    
    results <- data.frame(
      effect = c("Risk Difference", "Relative Risk", "Odds Ratio",
                 "Cohen's h (arcsine)", "Cohen's h (log RR)",
                 "d (from OR)", "r (from OR)", 
                 "R² (from OR)", "Cohen's f (from OR)", "Cohen's h (from OR)"),
      est = c(rd, rr, or, h_arcsine, h_logrr,
              d_from_or, r_from_or, r2_from_or, f_from_or, h_from_or),
      ci_lo = c(ci_rd[1], ci_rr[1], ci_or[1], NA, ci_h_logrr[1],
                ci_d_from_or[1], ci_r_from_or[1], ci_r2_from_or[1], NA, ci_h_from_or[1]),
      ci_hi = c(ci_rd[2], ci_rr[2], ci_or[2], NA, ci_h_logrr[2],
                ci_d_from_or[2], ci_r_from_or[2], ci_r2_from_or[2], NA, ci_h_from_or[2]),
      stringsAsFactors = FALSE
    )
    
    out <- rbind(out, results)
  }
  
  # ────────────────────────────────────────────────────────────────────────
  # F. КОРРЕЛЯЦИЯ (r)
  # ────────────────────────────────────────────────────────────────────────
  if (!is.null(r)) {
    r_est <- r
    r2_est <- r^2
    
    # Преобразования
    d_from_r <- conv_r_to_d(r_est)
    or_from_r <- conv_d_to_or(d_from_r)
    f_from_r <- calc_f(r2_est)
    h_from_r <- conv_r_to_h(r_est)
    
    # 95% ДИ для r
    if (!is.na(n1) && n1 > 3) {
      ci_r <- r_ci(r_est, n1, conf.level)
      ci_d_from_r <- conv_r_to_d(ci_r)
      ci_or_from_r <- conv_d_to_or(ci_d_from_r)
      ci_r2 <- ci_r^2
      ci_h_from_r <- conv_r_to_h(ci_r)
    } else {
      ci_r <- c(NA, NA)
      ci_d_from_r <- c(NA, NA)
      ci_or_from_r <- c(NA, NA)
      ci_r2 <- c(NA, NA)
      ci_h_from_r <- c(NA, NA)
    }
    
    results <- data.frame(
      effect = c("r (Pearson)", "R²", "d (from r)", 
                 "OR (from r)", "Cohen's f", "Cohen's h (from r)"),
      est = c(r_est, r2_est, d_from_r, or_from_r, f_from_r, h_from_r),
      ci_lo = c(ci_r[1], ci_r2[1], ci_d_from_r[1], 
                ci_or_from_r[1], NA, ci_h_from_r[1]),
      ci_hi = c(ci_r[2], ci_r2[2], ci_d_from_r[2],
                ci_or_from_r[2], NA, ci_h_from_r[2]),
      stringsAsFactors = FALSE
    )
    
    out <- rbind(out, results)
  }
  
  # ────────────────────────────────────────────────────────────────────────
  # G. COHEN'S D (явно заданный)
  # ────────────────────────────────────────────────────────────────────────
  if (!is.null(d)) {
    d_cohen <- d
    
    # Преобразования
    r_from_d <- conv_d_to_r(d_cohen)
    or_from_d <- conv_d_to_or(d_cohen)
    r2_from_d <- r_from_d^2
    f_from_d <- calc_f(r2_from_d)
    h_from_d <- conv_d_to_h(d_cohen)
    
    # 95% ДИ для d
    if (!is.na(n1) && !is.na(n2)) {
      ci_d_cohen <- cohen_d_ci_t(d_cohen, n1, n2, paired, conf.level)
      ci_r_from_d <- conv_d_to_r(ci_d_cohen)
      ci_or_from_d <- conv_d_to_or(ci_d_cohen)
      ci_r2_from_d <- ci_r_from_d^2
      ci_h_from_d <- conv_d_to_h(ci_d_cohen)
    } else if (!is.na(n1)) {
      # Одна выборка или парные данные
      ci_d_cohen <- one_sample_d(mean = d_cohen * sd, sd = sd, mu0 = 0, 
                                 n = n1, conf.level = conf.level)
      ci_d_cohen <- c(ci_d_cohen$ci_lo, ci_d_cohen$ci_hi)
      ci_r_from_d <- conv_d_to_r(ci_d_cohen)
      ci_or_from_d <- conv_d_to_or(ci_d_cohen)
      ci_r2_from_d <- ci_r_from_d^2
      ci_h_from_d <- conv_d_to_h(ci_d_cohen)
    } else {
      ci_d_cohen <- c(NA, NA)
      ci_r_from_d <- c(NA, NA)
      ci_or_from_d <- c(NA, NA)
      ci_r2_from_d <- c(NA, NA)
      ci_h_from_d <- c(NA, NA)
    }
    
    results <- data.frame(
      effect = c("Cohen's d", "r (from d)", "OR (from d)",
                 "R² (from d)", "Cohen's f (from d)", "Cohen's h (from d)"),
      est = c(d_cohen, r_from_d, or_from_d, r2_from_d, f_from_d, h_from_d),
      ci_lo = c(ci_d_cohen[1], ci_r_from_d[1], ci_or_from_d[1],
                ci_r2_from_d[1], NA, ci_h_from_d[1]),
      ci_hi = c(ci_d_cohen[2], ci_r_from_d[2], ci_or_from_d[2],
                ci_r2_from_d[2], NA, ci_h_from_d[2]),
      stringsAsFactors = FALSE
    )
    
    out <- rbind(out, results)
  }
  
  # ────────────────────────────────────────────────────────────────────────
  # H. ДВЕ ВЫБОРКИ: СРЕДНИЕ И СТАНДАРТНЫЕ ОТКЛОНЕНИЯ
  # ────────────────────────────────────────────────────────────────────────
  if (!is.null(mean1) && !is.null(mean2) && !is.null(sd1) && !is.null(sd2) && groups == "two") {
    if (paired) {
      if (is.null(rho_paired)) {
        stop("Для парного дизайна требуется rho_paired")
      }
      sd_diff <- sqrt(sd1^2 + sd2^2 - 2 * rho_paired * sd1 * sd2)
      d_cohen <- (mean1 - mean2) / sd_diff
    } else {
      # Две независимые выборки
      sp <- sqrt(((n1 - 1) * sd1^2 + (n2 - 1) * sd2^2) / (n1 + n2 - 2))
      d_cohen <- (mean1 - mean2) / sp
      
      # Glass's Δ
      if (!is.null(sd_control)) {
        delta_glass <- (mean1 - mean2) / sd_control
      } else if (control_group == 2) {
        delta_glass <- (mean1 - mean2) / sd2
        sd_control <- sd2
      } else {
        delta_glass <- (mean1 - mean2) / sd1
        sd_control <- sd1
      }
    }
    
    # Вызываем рекурсивно с вычисленным d
    recursive_result <- effect_convert(
      d = d_cohen,
      n1 = n1,
      n2 = n2,
      paired = paired,
      conf.level = conf.level
    )
    
    # Добавляем Glass's Δ если есть
    if (exists("delta_glass") && !paired && !is.null(n1) && !is.null(n2)) {
      # 95% ДИ для Glass's Δ
      ci_delta_glass <- glass_delta_ci(delta_glass, sd_control, n1, n2, conf.level)
      
      glass_row <- data.frame(
        effect = "Glass's Δ",
        est = delta_glass,
        ci_lo = ci_delta_glass[1],
        ci_hi = ci_delta_glass[2],
        stringsAsFactors = FALSE
      )
      recursive_result$effects <- rbind(recursive_result$effects, glass_row)
    }
    
    return(recursive_result)
  }
  
  # ────────────────────────────────────────────────────────────────────────
  # I. ПРЕОБРАЗОВАНИЕ RR ↔ OR
  # ────────────────────────────────────────────────────────────────────────
  if (!is.null(RR) && !is.null(base_rate_final)) {
    rr_est <- RR
    
    # RR → OR
    or_from_rr <- conv_rr_to_or(rr_est, base_rate_final)
    
    # RR → h
    h_from_rr <- cohens_h_logrr(rr_est)
    
    results <- data.frame(
      effect = c("Relative Risk", "OR (from RR)", "Cohen's h (log RR)"),
      est = c(rr_est, or_from_rr, h_from_rr),
      ci_lo = NA,
      ci_hi = NA,
      stringsAsFactors = FALSE
    )
    
    out <- rbind(out, results)
  }
  
  # ────────────────────────────────────────────────────────────────────────
  # J. СОРТИРОВКА И ВОЗВРАТ РЕЗУЛЬТАТОВ
  # ────────────────────────────────────────────────────────────────────────
  
  if (nrow(out) == 0) {
    warning("Не удалось вычислить ни одного размера эффекта. Проверьте входные параметры.")
    effects <- data.frame()
  } else {
    # Удаляем дубликаты
    out <- out[!duplicated(out$effect), ]
    # Сортируем по алфавиту
    effects <- out[order(out$effect), ]
    rownames(effects) <- NULL
  }
  
  # Интерпретация эффектов
  interpretation <- list()
  if (nrow(effects) > 0) {
    for (i in 1:nrow(effects)) {
      eff <- effects$effect[i]
      val <- abs(effects$est[i])
      
      # Правила интерпретации по Коэну
      interpretation[[eff]] <- if (grepl("Cohen's d|d \\(|Glass's", eff)) {
        if (val < 0.2) "Очень малый эффект"
        else if (val < 0.5) "Малый эффект"
        else if (val < 0.8) "Средний эффект"
        else if (val < 1.2) "Большой эффект"
        else "Очень большой эффект"
      } else if (grepl("r \\(|Pearson", eff)) {
        if (val < 0.1) "Очень слабая связь"
        else if (val < 0.3) "Слабая связь"
        else if (val < 0.5) "Умеренная связь"
        else if (val < 0.7) "Сильная связь"
        else "Очень сильная связь"
      } else if (grepl("R²", eff)) {
        if (val < 0.01) "Очень малая объясненная дисперсия"
        else if (val < 0.09) "Малая объясненная дисперсия"
        else if (val < 0.25) "Умеренная объясненная дисперсия"
        else if (val < 0.49) "Большая объясненная дисперсия"
        else "Очень большая объясненная дисперсия"
      } else if (grepl("Cohen's f", eff)) {
        if (val < 0.10) "Очень малый эффект"
        else if (val < 0.25) "Малый эффект"
        else if (val < 0.40) "Средний эффект"
        else if (val < 0.60) "Большой эффект"
        else "Очень большой эффект"
      } else if (grepl("Cohen's h", eff)) {
        if (val < 0.2) "Очень малый эффект"
        else if (val < 0.5) "Малый эффект"
        else if (val < 0.8) "Средний эффект"
        else if (val < 1.2) "Большой эффект"
        else "Очень большой эффект"
      } else if (grepl("OR|RR", eff) && val != 1) {
        log_val <- abs(log(val))
        if (log_val < log(1.5)) "Очень малый эффект"
        else if (log_val < log(2.5)) "Малый эффект"
        else if (log_val < log(4.3)) "Средний эффект"
        else if (log_val < log(7.4)) "Большой эффект"
        else "Очень большой эффект"
      } else {
        "Интерпретация недоступна"
      }
    }
  }
  
  # Возвращаем структурированный результат
  result <- list(
    input = list(
      groups = groups,
      paired = paired,
      control_group = if (groups == "two") control_group else NA,
      n1 = n1,
      n2 = n2,
      total_N = Ntot,
      ratio = ratio,
      conf.level = conf.level,
      mu0 = if (groups == "one") mu0 else NA
    ),
    effects = effects,
    interpretation = interpretation
  )
  
  class(result) <- "effect_size_conversion"
  return(result)
}

##############################################################################
#  1.  MEANS AND NUMERICS            ###
##############################################################################

# ---  ESTIMATE mean / sd FROM 5-NUMBER SUMMARY (user)                     ###

SummaryMeanSD <- function(n, median, q1 = NULL, q3 = NULL,
                          min = NULL, max = NULL, conf.level = .95){
  # Following Wan et al. (2014) / Hozo et al. (2005)
  if (!is.null(q1) && !is.null(q3)){                 # quartiles available
    mean_est <- if (n >= 25) (q1 + median + q3) / 3
                else if (!is.null(min) && !is.null(max))
                  (min + 2 * median + max) / 4
                else (q1 + median + q3) / 3
    sd_est   <- (q3 - q1) / 1.35                       # iqr / 1.35
  } else if (!is.null(min) && !is.null(max)){         # only min / max
    mean_est <- (min + 2 * median + max) / 4
    sd_est   <- (max - min) / 4
  } else {
    stop("Need at least (q1 & q3) or (min & max)")
  }

  # CI for mean
  tcrit <- qt(1 - (1 - conf.level) / 2, df = n - 1)
  se_m  <- sd_est / sqrt(n)
  ci_m  <- mean_est + c(-1, 1) * tcrit * se_m

  # Cohen d vs zero (descriptive)
  d_est <- mean_est / sd_est

  tibble(metric    = c("mean", "sd", "cohen_d"),
         estimate  = c(mean_est, sd_est, d_est),
         ci_low    = c(ci_m[1], NA, NA),
         ci_high   = c(ci_m[2], NA, NA))
}


# --- SAMPLE-SIZE CALCULATORS (user)                                       ###

make_ss_tibble <- function(method, n1, n2, alpha, power, d){
  tibble(method   = method,
         n1       = n1,
         n2       = n2,
         total_N  = ifelse(is.na(n2), n1, n1 + n2),
         ratio    = ifelse(is.na(n2), NA_real_, n2 / n1),
         alpha    = alpha,
         power    = power,
         cohen_d  = round(d, 3),
         effect_OR = round(conv_d_to_or(d), 3))
}

# --- Normal approximation ----------------------------------------------------
SSMeanNorm <- function(alpha = 0.05, power = 0.80,
                       design = c("one.sample", "paired", "two.sample"),
                       sided  = c("two", "one"),
                       tail   = c("upper", "lower"),
                       d, ratio = 1){

  design <- match.arg(design); sided <- match.arg(sided); tail <- match.arg(tail)
  z_a <- if (sided == "two") qnorm(1 - alpha / 2) else
         if (tail == "upper") qnorm(1 - alpha) else abs(qnorm(alpha))
  z_b <- qnorm(power)         # positive
  k   <- if (design == "two.sample") (1 + ratio) / ratio else 1

  n1  <- ceiling(k * (z_a + z_b)^2 / d^2)
  n2  <- if (design == "two.sample") ceiling(ratio * n1) else NA

  make_ss_tibble("Normal", n1, n2, alpha, power, d)
}

# --- Exact non-central t ------------------------------------------------------
SSMeanT <- function(alpha = 0.05, power = 0.80,
                    design = c("one.sample", "paired", "two.sample"),
                    sided  = c("two", "one"),
                    tail   = c("upper", "lower"),
                    d, ratio = 1,
                    lower = 2, upper = 128, tol = 1e-8){

  design <- match.arg(design); sided <- match.arg(sided); tail <- match.arg(tail)
  beta <- 1 - power

  ncp <- function(n1){
    if (design == "two.sample"){
      n2 <- ratio * n1
      d * sqrt(n1 * n2 / (n1 + n2))
    } else d * sqrt(n1)
  }
  f <- function(n1){
    delta <- ncp(n1)
    df <- if (design == "two.sample") (1 + ratio) * n1 - 2 else n1 - 1
    if (sided == "one"){
      tcrit <- qt(if (tail == "upper") 1 - alpha else alpha, df)
      beta_hat <- if (tail == "upper") pt(tcrit, df, ncp = delta)
                  else 1 - pt(tcrit, df, ncp = delta)
    } else {
      tcrit <- qt(1 - alpha / 2, df)
      beta_hat <- pt(tcrit, df, ncp = delta) -
                  pt(-tcrit, df, ncp = delta)
    }
    beta_hat - beta
  }
  while (sign(f(lower)) == sign(f(upper))) upper <- upper * 2
  n1 <- ceiling(uniroot(f, c(lower, upper), tol = tol)$root)
  while (f(n1) > 0) n1 <- n1 + 1
  n2 <- if (design == "two.sample") ceiling(ratio * n1) else NA

  make_ss_tibble("Exact-t", n1, n2, alpha, power, d)
}

##############################################################################
#  1.  PROPORTIONS            ###
##############################################################################

make_prop_tibble <- function(method, n_trt, n_control, alpha, power,
                               rd = NA, h = NA, rr = NA, or = NA, d = NA){
    tibble::tibble(
      method          = method,
      n_treatment     = n_trt,                    # размер экспериментальной группы
      n_control       = n_control,                # размер контрольной группы
      total_N         = ifelse(is.na(n_control), n_trt, n_trt + n_control),
      ratio           = ifelse(is.na(n_control), NA_real_, n_trt / n_control),
      alpha           = alpha,
      power           = power,
      risk_difference = rd,
      cohen_h         = h,
      cohen_d         = d,
      RR              = rr,
      OR              = or
    )
}


# Вспомогательная функция для расчета z-критического значения
z_alpha_val <- function(alpha, sided, tail){
  if (sided == "two"){
    qnorm(1 - alpha / 2)
  } else {
    if (tail == "upper") qnorm(1 - alpha) else abs(qnorm(alpha))
  }
}

# ──────────────────────────────────────────────────────────────────────────
# USER FUNCTION
# ──────────────────────────────────────────────────────────────────────────
SSProp <- function(alpha = 0.05,
                   power = 0.80,
                   sided = c("two", "one"),
                   tail  = c("upper", "lower"),
                   groups = c("one", "two"),
                   # ------ one-sample inputs ------------------------------
                   p       = NULL,       # anticipated proportion
                   p0      = NULL,       # hypothetical proportion (for one-sample)
                   epsilon = NULL,       # |p - p0|
                   # ------ two-sample inputs ------------------------------
                   p_trt = NULL,         # proportion in treatment group
                   p_control = NULL,     # proportion in control group
                   # ------ Cohen's h inputs -------------------------------
                   h = NULL,             # Cohen's h effect size
                   # ------ shared parameters ------------------------------
                   ratio = 1             # n_trt / n_control
){
  
  groups <- match.arg(groups)
  sided  <- match.arg(sided)
  tail   <- match.arg(tail)
  
  # Проверяем корректность входных параметров
  if (groups == "one" && !is.null(h) && is.null(p0)) {
    stop("For one-sample design with Cohen's h, supply `p0` (hypothetical proportion)")
  }
  
  if (groups == "two" && !is.null(h) && is.null(p_control)) {
    stop("For two-sample design with Cohen's h, supply `p_control` (control group proportion)")
  }
  
  z_a <- z_alpha_val(alpha, sided, tail)
  z_b <- qnorm(power)  # positive because power > 0.5
  
  if (groups == "one") {
    # ── one-sample (single group) ----------------------------------------------------------
    if (!is.null(h)) {
      # Если задан Cohen's h, вычисляем p и epsilon
      if (is.null(p0)) stop("For one-sample design with Cohen's h, supply `p0`")
      
      # Вычисляем ожидаемую пропорцию p из h и p0
      p_calc <- sin(asin(sqrt(p0)) + h/2)^2
      p_calc <- max(0, min(1, p_calc))
      
      p <- p_calc
      epsilon <- abs(p - p0)
    } else {
      # Используем обычные параметры
      if (is.null(p) || (is.null(epsilon) && is.null(p0))) {
        stop("For a one-sample design supply `p` and (`epsilon` or `p0`).")
      }
      
      if (is.null(epsilon) && !is.null(p0)) {
        epsilon <- abs(p - p0)
      }
    }
    
    # Проверяем корректность параметров
    if (p <= 0 || p >= 1) stop("`p` must be in (0,1)")
    if (epsilon <= 0) stop("`epsilon` must be > 0")
    
    # Расчет размера выборки для одной группы
    n <- ceiling(((z_a + z_b)^2 * p * (1 - p)) / epsilon^2)
    
    # Расчет Cohen's h для вывода
    h_calc <- if (!is.null(h)) h else 2 * (asin(sqrt(p)) - asin(sqrt(p0)))
    
    # Правильный вызов функции с именованными аргументами
    make_prop_tibble(
      method = "Prop-1sample", 
      n_trt = n, 
      n_control = NA,
      alpha = alpha, 
      power = power,
      rd = epsilon,
      h = h_calc
    )
    
  } else {
    # ── two-sample (treatment vs control) ----------------------------------------------------------
    if (!is.null(h)) {
      # Если задан Cohen's h, вычисляем p_trt из p_control
      if (is.null(p_control)) {
        stop("For two-sample design with Cohen's h, supply `p_control` (control group proportion)")
      }
      
      # Вычисляем p_trt из h и p_control
      p_trt_calc <- sin(asin(sqrt(p_control)) + h/2)^2
      p_trt_calc <- max(0, min(1, p_trt_calc))
      
      p_trt <- p_trt_calc
    } else {
      # Проверяем обязательные параметры
      if (is.null(p_trt) || is.null(p_control)) {
        stop("For a two-sample design supply `p_trt` and `p_control` (or use Cohen's h with `p_control`).")
      }
    }
    
    # Проверяем корректность параметров
    if (p_trt <= 0 || p_trt >= 1) stop("`p_trt` must be in (0,1)")
    if (p_control <= 0 || p_control >= 1) stop("`p_control` must be in (0,1)")
    
    # Вычисляем параметры эффекта
    rd  <- p_trt - p_control        # signed risk difference
    eps <- abs(rd)
    
    # Расчет размера выборки для двух групп
    n_control <- ((z_a + z_b)^2 / eps^2) *
      (p_trt * (1 - p_trt) / ratio + p_control * (1 - p_control))
    n_control <- ceiling(n_control)
    n_trt <- ceiling(ratio * n_control)
    
    # Расчет дополнительных метрик эффекта
    h_calc  <- if (!is.null(h)) h else 2 * asin(sqrt(p_trt)) - 2 * asin(sqrt(p_control))
    rr <- p_trt / p_control
    or_val <- (p_trt / (1 - p_trt)) / (p_control / (1 - p_control))
    d_val <- conv_or_to_d(or_val)
    
    # Правильный вызов функции с именованными аргументами
    make_prop_tibble(
      method = "Prop-2sample", 
      n_trt = n_trt, 
      n_control = n_control,
      alpha = alpha, 
      power = power,
      rd = rd, 
      h = h_calc, 
      d = d_val, 
      rr = rr, 
      or = or_val
    )
  }
}

# ──────────────────────────────────────────────────────────────────────────
##  Sample size for a given Odds Ratio (Wald, large-sample)  ––  SSOR()     ##
# ──────────────────────────────────────────────────────────────────────────

SSOR <- function(alpha = 0.05,
                 power = 0.80,
                 sided = c("two", "one"),
                 tail  = c("upper", "lower"),
                 OR,                     # alternative odds ratio to detect
                 pC,                     # event rate in control group
                 ratio = 1              # nT / nC  (k on the slide)
){
  
  sided <- match.arg(sided)
  tail  <- match.arg(tail)
  
  conv_or_to_d <- function(or) {
    log(or) * sqrt(3) / pi
  }
  
  if (OR <= 0) stop("OR must be > 0")
  if (pC <= 0 || pC >= 1) stop("pC must be in (0,1)")
  
  z_a <- z_alpha_val(alpha, sided, tail)
  z_b <- qnorm(power)                     # positive because power > 0.5
  
  # ---- derive event rate in treatment arm from OR and pC ------------------
  pT <- OR * pC / (1 - pC + OR * pC)
  if (pT <= 0 || pT >= 1)
    stop("Computed pT out of range; check OR and pC.")
  
  # ---- base formula -------------------------------------------------------
  logOR <- log(OR)
  nC <- ((z_a + z_b)^2 / logOR^2) *
    ( 1/(ratio * pT * (1 - pT)) + 1/(pC * (1 - pC)) )
  nC <- ceiling(nC)
  nT <- ceiling(ratio * nC)
  
  # ---- additional effect metrics -----------------------------------------
  rd <- pT - pC
  h  <- 2 * asin(sqrt(pT)) - 2 * asin(sqrt(pC))
  rr <- pT / pC
  d <- conv_or_to_d(OR)
  
  # ИСПРАВЛЕННЫЙ ВЫЗОВ: используем правильные имена параметров
  # n_trt = nT (размер экспериментальной/лечебной группы)
  # n_control = nC (размер контрольной группы)
  make_prop_tibble(
    method = "OR-2sample",
    n_trt = nT,              # экспериментальная группа
    n_control = nC,          # контрольная группа
    alpha = alpha,
    power = power,
    rd = rd,
    h = h,
    rr = rr,
    or = OR,
    d = d
  )
}

##############################################################################
# Generic permutation test
##############################################################################

perm_test <- function(data,
                      group,
                      stat_fun,                  # user-supplied statistic function
                      R           = 9999,        # # permutations
                      alternative = c("two.sided", "less", "greater"),
                      seed        = NULL,
                      ...)                         # extra args forwarded to stat_fun
{
  alternative <- match.arg(alternative)

  if (!is.null(seed)) set.seed(seed)

  # ------------------------------------------------------------------
  # 1. Observed statistic
  # ------------------------------------------------------------------
  T_obs <- stat_fun(data, group, ...)

  if (length(T_obs) != 1 || !is.numeric(T_obs))
    stop("stat_fun must return ONE numeric value")

  # ------------------------------------------------------------------
  # 2. Permutation loop
  # ------------------------------------------------------------------
  perm_stats <- numeric(R)
  for (i in seq_len(R)) {
    perm_group      <- sample(group, length(group), replace = FALSE)  # shuffle labels
    perm_stats[i]   <- stat_fun(data, perm_group, ...)
  }

  # ------------------------------------------------------------------
  # 3. P-value (add +1 so it can never be zero)
  # ------------------------------------------------------------------
  p_val <- switch(alternative,
                  two.sided = (sum(abs(perm_stats) >= abs(T_obs)) + 1) / (R + 1),
                  greater   = (sum(perm_stats  >=      T_obs)  + 1) / (R + 1),
                  less      = (sum(perm_stats  <=      T_obs)  + 1) / (R + 1))

  # ------------------------------------------------------------------
  # 4. Return a nice object
  # ------------------------------------------------------------------
  out <- list(statistic   = T_obs,
              perm_stats  = perm_stats,
              p.value     = p_val,
              alternative = alternative,
              R           = R,
              call        = match.call())
  class(out) <- "perm_test"
  return(out)
}




