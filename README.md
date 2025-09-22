# Medstats module - Python scripts

**Authors / Авторы**: 

- **`Suvorov Aleksandr Yu.`** - Senior biostatistician in Research Services Department of Sechenov Univercity
- **`Vergun Maria A.`** - biostatistician in Research Services Department of Sechenov Univercity

Collection of scripts and modules for biostatistics and data science simple automation for research process.
Module uses `rpy2`-module and basic `R` libraries.

Коллекция скриптов и функций для биостатистики в Python. В модуле используются некоторые функции из R через `rpy2` (например, точный тест Фишера для таблиц $r \times k$).

# Module contents

Ниже представлено структурированное описание всех основных функций из представленного скрипта отдельно на русском и английском языках.

---

#### Data preparation operations

- **column_normalizer(df, col_lst)**:
  Converts symbols and brings string data into uniform format across specified columns. Spaces and punctuation marks are normalized, and strings can optionally be transformed into numbers if feasible.

- **smart_imputer(df, thresh=0.10, random_state=0)**:
  An intelligent strategy for filling missing values in dataset. It fills gaps depending on the nature of the data (numerical features are imputed using Bayesian Ridge Regression, categorical ones are replaced with most frequent values).

- **factor_transformer(data_frame, min_factor_levels=7)**:
  Transforms columns with few unique values into categorical data type (`pd.Categorical`) for better processing efficiency.

- **glimpse(df)**:
  Generates a comprehensive overview of the entire dataset, showing types of data, number of non-null elements, missing patterns, and unique values per column.

- **plot_missing_by_combos(df, cat_cols=None, ...)**
  Visualizes maps of missing data across different categories, helping identify potential patterns and correlations between missing values.

#### Hypothesis testing and group comparison

- **summary_all(df, digits=1)**:
  Produces a general report summarizing information about all columns, including tests for normality distributions and measures of central tendency (mean, median, min/max values).

- **compare_all(df, group_var, digits=1, add_minmax=False)**:
  Comprehensive comparison of groups based on quantitative and qualitative indicators, selecting the appropriate hypothesis test (such as t-test, ANOVA, Wilcoxon-Mann-Whitney).

- **pairwise_comparisons(df, group_var, p_adjust_method="none")**:
  Pairwise testing of groups to detect statistically significant differences between subgroups. Results include p-values and utilized statistical methods.

- **numerics_95CI(df, num_vars, statistic="automatic")**:
  Calculation of the 95% confidence interval for chosen numerical variables, with automatic checking for normal distribution (uses either mean or median).

- **binary_95CI(df, cat_vars)**:
  Determination of the 95% confidence interval for binary attributes (typically used for proportions or rates).

#### Regression modeling and risk assessment

- **univariate_lineareg(df, target, predictors, normalize_numeric=True)**:
  Univariate linear regression model with an option to pre-standardize numeric inputs before performing regression.

- **univariate_logreg(df, target, predictors, normalize_numeric=True)**:
  Binary logistic regression model for classification tasks, similar to previous function but applied to binomial outcomes.

- **calc_vif(df, target, thresh=5.0)**:
  Evaluation of multicollinearity among predictor variables using the Variance Inflation Factor (VIF).

- **step_cox(df, group, time, vars, iterations=1000, penalty=.001)**:
  Stepwise implementation of the Cox proportional hazards survival model, supporting sequential elimination of irrelevant factors.

- **onedim_coxregr(df, group, time, adj=False, adj_cols_lst=None)**:
  Single-dimensional Cox regression for individual variables, potentially adjusting for additional covariates.

#### Graphical representation of data

- **dist_box(df, var, label=None, label_X=None, label_Y="Number of Observations")**:
  Combined density and box-plot visualization for a given numerical attribute.

- **draw_data_frame(df, col_lst, pict_sav=True)**:
  Universal plotting function for visualizing multiple variables simultaneously (histograms, scatter plots, bar charts etc.).

- **draw_data_frame_group(df, col_lst, group, pict_sav=True, add_number=True)**:
  Group-based visualization, separating data according to a specific factor (like categorical subgrouping).

- **bland_altman_plot(data1, data2, x_label="", y_label="", save=False, name=None)**:
  Specialized Bland-Altman plot for comparing measurements made by two different methods.

- **polar_plot_circular(df, cols, id_var, figsize=(8, 15), save=False, figname=None, title='')**:
  Polar circular diagram designed to visualize frequencies of categorical data effectively.

#### Подготовительные операции с данными

- **column_normalizer(df, col_lst)**:
  Преобразование символов и приведение строковых данных к одному формату в указанных колонках. Пробелы и знаки препинания унифицируются, и, если возможно, строки преобразуются в числовое представление.

- **smart_imputer(df, thresh=0.10, random_state=0)**:
  Умная стратегия заполнения пропусков в датасете. Пропущенные значения восполняются различными методами в зависимости от типа данных (числовые обрабатываются байесовской регрессией, категориальные – заменой наиболее частым значением).

- **factor_transformer(data_frame, min_factor_levels=7)**:
  Конвертирует колонки с малым количеством уникальных значений в категориальный тип данных (`pd.Categorical`) для оптимизации дальнейших операций над такими признаками.

- **glimpse(df)**:
  Генерирует сводный отчет по всей таблице, содержащий тип данных, количество ненулевых элементов, долю и категории пропусков, уникальные значения.

- **plot_missing_by_combos(df, cat_cols=None, ...)**
  Визуализирует карту пропусков данных в зависимости от комбинаций категориальных признаков, выявляя возможные паттерны и закономерности.

#### Тестирование гипотез и сравнение групп

- **summary_all(df, digits=1)**:
  Генерация общего отчета по всем столбцам таблицы, включая оценку нормальности распределений и расчет центральных тенденций (среднее, медиана, минимальное/максимальное значения).

- **compare_all(df, group_var, digits=1, add_minmax=False)**:
  Проведение всестороннего сравнения групп по количественным и качественным показателям, выбор соответствующего метода проверки гипотезы (например, t-тест, ANOVA, Wilcoxon-Mann-Whitney).

- **pairwise_comparisons(df, group_var, p_adjust_method="none")**:
  Парное тестирование групп, позволяющее определить существенные различия между подгруппами. Результаты включают p-значения и используемые статистические тесты.

- **numerics_95CI(df, num_vars, statistic="automatic")**:
  Рассчёт 95%-го доверительного интервала для выбранных численных показателей, с проверкой нормальности распределения (используется либо среднее, либо медиана).

- **binary_95CI(df, cat_vars)**:
  Определение 95%-го доверительного интервала для бинарных признаков (обычно используется для долей или пропорций).

#### Модели регрессии и оценка риска

- **univariate_lineareg(df, target, predictors, normalize_numeric=True)**:
  Линейная регрессия одной независимой переменной с возможностью предварительного центрирования и стандартизации числовых данных.

- **univariate_logreg(df, target, predictors, normalize_numeric=True)**:
  Логистическая регрессия для бинарного отклика, аналогично предыдущей функции, но с применением Logit-модели.

- **calc_vif(df, target, thresh=5.0)**:
  Оценка мультиколлинеарности факторов с использованием показателя Variance Inflation Factor (VIF).

- **step_cox(df, group, time, vars, iterations=1000, penalty=.001)**:
  Модель Кокса для анализа выживаемости с поддержкой пошагового отбора признаков, исключающего малозначимые факторы.

- **onedim_coxregr(df, group, time, adj=False, adj_cols_lst=None)**:
  Одномерная модель Кокса для единичных переменных с возможностью включения поправочных факторов (adjustment).

#### Графическое представление данных

- **dist_box(df, var, label=None, label_X=None, label_Y="Количество наблюдений")**:
  Совмещённый график плотности и box-plot для выбранного численного признака.

- **draw_data_frame(df, col_lst, pict_sav=True)**:
  Универсальная функция для визуализации любого количества переменных (гистограммы, scatter plots, bar charts и т.п.). 

- **draw_data_frame_group(df, col_lst, group, pict_sav=True, add_number=True)**:
  Визуализация группированных данных с выделением подгрупп по указанному фактору.

- **bland_altman_plot(data1, data2, x_label="", y_label="", save=False, name=None)**:
  Специализированный график Бланд-Алтомана для сравнения измерений двумя разными методами.

- **polar_plot_circular(df, cols, id_var, figsize=(8, 15), save=False, figname=None, title='')**:
  Круговая полярная диаграмма для представления категоричных данных, позволяет визуально сравнить частоты разных категорий.
