# utf8

# # Description of the Module
# # ---
# # This Python module provides various statistical tools related to sample size calculations, effect size conversions, and other utility functions useful in research studies and data analysis. The core functionalities include calculating effect sizes such as Cohen’s d, converting between different measures like correlations (r), odds ratios (OR), and risk ratios (RR), as well as computing confidence intervals for these metrics.

# # Key Features:
# # Effect Size Conversions: Functions that convert between different types of effect sizes (e.g., from correlation coefficients to standardized mean differences).
# # Confidence Intervals: Methods to compute confidence intervals for correlation coefficients (r), means, proportions, and ratios using techniques like Fisher’s transformation and non-central t-distributions.
# # Sample Size Calculations: Tools to determine appropriate sample sizes based on desired levels of significance (alpha), power, and expected effect sizes (such as Cohen’s d or relative risks).
# # Handling Different Types of Data: Support for single-sample designs, paired comparisons, and two-sample analyses involving independent groups.
# # Helper Functions: Utility methods for calculating intermediate statistics, such as determining necessary quantiles for hypothesis testing.
# # The module integrates common statistical distributions (like normal distribution, Student's t-distribution, etc.) through libraries like SciPy and NumPy, ensuring accurate computations for real-world scenarios.

# # Usage Scenarios:
# # Researchers planning experiments can use this toolkit to calculate the minimum number of participants needed to achieve adequate statistical power.
# # Analysts working with clinical trials may employ it to assess the reliability of their findings by estimating confidence intervals around key parameters.
# # Students learning about statistical inference will find its examples instructive for understanding how effect sizes are calculated and interpreted.

import numpy as np
import pandas as pd
from scipy.stats import norm, t as t_dist, nct
from scipy.optimize import brentq

###############################################################################
# Helper functions and effect sizes
###############################################################################

# ------------------------------------------------------------------------------
# Effect size simple transformers

def conv_d_to_r(d: np.ndarray | float) -> np.ndarray | float:
    """
    Convert Cohen's d to Pearson's r.
    """
    return d / np.sqrt(d**2 + 4)


def conv_r_to_d(r: np.ndarray | float) -> np.ndarray | float:
    """
    Convert Pearson's r to Cohen's d.
    """
    return 2 * r / np.sqrt(1 - r**2)


def conv_d_to_or(d: np.ndarray | float) -> np.ndarray | float:
    """
    Convert Cohen's d to an odds ratio (OR).
    """
    return np.exp(np.pi / np.sqrt(3) * d)


def conv_or_to_d(or_: np.ndarray | float) -> np.ndarray | float:
    """
    Convert an odds ratio (OR) to Cohen's d.
    """
    return np.log(or_) * np.sqrt(3) / np.pi


def zcrit(conf: float = 0.95) -> float:
    """
    Two-tailed critical z value for a given confidence level.
    """
    return norm.ppf(1 - (1 - conf) / 2)


# ------------------------------------------------------------------------------
# Effect size 

def cohens_d(mean1, mean2,
             sd1, sd2,
             n1, n2=None,
             *,
             paired=False,
             r=None,          # required for paired if sd_diff not supplied
             sd_diff=None,    # supply if you already have SD of differences
             conf_level=.95,
             ci_method="wald"   # "wald" | "none"   (can add "exact" later)
             ):
    """
    Cohen's d for independent or paired samples + Wald confidence interval.
    
    Returns
    -------
    dict with keys: {'d', 'ci_low', 'ci_high'}
    """

    # ------------------------------------------------------------------ #
    # 1. Point estimate                                                  #
    # ------------------------------------------------------------------ #
    if not paired:                               # Independent groups
        if n2 is None:
            raise ValueError("Need n2 for independent samples.")
        # pooled SD
        s_pooled = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2)
                           / (n1 + n2 - 2))
        d = (mean1 - mean2) / s_pooled
    else:                                        # Paired / repeated
        if sd_diff is None:
            if r is None:
                raise ValueError("Need either `sd_diff` or correlation `r` "
                                 "for paired samples.")
            sd_diff = np.sqrt(sd1**2 + sd2**2 - 2 * r * sd1 * sd2)
        d = (mean1 - mean2) / sd_diff
        n2 = n1                                  # for consistency later

    # ------------------------------------------------------------------ #
    # 2. Standard error & Wald CI                                         #
    # ------------------------------------------------------------------ #
    if ci_method.lower() == "none":
        return {"d": d, "ci_low": np.nan, "ci_high": np.nan}

    z = norm.ppf(1 - (1 - conf_level) / 2)

    if not paired:
        # Hedges & Olkin (1985) variance of d
        se_d = np.sqrt((n1 + n2) / (n1 * n2) +
                       (d**2) / (2 * (n1 + n2 - 2)))
    else:
        # Approximate variance for paired d ("dz")
        # Var(d) ≈ 1/n + d²/(2n)   (see e.g. Morris & DeShon, 2002)
        se_d = np.sqrt(1 / n1 + d**2 / (2 * n1))

    ci_low = d - z * se_d
    ci_high = d + z * se_d

    return {"d": d,
            "ci_low": ci_low,
            "ci_high": ci_high}


# ------------------------------------------------------------------------------
# Effect size CI

def r_ci(r: float | np.ndarray, n: int, conf: float = 0.95) -> dict:
    """
    Fisher Z-transformed confidence interval for Pearson's r.
    
    Parameters
    ----------
    r    : correlation coefficient (scalar or array)
    n    : sample size
    conf : confidence level (default 0.95)
    
    Returns
    -------
    dict with keys 'lo' and 'hi'
    """
    z      = 0.5 * np.log((1 + r) / (1 - r))
    se     = 1 / np.sqrt(n - 3)
    zc     = zcrit(conf)
    
    lo_z   = z - zc * se
    hi_z   = z + zc * se
    
    lo_r   = (np.exp(2 * lo_z) - 1) / (np.exp(2 * lo_z) + 1)
    hi_r   = (np.exp(2 * hi_z) - 1) / (np.exp(2 * hi_z) + 1)
    
    return {"lo": lo_r, "hi": hi_r}


def ratio_ci(est: float | np.ndarray,
             se: float | np.ndarray,
             conf: float = 0.95) -> dict:
    """
    Wald confidence interval for a ratio measure (OR or RR).
    
    Parameters
    ----------
    est  : point estimate of the ratio (e.g., odds ratio)
    se   : standard error of log(est)
    conf : confidence level (default 0.95)
    
    Returns
    -------
    dict with keys 'lo' and 'hi'
    """
    zc = zcrit(conf)
    lo = np.exp(np.log(est) - zc * se)
    hi = np.exp(np.log(est) + zc * se)
    return {"lo": lo, "hi": hi}

# ------------------------------------------------------------------------------
# Effect size main converter

def effect_convert(
    *,
    groups       = "one",       # "one" | "two"
    paired       = False,       # актуально только для two-sample
    # -- ровно один из этих параметров ---------------------------------------
    rho          = None,
    proportion   = None,
    prop1        = None, prop2 = None,
    OR           = None, OR_lo = None, OR_hi = None,
    RR           = None, RR_lo = None, RR_hi = None,
    d            = None,
    mean         = None, sd = None,              # one-sample или diff
    mean1        = None, mean2 = None,
    sd1          = None,  sd2  = None,
    rho_paired   = None,
    # -- размеры выборок ------------------------------------------------------
    n            = None,
    n1           = None, n2 = None,
    conf_level   = 0.95
):
    # --- проверки входа -----------------------------------------------------
    if groups not in ("one", "two"):
        raise ValueError("`groups` must be 'one' or 'two'")
    
    primary = [rho, proportion, prop1, OR, RR, d, mean, mean1]
    if sum(x is not None for x in primary) != 1:
        raise ValueError("Supply exactly ONE primary statistic")

    # --- согласовать n, n1, n2 ---------------------------------------------
    if groups == "one":
        if n is None and n1 is None:
            raise ValueError("Need `n` (or `n1`) for one-group input")
        n1 = n if n1 is None else n1
        n2 = np.nan
    else:  # two groups
        if n1 is None or n2 is None:
            raise ValueError("Need both `n1` and `n2` for two-group input")

    Ntot  = n1 if np.isnan(n2) else n1 + n2
    ratio = np.nan if np.isnan(n2) else n2 / n1
    zc    = zcrit(conf_level)
    
    rows = []                      # будем аккумулировать строки таблицы

    # ── a) корреляция -------------------------------------------------------
    if rho is not None:
        r   = rho
        d_  = conv_r_to_d(r)
        or_ = conv_d_to_or(d_)
        R2  = r**2; eta2 = R2

        if n1 is not None and n1 > 3:
            ci_r = r_ci(r, n1, conf_level)
            ci_d = {k: conv_r_to_d(v) for k, v in ci_r.items()}
            ci_o = {k: conv_d_to_or(v) for k, v in ci_d.items()}
        else:
            ci_r = ci_d = ci_o = {"lo": np.nan, "hi": np.nan}

        rows += [
            dict(effect="r",   est=r,   ci_lo=ci_r["lo"], ci_hi=ci_r["hi"]),
            dict(effect="d",   est=d_,  ci_lo=ci_d["lo"], ci_hi=ci_d["hi"]),
            dict(effect="OR",  est=or_, ci_lo=ci_o["lo"], ci_hi=ci_o["hi"]),
            dict(effect="R2",  est=R2,  ci_lo=np.nan,     ci_hi=np.nan),
            dict(effect="eta2",est=eta2,ci_lo=np.nan,     ci_hi=np.nan),
        ]

    # ── b) одна доля --------------------------------------------------------
    if proportion is not None:
        if groups != "one":
            raise ValueError("`proportion` only valid for one group")
        p = proportion; q = 1 - p
        h = 2*np.arcsin(np.sqrt(p)) - 2*np.arcsin(np.sqrt(.5))
        se_p = np.sqrt(p*q/n1)
        ci_p = np.clip(p + np.array([-1, 1]) * zc * se_p, 0, 1)

        rows += [
            dict(effect="proportion", est=p, ci_lo=ci_p[0], ci_hi=ci_p[1]),
            dict(effect="h",          est=h, ci_lo=np.nan,  ci_hi=np.nan),
        ]

    # ── c) две доли ---------------------------------------------------------
    if prop1 is not None:
        if groups != "two":
            raise ValueError("`prop1`/`prop2` require two groups")
        p1, p2 = prop1, prop2
        q1, q2 = 1 - p1, 1 - p2

        rd = p1 - p2
        rr = p1 / p2
        or_ = (p1/q1) / (p2/q2)
        h   = 2*np.arcsin(np.sqrt(p1)) - 2*np.arcsin(np.sqrt(p2))

        # ---------- доверительные интервалы --------------------------------
        se_rd   = np.sqrt(p1*q1/n1 + p2*q2/n2)
        ci_rd   = rd + np.array([-1, 1]) * zc * se_rd

        se_lnrr = np.sqrt(q1/(n1*p1) + q2/(n2*p2))
        ci_rr   = ratio_ci(rr, se_lnrr, conf_level)

        a,b,c,d_ = p1*n1, p2*n2, q1*n1, q2*n2
        se_lnor  = np.sqrt(1/a + 1/b + 1/c + 1/d_)
        ci_or    = ratio_ci(or_, se_lnor, conf_level)

        # ---------- новый расчёт Cohen's d ---------------------------------
        d_prop   = conv_or_to_d(or_)
        ci_dprop = {k: conv_or_to_d(v) for k, v in ci_or.items()}
        # -------------------------------------------------------------------

        rows += [
            dict(effect="risk_diff", est=rd,   ci_lo=ci_rd[0],   ci_hi=ci_rd[1]),
            dict(effect="RR",        est=rr,   ci_lo=ci_rr["lo"],ci_hi=ci_rr["hi"]),
            dict(effect="OR",        est=or_,  ci_lo=ci_or["lo"],ci_hi=ci_or["hi"]),
            dict(effect="d",         est=d_prop,                 
                                    ci_lo=ci_dprop["lo"],
                                    ci_hi=ci_dprop["hi"]),
            dict(effect="h",         est=h,    ci_lo=np.nan,     ci_hi=np.nan),
        ]

    # ── d) задан OR ---------------------------------------------------------
    if OR is not None and prop1 is None:
        or_ = OR
        d_  = conv_or_to_d(or_)
        r   = conv_d_to_r(d_)
        R2  = r**2; eta2 = R2

        ci_or = {"lo": OR_lo, "hi": OR_hi} if OR_lo is not None else {"lo": np.nan, "hi": np.nan}
        if not np.isnan(ci_or["lo"]):
            ci_d  = {k: conv_or_to_d(v) for k, v in ci_or.items()}
            ci_r  = {k: conv_d_to_r(v)  for k, v in ci_d.items()}
        else:
            ci_d = ci_r = {"lo": np.nan, "hi": np.nan}

        rows += [
            dict(effect="OR",  est=or_, ci_lo=ci_or["lo"], ci_hi=ci_or["hi"]),
            dict(effect="d",   est=d_,  ci_lo=ci_d["lo"],  ci_hi=ci_d["hi"]),
            dict(effect="r",   est=r,   ci_lo=ci_r["lo"],  ci_hi=ci_r["hi"]),
            dict(effect="R2",  est=R2,  ci_lo=np.nan,      ci_hi=np.nan),
            dict(effect="eta2",est=eta2,ci_lo=np.nan,      ci_hi=np.nan),
        ]

    # ── e) задан RR ---------------------------------------------------------
    if RR is not None and OR is None and prop1 is None:
        rr = RR
        # приближение через функцию для OR – как в R-коде
        d_  = conv_or_to_d(rr)
        r   = conv_d_to_r(d_)
        R2  = r**2; eta2 = R2

        ci_rr = {"lo": RR_lo, "hi": RR_hi} if RR_lo is not None else {"lo": np.nan, "hi": np.nan}
        if not np.isnan(ci_rr["lo"]):
            ci_d = {k: conv_or_to_d(v) for k, v in ci_rr.items()}
            ci_r = {k: conv_d_to_r(v)  for k, v in ci_d.items()}
        else:
            ci_d = ci_r = {"lo": np.nan, "hi": np.nan}

        rows += [
            dict(effect="RR",  est=rr, ci_lo=ci_rr["lo"], ci_hi=ci_rr["hi"]),
            dict(effect="d",   est=d_,  ci_lo=ci_d["lo"], ci_hi=ci_d["hi"]),
            dict(effect="r",   est=r,   ci_lo=ci_r["lo"], ci_hi=ci_r["hi"]),
            dict(effect="R2",  est=R2,  ci_lo=np.nan,     ci_hi=np.nan),
            dict(effect="eta2",est=eta2,ci_lo=np.nan,     ci_hi=np.nan),
        ]

    # ── f) d или сырые средние/SD ------------------------------------------
    if (mean is not None) or (mean1 is not None) or (d is not None):
        # вычислить d_calc
        if d is not None:
            d_calc = d
        elif groups == "one":
            if mean is None or sd is None:
                raise ValueError("Need mean & sd for one-sample effect")
            d_calc = mean / sd
        elif paired:
            if None in (mean1, mean2, sd1, sd2):
                raise ValueError("Need mean1/2 & sd1/2 for paired design")
            if rho_paired is None:
                raise ValueError("Provide `rho_paired` for paired design")
            sd_diff = np.sqrt(sd1**2 + sd2**2 - 2 * rho_paired * sd1 * sd2)
            d_calc  = (mean1 - mean2) / sd_diff
        else:  # two-sample, независимые
            if None in (mean1, mean2, sd1, sd2):
                raise ValueError("Need mean1/2 & sd1/2 for two-sample design")
            sp = np.sqrt((sd1**2 + sd2**2) / 2)
            d_calc = (mean1 - mean2) / sp

        r   = conv_d_to_r(d_calc)
        or_ = conv_d_to_or(d_calc)
        R2  = r**2; eta2 = R2

        rows += [
            dict(effect="d",    est=d_calc, ci_lo=np.nan, ci_hi=np.nan),
            dict(effect="r",    est=r,      ci_lo=np.nan, ci_hi=np.nan),
            dict(effect="OR",   est=or_,    ci_lo=np.nan, ci_hi=np.nan),
            dict(effect="R2",   est=R2,     ci_lo=np.nan, ci_hi=np.nan),
            dict(effect="eta2", est=eta2,   ci_lo=np.nan, ci_hi=np.nan),
        ]

    # --- итоговая таблица ----------------------------------------------------
    effects_df = (pd.DataFrame(rows)
                    .drop_duplicates(subset=["effect"])        # на всякий
                    .sort_values("effect")
                    .reset_index(drop=True))

    return {
        "input":  dict(groups=groups, paired=paired,
                       n1=n1, n2=n2, total_N=Ntot, ratio=ratio),
        "effects": effects_df
    }

###############################################################################
# Means and numerics
###############################################################################


def summary_mean_sd(n, median, q1=None, q3=None,
                    min_=None, max_=None, conf_level=.95):
    """
    Оценка mean / SD из медианы, квартилей и/или min/max
    (Wan et al., 2014; Hozo et al., 2005).

    Возвращает pandas.DataFrame с колонками:
    metric ∈ {"mean", "sd", "cohen_d"}, estimate, ci_low, ci_high
    """

    # --- определить, какие числа доступны
    have_quart = (q1 is not None) and (q3 is not None)
    have_minmax = (min_ is not None) and (max_ is not None)

    if have_quart:
        if n >= 25:
            mean_est = (q1 + median + q3) / 3
        elif have_minmax:                       # n < 25  +  есть min/max
            mean_est = (min_ + 2*median + max_) / 4
        else:
            mean_est = (q1 + median + q3) / 3
        sd_est = (q3 - q1) / 1.35               # IQR / 1.35
    elif have_minmax:                           # только min/max
        mean_est = (min_ + 2*median + max_) / 4
        sd_est = (max_ - min_) / 4
    else:
        raise ValueError("Need at least (q1 & q3) or (min & max)")

    # --- доверительный интервал для mean
    tcrit = t_dist.ppf(1 - (1 - conf_level) / 2, df=n-1)
    se_m  = sd_est / np.sqrt(n)
    ci_m  = mean_est + np.array([-1, 1]) * tcrit * se_m

    # --- Cohen's d (среднее против нуля, дескриптивно)
    d_est = mean_est / sd_est

    df_out = pd.DataFrame({
        "metric":   ["mean", "sd", "cohen_d"],
        "estimate": [mean_est, sd_est, d_est],
        "ci_low":   [ci_m[0],  np.nan, np.nan],
        "ci_high":  [ci_m[1],  np.nan, np.nan]
    })

    return df_out


def _make_ss_table(method, n1, n2, alpha, power, d):
    """
    Вспомогательная «табличка результата» для 
    sample-size калькуляторов
    """

    return pd.DataFrame({
        "method":    [method],
        "n1":        [int(n1)],
        "n2":        [np.nan if n2 is None else int(n2)],
        "total_N":   [int(n1) if n2 is None else int(n1 + n2)],
        "ratio":     [np.nan if n2 is None else n2 / n1],
        "alpha":     [alpha],
        "power":     [power],
        "cohen_d":   [round(d, 3)],
        "effect_OR": [round(conv_d_to_or(d), 3)]
    })

# ---------------------------------------------------------------------------
# Нормальная аппроксимация (z-тест)


def ss_mean_norm(alpha=.05, power=.80,
                 design="one.sample",      # "one.sample" | "paired" | "two.sample"
                 sided="two",              # "two" | "one"
                 tail="upper",             # only if sided == "one"
                 d=None, ratio=1):
    if d is None:
        raise ValueError("Effect size `d` must be provided")

    design  = {"one.sample", "paired", "two.sample"}.intersection([design]).pop()
    sided   = {"two", "one"}.intersection([sided]).pop()
    tail    = {"upper", "lower"}.intersection([tail]).pop()

    # --- критические квантилы
    if sided == "two":
        z_a = norm.ppf(1 - alpha / 2)
    else:  # one-sided
        z_a = norm.ppf(1 - alpha) if tail == "upper" else abs(norm.ppf(alpha))
    z_b = norm.ppf(power)          # положителен

    k   = (1 + ratio) / ratio if design == "two.sample" else 1
    n1  = np.ceil(k * (z_a + z_b)**2 / d**2).astype(int)
    n2  = int(np.ceil(ratio * n1)) if design == "two.sample" else None

    return _make_ss_table("Normal", n1, n2, alpha, power, d)

# ---------------------------------------------------------------------------
# Точная модель с нецентральным t-распределением


def ss_mean_t(alpha=.05, power=.80,
              design="one.sample",
              sided="two",
              tail="upper",
              d=None, ratio=1,
              lower=2, upper=128, tol=1e-8):
    """
    Точный расчёт размера выборки через нецентральное t-распределение.
    Ровно те же аргументы, что и у оригинальной R-функции.
    """
    if d is None:
        raise ValueError("Effect size `d` must be provided")

    design  = {"one.sample", "paired", "two.sample"}.intersection([design]).pop()
    sided   = {"two", "one"}.intersection([sided]).pop()
    tail    = {"upper", "lower"}.intersection([tail]).pop()
    beta = 1 - power

    # --- функция нецентральности
    def ncp(n1):
        n1 = float(n1)
        if design == "two.sample":
            n2 = ratio * n1
            return d * np.sqrt(n1 * n2 / (n1 + n2))
        else:
            return d * np.sqrt(n1)

    # --- функция, ноль которой ищем: f(n1) = beta_hat - beta
    def f(n1):
        delta = ncp(n1)
        if design == "two.sample":
            df = (1 + ratio) * n1 - 2
        else:
            df = n1 - 1

        if sided == "one":
            tcrit = t_dist.ppf(1 - alpha, df) if tail == "upper" \
                    else t_dist.ppf(alpha, df)
            if tail == "upper":
                beta_hat = nct.cdf(tcrit, df, nc=delta)
            else:  # lower tail
                beta_hat = 1 - nct.cdf(tcrit, df, nc=delta)
        else:  # two-sided
            tcrit = t_dist.ppf(1 - alpha / 2, df)
            beta_hat = nct.cdf(tcrit, df, nc=delta) - nct.cdf(-tcrit, df, nc=delta)

        return beta_hat - beta

    # --- подобрать интервал, в котором знаки разные
    while np.sign(f(lower)) == np.sign(f(upper)):
        upper *= 2                               # растягиваем диапазон

    # --- корень
    n1_root = brentq(f, lower, upper, xtol=tol)
    n1 = int(np.ceil(n1_root))
    # иногда после округления beta_hat чуть > beta – сделать +1
    while f(n1) > 0:
        n1 += 1

    n2 = int(np.ceil(ratio * n1)) if design == "two.sample" else None
    return _make_ss_table("Exact-t", n1, n2, alpha, power, d)

###############################################################################
# Proportions and odds
###############################################################################

# ---------------------------------------------------------------------------
# Вспомогательные функции

def z_alpha_val(alpha: float, sided: str, tail: str) -> float:
    """
    Критическое Z-значение для заданных alpha / sided / tail.
    sided ∈ {"two", "one"}
    tail  ∈ {"upper", "lower"} (используется, если sided == "one")
    """
    if sided == "two":
        return norm.ppf(1 - alpha / 2)          # qnorm(1-α/2)
    else:                                       # one-sided
        return norm.ppf(1 - alpha) if tail == "upper" else abs(norm.ppf(alpha))


def _make_prop_table(method, n1, n2, alpha, power,
                     rd=np.nan, h=np.nan, rr=np.nan, or_=np.nan, d=np.nan):
    """
    Формирует tidy-таблицу (pandas.DataFrame) с результатами.
    """
    total_N = n1 if n2 is None or np.isnan(n2) else n1 + n2
    ratio   = np.nan if n2 is None or np.isnan(n2) else n2 / n1

    return pd.DataFrame({
        "method":          [method],
        "n1":              [int(n1) if not np.isnan(n1) else np.nan],
        "n2":              [np.nan if n2 is None else int(n2)],
        "total_N":         [total_N],
        "ratio":           [ratio],
        "alpha":           [alpha],
        "power":           [power],
        "risk_difference": [rd],
        "cohen_h":         [h],
        "cohen_d":         [d],
        "RR":              [rr],
        "OR":              [or_]
    })


# ---------------------------------------------------------------------------
# Proportions - main function


def ss_prop(alpha=.05,
            power=.80,
            sided="two",          # "two" | "one"
            tail="upper",         # "upper" | "lower"  (если sided == "one")
            groups="one",         # "one" | "two"
            # ---- параметры для one-sample ----------------------------------
            p=None,               # ожидаемая пропорция
            epsilon=None,         # |p - p0|
            # ---- параметры для two-sample ----------------------------------
            p1=None, p2=None,
            ratio=1               # n2 / n1
            ) -> pd.DataFrame:

    groups = {"one", "two"}.intersection([groups]).pop()
    sided  = {"two", "one"}.intersection([sided]).pop()
    tail   = {"upper", "lower"}.intersection([tail]).pop()

    z_a = z_alpha_val(alpha, sided, tail)
    z_b = norm.ppf(power)     # положителен, т.к. power > 0.5

    # ------------------------------------------------------------------ #
    # ONE-SAMPLE                                                         #
    # ------------------------------------------------------------------ #
    if groups == "one":
        if (p is None) or (epsilon is None):
            raise ValueError("For a one-sample design supply `p` and `epsilon`.")

        n = np.ceil(((z_a + z_b)**2 * p * (1 - p)) / epsilon**2).astype(int)

        return _make_prop_table("Prop-1sample",
                                n1=n, n2=None,
                                alpha=alpha, power=power,
                                rd=epsilon)

    # ------------------------------------------------------------------ #
    # TWO-SAMPLE                                                         #
    # ------------------------------------------------------------------ #
    else:
        if (p1 is None) or (p2 is None):
            raise ValueError("For a two-sample design supply `p1` and `p2`.")

        rd  = p1 - p2
        eps = abs(rd)

        # сначала n2 (как в оригинальном R-коде), потом n1 = ratio * n2
        n2 = np.ceil(((z_a + z_b)**2 / eps**2) *
                     (p1 * (1 - p1) / ratio + p2 * (1 - p2))).astype(int)
        n1 = int(np.ceil(ratio * n2))

        # дополнительные эффекты
        h  = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
        rr = p1 / p2
        or_ = (p1 / (1 - p1)) / (p2 / (1 - p2))
        d   = conv_or_to_d(or_)

        return _make_prop_table("Prop-2sample",
                                n1=n1, n2=int(n2),
                                alpha=alpha, power=power,
                                rd=rd, h=h, d=d, rr=rr, or_=or_)
    

# ---------------------------------------------------------------------------
# Odds - main function


def ss_or(alpha=.05,
          power=.80,
          sided="two",          # "two" | "one"
          tail="upper",         # "upper" | "lower" (если sided == "one")
          OR=None,              # альтернативное отношение шансов (>0)
          pC=None,              # ожидаемая частота события в контроле (0–1)
          ratio=1               # nT / nC  (treatment : control)
          ):
    """
    Sample-size for detecting a given odds ratio (large-sample Wald test).

    Returns
    -------
    pandas.DataFrame  ( tidy-style, совместимо с предыдущими функциями)
    """

    # ---- проверки аргументов ---------------------------------------------
    if OR is None:
        raise ValueError("`OR` must be provided")
    if OR <= 0:
        raise ValueError("`OR` must be > 0")

    if pC is None:
        raise ValueError("`pC` must be provided")
    if not (0 < pC < 1):
        raise ValueError("`pC` must lie in (0, 1)")

    sided = {"two", "one"}.intersection([sided]).pop()
    tail  = {"upper", "lower"}.intersection([tail]).pop()

    # ---- критические квантилы --------------------------------------------
    z_a = z_alpha_val(alpha, sided, tail)
    z_b = norm.ppf(power)                 # всегда > 0, т.к. power > .5

    # ---- частота события в treatment-группе ------------------------------
    pT = OR * pC / (1 - pC + OR * pC)
    if not (0 < pT < 1):
        raise ValueError("Computed pT out of (0,1); check OR & pC.")

    # ---- основная формула Уолда ------------------------------------------
    logOR = np.log(OR)
    nC = np.ceil(((z_a + z_b)**2 / logOR**2) *
                 (1 / (ratio * pT * (1 - pT)) +
                  1 / (pC * (1 - pC)))).astype(int)
    nT = int(np.ceil(ratio * nC))

    # ---- дополнительные метрики эффекта ----------------------------------
    rd = pT - pC
    h  = 2 * np.arcsin(np.sqrt(pT)) - 2 * np.arcsin(np.sqrt(pC))
    rr = pT / pC
    d  = conv_or_to_d(OR)

    # ---- tidy-вывод -------------------------------------------------------
    return _make_prop_table(method="OR-2sample",
                            n1=nC,         # control = n1
                            n2=nT,         # treatment = n2
                            alpha=alpha,
                            power=power,
                            rd=rd,
                            h=h,
                            rr=rr,
                            or_=OR,
                            d=d)
