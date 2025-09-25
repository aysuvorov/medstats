
library(tidyverse)

# +-----------------------------------------------------------------------------
# +-----------------------------------------------------------------------------
# Одна выборка - известна дисперсия генеральной совокупности
# Одновыборочный и парный тест
# +-----------------------------------------------------------------------------
# +-----------------------------------------------------------------------------

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers ────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
library(tidyverse)

##############################################################################
#  0.  UNIVERSAL HELPER FUNCTIONS                                           ###
##############################################################################
conv_d_to_r   <- function(d) d / sqrt(d^2 + 4)
conv_r_to_d   <- function(r) 2 * r / sqrt(1 - r^2)
conv_d_to_or  <- function(d) exp(pi / sqrt(3) * d)
conv_or_to_d  <- function(or) log(or) * sqrt(3) / pi
zcrit         <- function(conf = .95) qnorm(1 - (1 - conf) / 2)

# Fisher-Z CI for Pearson r
r_ci <- function(r, n, conf = .95){
  z  <- 0.5 * log((1 + r)/(1 - r))
  se <- 1 / sqrt(n - 3)
  zc <- zcrit(conf)
  lo <- z - zc * se
  hi <- z + zc * se
  c(lo = (exp(2 * lo) - 1)/(exp(2 * lo) + 1),
    hi = (exp(2 * hi) - 1)/(exp(2 * hi) + 1))
}

# log-ratio (OR / RR) Wald CI
ratio_ci <- function(est, se, conf = .95){
  zc <- zcrit(conf)
  c(lo = exp(log(est) - zc * se),
    hi = exp(log(est) + zc * se))
}

cohen_d_ci <- function(d, n1, n2 = NULL, paired = FALSE,
                       conf.level = .95)
{
  if (is.null(n2) || paired) {                 # one-sample or paired
    n  <- n1
    se <- if (n > 2) sqrt(1/n + d^2 / (2*(n - 1))) else NA
  } else {                                     # two independent groups
    se <- if (n1 + n2 > 2)
            sqrt((n1 + n2)/(n1*n2) + d^2 / (2*(n1 + n2 - 2)))
          else NA
  }

  z  <- qnorm(1 - (1 - conf.level)/2)
  ci <- if (!is.na(se)) d + c(-1, 1) * z * se else c(NA, NA)
  c(ci.lo = ci[1], ci.hi = ci[2])
}


##############################################################################
#  1.  EFFECT-SIZE CONVERTER  (returns list: $input + $effects)            ###
##############################################################################
effect_convert <- function(
        groups      = c("one", "two"),
        paired      = FALSE,
        rho         = NULL,
        proportion  = NULL,
        prop1       = NULL, prop2 = NULL,
        OR          = NULL, OR_lo = NULL, OR_hi = NULL,
        RR          = NULL, RR_lo = NULL, RR_hi = NULL,
        d           = NULL,
        mean        = NULL, sd = NULL,
        mean1       = NULL, mean2 = NULL,
        sd1         = NULL,  sd2  = NULL,
        rho_paired  = NULL,
        n           = NULL,
        n1          = NULL, n2 = NULL,
        conf.level  = .95)
{
  ## ── 0. detect groups if user forgot to set it ---------------------------
  # if (missing(groups)) {
  #   groups <- if (!is.null(n2)) "two" else "one"
  # }
  groups <- match.arg(groups)

  if (groups == "one" && !is.null(n2)) {
    groups <- "two"
  }

  ## ── 1. check exactly ONE primary statistic -----------------------------
  primary <- list(rho, proportion, prop1, OR, RR, d, mean, mean1)
  if (sum(!vapply(primary, is.null, FALSE)) != 1)
    stop("Supply exactly ONE primary statistic")

  ## ── 2. harmonise sample sizes ------------------------------------------
  if (groups == "one") {
    if (is.null(n) && is.null(n1))
      stop("Need `n` for one-group input")
    n1 <- ifelse(is.null(n1), n, n1);  n2 <- NA
  } else {
    if (is.null(n1) || is.null(n2))
      stop("Need `n1` and `n2` for two-group input")
  }
  Ntot  <- ifelse(is.na(n2), n1, n1 + n2)
  ratio <- ifelse(is.na(n2), NA, n2 / n1)

  zc  <- zcrit(conf.level)
  out <- list()

  # ‑- a) correlation ---------------------------------------------------------
  if (!is.null(rho)){
    r <- rho
    d  <- conv_r_to_d(r)
    or <- conv_d_to_or(d)
    R2 <- r^2; eta2 <- R2
    ci_r  <- if (!is.null(n1) && n1 > 3) r_ci(r, n1, conf.level) else c(NA, NA)
    ci_d  <- conv_r_to_d(ci_r);  ci_or <- conv_d_to_or(ci_d)
    out <- bind_rows(out,
      tibble(effect = c("r", "d", "OR", "R2", "eta2"),
             est    = c(r,  d,  or,  R2,   eta2),
             ci_lo  = c(ci_r[1], ci_d[1], ci_or[1], NA, NA),
             ci_hi  = c(ci_r[2], ci_d[2], ci_or[2], NA, NA)))
  }

  # ‑- b) single proportion ---------------------------------------------------
  if (!is.null(proportion)){
    if (groups != "one") stop("`proportion` only valid for one group")
    p <- proportion; q <- 1 - p
    h <- 2 * asin(sqrt(p)) - 2 * asin(sqrt(.5))
    se_p <- sqrt(p * q / n1)
    ci_p <- p + c(-1, 1) * zc * se_p
    ci_p <- pmax(0, pmin(1, ci_p))
    out <- bind_rows(out,
      tibble(effect = c("proportion", "h"),
             est    = c(p, h),
             ci_lo  = c(ci_p[1], NA),
             ci_hi  = c(ci_p[2], NA)))
  }

  # ­- c) two proportions -------------------------------------------------
  if (!is.null(prop1)){                         # groups must be "two"
    if (groups != "two")
      stop("`prop1`/`prop2` require two groups")

    p1 <- prop1;  p2 <- prop2
    q1 <- 1 - p1; q2 <- 1 - p2

    rd <- p1 - p2                  # risk difference
    rr <- p1 / p2                  # risk ratio
    or <- (p1 / q1) / (p2 / q2)    # odds ratio
    h  <- 2*asin(sqrt(p1)) - 2*asin(sqrt(p2))  # Cohen's h (arcsine)

    ## --- Wald SE + CI for each primary ratio -----------------------------
    se_rd   <- sqrt(p1*q1/n1 + p2*q2/n2)
    ci_rd   <- rd + c(-1,1)*zc*se_rd

    se_lnrr <- sqrt(q1/(n1*p1) + q2/(n2*p2))
    ci_rr   <- ratio_ci(rr, se_lnrr, conf.level)

    a <- p1*n1; b <- p2*n2; c <- q1*n1; d_ <- q2*n2
    se_lnor <- sqrt(1/a + 1/b + 1/c + 1/d_)
    ci_or   <- ratio_ci(or, se_lnor, conf.level)   # LOG-scale Wald CI

    ## --- convert OR  →  d → r → R²/η²  (+ propagate CI) ------------------
    d_est    <- conv_or_to_d(or)
    ci_d     <- conv_or_to_d(ci_or)
    r_est    <- conv_d_to_r(d_est)
    ci_r     <- conv_d_to_r(ci_d)
    R2_est   <- r_est^2
    ci_R2    <- sort(ci_r^2)       # keep ascending order
    eta2_est <- R2_est             # identical for 2-groups

    ## --- assemble tibble -------------------------------------------------
    out <- bind_rows(out,
      tibble(effect = c("risk_diff", "RR", "OR", "h",
                        "d", "r", "R2", "eta2"),
            est    = c(rd, rr, or, h,
                        d_est, r_est, R2_est, eta2_est),
            ci_lo  = c(ci_rd[1],  ci_rr[1],  ci_or[1],  NA,
                        ci_d[1],   ci_r[1],   ci_R2[1],  ci_R2[1]),
            ci_hi  = c(ci_rd[2],  ci_rr[2],  ci_or[2],  NA,
                        ci_d[2],   ci_r[2],   ci_R2[2],  ci_R2[2])))
  }

  ## ── f) RAW MEANS / SDs  →  Cohen d  (with CI propagation) --------------
  if (!is.null(mean) || !is.null(mean1) || !is.null(d)) {

    ## 1. point estimate ----------------------------------------------------
    if (!is.null(d)) {
      d_calc <- d
    } else if (groups == "one") {
      if (any(is.null(c(mean, sd))))
        stop("Need mean & sd for one-sample effect")
      d_calc <- mean / sd
    } else if (paired) {
      if (any(is.null(c(mean1, mean2, sd1, sd2))))
        stop("Need mean1/2 & sd1/2 for paired design")
      if (is.null(rho_paired))
        stop("Provide `rho_paired` for paired design")
      sd_diff <- sqrt(sd1^2 + sd2^2 - 2*rho_paired*sd1*sd2)
      d_calc  <- (mean1 - mean2) / sd_diff
    } else {                            # two independent
      if (any(is.null(c(mean1, mean2, sd1, sd2))))
        stop("Need mean1/2 & sd1/2 for two-sample design")
      sp     <- sqrt((sd1^2 + sd2^2)/2)
      d_calc <- (mean1 - mean2) / sp
    }

    ## 2. 95 % CI for d -----------------------------------------------------
    ci_vec <- cohen_d_ci(d_calc,
                        n1     = n1,
                        n2     = if (groups == "two" && !paired) n2 else NULL,
                        paired = (groups == "two" &&  paired),
                        conf.level = conf.level)

    ci_d <- sort(ci_vec[c("ci.lo", "ci.hi")]) 

    ## 3. propagate CI to r, OR, R2, eta2 -----------------------------------
    ci_r  <- conv_d_to_r(ci_d)
    ci_or <- conv_d_to_or(ci_d)
    ci_R2 <- range(ci_r^2)   # keep order even if r could be negative

    ## 4. point estimates ---------------------------------------------------
    r   <- conv_d_to_r(d_calc)
    or  <- conv_d_to_or(d_calc)
    R2  <- r^2;  eta2 <- R2

    out <- bind_rows(out,
      tibble(effect = c("d", "r", "OR", "R2", "eta2"),
             est    = c(d_calc, r, or,  R2,  eta2),
             ci_lo  = c(ci_d["ci.lo"],  ci_r[1], ci_or[1], ci_R2[1], ci_R2[1]),
             ci_hi  = c(ci_d["ci.hi"],  ci_r[2], ci_or[2], ci_R2[2], ci_R2[2])))
  }

  ## -----------------------------------------------------------------------
  effects <- dplyr::arrange(out, effect)

  list(
    input   = list(groups = groups, paired = paired,
                   n1 = n1, n2 = n2, total_N = Ntot, ratio = ratio),
    effects = effects
  )
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
z_alpha_val <- function(alpha, sided, tail){
  if (sided == "two"){
    zcrit(conf = 1 - alpha)           # same as qnorm(1-α/2)
  } else {
    if (tail == "upper") qnorm(1 - alpha) else abs(qnorm(alpha))
  }
}

# tidy output formatter
make_prop_tibble <- function(method, n1, n2, alpha, power,
                             rd = NA, h = NA, rr = NA, or = NA, d = NA){
  tibble(method          = method,
         n1              = n1,
         n2              = n2,
         total_N         = ifelse(is.na(n2), n1, n1 + n2),
         ratio           = ifelse(is.na(n2), NA_real_, n2 / n1),
         alpha           = alpha,
         power           = power,
         risk_difference = rd,
         cohen_h         = h,
         cohen_d         = d,
         RR              = rr,
         OR              = or)
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
                   epsilon = NULL,       # |p − p0|
                   # ------ two-sample inputs ------------------------------
                   p1 = NULL, p2 = NULL,
                   ratio = 1             # n2 / n1
                   ){

  groups <- match.arg(groups)
  sided  <- match.arg(sided)
  tail   <- match.arg(tail)

  z_a <- z_alpha_val(alpha, sided, tail)
  z_b <- qnorm(power)                    # positive because power > 0.5

  if (groups == "one"){
    # ── one-sample ----------------------------------------------------------
    if (is.null(p) || is.null(epsilon))
      stop("For a one-sample design supply `p` and `epsilon`.")

    n <- ceiling(((z_a + z_b)^2 * p * (1 - p)) / epsilon^2)

    make_prop_tibble("Prop-1sample", n1 = n, n2 = NA,
                     alpha, power,
                     rd = epsilon)
  } else {
    # ── two-sample ----------------------------------------------------------
    if (is.null(p1) || is.null(p2))
      stop("For a two-sample design supply `p1` and `p2`.")

    rd  <- p1 - p2             # signed risk difference
    eps <- abs(rd)

    n2 <- ((z_a + z_b)^2 / eps^2) *
          (p1 * (1 - p1) / ratio + p2 * (1 - p2))
    n2 <- ceiling(n2)
    n1 <- ceiling(ratio * n2)

    h  <- 2 * asin(sqrt(p1)) - 2 * asin(sqrt(p2))
    rr <- p1 / p2
    or <- (p1 / (1 - p1)) / (p2 / (1 - p2))
    d <- conv_or_to_d(or)

    make_prop_tibble("Prop-2sample", n1, n2,
                     alpha, power,
                     rd = rd, h = h, d = d, rr = rr, or = or)
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

  make_prop_tibble(method = "OR-2sample",
                   n1 = nC,              # control first
                   n2 = nT,
                   alpha = alpha,
                   power = power,
                   rd = rd,
                   h  = h,
                   rr = rr,
                   or = OR,
                   d = d)
}

##############################################################################
##  Quick illustration (uncomment to run)                                   ##
##############################################################################
# # Detect OR = 1.8,  control risk = 0.25, one-sided α = 0.025,  power 90 %
# # 2 : 1 allocation (treatment twice the control)
SSOR(alpha = 0.025, power = 0.90,
     sided = "one", tail = "upper",
     OR = .5, pC = 0.25,
     ratio = 2)


##############################################################################
##  Survival analisis sample size                                           ##
##############################################################################


library(gsDesign)

# 1. Входные параметры -------------------------------------------
alpha   <- 0.025          # односторонний уровень значимости
power   <- 0.926          # требуемая мощность (1-beta)
beta    <- 1 - power

hr      <- 0.6667         # ожидаемое отношение рисков
ratio   <- 1              # 1:1 (k)

medC    <- 6             # медиана выживания плацебо, нед
medT    <- 9             # медиана выживания лечения, нед  (проверка: 24/36 = 0.6667)

accrual <- 18.5             # набор, нед (18,5 мес)
follow  <- 9.75             # дополнительное наблюдение, нед (9,75 мес)

# 2. Перевод медианы в интенсивность (λ = -ln(.5)/MST) -------------
lambdaC <- -log(.5) / medC      # контроль
lambdaT <- lambdaC * hr     # можно посчитать, но функции достаточно HR

# 3. Расчёт фиксированного объёма --------------------------------
n.fix <- nSurvival(
      lambda1 = -log(0.5)/6,
      lambda2 = -log(0.5)/9,
      Ts = 18.5 + 9.75,
      Tr = 18.5,
      ratio = 1,
      alpha = alpha,
      beta = 1-0.926,
      sided = 1
    )

n.fix

##############################################################################
options(scipen = 999)

p = c(0.3, 0.03, 0.003, 0.0003)
p.adjust.methods = c("bonferroni","holm", "hochberg", "hommel",  "BH", "BY",
  "fdr", "none")

for (method in p.adjust.methods) {
  print(method)
  print(p.adjust(p, method = method, n = length(p)) |> round(5))
}











# ##############################################################################
# ##  Number-of-events calculator for a log-rank test                         ##
# ##  – one or two-sided                                                      ##
# ##  – unequal allocation allowed (ratio = nT / nC)                          ##
# ##############################################################################
# # prerequisite utility already in the workspace:
# # zcrit <- function(conf = .95) qnorm(1 - (1 - conf) / 2)

# # main function --------------------------------------------------------------

# ## tidy output formatter
# make_evt_tbl <- function(method, events, alpha, power, HR, HR0, ratio){
#   tibble(method          = method,
#          n_events        = events,
#          ratio           = ratio,
#          alpha           = alpha,
#          power           = power,
#          HR              = HR,
#          HR0             = HR0,
#          treatment_HR    = HR / HR0,
#          risk_reduction  = 1 - HR / HR0)
# }

# ##############################################################################
# #  USER FUNCTION  ------------------------------------------------------------
# ##############################################################################
# SSEvents <- function(alpha = 0.05,
#                      power = 0.80,
#                      sided = c("two", "one"),
#                      tail  = c("upper", "lower"),
#                      groups = c("one", "two"),
#                      # ---- effect definition --------------------------------
#                      HR      = NULL, HR0 = 1,
#                      mst_t   = NULL, mst_c = NULL,   # optional, derive HR
#                      ratio   = 1,                    # nT / nC  (k)
#                      inflate = 1){                   # optional inflation

#   sided  <- match.arg(sided)
#   tail   <- match.arg(tail)
#   groups <- match.arg(groups)

#   if (!is.null(HR) && (HR <= 0)) stop("HR must be > 0.")
#   if (HR0 <= 0) stop("HR0 must be > 0.")

#   ## ── derive HR from two medians if needed ---------------------------------
#   if (is.null(HR)){
#     if (is.null(mst_t) || is.null(mst_c))
#       stop("Supply HR, or both mst_t and mst_c.")
#     HR <- mst_c / mst_t                         # since λ ∝ 1/MST
#   }
#   if (HR == HR0) stop("No effect: HR equals HR0.")

#   ## z-values ---------------------------------------------------------------
#   z_alpha <- z_alpha_val(alpha, sided, tail)
#   z_beta  <- qnorm(power)                       # positive

#   log_delta <- log(HR / HR0)                    # effect on ln scale

#   ## events -----------------------------------------------------------------
#   if (groups == "one"){
#     events <- (z_alpha + z_beta)^2 / log_delta^2
#   } else {
#     k      <- ratio
#     events <- ((1 + k)^2 / (k * log_delta^2)) * (z_alpha + z_beta)^2
#   }
#   events <- ceiling(events * inflate)

#   make_evt_tbl(ifelse(groups == "one", "HR-1arm", "HR-2arm"),
#                events, alpha, power, HR, HR0, ratio)
# }

# ##############################################################################
# ##  EXAMPLES  ── reproduce 191 vs 282 discussion                            ##
# ##############################################################################

# # A) original slide numbers  (HR = 0.67 from 36 vs 24 months, one-sided)
# SSEvents(alpha = 0.025, power = 0.80,
#          sided = "one", tail = "lower",
#          groups = "two", ratio = 1,
#          mst_t = 36, mst_c = 24)
# #> # A tibble: 1 × 8
# #>   method  n_events ratio alpha power    HR  HR0 treatment_HR risk_reduction
# #>   <chr>      <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>       <dbl>          <dbl>
# #> 1 HR-2arm      191     1 0.025    0.8 0.667     1       0.667          0.333

# # B) to reach 282 events at same α/power you could, for instance,
# #    assume HR = 0.63 (stronger effect)
# SSEvents(alpha = 0.025, power = 0.80,
#          sided = "one", tail = "lower",
#          groups = "two", ratio = 1,
#          HR = 0.63)
# #> n_events ≈ 282

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

# --------------------------------------------------------------------
# Optional print / plot methods
# --------------------------------------------------------------------
print.perm_test <- function(x, ...) {
  cat("Permutation test\n")
  cat("  Statistic    :", x$statistic, "\n")
  cat("  p-value      :", x$p.value,   "\n")
  cat("  Alternative  :", x$alternative, "\n")
  invisible(x)
}

plot.perm_test <- function(x, ...) {
  hist(x$perm_stats, breaks = "FD",
       main = "Permutation distribution",
       xlab = "Statistic", col = "#9ecae1", border = "white")
  abline(v = x$statistic, col = "red", lwd = 2)
  legend("topright", legend = "Observed", col = "red", lwd = 2, bty = "n")
  invisible(x)
}

set.seed(1)
x <- rnorm(100, 0)
y <- rnorm(100, 0.1)

dat   <- c(x, y)
group <- factor(rep(c("x","y"), c(length(x), length(y))))

mean_diff <- function(d, g) {
  tapply(d, g, mean)[2] - tapply(d, g, mean)[1]
}

test <- perm_test(dat, group, mean_diff, R = 9999, seed = 20)
test$p.value     # должно быть что-то мелкое (~0.02)
print(test)


##############################################################################
# Effect size
##############################################################################

effect_convert(n1= 100, d = .5)


effect_convert(groups = 'two',        mean1       = 0, mean2 = 1,          # two groups
        sd1         = 1,  sd2  = 1,n1= 100, n2 = 100)

effect_convert(
  groups = 'two',       
  prop1       = .5, 
  prop2 = .1,
  n1= 100, 
  n2 = 1000)
