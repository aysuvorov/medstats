
library(tidyverse)

# Одна выборка - известна дисперсия генеральной совокупности
# Одновыборочный и парный тест

one_sample_mean <- function(alpha = 0.05, beta = 0.2, sigma, epsilon, sided = "two") {
  
    #' Calculate Sample Size for Mean Comparison Tests
    #'
    #' This function calculates the required sample size for one-sample or paired 
    #' t-tests for means with known standard deviation, specified power, and 
    #' significance level.
    #'
    #' @param alpha Significance level (Type I error rate). Default is 0.05.
    #' @param beta Probability of Type II error (1 - power). Default is 0.2.
    #' @param sigma Population standard deviation. Must be positive.
    #' @param epsilon Margin of error (desired precision/effect size). Must be positive.
    #' @param sided Character string indicating test type: "one" for one-sided, 
    #'              "two" for two-sided. Default is "two".
    #'
    #' @return Integer value representing the minimum required sample size.
    #'
    #' @details 
    #' The sample size is calculated using the normal approximation method:
    #' \itemize{
    #'   \item One-sided test: n = ((z_α + z_β)² × σ²) / ε²
    #'   \item Two-sided test: n = ((z_{α/2} + z_β)² × σ²) / ε²
    #' }
    #' Where z_α and z_{α/2} are critical values from the standard normal distribution.
    #'
    #' @examples
    #' # Calculate sample size for a two-sided test with 80% power
    #' one_sample_mean(alpha = 0.05, beta = 0.2, sigma = 1.5, epsilon = 0.5, sided = "two")
    #' 
    #' # Calculate sample size for a one-sided test with 90% power
    #' one_sample_mean(alpha = 0.05, beta = 0.1, sigma = 2.0, epsilon = 0.3, sided = "one")
    #' 
    #' # Compare one-sided vs two-sided sample sizes
    #' n_one <- one_sample_mean(0.05, 0.2, 1.5, 0.5, "one")
    #' n_two <- one_sample_mean(0.05, 0.2, 1.5, 0.5, "two")
    #' cat("One-sided sample size:", n_one, "\n")
    #' cat("Two-sided sample size:", n_two, "\n")
    #'
    #' @author Assistant
    #' @export

  # Input validation
  if (!is.numeric(alpha) || alpha <= 0 || alpha >= 1) {
    stop("alpha must be a numeric value between 0 and 1")
  }
  if (!is.numeric(beta) || beta <= 0 || beta >= 1) {
    stop("beta must be a numeric value between 0 and 1")
  }
  if (!is.numeric(sigma) || sigma <= 0) {
    stop("sigma must be a positive numeric value")
  }
  if (!is.numeric(epsilon) || epsilon <= 0) {
    stop("epsilon must be a positive numeric value")
  }
  if (!sided %in% c("one", "two")) {
    stop("sided must be either 'one' or 'two'")
  }
  
  # Get z-values
  z_alpha <- qnorm(1 - alpha)
  if (sided == "two") {
    z_alpha <- qnorm(1 - alpha/2)  # Two-sided: use alpha/2
  }
  
  z_beta <- qnorm(1 - beta)  # Power = 1 - beta
  
  # Compute n
  n <- ((z_alpha + z_beta)^2 * sigma^2) / (epsilon^2)
  
  # Round up to nearest integer
  ceiling(n)
}



one_sample_means_t <- function(alpha  = 0.05,
                               beta   = 0.20,
                               sigma,
                               epsilon,
                               sided  = c("two", "one"),
                               tail   = c("upper", "lower"),
                               lower_bound = 2,
                               upper_bound = 128,
                               tol         = 1e-8,
                               max_expand  = 1e3) {

  sided <- match.arg(sided)
  tail  <- match.arg(tail)

  if (!is.numeric(alpha)  || alpha <= 0 || alpha >= 1) stop("alpha must be in (0,1)")
  if (!is.numeric(beta)   || beta  <= 0 || beta  >= 1) stop("beta must be in (0,1)")
  if (!is.numeric(sigma)  || sigma <= 0) stop("sigma must be > 0")
  if (!is.numeric(epsilon)|| epsilon <= 0) stop("epsilon must be > 0")

  ## power-deficiency function  f(n) = β̂(n) − β_target
  f <- function(n) {
    df    <- n - 1
    delta <- sqrt(n) * abs(epsilon) / sigma

    if (sided == "one") {
      if (tail == "upper") {
        tcrit <- qt(1 - alpha, df)
        beta_hat <- pt(tcrit, df, ncp = delta)
      } else {                            # lower tail
        tcrit <- qt(alpha, df)
        beta_hat <- 1 - pt(tcrit, df, ncp = delta)
      }
    } else {                              # two-sided
      tcrit <- qt(1 - alpha/2, df)
      beta_hat <- pt( tcrit, df, ncp = delta) -
                  pt(-tcrit, df, ncp = delta)
    }
    beta_hat - beta
  }

  ## 1) bracket the root ----------------------------------------------------
  expand <- 0
  while (sign(f(lower_bound)) == sign(f(upper_bound))) {
    upper_bound <- upper_bound * 2
    expand <- expand + 1
    if (expand > max_expand)
      stop("Failed to bracket a root; check your alpha/beta/epsilon settings.")
  }

  ## 2) real-valued root ----------------------------------------------------
  root  <- uniroot(f, interval = c(lower_bound, upper_bound), tol = tol)$root

  ## 3) round up and final integer tweak ------------------------------------
  n_int <- ceiling(root)
  while (f(n_int) > 0) n_int <- n_int + 1L   # UNDER-powered → add 1
  n_int
}

# Optional: Normal approximation version (for comparison)
one_sample_means_t(alpha = 0.05, beta = 0.20,
                        sigma = 2.25, epsilon = 0.75,
                        sided = "two")

