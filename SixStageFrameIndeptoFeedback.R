# ============================================================
# A Six-Stage Framework from Independence to Feedback
# Reproducible simulation script
# ============================================================

# -------------------------------
# Packages
# -------------------------------
library(dplyr)
library(geepack)
library(brms)
library(ggplot2)

# -------------------------------
# Reproducibility
# -------------------------------
set.seed(2025)

# -------------------------------
# Simulation settings
# -------------------------------
nsim <- 120
N <- 125
T <- 4
G <- 12

# True model parameters
beta0 <- -1
beta1 <- 0.5
beta2 <- 0.6
beta3 <- 1.2

gamma0 <- 0.2
gamma1 <- -0.8
gamma2 <- 0.4
gamma3 <- 0.3

# -------------------------------
# Storage
# -------------------------------
beta3_s1 <- numeric(nsim)
se_s1    <- numeric(nsim)

beta3_s2 <- numeric(nsim)
se_s2    <- numeric(nsim)

beta3_s3 <- numeric(nsim)
se_s3    <- numeric(nsim)

beta3_s4 <- numeric(nsim)
lower_s4 <- numeric(nsim)
upper_s4 <- numeric(nsim)

beta3_s5 <- numeric(nsim)
lower_s5 <- numeric(nsim)
upper_s5 <- numeric(nsim)

# -------------------------------
# Optional: explicit priors for Stage 5
# Replace these with the exact priors you want to document in Appendix A.
# brms prior specification is handled via set_prior()/prior().
# -------------------------------
priors_stage5 <- c(
  # Weakly informative priors for regression coefficients
  set_prior("normal(0, 1000)", class = "b", resp = "Y"),
  set_prior("normal(0, 1000)", class = "b", resp = "Z"),

  # Weakly informative priors for group-level SDs
  # Replace if you want the appendix to say something different.
  set_prior("student_t(3, 0, 2.5)", class = "sd", resp = "Y"),
  set_prior("student_t(3, 0, 2.5)", class = "sd", resp = "Z"),

  # Residual SD for the Gaussian Z model
  set_prior("student_t(3, 0, 2.5)", class = "sigma", resp = "Z")
)

# ============================================================
# Monte Carlo loop
# ============================================================
for (s in 1:nsim) {

  # --------------------------------
  # Stage 0: simulate data
  # --------------------------------
  b_i <- rnorm(N, 0, 0.6)
  v_i <- rnorm(N, 0, 0.4)
  u_g <- rnorm(G, 0, 0.5)
  delta_g <- rnorm(G, 0, 0.3)

  data <- data.frame()

  for (i in 1:N) {
    Z_prev <- 0
    Y_prev <- 0

    for (t in 1:T) {

      # Contextual membership used in the data-generating process
      k <- sample(1:2, 1)
      groups <- sample(1:G, k)
      w <- rep(1 / k, k)

      g1 <- groups[1]
      g2 <- if (k == 2) groups[2] else NA
      w1 <- w[1]
      w2 <- if (k == 2) w[2] else 0

      X1 <- rnorm(1)
      X2 <- mean(u_g[groups]) + rnorm(1, 0, 0.3)

      # Feedback-driven covariate
      Z <- gamma0 + gamma1 * Y_prev + gamma2 * Z_prev + gamma3 * X1 +
        v_i[i] + sum(delta_g[groups] * w) + rnorm(1)

      # Binary outcome
      linpred <- beta0 + beta1 * X1 + beta2 * X2 + beta3 * Z +
        b_i[i] + sum(u_g[groups] * w)

      p <- 1 / (1 + exp(-linpred))
      Y <- rbinom(1, 1, p)

      data <- rbind(
        data,
        data.frame(
          id = i, time = t,
          X1 = X1, X2 = X2, Z = Z, Y = Y,
          g1 = g1, g2 = g2, w1 = w1, w2 = w2
        )
      )

      Z_prev <- Z
      Y_prev <- Y
    }
  }

  # --------------------------------
  # Stage 1: Independent logistic regression
  # --------------------------------
  model1 <- glm(Y ~ X1 + X2 + Z, data = data, family = binomial)

  beta3_s1[s] <- coef(model1)["Z"]
  se_s1[s]    <- summary(model1)$coefficients["Z", "Std. Error"]

  # --------------------------------
  # Stage 2: GEE for within-subject correlation
  # --------------------------------
  model2 <- geeglm(
    Y ~ X1 + X2 + Z,
    id = id,
    data = data,
    family = binomial,
    corstr = "exchangeable"
  )

  beta3_s2[s] <- coef(model2)["Z"]
  se_s2[s]    <- summary(model2)$coefficients["Z", "Std.err"]

  # --------------------------------
  # Stage 3: Control-function model for endogeneity
  # --------------------------------
  data_s3 <- data %>%
    arrange(id, time) %>%
    group_by(id) %>%
    mutate(
      Y_lag = lag(Y),
      Z_lag = lag(Z)
    ) %>%
    ungroup()

  stage1_Z <- lm(
    Z ~ Y_lag + Z_lag + X1,
    data = data_s3,
    na.action = na.exclude
  )

  data_s3$resid_Z <- resid(stage1_Z)

  model3 <- glm(
    Y ~ X1 + X2 + Z + resid_Z,
    data = data_s3,
    family = binomial
  )

  beta3_s3[s] <- coef(model3)["Z"]
  se_s3[s]    <- summary(model3)$coefficients["Z", "Std. Error"]

  # --------------------------------
  # Stage 4: Joint hierarchical feedback model
  # --------------------------------
  data_s4 <- data %>%
    arrange(id, time) %>%
    group_by(id) %>%
    mutate(
      Y_lag = lag(Y, default = 0),
      Z_lag = lag(Z, default = 0)
    ) %>%
    ungroup()

  # Replace NA group IDs so the multilevel term is defined for all rows
  data_s4$g2 <- ifelse(is.na(data_s4$g2), data_s4$g1, data_s4$g2)

  f_y <- bf(
    Y ~ X1 + X2 + Z +
      (1 | id) +
      (1 | mm(g1, g2, weights = cbind(w1, w2))),
    family = bernoulli()
  )

  f_z <- bf(
    Z ~ Y_lag + Z_lag + X1 +
      (1 | id) +
      (1 | mm(g1, g2, weights = cbind(w1, w2))),
    family = gaussian()
  )

  model4 <- brm(
    f_y + f_z + set_rescor(FALSE),
    data = data_s4,
    chains = 4,
    iter = 10000,
    warmup = 2500,
    cores = 4,
    seed = 2025 + s,
    control = list(adapt_delta = 0.95)
  )

  ps4 <- posterior_summary(model4, variable = "b_Y_Z")
  beta3_s4[s] <- ps4[1, "Estimate"]
  lower_s4[s] <- ps4[1, "Q2.5"]
  upper_s4[s] <- ps4[1, "Q97.5"]

  # --------------------------------
  # Stage 5: Bayesian joint hierarchical model
  # --------------------------------
  # Same joint model structure, but with explicit priors for the appendix.
  model5 <- brm(
    f_y + f_z + set_rescor(FALSE),
    data = data_s4,
    prior = priors_stage5,
    chains = 4,
    iter = 10000,
    warmup = 2500,
    cores = 4,
    seed = 2025 + s,
    control = list(adapt_delta = 0.95)
  )

  ps5 <- posterior_summary(model5, variable = "b_Y_Z")
  beta3_s5[s] <- ps5[1, "Estimate"]
  lower_s5[s] <- ps5[1, "Q2.5"]
  upper_s5[s] <- ps5[1, "Q97.5"]
}

# ============================================================
# Monte Carlo summaries
# ============================================================
true_beta3 <- 1.2

# Stage 1
bias_s1 <- mean(beta3_s1 - true_beta3)
rmse_s1 <- sqrt(mean((beta3_s1 - true_beta3)^2))
ci_lower_s1 <- beta3_s1 - 1.96 * se_s1
ci_upper_s1 <- beta3_s1 + 1.96 * se_s1
coverage_s1 <- mean(ci_lower_s1 <= true_beta3 & ci_upper_s1 >= true_beta3)

# Stage 2
bias_s2 <- mean(beta3_s2 - true_beta3)
rmse_s2 <- sqrt(mean((beta3_s2 - true_beta3)^2))
ci_lower_s2 <- beta3_s2 - 1.96 * se_s2
ci_upper_s2 <- beta3_s2 + 1.96 * se_s2
coverage_s2 <- mean(ci_lower_s2 <= true_beta3 & ci_upper_s2 >= true_beta3)

# Stage 3
bias_s3 <- mean(beta3_s3 - true_beta3)
rmse_s3 <- sqrt(mean((beta3_s3 - true_beta3)^2))
ci_lower_s3 <- beta3_s3 - 1.96 * se_s3
ci_upper_s3 <- beta3_s3 + 1.96 * se_s3
coverage_s3 <- mean(ci_lower_s3 <= true_beta3 & ci_upper_s3 >= true_beta3)

# Stage 4
bias_s4 <- mean(beta3_s4 - true_beta3, na.rm = TRUE)
rmse_s4 <- sqrt(mean((beta3_s4 - true_beta3)^2, na.rm = TRUE))
coverage_s4 <- mean(lower_s4 <= true_beta3 & upper_s4 >= true_beta3, na.rm = TRUE)

# Stage 5
bias_s5 <- mean(beta3_s5 - true_beta3, na.rm = TRUE)
rmse_s5 <- sqrt(mean((beta3_s5 - true_beta3)^2, na.rm = TRUE))
coverage_s5 <- mean(lower_s5 <= true_beta3 & upper_s5 >= true_beta3, na.rm = TRUE)

# Print summaries
list(
  stage1 = c(bias = bias_s1, rmse = rmse_s1, coverage = coverage_s1),
  stage2 = c(bias = bias_s2, rmse = rmse_s2, coverage = coverage_s2),
  stage3 = c(bias = bias_s3, rmse = rmse_s3, coverage = coverage_s3),
  stage4 = c(bias = bias_s4, rmse = rmse_s4, coverage = coverage_s4),
  stage5 = c(bias = bias_s5, rmse = rmse_s5, coverage = coverage_s5)
)

# ============================================================
# Optional: Stage 1 sampling distribution plot
# ============================================================
df_plot <- data.frame(beta3_hat = beta3_s1)

ggplot(df_plot, aes(x = beta3_hat)) +
  geom_histogram(aes(y = after_stat(density)),
                 bins = 20, color = "black", fill = "steelblue") +
  geom_density(linewidth = 1) +
  geom_vline(xintercept = mean(beta3_s1), col = "red", lwd = 2) +
  geom_vline(xintercept = 1.2, linetype = "dashed", linewidth = 1) +
  labs(
    x = expression(hat(beta)[3]),
    y = "Density"
  ) +
  theme_classic(base_size = 14)


