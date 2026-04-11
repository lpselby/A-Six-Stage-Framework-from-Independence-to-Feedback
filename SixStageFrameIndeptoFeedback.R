#this is for paper for education journal
#1. Simulate Data (Stage 0)
set.seed(2025)

nsim <- 120
N <- 125
T <- 4
G <- 12

# Parameters
beta0 <- -1
beta1 <- 0.5
beta2 <- 0.6
beta3 <- 1.2

gamma0 <- 0.2
gamma1 <- -0.8
gamma2 <- 0.4
gamma3 <- 0.3

#Storage
beta3_hat <- numeric(nsim)
se_beta3 <- numeric(nsim)
beta3_s2 <- numeric(nsim)
se_s2 <- numeric(nsim)
beta3_s3 <- numeric(nsim)
se_s3    <- numeric(nsim)
beta3_s4 <- numeric(nsim)
se_s4    <- numeric(nsim)

for (s in 1:nsim) {

# Random effects
b_i <- rnorm(N, 0, 0.6)
v_i <- rnorm(N, 0, 0.4)
u_g <- rnorm(G, 0, 0.5)
delta_g <- rnorm(G, 0, 0.3)

# Data container
data <- data.frame()

for (i in 1:N) {
  Z_prev <- 0
  Y_prev <- 0
  
  for (t in 1:T) {
    
    # group membership
    k <- sample(1:2, 1)
    groups <- sample(1:G, k)
    w <- rep(1/k, k)
    
    X1 <- rnorm(1)
    X2 <- mean(u_g[groups]) + rnorm(1, 0, 0.3)
 
    
    # Stage 3 process (feedback)
    Z <- gamma0 + gamma1*Y_prev + gamma2*Z_prev + gamma3*X1 +
      v_i[i] + sum(delta_g[groups]*w) + rnorm(1)
    
    # Outcome
    linpred <- beta0 + beta1*X1 + beta2*X2 + beta3*Z +
      b_i[i] + sum(u_g[groups]*w)
    
    p <- 1 / (1 + exp(-linpred))
    Y <- rbinom(1, 1, p)
    
    data <- rbind(data, data.frame(id=i, time=t, X1=X1, X2=X2, Z=Z, Y=Y))
    
    Z_prev <- Z
    Y_prev <- Y
  }
}


# --- Add group membership variables ---
set.seed(2025 + s)  # ensures variation across simulations

data$k <- sample(1:2, nrow(data), replace = TRUE)

data$g1 <- sample(1:G, nrow(data), replace = TRUE)
data$g2 <- ifelse(data$k == 2,
                  sample(1:G, nrow(data), replace = TRUE),
                  NA)

data$w1 <- ifelse(data$k == 1, 1, 0.5)
data$w2 <- ifelse(data$k == 2, 0.5, 0)

head(data[, c("g1","g2","w1","w2")])

#______________________________________________________________________________________________
#2. STAGE 1 — Independent Model
model1 <- glm(Y ~ X1 + X2 + Z, data=data, family=binomial)
summary(model1)

# Save estimate and SE for beta3 = coefficient of Z
beta3_hat[s] <- coef(model1)["Z"]
se_beta3[s]   <- summary(model1)$coefficients["Z", "Std. Error"]
}

# Monte Carlo summaries
true_beta3 <- 1.2

bias <- mean(beta3_hat - true_beta3)
rmse <- sqrt(mean((beta3_hat - true_beta3)^2))

# Wald 95% CI for each replication
ci_lower <- beta3_hat - 1.96 * se_beta3
ci_upper <- beta3_hat + 1.96 * se_beta3

# Coverage probability
coverage <- mean(ci_lower <= true_beta3 & ci_upper >= true_beta3)

# Average estimated 95% CI across replications
avg_ci <- c(mean(ci_lower), mean(ci_upper))

# Results
bias
rmse
coverage
avg_ci


#plot distribution of estimated beta3
library(ggplot2)

df_plot <- data.frame(beta3_hat = beta3_hat)

ggplot(df_plot, aes(x = beta3_hat)) +
  geom_histogram(aes(y = ..density..),
                 bins = 20,
                 color = "black",
                 fill = "steelblue") +
  geom_density(linewidth = 1) +
  abline(v = mean(beta3_hat), col = 2, lwd = 2)+
  geom_vline(xintercept = 1.2,
             linetype = "dashed",
             linewidth = 1) +
  labs(
    
    x = expression(hat(beta)[3]),
    y = "Density"
  ) +
  theme_classic(base_size = 14)
#___________________________________________________________________________________________________________
#3. STAGE 2 — Correlation (GEE)
# 3. STAGE 2 — Correlation (GEE)
library(geepack)

model2 <- geeglm(
  Y ~ X1 + X2 + Z,
  id = id,
  data = data,
  family = binomial
)

summary(model2)

# Save estimate and SE for beta3 = coefficient of Z
beta3_s2[s] <- coef(model2)["Z"]
se_s2[s]    <- summary(model2)$coefficients["Z", "Std.err"]

true_beta3 <- 1.2
bias_s2 <- mean(beta3_s2 - true_beta3)
rmse_s2 <- sqrt(mean((beta3_s2 - true_beta3)^2))
ci_lower_s2 <- beta3_s2 - 1.96 * se_s2
ci_upper_s2 <- beta3_s2 + 1.96 * se_s2
coverage_s2 <- mean(ci_lower_s2 <= true_beta3 & ci_upper_s2 >= true_beta3)


#__________________________________________________________________________________________________________
# 4. STAGE 3 — Feedback (Control Function)
library(dplyr)

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

summary(model3)

beta3_s3[s] <- coef(model3)["Z"]
se_s3[s]    <- summary(model3)$coefficients["Z", "Std. Error"]

print(head(data))

#________________________________________________________________________________________________
# 5. STAGE 4 — Direct Modeling of Dependence
library(brms)
library(dplyr)

data_s4 <- data %>%
  arrange(id, time) %>%
  group_by(id) %>%
  mutate(
    Y_lag = lag(Y, default = 0),
    Z_lag = lag(Z, default = 0)
  ) %>%
  ungroup()

# make sure there are no NA group IDs
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
  chains = 2,
  iter = 2000,
  warmup = 1000,
  control = list(adapt_delta=0.95)
  
)

summary(model4)

#beta3_s4[s] <- fixef(model4)["Z"]
#se_s4[s]    <- summary(model4)$coefficients["Z", "Std. Error"]



#______________________________________________________________________________________________
#6. Stage 5 — Bayesian Model
library(brms)
library(dplyr)

nsim <- 120

beta3_s5 <- numeric(nsim)
lower_s5 <- numeric(nsim)
upper_s5 <- numeric(nsim)

for (s in 1:nsim) {
  
  data_s5 <- data %>%
    arrange(id, time) %>%
    group_by(id) %>%
    mutate(
      Y_lag = lag(Y, default = 0),
      Z_lag = lag(Z, default = 0)
    ) %>%
    ungroup()
  
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
  
  model5 <- brm(
    f_y + f_z + set_rescor(FALSE),
    data = data_s5,
    chains = 2,
    iter = 2000,
    warmup = 1000,
    control = list(adapt_delta = 0.95)
  )
  
  ps <- posterior_summary(model5, variable = "b_Y_Z")
  
  beta3_s5[s] <- ps[1, "Estimate"]
  lower_s5[s] <- ps[1, "Q2.5"]
  upper_s5[s] <- ps[1, "Q97.5"]
}

true_beta3 <- 1.2

bias_s5 <- mean(beta3_s5 - true_beta3, na.rm = TRUE)
rmse_s5 <- sqrt(mean((beta3_s5 - true_beta3)^2, na.rm = TRUE))
coverage_s5 <- mean(lower_s5 <= true_beta3 & upper_s5 >= true_beta3, na.rm = TRUE)

bias_s5
rmse_s5
coverage_s5


