library(dplyr)
library(forecast)   # for Arima()

# --- Create lags for returns and sentiment ---
model_data <- model_data %>%
  dplyr::mutate(
    L1_return = dplyr::lag(adj_return, 1),
    L2_return = dplyr::lag(adj_return, 2),
    L1_sentiment = dplyr::lag(sentiment_index, 1),
    L2_sentiment = dplyr::lag(sentiment_index, 2)
  ) %>%
  tidyr::drop_na()

# --- Fit ARIMAX: AR(2) with exogenous regressors ---
y <- model_data$adj_return
xreg <- as.matrix(model_data[, c("log_adj_close","log_volume",
                                 "log_p_s_ttm_x","log_eps_ttm_x",
                                 "L1_return","L2_return",
                                 "L1_sentiment","L2_sentiment")])

fit_arimax <- forecast::Arima(y, order = c(2,0,0), xreg = xreg)

summary(fit_arimax)

# --- Residual diagnostics ---
checkresiduals(fit_arimax)

# Optional: test for ARCH effects (heteroskedasticity in ARIMA residuals)
library(FinTS)
ArchTest(residuals(fit_arimax), lags = 5)

# --- Fit ARIMAX: AR(3) with exogenous regressors ---
fit_arimax_ar3 <- forecast::Arima(y, order = c(3,0,0), xreg = xreg)

# Model summary
summary(fit_arimax_ar3)

# Residual diagnostics
cat("\n=== Residual diagnostics for AR(3) ===\n")
forecast::checkresiduals(fit_arimax_ar3)

# ARCH test for heteroskedasticity
cat("\n=== ARCH LM test (AR(3) residuals) ===\n")
print(FinTS::ArchTest(residuals(fit_arimax_ar3), lags = 5))