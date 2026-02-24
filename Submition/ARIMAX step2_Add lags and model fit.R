options(scipen=10)

### Libraries ###
library(forecast)   # Arima(), checkresiduals()
library(FinTS)      # ArchTest()

### 1) Build lags (base R, no pipes) ###
# Expect: model_data already in memory with adj_return, sentiment_index, logs...
model_data$L1_return     = c(NA, head(model_data$adj_return, -1))
model_data$L2_return     = c(NA, NA, head(model_data$adj_return, -2))
model_data$L1_sentiment  = c(NA, head(model_data$sentiment_index, -1))
model_data$L2_sentiment  = c(NA, NA, head(model_data$sentiment_index, -2))

### 2) Keep complete cases for modeling ###
xreg_cols = c(
  "log_adj_close","log_volume",
  "log_p_s_ttm_x","log_eps_ttm_x",
  "L1_return","L2_return",
  "L1_sentiment","L2_sentiment"
)
keep_cols = c("adj_return", xreg_cols)
model_df  = model_data[complete.cases(model_data[, keep_cols]), ]

### 3) Fit ARIMAX: AR(2) with exogenous regressors ###
y    = model_df$adj_return
xreg = as.matrix(model_df[, xreg_cols])

fit_arimax = Arima(y, order=c(2,0,0), xreg=xreg)
print(summary(fit_arimax))

### 4) Residual diagnostics (AR(2)) ###
checkresiduals(fit_arimax)

### 5) Optional: ARCH LM test on AR(2) residuals ###
print(ArchTest(residuals(fit_arimax), lags=5))

### 6) Fit ARIMAX: AR(3) with the same exogenous regressors ###
fit_arimax_ar3 = Arima(y, order=c(3,0,0), xreg=xreg)
print(summary(fit_arimax_ar3))

### 7) Residual diagnostics (AR(3)) ###
cat("\n=== Residual diagnostics for AR(3) ===\n")
checkresiduals(fit_arimax_ar3)

### 8) ARCH LM test on AR(3) residuals ###
cat("\n=== ARCH LM test (AR(3) residuals) ===\n")
print(ArchTest(residuals(fit_arimax_ar3), lags=5))
