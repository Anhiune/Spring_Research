options(scipen = 10)

### ============================
### OLS with robust/HAC SEs, alias handling, Box–Cox, diagnostics
### ============================

### Libraries (install once if needed)
# install.packages(c("ggplot2","tseries","lmtest","car","sandwich","MASS","lubridate"))

library(ggplot2)    # plots
library(tseries)    # ADF, Jarque–Bera tests
library(lmtest)     # BP, DW, BG, coeftest
library(car)        # VIF, ncvTest
library(sandwich)   # HC/HAC covariances
library(MASS)       # boxcox(), rlm()
library(lubridate)  # robust date parsing

### ---------------------------
### 0) Output plots folder
### ---------------------------
plots_dir = "C:/Users/hoang/OneDrive - University of St. Thomas/Summer Research/Plot"
if (!dir.exists(plots_dir)) dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)

### ---------------------------
### 1) Load data
### ---------------------------
df = read.csv(
  "C:/Users/hoang/OneDrive - University of St. Thomas/Summer Research/Data/ARIMAX modeling (version2).csv",
  check.names = FALSE, stringsAsFactors = FALSE
)

### ---------------------------
### 2) Dates & unnamed columns (robust parsing)
### ---------------------------

# Helper: robust date parsing (strings, timestamps, excel serials)
parse_dates_robust = function(x) {
  if (inherits(x, "Date"))   return(x)
  if (inherits(x, "POSIXt")) return(as.Date(x))

  if (is.numeric(x)) {
    return(as.Date(x, origin = "1899-12-30"))  # common Excel origin
  }

  if (is.character(x)) {
    x_trim = trimws(x)

    parsed = suppressWarnings(
      parse_date_time(
        x_trim,
        orders = c(
          "ymd","mdy","dmy",
          "ymd HMS","mdy HMS","dmy HMS",
          "ymd HM","mdy HM","dmy HM",
          "ymd HMS p","mdy HMS p","dmy HMS p"
        ),
        tz = "UTC", exact = FALSE
      )
    )

    needs_excel = is.na(parsed) & grepl("^\\d+$", x_trim)
    if (any(needs_excel)) {
      serial_vals = suppressWarnings(as.numeric(x_trim[needs_excel]))
      parsed[needs_excel] = as.POSIXct(as.Date(serial_vals, origin = "1899-12-30"))
    }
    return(as.Date(parsed))
  }

  as.Date(rep(NA, length(x)))
}

# Remove "Unnamed" columns if present
unnamed_cols = grepl("^Unnamed", names(df))
if (any(unnamed_cols)) {
  df = df[, !unnamed_cols, drop = FALSE]
}

# Parse a 'date' column if present (keep raw copy)
if ("date" %in% names(df)) {
  df$date_raw = df$date
  df$date     = parse_dates_robust(df$date)

  na_pct = mean(is.na(df$date))
  if (!is.na(na_pct) && na_pct > 0.1) {
    message(sprintf("Warning: %.1f%% of 'date' values could not be parsed. Check df$date_raw.", 100 * na_pct))
  }
}

### ---------------------------
### 3) Sentiment index (if available)
### ---------------------------
if (!("sentiment_index" %in% names(df)) && all(c("positive","negative") %in% names(df))) {
  df$sentiment_index = df$positive - df$negative
}

### ---------------------------
### 4) Numeric columns (for ADF differencing scan)
### ---------------------------
num_cols = names(df)[sapply(df, is.numeric)]
num_cols = setdiff(num_cols, "diluted_average_shares")

### ---------------------------
### 5) Difference non-stationary series (ADF p > 0.05)
### ---------------------------
for (var in num_cols) {
  series = df[[var]]
  if (sum(!is.na(series)) > 10) {
    adf_p = tseries::adf.test(stats::na.omit(series), k = 0)$p.value
    if (adf_p > 0.05) {
      df[[paste0("d_", var)]] = c(NA, diff(series))
    }
  }
}

### ---------------------------
### 6) Log transforms to stabilize variance
### ---------------------------
vars_to_log = c(
  "adj_close","volume","marketcap_daily","ev_daily",
  "p_s_ttm_x","eps_ttm_x","p_e_ttm_x","debt_equity_x",
  "total_revenue_ttm","operating_income_ttm","gross_profit_ttm",
  "net_income_ttm","total_debt","cash_and_cash_equivalents",
  "price_to_sales_ratio","price_to_book_ratio","pe_ratio"
)
for (v in vars_to_log) {
  if (v %in% names(df)) df[[paste0("log_", v)]] = log1p(df[[v]])
}

### ---------------------------
### 7) Define Y and X
### ---------------------------
response_var = "adj_return"
predictors = c(
  "log_adj_close","log_volume",
  "log_p_s_ttm_x","log_eps_ttm_x","log_p_e_ttm_x",
  "log_debt_equity_x","sentiment_index"
)
predictors = predictors[predictors %in% names(df)]

### ---------------------------
### 8) Build model data (complete cases)
### ---------------------------
keep_cols   = c(response_var, predictors)
complete_ix = complete.cases(df[, keep_cols, drop = FALSE])
model_data  = df[complete_ix, keep_cols, drop = FALSE]

stopifnot(nrow(model_data) > 10, length(predictors) > 0)

### ---------------------------
### 9) Baseline OLS and alias handling
### ---------------------------
form_str  = paste(response_var, "~", paste(predictors, collapse = "+"))
form      = stats::as.formula(form_str)
ols_model = stats::lm(form, data = model_data)

# Detect aliased coefficients (NA) and refit without them
aliased_coefs = names(coef(ols_model))[is.na(coef(ols_model))]
aliased_terms = setdiff(aliased_coefs, "(Intercept)")

if (length(aliased_terms) > 0) {
  cat("\nAliased (perfectly collinear) terms detected and removed:\n")
  print(aliased_terms)
  predictors = setdiff(predictors, aliased_terms)
  form       = stats::as.formula(paste(response_var, "~", paste(predictors, collapse = "+")))
  ols_model  = stats::lm(form, data = model_data)
} else {
  cat("\nNo aliased terms detected.\n")
}

# VIF (after alias cleanup)
cat("\n=== VIF (multicollinearity check) ===\n")
suppressWarnings(print(car::vif(ols_model)))

# Heteroskedasticity tests
cat("\n=== Breusch–Pagan test (OLS) ===\n")
print(lmtest::bptest(ols_model))
cat("\n=== car::ncvTest (OLS; non-constant variance score test) ===\n")
print(car::ncvTest(ols_model))

### ---------------------------
### 10) Robust inference (no weighting)
### ---------------------------
cat("\n=== OLS with HC1 robust SEs ===\n")
print(lmtest::coeftest(ols_model, vcov = sandwich::vcovHC(ols_model, type = "HC1")))
cat("\n=== OLS with HC3 robust SEs (recommended) ===\n")
print(lmtest::coeftest(ols_model, vcov = sandwich::vcovHC(ols_model, type = "HC3")))
cat("\n=== OLS with HC4 robust SEs (leverage/small n) ===\n")
print(lmtest::coeftest(ols_model, vcov = sandwich::vcovHC(ols_model, type = "HC4")))

# HAC (het + autocorr)
Tn     = nrow(model_data)
nw_lag = max(1, floor(0.75 * Tn^(1/3)))
cat("\n=== OLS with Newey–West (HAC) SEs; lag =", nw_lag, "===\n")
nw_cov = sandwich::NeweyWest(ols_model, lag = nw_lag, prewhite = FALSE, adjust = TRUE)
print(lmtest::coeftest(ols_model, vcov = nw_cov))

# Autocorrelation diagnostics
cat("\n=== Breusch–Godfrey test (lag 1 and 4) ===\n")
print(lmtest::bgtest(ols_model, order = 1))
print(lmtest::bgtest(ols_model, order = 4))
cat("\n=== Durbin–Watson test ===\n")
print(lmtest::dwtest(ols_model))

### ---------------------------
### 11) Box–Cox transform of Y (using cleaned predictors)
### ---------------------------
y_shifted = model_data[[response_var]] - min(model_data[[response_var]], na.rm = TRUE) + 1
tmp_bc    = model_data[, predictors, drop = FALSE]
tmp_bc$y_shifted = y_shifted

png(file.path(plots_dir, "boxcox_profile.png"), width = 1400, height = 1000, res = 150)
bc = MASS::boxcox(y_shifted ~ ., data = tmp_bc, lambda = seq(-2, 2, 0.1))
dev.off()

lambda_opt = bc$x[which.max(bc$y)]
cat("\nOptimal Box–Cox lambda:", lambda_opt, "\n")

y_bc = if (abs(lambda_opt) < 1e-6) log(y_shifted) else (y_shifted^lambda_opt - 1) / lambda_opt
model_data$y_bc = y_bc

bc_model = stats::lm(y_bc ~ ., data = model_data[, c(predictors, "y_bc"), drop = FALSE])

cat("\n=== Box–Cox model with HC3 robust SEs ===\n")
print(lmtest::coeftest(bc_model, vcov = sandwich::vcovHC(bc_model, type = "HC3")))
cat("\n=== Jarque–Bera test (Box–Cox residuals) ===\n")
print(tseries::jarque.bera.test(stats::residuals(bc_model)))

# HAC on Box–Cox
cat("\n=== Box–Cox model with Newey–West (lag =", nw_lag, ") ===\n")
bc_nw_cov = sandwich::NeweyWest(bc_model, lag = nw_lag, prewhite = FALSE, adjust = TRUE)
print(lmtest::coeftest(bc_model, vcov = bc_nw_cov))

### ---------------------------
### 12) Diagnostics & plots
### ---------------------------
augment_resids = function(fit, nrows) {
  data.frame(
    fitted = as.numeric(stats::fitted(fit)),
    resid  = as.numeric(stats::residuals(fit)),
    stdres = as.numeric(rstandard(fit)),
    sqrt_abs_stdres = sqrt(abs(as.numeric(rstandard(fit)))),
    idx = seq_len(nrows)
  )
}

diag_ols = augment_resids(ols_model, nrow(model_data))
diag_bc  = augment_resids(bc_model,  nrow(model_data))

# Residuals vs Fitted (OLS)
ggplot2::ggplot(diag_ols, ggplot2::aes(fitted, resid)) +
  ggplot2::geom_point(alpha = 0.7) +
  ggplot2::geom_hline(yintercept = 0, linetype = 2) +
  ggplot2::labs(title = "Residuals vs Fitted (OLS)", x = "Fitted Values", y = "Residuals")
ggplot2::ggsave(file.path(plots_dir, "residuals_vs_fitted_OLS.png"), width = 8, height = 6, dpi = 150)

# Scale-Location (OLS)
ggplot2::ggplot(diag_ols, ggplot2::aes(fitted, sqrt_abs_stdres)) +
  ggplot2::geom_point(alpha = 0.7) +
  ggplot2::labs(title = "Scale-Location (OLS)", x = "Fitted Values", y = "√|Standardized Residuals|")
ggplot2::ggsave(file.path(plots_dir, "scale_location_OLS.png"), width = 8, height = 6, dpi = 150)

# QQ plot (OLS)
png(file.path(plots_dir, "qqplot_OLS.png"), width = 1200, height = 900, res = 150)
qqnorm(stats::residuals(ols_model), main = "Normal Q-Q Plot (OLS Residuals)",
       xlab = "Theoretical Quantiles", ylab = "Sample Quantiles")
qqline(stats::residuals(ols_model))
dev.off()

# Residual histogram + density (OLS)
ggplot2::ggplot(diag_ols, ggplot2::aes(resid)) +
  ggplot2::geom_histogram(bins = 30, fill = "skyblue", alpha = 0.7) +
  ggplot2::geom_density(size = 1) +
  ggplot2::labs(title = "Residuals Histogram (OLS)", x = "Residuals", y = "Count / Density")
ggplot2::ggsave(file.path(plots_dir, "hist_residuals_OLS.png"), width = 8, height = 6, dpi = 150)

# ACF of residuals (OLS)
png(file.path(plots_dir, "ACF_residuals_OLS.png"), width = 1200, height = 900, res = 150)
stats::acf(stats::residuals(ols_model), main = "ACF of OLS Residuals",
           xlab = "Lag", ylab = "Autocorrelation")
dev.off()

# Cook's distance (OLS)
png(file.path(plots_dir, "cooks_distance_OLS.png"), width = 1400, height = 1000, res = 150)
plot(stats::cooks.distance(ols_model), type = "h",
     main = "Cook's Distance (OLS)", xlab = "Observation Index", ylab = "Cook's D")
dev.off()

# Residuals vs Fitted (Box–Cox)
ggplot2::ggplot(diag_bc, ggplot2::aes(fitted, resid)) +
  ggplot2::geom_point(alpha = 0.7) +
  ggplot2::geom_hline(yintercept = 0, linetype = 2) +
  ggplot2::labs(title = "Residuals vs Fitted (Box–Cox Transformed Y)",
                x = "Fitted Values", y = "Residuals")
ggplot2::ggsave(file.path(plots_dir, "residuals_vs_fitted_BoxCox.png"), width = 8, height = 6, dpi = 150)

# Scale-Location (Box–Cox)
ggplot2::ggplot(diag_bc, ggplot2::aes(fitted, sqrt_abs_stdres)) +
  ggplot2::geom_point(alpha = 0.7) +
  ggplot2::labs(title = "Scale-Location (Box–Cox Transformed Y)",
                x = "Fitted Values", y = "√|Standardized Residuals|")
ggplot2::ggsave(file.path(plots_dir, "scale_location_BoxCox.png"), width = 8, height = 6, dpi = 150)

# QQ plot (Box–Cox)
png(file.path(plots_dir, "qqplot_BoxCox.png"), width = 1200, height = 900, res = 150)
qqnorm(stats::residuals(bc_model), main = "Normal Q-Q Plot (Box–Cox Residuals)",
       xlab = "Theoretical Quantiles", ylab = "Sample Quantiles")
qqline(stats::residuals(bc_model))
dev.off()

### ---------------------------
### 13) Influence summary + optional sensitivity
### ---------------------------
infl = stats::influence.measures(ols_model)
cat("\n=== Potentially influential points (OLS) ===\n")
print(summary(infl))

infl_sum  = infl$infmat   # columns: dffit, cook.d, hat, dfbetas...
n         = nrow(model_data)
p         = length(coef(ols_model))
cook_cut  = 4 / n
dffits_cut = 2 * sqrt(p / n)

flag_idx = which(infl_sum[, "cook.d"] > cook_cut | abs(infl_sum[, "dffit"]) > dffits_cut)

if (length(flag_idx) > 0) {
  cat("\nFlagged influential rows removed for sensitivity run:\n"); print(flag_idx)
  model_data_trim = model_data[-flag_idx, , drop = FALSE]
  form_trim       = stats::as.formula(paste(response_var, "~", paste(predictors, collapse = "+")))
  ols_trim        = stats::lm(form_trim, data = model_data_trim)
  cat("\n=== Trimmed OLS with HC3 SEs (sensitivity) ===\n")
  print(lmtest::coeftest(ols_trim, vcov = sandwich::vcovHC(ols_trim, type = "HC3")))
}

cat("\nAll plots saved to:\n", plots_dir, "\n")
