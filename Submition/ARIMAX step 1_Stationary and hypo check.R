options(scipen=10)

# Libraries:
# Add all packages used below (no tidyverse pipes)
library(ggplot2)
library(tseries)     # ADF, Jarque–Bera tests
library(lmtest)      # BP, DW, BG, coeftest
library(car)         # VIF, ncvTest
library(sandwich)    # HC/HAC covariances
library(MASS)        # boxcox()
library(lubridate)   # parse_date_time()

# Paths:
# Set the folder where plots and data live
base_dir = "C:/Users/hoang/OneDrive - University of St. Thomas/Summer Research/Submition/"
plots_dir = paste0(base_dir, "Plot/")
data_csv  = paste0(base_dir, "Data/ARIMAX modeling (version2).csv")

# Ensure plot folder exists
if (!dir.exists(plots_dir)) dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)

# Data:
# Read CSV with original names preserved
df = read.csv(data_csv, check.names = FALSE, stringsAsFactors = FALSE)

# Utilities:
# Robust date parsing that handles Date, POSIX, Excel serials, and common string formats
parse_dates_robust = function(x) {
  if (inherits(x, "Date"))   return(x)
  if (inherits(x, "POSIXt")) return(as.Date(x))
  if (is.numeric(x))         return(as.Date(x, origin = "1899-12-30"))
  
  if (is.character(x)) {
    x_trim = trimws(x)
    parsed = suppressWarnings(
      lubridate::parse_date_time(
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
  
  return(as.Date(rep(NA, length(x))))
}

# Cleanup:
# Drop any "Unnamed..." columns created by Excel/CSV export
drop_ix = grepl("^Unnamed", names(df))
if (any(drop_ix)) df = df[, !drop_ix, drop = FALSE]

# Dates:
# If a 'date' column exists, parse and keep a raw copy for audit
if ("date" %in% names(df)) {
  df$date_raw = df$date
  df$date     = parse_dates_robust(df$date)
  
  na_pct = mean(is.na(df$date))
  if (!is.na(na_pct) && na_pct > 0.1) {
    message(sprintf("Warning: %.1f%% of 'date' could not be parsed; see df$date_raw.", 100 * na_pct))
  }
}

# Sentiment:
# If sentiment_index is missing but positive/negative exist, construct it
if (!("sentiment_index" %in% names(df)) && all(c("positive","negative") %in% names(df))) {
  df$sentiment_index = df$positive - df$negative
}

# Stationarity Scan:
# Identify numeric columns and difference non-stationary ones (ADF p > 0.05)
num_cols = names(df)[sapply(df, is.numeric)]
num_cols = setdiff(num_cols, "diluted_average_shares")

for (v in num_cols) {
  series = df[[v]]
  if (sum(!is.na(series)) > 10) {
    adf_p = tseries::adf.test(stats::na.omit(series), k = 0)$p.value
    if (adf_p > 0.05) {
      df[[paste0("d_", v)]] = c(NA, diff(series))
    }
  }
}

# Variance Stabilization:
# Create log1p-transforms for selected variables if present
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

# Model Spec:
# Response and candidate predictors (keep only those that exist)
response_var = "adj_return"
predictors = c(
  "log_adj_close","log_volume",
  "log_p_s_ttm_x","log_eps_ttm_x","log_p_e_ttm_x",
  "log_debt_equity_x","sentiment_index"
)
predictors = predictors[predictors %in% names(df)]

# Model Data:
# Keep complete cases for Y and X
keep_cols   = c(response_var, predictors)
complete_ix = complete.cases(df[, keep_cols, drop = FALSE])
model_data  = df[complete_ix, keep_cols, drop = FALSE]

# Guardrails:
# Require enough rows and at least one predictor
stopifnot(nrow(model_data) > 10, length(predictors) > 0)

# OLS:
# Fit baseline OLS and handle aliased (collinear) terms
form_str  = paste(response_var, "~", paste(predictors, collapse = "+"))
form      = stats::as.formula(form_str)
ols_model = stats::lm(form, data = model_data)

aliased_coefs = names(coef(ols_model))[is.na(coef(ols_model))]
aliased_terms = setdiff(aliased_coefs, "(Intercept)")

if (length(aliased_terms) > 0) {
  cat("\nAliased (perfect collinearity) removed:\n"); print(aliased_terms)
  predictors = setdiff(predictors, aliased_terms)
  form       = stats::as.formula(paste(response_var, "~", paste(predictors, collapse = "+")))
  ols_model  = stats::lm(form, data = model_data)
} else {
  cat("\nNo aliased terms detected.\n")
}

# Multicollinearity:
# VIF after alias cleanup
cat("\n=== VIF (multicollinearity) ===\n")
suppressWarnings(print(car::vif(ols_model)))

# Heteroskedasticity:
# Breusch–Pagan and NCV score tests
cat("\n=== Breusch–Pagan (OLS) ===\n")
print(lmtest::bptest(ols_model))
cat("\n=== car::ncvTest (OLS) ===\n")
print(car::ncvTest(ols_model))

# Robust SE:
# HC1 / HC3 / HC4
cat("\n=== OLS + HC1 ===\n")
print(lmtest::coeftest(ols_model, vcov = sandwich::vcovHC(ols_model, type = "HC1")))
cat("\n=== OLS + HC3 (recommended) ===\n")
print(lmtest::coeftest(ols_model, vcov = sandwich::vcovHC(ols_model, type = "HC3")))
cat("\n=== OLS + HC4 (high leverage/small n) ===\n")
print(lmtest::coeftest(ols_model, vcov = sandwich::vcovHC(ols_model, type = "HC4")))

# HAC SE:
# Newey–West lag using 0.75 * T^(1/3)
Tn     = nrow(model_data)
nw_lag = max(1, floor(0.75 * Tn^(1/3)))
cat("\n=== OLS + Newey–West (lag = ", nw_lag, ") ===\n", sep = "")
nw_cov = sandwich::NeweyWest(ols_model, lag = nw_lag, prewhite = FALSE, adjust = TRUE)
print(lmtest::coeftest(ols_model, vcov = nw_cov))

# Autocorrelation:
# BG(1), BG(4), and Durbin–Watson
cat("\n=== Breusch–Godfrey ===\n")
print(lmtest::bgtest(ols_model, order = 1))
print(lmtest::bgtest(ols_model, order = 4))
cat("\n=== Durbin–Watson ===\n")
print(lmtest::dwtest(ols_model))

# Box–Cox:
# Shift Y to be positive, profile lambda, refit transformed model
y_shifted = model_data[[response_var]] - min(model_data[[response_var]], na.rm = TRUE) + 1
tmp_bc    = model_data[, predictors, drop = FALSE]
tmp_bc$y_shifted = y_shifted

png(paste0(plots_dir, "boxcox_profile.png"), width = 1400, height = 1000, res = 150)
bc = MASS::boxcox(y_shifted ~ ., data = tmp_bc, lambda = seq(-2, 2, 0.1))
dev.off()

lambda_opt = bc$x[which.max(bc$y)]
cat("\nOptimal Box–Cox lambda: ", lambda_opt, "\n", sep = "")

y_bc = if (abs(lambda_opt) < 1e-6) log(y_shifted) else (y_shifted^lambda_opt - 1) / lambda_opt
model_data$y_bc = y_bc

bc_model = stats::lm(y_bc ~ ., data = model_data[, c(predictors, "y_bc"), drop = FALSE])

cat("\n=== Box–Cox + HC3 ===\n")
print(lmtest::coeftest(bc_model, vcov = sandwich::vcovHC(bc_model, type = "HC3")))
cat("\n=== Jarque–Bera (Box–Cox residuals) ===\n")
print(tseries::jarque.bera.test(stats::residuals(bc_model)))

cat("\n=== Box–Cox + Newey–West (lag = ", nw_lag, ") ===\n", sep = "")
bc_nw_cov = sandwich::NeweyWest(bc_model, lag = nw_lag, prewhite = FALSE, adjust = TRUE)
print(lmtest::coeftest(bc_model, vcov = bc_nw_cov))

# Diagnostics:
# Helper to collect fitteds and residuals
augment_resids = function(fit, df_used) {
  out = data.frame(
    fitted = as.numeric(stats::fitted(fit)),
    resid  = as.numeric(stats::residuals(fit)),
    stdres = as.numeric(rstandard(fit)),
    sqrt_abs_stdres = sqrt(abs(as.numeric(rstandard(fit)))),
    idx = seq_len(nrow(df_used))
  )
  return(out)
}

diag_ols = augment_resids(ols_model, model_data)
diag_bc  = augment_resids(bc_model,  model_data)

# Residuals vs Fitted (OLS)
p1 = ggplot(diag_ols, aes(x = fitted, y = resid)) +
  geom_point(alpha = 0.7) +
  geom_hline(yintercept = 0, linetype = 2) +
  labs(title = "Residuals vs Fitted (OLS)", x = "Fitted Values", y = "Residuals")
ggsave(filename = paste0(plots_dir, "residuals_vs_fitted_OLS.png"), plot = p1, width = 8, height = 6, dpi = 150)

# Scale–Location (OLS)
p2 = ggplot(diag_ols, aes(x = fitted, y = sqrt_abs_stdres)) +
  geom_point(alpha = 0.7) +
  labs(title = "Scale-Location (OLS)", x = "Fitted Values", y = "√|Standardized Residuals|")
ggsave(filename = paste0(plots_dir, "scale_location_OLS.png"), plot = p2, width = 8, height = 6, dpi = 150)

# QQ (OLS)
png(paste0(plots_dir, "qqplot_OLS.png"), width = 1200, height = 900, res = 150)
qqnorm(stats::residuals(ols_model), main = "Normal Q-Q Plot (OLS Residuals)",
       xlab = "Theoretical Quantiles", ylab = "Sample Quantiles")
qqline(stats::residuals(ols_model))
dev.off()

# Histogram + density (OLS)
p3 = ggplot(diag_ols, aes(x = resid)) +
  geom_histogram(bins = 30, fill = "skyblue", alpha = 0.7) +
  geom_density(linewidth = 1) +
  labs(title = "Residuals Histogram (OLS)", x = "Residuals", y = "Count / Density")
ggsave(filename = paste0(plots_dir, "hist_residuals_OLS.png"), plot = p3, width = 8, height = 6, dpi = 150)

# ACF (OLS)
png(paste0(plots_dir, "ACF_residuals_OLS.png"), width = 1200, height = 900, res = 150)
stats::acf(stats::residuals(ols_model), main = "ACF of OLS Residuals", xlab = "Lag", ylab = "Autocorrelation")
dev.off()

# Cook's D (OLS)
png(paste0(plots_dir, "cooks_distance_OLS.png"), width = 1400, height = 1000, res = 150)
plot(stats::cooks.distance(ols_model), type = "h", main = "Cook's Distance (OLS)",
     xlab = "Observation Index", ylab = "Cook's D")
dev.off()

# Residuals vs Fitted (Box–Cox)
p4 = ggplot(diag_bc, aes(x = fitted, y = resid)) +
  geom_point(alpha = 0.7) +
  geom_hline(yintercept = 0, linetype = 2) +
  labs(title = "Residuals vs Fitted (Box–Cox Transformed Y)", x = "Fitted Values", y = "Residuals")
ggsave(filename = paste0(plots_dir, "residuals_vs_fitted_BoxCox.png"), plot = p4, width = 8, height = 6, dpi = 150)

# Scale–Location (Box–Cox)
p5 = ggplot(diag_bc, aes(x = fitted, y = sqrt_abs_stdres)) +
  geom_point(alpha = 0.7) +
  labs(title = "Scale-Location (Box–Cox Transformed Y)", x = "Fitted Values", y = "√|Standardized Residuals|")
ggsave(filename = paste0(plots_dir, "scale_location_BoxCox.png"), plot = p5, width = 8, height = 6, dpi = 150)

# QQ (Box–Cox)
png(paste0(plots_dir, "qqplot_BoxCox.png"), width = 1200, height = 900, res = 150)
qqnorm(stats::residuals(bc_model), main = "Normal Q-Q Plot (Box–Cox Residuals)",
       xlab = "Theoretical Quantiles", ylab = "Sample Quantiles")
qqline(stats::residuals(bc_model))
dev.off()

# Influence:
# Flag potential influential rows and optionally re-fit without them
infl = stats::influence.measures(ols_model)
cat("\n=== Potentially influential points (OLS) ===\n")
print(summary(infl))

infl_mat  = infl$infmat   # dffit, cook.d, hat, dfbetas...
n         = nrow(model_data)
p         = length(coef(ols_model))
cook_cut  = 4 / n
dffit_cut = 2 * sqrt(p / n)

flag_idx = which(infl_mat[, "cook.d"] > cook_cut | abs(infl_mat[, "dffit"]) > dffit_cut)

if (length(flag_idx) > 0) {
  cat("\nFlagged rows removed for sensitivity:\n"); print(flag_idx)
  model_data_trim = model_data[-flag_idx, , drop = FALSE]
  form_trim       = stats::as.formula(paste(response_var, "~", paste(predictors, collapse = "+")))
  ols_trim        = stats::lm(form_trim, data = model_data_trim)
  cat("\n=== Trimmed OLS + HC3 (sensitivity) ===\n")
  print(lmtest::coeftest(ols_trim, vcov = sandwich::vcovHC(ols_trim, type = "HC3")))
}

# Done:
cat("\nAll plots saved to: ", plots_dir, "\n", sep = "")

