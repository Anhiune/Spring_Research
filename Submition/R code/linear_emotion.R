# ================================
# Linear regressions: returns ~ emotions (level and lag1)
# Validations + robust inference + diagnostics + plots
# ================================

# Install once if needed:
# install.packages(c("tidyverse","broom","lmtest","sandwich","car"))

suppressPackageStartupMessages({
  library(tidyverse)
  library(broom)
  library(lmtest)     # bptest, bgtest, dwtest, coeftest
  library(sandwich)   # vcovHC, NeweyWest
  library(car)        # vif
})

# ---------------------------
# Paths
# ---------------------------
data_path <- "C:/Users/hoang/OneDrive - University of St. Thomas/Summer Research/Data/Data file for Granger test.csv"
out_dir   <- "C:/Users/hoang/OneDrive - University of St. Thomas/Summer Research/Linear_Emotion"
plot_dir  <- file.path(out_dir, "plots")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

# ---------------------------
# Load & basic clean
# ---------------------------
df <- read.csv(data_path, check.names = FALSE, stringsAsFactors = FALSE)

# date column (optional)
if ("date" %in% names(df)) {
  df$date <- as.Date(df$date)
}

# drop Unnamed
if (any(grepl("^Unnamed", names(df)))) {
  df <- df %>% dplyr::select(-dplyr::starts_with("Unnamed"))
}

# ---------------------------
# Ensure returns exist (adj_return); build from price if needed
# ---------------------------
return_candidates <- c("adj_return","return","ret","returns","daily_return")
ret_col <- intersect(return_candidates, names(df))

if (length(ret_col) == 0) {
  price_candidates <- c("adj_close","close","price","Adj Close","Adj_Close","Close","PX_LAST")
  price_col <- intersect(price_candidates, names(df))
  if (length(price_col) == 0) {
    price_guess <- names(df)[sapply(df, is.numeric) & grepl("(close|price)", tolower(names(df)))]
    if (length(price_guess)) price_col <- price_guess[1]
  } else {
    price_col <- price_col[1]
  }
  if (!length(price_col)) stop("No returns column and no price column found. Please provide one.")
  df <- df %>% mutate(adj_return = c(NA, diff(log(as.numeric(.data[[price_col]])))))
  ret_col <- "adj_return"
  message("Built returns as log-diff from '", price_col, "' into '", ret_col, "'.")
} else {
  ret_col <- ret_col[1]
  message("Using existing returns column: ", ret_col)
}

# ---------------------------
# Collect emotion variables (numeric)
# ---------------------------
known_emotions <- c(
  "emotion","anger","joy","fear","sadness","surprise","disgust","trust",
  "anticipation","positive","negative","valence","arousal","dominance"
)

emotion_cols <- unique(c(
  intersect("emotion", names(df)),
  names(df)[grepl("emotion", tolower(names(df)))],
  intersect(known_emotions, names(df))
))
emotion_cols <- emotion_cols[emotion_cols %in% names(df)]

# Coerce to numeric where possible
for (v in emotion_cols) {
  if (!is.numeric(df[[v]])) {
    suppressWarnings(num_try <- as.numeric(df[[v]]))
    if (sum(!is.na(num_try)) > 0) df[[v]] <- num_try
  }
}
# keep numeric only
emotion_cols <- emotion_cols[sapply(df[emotion_cols], is.numeric)]

if (!length(emotion_cols)) stop("No numeric emotion variables found.")

# Drop constant/near-constant emotion columns (variance == 0 or 0/1 with one level)
is_const <- function(x) {
  x <- x[is.finite(x)]
  if (!length(x)) return(TRUE)
  length(unique(x)) <= 1
}
const_flags <- vapply(df[emotion_cols], is_const, logical(1))
if (any(const_flags)) {
  message("Dropping constant emotion columns: ",
          paste(emotion_cols[const_flags], collapse = ", "))
  emotion_cols <- emotion_cols[!const_flags]
}
if (!length(emotion_cols)) stop("All emotion variables were constant; nothing to regress.")

# ---------------------------
# Build modeling frames
# ---------------------------
# Remove rows with NA in returns or emotions
base_data <- df %>%
  dplyr::select(dplyr::any_of(c("date", ret_col, emotion_cols))) %>%
  tidyr::drop_na(dplyr::all_of(c(ret_col, emotion_cols)))

stopifnot(nrow(base_data) >= 30)

# Create lag1 emotions
lag1_names <- paste0(emotion_cols, "_lag1")
for (i in seq_along(emotion_cols)) {
  base_data[[lag1_names[i]]] <- dplyr::lag(base_data[[emotion_cols[i]]], 1)
}
lag_data <- base_data %>% tidyr::drop_na(dplyr::all_of(c(ret_col, lag1_names)))

# ---------------------------
# Helper: fit OLS with alias cleanup + robust inference
# ---------------------------
fit_lm_clean <- function(formula, data) {
  mod <- lm(formula, data = data)
  # Handle aliased coefficients
  aliased <- names(coef(mod))[is.na(coef(mod))]
  aliased <- setdiff(aliased, "(Intercept)")
  if (length(aliased)) {
    message("Aliased (perfectly collinear) terms removed: ",
            paste(aliased, collapse = ", "))
    rhs_terms <- attr(terms(formula), "term.labels")
    rhs_keep  <- setdiff(rhs_terms, aliased)
    if (!length(rhs_keep)) stop("After removing aliased terms, no predictors remain.")
    formula <- as.formula(paste(all.vars(formula)[1], "~", paste(rhs_keep, collapse = " + ")))
    mod <- lm(formula, data = data)
  }
  list(model = mod, formula = formula)
}

tidy_robust <- function(mod, type = c("HC3","NW"), nw_lag = NULL) {
  type <- match.arg(type)
  if (type == "HC3") {
    vc <- vcovHC(mod, type = "HC3")
  } else {
    if (is.null(nw_lag)) {
      Tn <- nrow(model.frame(mod))
      nw_lag <- max(1, floor(0.75 * Tn^(1/3)))
    }
    vc <- NeweyWest(mod, lag = nw_lag, prewhite = FALSE, adjust = TRUE)
  }
  ct <- lmtest::coeftest(mod, vcov = vc)
  tibble::tibble(
    term      = rownames(ct),
    estimate  = ct[, "Estimate"],
    std.error = ct[, "Std. Error"],
    statistic = ct[, "t value"],
    p.value   = ct[, "Pr(>|t|)"],
    se_type   = type
  )
}

# ---------------------------
# Fit models
# ---------------------------
# (A) Contemporaneous emotions
form_now <- as.formula(paste(ret_col, "~", paste(emotion_cols, collapse = " + ")))
fit_now  <- fit_lm_clean(form_now, base_data)
mod_now  <- fit_now$model

# (B) Lagged emotions (t-1)
form_lag <- as.formula(paste(ret_col, "~", paste(lag1_names, collapse = " + ")))
fit_lg   <- fit_lm_clean(form_lag, lag_data)
mod_lag  <- fit_lg$model

# ---------------------------
# Multicollinearity (VIF)
# ---------------------------
cat("\n=== VIF (contemporaneous) ===\n")
suppressWarnings(print(car::vif(mod_now)))
cat("\n=== VIF (lag1) ===\n")
suppressWarnings(print(car::vif(mod_lag)))

# ---------------------------
# Residual diagnostics
# ---------------------------
diag_block <- function(m, label) {
  cat("\n============================\n", label, "\n============================\n", sep = "")
  print(summary(m))
  cat("\nBreusch–Pagan (het):\n"); print(lmtest::bptest(m))
  cat("\nDurbin–Watson:\n");      print(lmtest::dwtest(m))
  cat("\nBreusch–Godfrey (lag 1):\n"); print(lmtest::bgtest(m, order = 1))
  cat("\nBreusch–Godfrey (lag 4):\n"); print(lmtest::bgtest(m, order = 4))
}
diag_block(mod_now, "Diagnostics: contemporaneous")
diag_block(mod_lag, "Diagnostics: lag1")

# ---------------------------
# Robust inference (HC3 + Newey–West)
# ---------------------------
hc3_now <- tidy_robust(mod_now, "HC3")
nw_now  <- tidy_robust(mod_now, "NW")

hc3_lag <- tidy_robust(mod_lag, "HC3")
nw_lag  <- tidy_robust(mod_lag, "NW")

# Save tidy tables
readr::write_csv(hc3_now, file.path(out_dir, "lm_now_coef_HC3.csv"))
readr::write_csv(nw_now,  file.path(out_dir, "lm_now_coef_NeweyWest.csv"))
readr::write_csv(hc3_lag, file.path(out_dir, "lm_lag1_coef_HC3.csv"))
readr::write_csv(nw_lag,  file.path(out_dir, "lm_lag1_coef_NeweyWest.csv"))

# ---------------------------
# Coefficient plot (HC3) for both models
# ---------------------------
plot_coefs <- function(tidy_tbl, title) {
  dfp <- tidy_tbl %>%
    filter(term != "(Intercept)") %>%
    mutate(
      lower = estimate - 1.96 * std.error,
      upper = estimate + 1.96 * std.error,
      term  = factor(term, levels = rev(term))
    )
  ggplot(dfp, aes(x = term, y = estimate)) +
    geom_hline(yintercept = 0, linetype = 2) +
    geom_point() +
    geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.15) +
    coord_flip() +
    labs(title = title, x = "Predictor", y = "Coefficient (HC3 95% CI)") +
    theme_minimal(base_size = 12)
}

p_now <- plot_coefs(hc3_now, "Returns ~ Emotions (same-day) — HC3 SEs")
p_lag <- plot_coefs(hc3_lag, "Returns ~ Emotions (lag-1) — HC3 SEs")

ggsave(file.path(plot_dir, "lm_now_HC3_coefplot.png"), p_now, width = 8, height = 6, dpi = 150)
ggsave(file.path(plot_dir, "lm_lag1_HC3_coefplot.png"), p_lag, width = 8, height = 6, dpi = 150)

# ---------------------------
# Console summary (HC3 significant terms)
# ---------------------------
sig_line <- function(tb, alpha = 0.05) {
  tb %>% filter(term != "(Intercept)", p.value < alpha) %>% arrange(p.value) %>%
    transmute(term, estimate = round(estimate, 4), p.value = signif(p.value, 3))
}

cat("\n=== Significant (HC3) — contemporaneous (alpha=0.05) ===\n")
print(sig_line(hc3_now))
cat("\n=== Significant (HC3) — lag1 (alpha=0.05) ===\n")
print(sig_line(hc3_lag))

message("\nAll outputs written to:\n- ", out_dir, "\n- ", plot_dir, "\n")
# Assumes you already ran everything up to hc3_now / hc3_lag and defined plot_dir
# from your script in the message.

suppressPackageStartupMessages({
  library(tidyverse)
  library(ggplot2)
})

# Helper to build a clean coefficient dataframe (excluding intercept)
prep_coef_df <- function(tidy_tbl) {
  tidy_tbl %>%
    filter(term != "(Intercept)") %>%
    mutate(
      lower = estimate - 1.96 * std.error,
      upper = estimate + 1.96 * std.error,
      term  = factor(term, levels = rev(term))
    )
}

# Plot function (HC3 95% CI)
plot_coefs <- function(dfp, title_text) {
  ggplot(dfp, aes(x = term, y = estimate)) +
    geom_hline(yintercept = 0, linetype = 2) +
    geom_point(size = 2) +
    geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.15) +
    coord_flip() +
    labs(title = title_text,
         x = "Predictor",
         y = "Coefficient (HC3 95% CI)") +
    theme_minimal(base_size = 12)
}

# Prepare data frames
df_now <- prep_coef_df(hc3_now)
df_lag <- prep_coef_df(hc3_lag)

# Build plots with clear, manuscript-friendly titles
p_now <- plot_coefs(df_now, "Figure 1: Returns ~ Same-day Emotions (HC3 standard errors)")
p_lag <- plot_coefs(df_lag, "Figure 2: Returns ~ Lagged Emotions (t-1, HC3 standard errors)")

# Save figures (also save with simple names for easy insertion)
ggsave(file.path(plot_dir, "Figure_1_Returns_SameDay_Emotions_HC3.png"),
       p_now, width = 8, height = 6, dpi = 150)
ggsave(file.path(plot_dir, "Figure_2_Returns_Lag1_Emotions_HC3.png"),
       p_lag, width = 8, height = 6, dpi = 150)

# (Optional) If you also want the earlier filenames:
ggsave(file.path(plot_dir, "lm_now_HC3_coefplot.png"),
       p_now, width = 8, height = 6, dpi = 150)
ggsave(file.path(plot_dir, "lm_lag1_HC3_coefplot.png"),
       p_lag, width = 8, height = 6, dpi = 150)

