library(MatchIt)
library(tidyverse)
library(ggplot2)
library(tableone)
library(knitr)
library(broom)
library(lme4)
library(stringr)
library(car)
library(MuMIn)
library(texreg)
# ------------- utility functions -----------
backward_selection <- function(model) {
    while(TRUE) {
        # Get current fixed effects
        fixed_effects <- names(fixef(model))[-1]  # Exclude intercept
        
        # Try removing each term
        models <- list()
        for(term in fixed_effects) {
            formula <- update(formula(model), paste0(". ~ . -", term))
            models[[term]] <- update(model, formula)
        }
        
        # Compare AICs
        aics <- sapply(models, AIC)
        best_aic <- min(aics)
        
        # If removing a term improves AIC, update the model
        if(best_aic < AIC(model)) {
            best_term <- names(which.min(aics))
            model <- models[[best_term]]
            cat("Removed term:", best_term, "\n")
        } else {
            break
        }
    }
    return(model)
}
compare_single_covariate <- function(df1, df2, covariate, dx_column="dx"){
    format_p <- function(p) {
        if(is.na(p)) return(NA)
        return(format(round(p, 3), nsmall = 3, scientific = FALSE))
    }
    wilcox_test <- function(df, covariate, dx_column) {
        tryCatch({
            formula <- as.formula(paste(covariate, "~", dx_column))
            test_result <- wilcox.test(formula, data = df)
            return(tidy(test_result))
        }, error = function(e) {
            return(data.frame(p.value = NA, error = as.character(e)))
        })
    }
    
    # Before matching
    result_df1 <- wilcox_test(df1, covariate, dx_column)
    
    # After matching
    result_df2 <- wilcox_test(df2, covariate, dx_column)
    
    result <- data.frame(
        covariate = covariate,
        p_value_before = format_p(result_df1$p.value),
        p_value_after = format_p(result_df2$p.value)
    )
    
    return(result)
}

get_acc <- function(model, df){
    test_pred <- predict(
        model, newdata = df, 
        type = "response", allow.new.levels = TRUE)
    predicted_classes <- ifelse(test_pred > 0.5, 1, 0)
    actual_classes <- df$dx
    accuracy <- mean(predicted_classes == actual_classes)
    return(round(accuracy, 4))
}

get_metrics <- function(model, df) {
    test_pred <- predict(
        model, newdata = df, 
        type = "response", allow.new.levels = TRUE)
    predicted_classes <- ifelse(test_pred > 0.5, 1, 0)
    actual_classes <- df$dx
    
    # Calculate confusion matrix
    cm <- table(Actual = actual_classes, Predicted = predicted_classes)
    cm
    
    # Extract values from confusion matrix
    TP <- cm[2,2]
    TN <- cm[1,1]
    FP <- cm[1,2]
    FN <- cm[2,1]
    
    # Calculate metrics
    accuracy <- (TP + TN) / (TP + TN + FP + FN)
    precision <- TP / (TP + FP)
    recall <- TP / (TP + FN)
    f1_score <- 2 * (precision * recall) / (precision + recall)
    
    # Return all metrics
    return(list(
        accuracy = round(accuracy, 4),
        precision = round(precision, 4),
        recall = round(recall, 4),
        f1_score = round(f1_score, 4)
    ))
}
set.seed(98103)
# ---------- Pitt corpus -------------
covariates <- c("PRON", "AUX", "VERB", "ADP", 
                "DET", "NOUN", "CCONJ", "ADJ",
                "ADV","PART", "SCONJ", "INTJ",
                "PROPN", "CLAUSE", "educ", "LF", "TTR")
full_df <- read.csv("pitt_total.csv")
filtered_pitt <- full_df[which(full_df$dx %in%  c("Control", "ProbableAD")), ]
filtered_pitt$dx <- ifelse(
    filtered_pitt$dx == "ProbableAD", 1, 0)
col_names <- colnames(filtered_pitt)
factorVals <- c("gender", "dx")
vars <- col_names[-1]
# --------- Pitt table 1 -------------
tab1 <- CreateTableOne(
    data = filtered_pitt,
    factorVars = factorVals,
    vars = vars[! vars == "dx"],
    strata = "dx",
    test = FALSE)
p <- print(
    tab1,
    printToggle = FALSE,
    noSpaces = TRUE,
    smd = TRUE,
    varLabels = TRUE,
    showAllLevels = TRUE)
k <- kable(p, format = "latex", booktabs=TRUE,
           caption = "Covariables before Matching, Pitt corpus")
k
#---------- Pitt matching -------------
mod_ps1 <- matchit(
    dx ~ inv_turns + educ,
    data = filtered_pitt,
    method = "optimal",
    distance = "logit",
    ratio = 1,
    discard = "both",
    std.caliper = FALSE
)
plot(summary(mod_ps1))
matched_pitt1 <- match.data(mod_ps1)
tab1 <- CreateTableOne(
    data = matched_pitt1,
    factorVars = factorVals,
    vars = vars[! vars == "dx"],
    strata = "dx",
    test = FALSE)
p1 <- print(
    tab1,
    printToggle = FALSE,
    noSpaces = TRUE,
    smd = TRUE,
    varLabels = TRUE,
    showAllLevels = TRUE)
k1 <- kable(p1, format = "latex", booktabs=TRUE, 
            caption = "Covariates after Matching with INV Turns, Pitt Corpus")
k1
pitt_results <- do.call(rbind, lapply(covariates, function(item) {
    compare_single_covariate(filtered_pitt, matched_pitt1, item)
}))
write.csv(
    setNames(data.frame(matched_pitt1[['pid']]), "pid"), 
    "matched_pitt.csv",
    row.names = FALSE)
# -------------- WLS ----------------
wls <- read.csv("wls_total.csv")
# revert to 2011 age
wls$age <-wls$age - (2020 - 2011)
wls_vars <- colnames(wls)[-1]
tab2 <- CreateTableOne(
    data = wls, factorVars = factorVals, 
    vars = vars[!vars == "dx"],
    strata = "dx",
    test = FALSE)
p2 <- print(
    tab2, printToggle = FALSE, 
    noSpaces = TRUE, smd = TRUE, 
    varLabels = TRUE, showAllLevels = TRUE)
k2 <- kable(
    p2, format = "latex", booktabs=TRUE,
    caption = "Covariables before Matching, WLS dataset")
k2
# ----------- WLS matching ----------
mod_ps2 <- matchit(
    dx ~ inv_turns + educ,
    data = wls,
    method = "optimal",
    distance = "logit",
    ratio = 1,
    discard = "both",
    std.caliper = FALSE
)
plot(summary(mod_ps2))
matched_wls <- match.data(mod_ps2)
tab3 <- CreateTableOne(
    data = matched_wls,
    factorVars = factorVals,
    vars = vars[! vars == "dx"],
    strata = "dx",
    test = FALSE)
p3 <- print(
    tab3,
    printToggle = FALSE,
    noSpaces = TRUE,
    smd = TRUE,
    varLabels = TRUE,
    showAllLevels = TRUE)
k3 <- kable(p3, format = "latex", booktabs=TRUE, 
            caption = "Covariates after Matching with INV Turns, WLS corpus")
k3
wls_results <- do.call(rbind, lapply(covariates, function(item) {
    compare_single_covariate(wls, matched_wls, item)
}))
write.csv(
    setNames(data.frame(matched_wls[c('pid', 'dx')]), c("pid", "dx")),
    "matched_wls.csv",
    row.names = FALSE
)
# -------------- test ------------
wilcox.test(
    inv_turns ~ dx,
    data = wls,
    alternative = "less",
    exact = FALSE)

wilcox.test(
    inv_turns ~ dx,
    data = filtered_pitt,
    alternative = "less",
    exact = FALSE)

wilcox.test(
    par_turns ~ dx,
    data = filtered_pitt,
    #alternative = "less",
    exact = FALSE)

wilcox.test(
    par_turns ~ dx,
    data = wls,
    alternative = "less",
    exact = FALSE)

wilcox.test(
    par_turns ~ dx,
    data = matched_wls,
    alternative = "less",
    exact = FALSE)

wilcox.test(filtered_pitt$educ, wls$educ, alternative = "less")
cor(filtered_pitt$inv_turns, filtered_pitt$mmse, method = "spearman")
cor(wls$inv_turns, wls$score, method = "spearman")
filtered_pitt[covariates] <- scale(filtered_pitt[covariates])
wls[covariates] <- scale(wls[covariates])
# -------------- GLM on original Pitt ---------------
sample <- sample(
    c(TRUE, FALSE), nrow(filtered_pitt),
    replace=TRUE, prob=c(0.7,0.3))
filtered_pitt_train <- filtered_pitt[sample,]
filtered_pitt_test <- filtered_pitt[!sample,]
matched_pitt1[covariates] <- scale(matched_pitt1[covariates])
sample <- sample(
    c(TRUE, FALSE), nrow(matched_pitt1),
    replace=TRUE, prob=c(0.7,0.3))
matched_pitt_train <- matched_pitt1[sample,]
matched_pitt_test <- matched_pitt1[!sample,]
matched_wls[covariates] <- scale(matched_wls[covariates])
# -------- mixed effect GLM on Pitt -------------
pitt_model <- glmer(
    dx ~ inv_turns * (LF + TTR + PRON + AUX + ADP + DET + NOUN +
        CCONJ + ADJ + ADV + PART + SCONJ + PROPN + VERB  + 
        CLAUSE) + (1 | inv_turns),
    data = filtered_pitt_train, family = binomial,
    control = glmerControl(
        optimizer = "bobyqa", 
        optCtrl = list(maxfun = 2e5))
)
summary(pitt_model)
matched_pitt_model <- glmer(
    dx ~ inv_turns * (LF + TTR + PRON + AUX + ADP + DET + NOUN +
                          CCONJ + ADJ + ADV + PART + SCONJ + PROPN + VERB  + 
                          CLAUSE) + (1 | inv_turns),
    data = matched_pitt_train, family = binomial,
    control = glmerControl(
        optimizer = "bobyqa", 
        optCtrl = list(maxfun = 2e5))
)
summary(matched_pitt_model)
final_pitt_model <- backward_selection(pitt_model)
final_matched_pitt <- backward_selection(matched_pitt_model)
summary(final_pitt_model)
summary(final_matched_pitt)

# ----------- visualization of interaction terms ----------
inv_turns_range <- range(matched_pitt_test$inv_turns)
PRON_range <- range(matched_pitt_test$PRON)
plot_data <- expand.grid(
    inv_turns = seq(inv_turns_range[1], inv_turns_range[2], length.out = 100),
    PRON = seq(PRON_range[1], PRON_range[2], length.out = 100)
)
other_predictors <- c("AUX", "ADP", "VERB", "CLAUSE", "TTR", "PART")
for(pred in other_predictors) {
    plot_data[[pred]] <- mean(matched_pitt_test[[pred]], na.rm = TRUE)
}

# Add predicted probabilities
plot_data$predicted_prob <- predict(final_matched_pitt, newdata = plot_data, 
                                    type = "response", 
                                    re.form = NA)

# Create the plot
ggplot(plot_data, aes(x = PRON, y = predicted_prob, color = inv_turns)) +
    geom_line() +
    scale_color_viridis_c(name = "Turns") +
    labs(x = "Pronoun Usage", 
         y = "Predicted Probability of Diagnosis") +
    theme_minimal()

# ---------- classification performance ---------
metrics <- get_metrics(pitt_model, filtered_pitt_test)
cat(paste(
    "pitt model on pitt test:",
    "\nAccuracy:", metrics$accuracy,
    "\nPrecision:", metrics$precision,
    "\nRecall:", metrics$recall,
    "\nF1 Score:", metrics$f1_score
))

metrics <- get_metrics(final_matched_pitt, matched_pitt_test)
cat(paste(
    "final matched pitt model on matched pitt test:",
    "\nAccuracy:", metrics$accuracy,
    "\nPrecision:", metrics$precision,
    "\nRecall:", metrics$recall,
    "\nF1 Score:", metrics$f1_score
))

metrics <- get_metrics(final_matched_pitt, wls)
cat(paste(
    "final matched pitt model on wls:",
    "\nAccuracy:", metrics$accuracy,
    "\nPrecision:", metrics$precision,
    "\nRecall:", metrics$recall,
    "\nF1 Score:", metrics$f1_score
))

metrics <- get_metrics(final_matched_pitt, matched_wls)
cat(paste(
    "final matched pitt model on matched wls:",
    "\nAccuracy:", metrics$accuracy,
    "\nPrecision:", metrics$precision,
    "\nRecall:", metrics$recall,
    "\nF1 Score:", metrics$f1_score
))

metrics <- get_metrics(final_pitt_model, filtered_pitt_test)
cat(paste(
    "final pitt model on pitt test:",
    "\nAccuracy:", metrics$accuracy,
    "\nPrecision:", metrics$precision,
    "\nRecall:", metrics$recall,
    "\nF1 Score:", metrics$f1_score
))

metrics <- get_metrics(pitt_model, wls)
cat(paste(
    "pitt model on wls:",
    "\nAccuracy:", metrics$accuracy,
    "\nPrecision:", metrics$precision,
    "\nRecall:", metrics$recall,
    "\nF1 Score:", metrics$f1_score
))

metrics <- get_metrics(final_pitt_model, wls)
cat(paste(
    "final pitt model on wls:",
    "\nAccuracy:", metrics$accuracy,
    "\nPrecision:", metrics$precision,
    "\nRecall:", metrics$recall,
    "\nF1 Score:", metrics$f1_score
))

# -------- mixed effect GLM on WLS, no classification -------------
wls_model <- glmer(
    dx ~ inv_turns * (LF + TTR + PRON + AUX + ADP + DET + NOUN +
                          CCONJ + ADJ + ADV + PART + SCONJ + PROPN + VERB  + 
                          CLAUSE) + (1 | inv_turns),
    data = wls, family = binomial,
    control = glmerControl(
        optimizer = "bobyqa", 
        optCtrl = list(maxfun = 2e5))
)
summary(wls_model)
matched_wls_model <- glmer(
    dx ~ inv_turns * (LF + TTR + PRON + AUX + ADP + DET + NOUN +
                          CCONJ + ADJ + ADV + PART + SCONJ + PROPN + VERB  + 
                          CLAUSE) + (1 | inv_turns),
    data = matched_wls, family = binomial,
    control = glmerControl(
        optimizer = "bobyqa", 
        optCtrl = list(maxfun = 2e5))
)
final_wls_model <- backward_selection(wls_model)
final_matched_wls <- backward_selection(matched_wls_model)
summary(final_wls_model)
summary(final_matched_wls)

texreg(list(final_matched_pitt, final_matched_wls), booktabs = TRUE)

metrics <- get_metrics(final_matched_wls, filtered_pitt_test)
cat(paste(
    "final wls model on orginal pitt test:",
    "\nAccuracy:", metrics$accuracy,
    "\nPrecision:", metrics$precision,
    "\nRecall:", metrics$recall,
    "\nF1 Score:", metrics$f1_score
))

metrics <- get_metrics(final_matched_wls, matched_pitt_test)
cat(paste(
    "final wls model on matched pitt test:",
    "\nAccuracy:", metrics$accuracy,
    "\nPrecision:", metrics$precision,
    "\nRecall:", metrics$recall,
    "\nF1 Score:", metrics$f1_score
))

# ----------- stats ------------
filtered_pitt[c("pid", "visit")] <- str_split_fixed(filtered_pitt$pid, "-", 2)
# number of participants
n_distinct(filtered_pitt$pid[filtered_pitt$dx == 0])
n_distinct(filtered_pitt$pid[filtered_pitt$dx == 0 & filtered_pitt$gender == "female"])
n_distinct(filtered_pitt$pid[filtered_pitt$dx == 0 & filtered_pitt$gender == "male"])
# age
mean(filtered_pitt$age[filtered_pitt$dx == 0])
sd(filtered_pitt$age[filtered_pitt$dx == 0])
mean(filtered_pitt$age[filtered_pitt$dx == 1])
sd(filtered_pitt$age[filtered_pitt$dx == 1])

n_distinct(filtered_pitt$pid[filtered_pitt$dx == 1])
n_distinct(filtered_pitt$pid[filtered_pitt$dx == 1 & filtered_pitt$gender == "female"])
n_distinct(filtered_pitt$pid[filtered_pitt$dx == 1 & filtered_pitt$gender == "male"])

nrow(filtered_pitt[filtered_pitt$dx == 0,])
nrow(filtered_pitt[filtered_pitt$dx == 1,])

nrow(wls[wls$dx == 0,])
nrow(wls[wls$dx == 1,])

n_distinct(wls$pid[wls$dx == 0])
n_distinct(wls$pid[wls$dx == 1])
# ------------ INV utterance analysis ----------
pitt_sim <- read.csv('pitt_sim.csv')
wls_sim <- read.csv('wls_sim.csv')
pitt_sim <- pitt_sim %>% 
    filter(str_detect(utterance1, "anything else") & 
               str_detect(utterance2, "anything else"))
wls_sim <- wls_sim %>% 
    filter(str_detect(utterance1, "anything else") & 
               str_detect(utterance2, "anything else"))
pitt_sep <- read.csv("pitt_inv_sep.csv")
wls_sep <- read.csv("wls_inv_sep.csv")
pitt_sep <- pitt_sep %>% 
    filter(str_detect(text, "anything else"))
wls_sep <- wls_sep %>% 
    filter(str_detect(text, "anything else"))
nrow(pitt_sep[pitt_sep$dx == 1,])
nrow(pitt_sep[pitt_sep$dx == 0,])
nrow(wls_sep[wls_sep$dx == 1,])
nrow(wls_sep[wls_sep$dx == 0,])
mean(pitt_sim$score)
mean(wls_sim$score)
