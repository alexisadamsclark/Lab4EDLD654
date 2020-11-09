#using Daniel's recommended oraganization structure

source("common.R")

model_tuned <- readRDS("models/tune.Rds")

model <- nearest_neighbor() %>%
  set_mode("classification") %>%
  set_engine("kknn")

param_tuned <- select_best(model_tuned, metric = "roc_auc")
model_final <- finalize_model(model, param_tuned)
recipe_final <- finalize_recipe(knn_recipe, param_tuned)

final_fit <- last_fit(model_final, preprocessor = recipe_final, split = splt)
saveRDS(final_fit, "models/final.Rds")