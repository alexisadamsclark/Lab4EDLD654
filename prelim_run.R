#using Daniel's oraganization suggestion

source("common.R")

model <- nearest_neighbor() %>%
  set_mode("classification") %>%
  set_engine("kknn")

resamples <- fit_resamples(model, knn_recipe, cv)
saveRDS(resamples, "models/prelim_run.Rds")