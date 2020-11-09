#load libraries
library(tidyverse)
library(tidymodels)
library(kknn)
library(doParallel)

#lab4 code
full_train <- read_csv("data/train.csv") %>%
  mutate(classification = factor(classification,
                                 levels = 1:4,
                                 labels = c("far below", "below",
                                            "meets", "exceeds"),
                                 ordered = TRUE))

set.seed(1000)
full_train <- slice_sample(full_train, prop = 0.005)
splt <- initial_split(full_train)

set.seed(1000)
train <- training(splt)
test <- testing(splt)

set.seed(1000)
cv <- vfold_cv(test)

knn_recipe <-
  recipe(
    classification ~ lat + lon + econ_dsvntg + gndr + ethnic_cd,
    data = train
  ) %>%
  step_medianimpute(all_numeric(), -all_outcomes(), -has_role("id vars")) %>%
  step_novel(all_nominal(), -all_outcomes()) %>%
  step_unknown(all_nominal(), -all_outcomes()) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_nzv(all_predictors()) %>%
  step_normalize(all_predictors())

#parallel processing
all_cores <- parallel::detectCores(logical = FALSE)

library(doParallel)
cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)
foreach::getDoParWorkers()
clusterEvalQ(cl, {library(tidymodels)})