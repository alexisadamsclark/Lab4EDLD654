#using Daniel's oraganization suggestion

source("common.R")

model <- nearest_neighbor() %>%
  set_mode("classification") %>%
  set_engine("kknn") %>%
  set_args(neighbors = tune(), 
  dist_power = tune())

hpar <- parameters(
  neighbors(range = c(1,20)),
  dist_power()
)

grid <- grid_max_entropy(hpar, size = 25)

#ggplot(grid, aes(neighbors, dist_power)) + 
  #geom_point() + 
  #theme_light()

#ggsave(here::here("plots", "grid.pdf"))

tune <- tune_grid(model, preprocessor = knn_recipe, resamples = cv, grid = grid)
saveRDS(tune, "models/tune.Rds")