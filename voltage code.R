rm(list = ls())
dev.off()
library(forecastML)
library(dplyr)
library(DT)
library(ggplot2)
library(xgboost)
library(hablar)
library(forecastML)

data("data_buoy_gaps")
str(data_buoy_gaps)


data_buoy_gaps <- read.csv(file.choose(),header = T ,sep = ",")

DT::datatable(head(data_buoy_gaps), options = list(scrollX = TRUE))
head(data_buoy_gaps)
data <- data_buoy_gaps

str(data)

# %Y-%m-%d
data$Date <- as.POSIXct(data$Date, format=" %Y-%m-%d %H:%M:%S")
str(data)


print(list(paste0("The original dataset with gaps in data collection is ", nrow(data_buoy_gaps), " rows."),
           paste0("The modified dataset with no gaps in data collection from fill_gaps() is ", nrow(data), " rows.")))

data$day <- as.integer(lubridate::mday(data$Date))


data$year <- as.numeric(lubridate::year(data$Date))


p <- ggplot(data, aes(x = Date, y = Voltage))
p <- p + geom_line()
p <- p + theme_bw() + theme(
  legend.position = "none"
) + xlab(NULL)
p+ theme(legend.position = "none",panel.grid.major = element_blank(),
         panel.grid.minor = element_blank(),
         strip.text = element_text(size = 14, face = "bold"),
         axis.title = element_text(size = 14, face = "bold"),
         axis.text = element_text(size = 14)
)


outcome_col <- 1

horizons <- c(1,7,30)    # Features from 1 to 30 days in the past and annually.

lookback <- c(1:30,360:370)

dates <- data$Date


data$Date <- NULL  # Dates, however, don't need to be in the input data.

outcome_col <- 1

frequency <- "1 min "  # A string that works in base::seq(..., by = "frequency").


dynamic_features <- c( "day", "year")  # Features that change through time but which will not be lagged.


type <- "train"  # Create a model-training dataset.



data_train <- forecastML::create_lagged_df(data, type = type, outcome_col = outcome_col,
                                           horizons = horizons, lookback = lookback,
                                           dates = dates,
                                           frequency = frequency,
                                           dynamic_features = dynamic_features,
                                           # groups = groups,
                                           # static_features = static_features,
                                           use_future = FALSE)

DT::datatable(head(data_train$horizon_1), options = list(scrollX = TRUE))
p <- plot(data_train)  # plot.lagged_df() returns a ggplot object.
p <- p + geom_tile(NULL)  # Remove the gray border for a cleaner plot.
p

windows <- forecastML::create_windows(data_train, window_length = 0,
                                      include_partial_window = FALSE)
windows

p <- plot(windows, data_train) + theme(legend.position = "none")+ylim(2.5,4.5)
p

p <- plot(windows, data_train) +
  theme(legend.position = "none")+ylim(0,1)
p


model_function <- function(data, outcome_col = 1) {

  # xgboost cannot handle missing outcomes data.
  data <- data[!is.na(data[, outcome_col]), ]

  indices <- 1:nrow(data)

  set.seed(224)
  train_indices <- sample(1:nrow(data), ceiling(nrow(data) * .8), replace = FALSE)
  test_indices <- indices[!(indices %in% train_indices)]

  data_train <- xgboost::xgb.DMatrix(data = as.matrix(data[train_indices,
                                                           -(outcome_col), drop = FALSE]),
                                     label = as.matrix(data[train_indices,
                                                            outcome_col, drop = FALSE]))

  data_test <- xgboost::xgb.DMatrix(data = as.matrix(data[test_indices,
                                                          -(outcome_col), drop = FALSE]),
                                    label = as.matrix(data[test_indices,
                                                           outcome_col, drop = FALSE]))

  params <- list("objective" = "reg:linear")
  watchlist <- list(train = data_train, test = data_test)

  set.seed(224)
  model <- xgboost::xgb.train(data = data_train, params = params,
                              max.depth = 8, nthread = 2, nrounds = 30,
                              metrics = "rmse", verbose = 0,
                              early_stopping_rounds = 5,
                              watchlist = watchlist)

  return(model)
}

# str(data)
model_results_cv <- forecastML::train_model(lagged_df = data_train,
                                            windows = windows,
                                            model_name = "xgboost",
                                            model_function = model_function,
                                            use_future = FALSE)


summary(model_results_cv$horizon_1$window_1$model)

##                 Length Class              Mode
## handle               1 xgb.Booster.handle externalptr
## raw             309461 -none-             raw
## best_iteration       1 -none-             numeric
## best_ntreelimit      1 -none-             numeric
## best_score           1 -none-             numeric
## niter                1 -none-             numeric
## evaluation_log       3 data.table         list
## call                10 -none-             call
## params               5 -none-             list
## callbacks            2 -none-             list
## feature_names      128 -none-             character
## nfeatures            1 -none-             numeric

# If 'model' is passed as a named list, the prediction model would be accessed with model$model or model["model"].
prediction_function <- function(model, data_features) {
  x <- xgboost::xgb.DMatrix(data = as.matrix(data_features))
  data_pred <- data.frame("y_pred" = predict(model, x),
                          "y_pred_lower" = predict(model, x) - 2,  # Optional; in practice, forecast bounds are not hard coded.
                          "y_pred_upper" = predict(model, x) + 2)  # Optional; in practice, forecast bounds are not hard coded.
  return(data_pred)
}

data_pred_cv <- predict(model_results_cv, prediction_function = list(prediction_function), data = data_train)

plot(data_pred_cv,panel.grid.major = element_blank(),
     panel.grid.minor = element_blank(),lwd = 0.001)+ylim(2.5,4.5) + theme(legend.position = "none",panel.grid.major = element_blank(),
                                                               panel.grid.minor = element_blank(),
                                                               strip.text = element_text(size = 14, face = "bold"),
                                                               axis.title = element_text(size = 14, face = "bold"),
                                                               axis.text = element_text(size = 14)
     )


plot(data_pred_cv, facet = group ~ model,  windows = 1,panel.grid.major = element_blank(),
     panel.grid.minor = element_blank())+ylim(2.5,4.5)+ theme(legend.position = "topright",panel.grid.major = element_blank(),
                                                              panel.grid.minor = element_blank(),
                                                              strip.text = element_text(size = 14, face = "bold"),
                                                              axis.title = element_text(size = 14, face = "bold"),
                                                              axis.text = element_text(size = 14)
     )

plot(data_pred_cv, facet = group ~ model, windows = 1,panel.grid.major = element_blank(),
     panel.grid.minor = element_blank())+ylim(2.5,4.5)+ theme(legend.position = "none",panel.grid.major = element_blank(),
                                                              panel.grid.minor = element_blank(),
                                                              strip.text = element_text(size = 14, face = "bold"),
                                                              axis.title = element_text(size = 14, face = "bold"),
                                                              axis.text = element_text(size = 14)
     )



plot(data_pred_cv, facet = group ~ horizon ,windows = 1,panel.grid.major = element_blank(),
     panel.grid.minor = element_blank())+ylim(2.5,4.5)+ theme(legend.position = "none",panel.grid.major = element_blank(),
                                                          panel.grid.minor = element_blank(),
                                                          strip.text = element_text(size = 10, face = "bold"),
                                                          axis.title = element_text(size = 10, face = "bold"),
                                                          axis.text = element_text(size = 10))

data_error <- forecastML::return_error(data_pred_cv)
data_error
plot(data_error, type = "window", metric = "mae")+ theme(legend.position = "none",panel.grid.major = element_blank(),
                                                         panel.grid.minor = element_blank(),
                                                         strip.text = element_text(size = 14, face = "bold"),
                                                         axis.title = element_text(size = 14, face = "bold"),
                                                         axis.text = element_text(size = 14)
)



plot(data_error, type = "horizon", metric = "mae")+ theme(legend.position = "none",panel.grid.major = element_blank(),
                                                          panel.grid.minor = element_blank(),
                                                          strip.text = element_text(size = 12, face = "bold"),
                                                          axis.title = element_text(size = 12, face = "bold"),
                                                          axis.text = element_text(size = 12)
)

plot(data_error, type = "global", metric = "mae")+ theme(legend.position = "none",panel.grid.major = element_blank(),
                                                         panel.grid.minor = element_blank(),
                                                         strip.text = element_text(size = 14, face = "bold"),
                                                         axis.title = element_text(size = 14, face = "bold"),
                                                         axis.text = element_text(size = 14)
)

type <- "forecast"  # Create a forecasting dataset for our predict() function.

data_forecast <- forecastML::create_lagged_df(data, type = type, outcome_col = outcome_col,
                                              horizons = horizons, lookback = lookback,
                                              dates = dates, frequency = frequency,
                                              dynamic_features = dynamic_features,
                                              groups = groups, static_features = static_features,
                                              use_future = FALSE)

DT::datatable(head(data_forecast$horizon_1), options = list(scrollX = TRUE))

for (i in seq_along(data_forecast)) {
  data_forecast[[i]]$day <- lubridate::mday(data_forecast[[i]]$index)  # When dates are given, the 'index` is date-based.
  data_forecast[[i]]$year <- lubridate::year(data_forecast[[i]]$index)
}
# mae

data_forecasts <- predict(model_results_cv, prediction_function = list(prediction_function), data = data_forecast)

plot(data_forecasts)+ylim(3,4.5)

plot(data_forecasts, facet = group ~ ., group_filter = "group %in% 1:3")+ylim(2.5,4.5)

windows <- forecastML::create_windows(data_train, window_length = 0)

p <- plot(windows, data_train) + theme(legend.position = "none")
p

# Un-comment the code below and set 'use_future' to TRUE.
#future::plan(future::multiprocess)

model_results_no_cv <- forecastML::train_model(lagged_df = data_train,
                                               windows = windows,
                                               model_name = "xgboost",
                                               model_function = model_function,
                                               use_future = FALSE)

data_forecasts <- predict(model_results_no_cv, prediction_function = list(prediction_function), data = data_forecast)


DT::datatable(head(data_forecasts), options = list(scrollX = TRUE))

data_combined <- forecastML::combine_forecasts(data_forecasts)

# Plot a background dataset of actuals using the most recent data.
data_actual <- data[dates >= as.Date("2016-05-13"), ]
actual_indices <- dates[dates >= as.Date("2016-05-13")]

# Plot all final forecasts plus historical data.
plot(data_combined, data_actual = data_actual, actual_indices = actual_indices)+ylim(2.5,4.5)+ theme(legend.position = "none",panel.grid.major = element_blank(),
                                                                                                     panel.grid.minor = element_blank(),
                                                                                                     strip.text = element_text(size = 14, face = "bold"),
                                                                                                     axis.title = element_text(size = 14, face = "bold"),
                                                                                                     axis.text = element_text(size = 14)
)


plot(data_combined, data_actual = data_actual, actual_indices = actual_indices,
     facet = group ~ .)+ylim(2.5,4.5)+ theme(legend.position = "none",panel.grid.major = element_blank(),
                                                                                       panel.grid.minor = element_blank(),
                                                                                       strip.text = element_text(size = 14, face = "bold"),
                                                                                       axis.title = element_text(size = 14, face = "bold"),
                                                                                       axis.text = element_text(size = 14)
     )







