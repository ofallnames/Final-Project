# Load necessary libraries
library(readr)
library(caret)
library(keras)
library(dplyr)

# Set seed for reproducibility
set.seed(123)

# Import data from CSV file
data <- read_csv("C:/Users/Thomas/Desktop/boston-housing/train.csv")


# Standardize the data
scaled_data <- scale(data)

# Split the data into training and testing sets (80% training, 20% testing)
train_index <- createDataPartition(data$medv, p = 0.8, list = FALSE)
train_data <- scaled_data[train_index, ]
test_data <- scaled_data[-train_index, ]

# Define the neural network model
model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = ncol(train_data)) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1)

# Compile the model
model %>% compile(
  loss = "mse",
  optimizer = optimizer_adam(),
  metrics = c("mae")
)

# Train the model
history <- model %>% fit(
  x = as.matrix(train_data[, -which(colnames(train_data) == "medv")]),
  y = train_data$medv,
  epochs = 100,
  batch_size = 32,
  validation_split = 0.2
)

# Evaluate the model on the testing set
predictions <- model %>% predict(as.matrix(test_data[, -which(colnames(test_data) == "medv")]))

# Revert the scaling on predictions
unscaled_predictions <- predictions * sd(data$medv[train_index]) + mean(data$medv[train_index])

# Calculate RMSE
rmse <- sqrt(mean((unscaled_predictions - data$medv[-train_index])^2))
print(paste("RMSE:", rmse))
