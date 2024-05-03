# Load required libraries
library(neuralnet)
library(readr)
# Read the CSV file
data <- read_csv("pcaAnalysis.csv")
library(caret)
data <- data[, -1]
# Standardize the data (subtract mean and divide by standard deviation)
standardized_data <- scale(data)

# Define the percentage of data to use for testing
test_size <- 0.7  # Adjust as needed

# Split the data into training and testing sets
set.seed(123)  # for reproducibility
index <- createDataPartition(data$medv, p = test_size, list = FALSE)
training_data <- standardized_data[index, ]
testing_data <- standardized_data[-index, ]

# Define the formula for the neural network (assuming medv is the last column)
formula <- as.formula(paste("medv ~", paste(names(data)[-ncol(data)], collapse = " + ")))

# Train the neural network
neural_network <- neuralnet(formula, data = training_data, hidden = c(10, 5)) # You can adjust the number of hidden layers and neurons as needed

# Make predictions on the testing set
predictions <- predict(neural_network, testing_data)

# Rescale the predictions back to the original scale
medv_mean <- mean(data$medv)
medv_sd <- sd(data$medv)
rescaled_predictions <- predictions * medv_sd + medv_mean

# Calculate RMSE (Root Mean Squared Error)
rmseNN <- sqrt(mean((rescaled_predictions - data$medv[-index])^2))
print(paste("RMSE:", rmseNN))


library(randomForest)

trainIndex <- createDataPartition(data$medv, p = .7, 
                                  list = FALSE, 
                                  times = 1)
train_data <- data[trainIndex, ]
test_data <- data[-trainIndex, ]

# Train the random forest model
rf_model <- randomForest(medv ~ ., data = train_data)

# Make predictions on the testing set
predictions2 <- predict(rf_model, test_data)

# Calculate RMSE
rmseRF <- sqrt(mean((predictions2 - test_data$medv)^2))
rmseRF


library(dplyr)

train_index3 <- createDataPartition(data$medv, p = 0.70, list = FALSE)
train_data3 <- standardized_data[train_index3, ]
test_data3 <- standardized_data[-train_index3, ]
train_data3 <- as.data.frame(train_data3)
test_data3 <- as.data.frame(test_data3)

# Train multiple linear regression model
model <- lm(medv ~ ., data = train_data3)

# Make predictions on the testing set
predictions3 <- predict(model, newdata = test_data3)
rescaled_predictions3 <- predictions3 * medv_sd + medv_mean


# Calculate RMSE
rmseMLM <- sqrt(mean((rescaled_predictions3 - data$medv[-train_index3])^2))
print(paste("RMSE:", rmseMLM))

library(glmnet)

# Split the data into training and testing sets (70% training, 30% testing)
train_index4 <- createDataPartition(data$medv, p = 0.7, list = FALSE)
train_data4 <- standardized_data[train_index4, ]
test_data4 <- standardized_data[-train_index4, ]

# Fit Lasso regression model
lasso_model <- cv.glmnet(x = as.matrix(train_data4[, -which(colnames(train_data) == "medv")]), 
                         y = train_data4[, which(colnames(train_data) == "medv")], alpha = 1)
# Make predictions on the testing set
lasso_predictions <- predict(lasso_model, newx = as.matrix(test_data4[, 2:8]), s = "lambda.min")

# Revert the scaling on predictions
unscaled_predictions <- lasso_predictions * sd(data$medv) + mean(data$medv)

# Calculate RMSE
rmseLR <- sqrt(mean((unscaled_predictions - data$medv[-train_index4])^2))
print(paste("RMSE:", rmseLR))
