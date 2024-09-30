
imdb_data=imdb_top_1000
#DATA PREPROCESSING
imdb_data$Runtime <- as.numeric(gsub("[^0-9]", "", imdb_data$Runtime))
summary(imdb_data)
unique(imdb_data$Released_Year)
# Identify rows with "PG" in Released_Year
pg_rows <- grep("PG", imdb_data$Released_Year, ignore.case = TRUE)
# Display the rows with all columns
print(imdb_data[pg_rows, ], row.names = FALSE)
View(imdb_data[pg_rows, ])
# Replace "PG" with "1995" in Released_Year
imdb_data$Released_Year <- gsub("PG", "1995", imdb_data$Released_Year, ignore.case = TRUE)
# Convert Released_Year to numeric
imdb_data$Released_Year <- as.numeric(imdb_data$Released_Year)
imdb_data=imdb_data[,-1]
summary(imdb_data)

h=hist(imdb_data$Gross,prob=TRUE,xlab="Gross",ylab="Freq.",main="Gross Box Office")
h
hist(imdb_data$Runtime,prob=TRUE,xlab="Runtime",ylab="Freq.",main="Runtime of films")
hist(imdb_data$IMDB_Rating,prob=TRUE)
hist(imdb_data$Meta_score,prob=TRUE)
hist(imdb_data$No_of_Votes,prob=TRUE)
plot(imdb_data$Meta_score,imdb_data$No_of_Votes)
plot(imdb_data$Meta_score,imdb_data$Gross)
plot(imdb_data$IMDB_Rating*10,imdb_data$Meta_score)
#data cleaning
colSums(is.na(imdb_data))
imdb_data <- subset(imdb_data, !is.na(Meta_score)) ##needs to be imputed

library(stats)
library(dplyr)
library(ggplot2)
library(corrplot)
library(reshape2)
library(caret) 
#for a unified interface for numerous machine learning algorithms and includes tools for data splitting, pre-processing, feature selection, model tuning using resampling, and variable importance estimation.
#EDA

table(imdb_data$Certificate)
unique(imdb_data$Certificate)
# Plot the distribution of movie certificates
ggplot(imdb_data, aes(x = Certificate)) +
  geom_bar(fill = "skyblue") +
  theme_minimal() +
  labs(title = "Distribution of Movie Certificates",
       x = "Certificate",
       y = "Count")
# Subset numeric variables
numeric_vars <- imdb_data[, sapply(imdb_data, is.numeric) & names(imdb_data) != "Gross"]

# Compute correlation matrix
correlation_matrix <- cor(numeric_vars)

# Plot correlation heatmap using corrplot
c=corrplot(correlation_matrix, method = "color", type = "upper", 
           addCoef.col = "black", tl.col = "black", tl.srt = 45)
c
plot(imdb_data$Released_Year)
sort(unique(imdb_data$Released_Year))
# Calculate median runtime by year
med_runtime_by_year <- imdb_data %>%
  group_by(Released_Year) %>%
  summarise(Avg_Runtime = median(Runtime, na.rm = TRUE))
# Create interactive plot with plot_ly
library(plotly)
plot_ly(data = med_runtime_by_year, x = ~Released_Year, y = ~Avg_Runtime, type = 'scatter', mode = 'lines+markers',
        marker = list(color = 'blue', size = 10),
        line = list(color = 'blue', width = 2)) %>%
  layout(title = "Median Runtime of Movies by Released Year",
         xaxis = list(title = "Released Year"),
         yaxis = list(title = "Median Runtime (minutes)"),
         hovermode = "closest")
# Scatter plot to visualize relationship between IMDb ratings and No_of_Votes
ggplot(imdb_data, aes(x = No_of_Votes, y = IMDB_Rating)) +
  geom_point(color = "blue", alpha = 0.6) +  # Points with transparency
  labs(title = "Relationship between IMDb Ratings and Number of Votes",
       x = "Number of Votes",
       y = "IMDb Rating") +
  theme_minimal()
min(imdb_data$No_of_Votes)

threshold_percent <- quantile(imdb_data$IMDB_Rating, probs = 0.8)
threshold_percent
# Count movies in top 20%
movies_in_top_percent <- sum(imdb_data$IMDB_Rating >= threshold_percent)
movies_in_top_percent
imdb_data$High_Rating <- as.factor(ifelse(imdb_data$IMDB_Rating >= threshold_percent, 1, 0))
# Split the data into training and testing sets
set.seed(120)
trainIndex <- createDataPartition(imdb_data$High_Rating, p = .85, 
                                  list = FALSE, 
                                  times = 1)
imdb_train <- imdb_data[trainIndex,]
imdb_test <- imdb_data[-trainIndex,]
log_reg_model <- train(High_Rating ~  Released_Year+Runtime + Meta_score + No_of_Votes , 
                       data = imdb_train, 
                       method = "glm", 
                       family = "binomial")
summary(log_reg_model$finalModel)

log_reg_model1 <- glm(High_Rating ~ Released_Year + Runtime + Meta_score + No_of_Votes + Gross, 
                      data = imdb_train, 
                      family = binomial)
summary(log_reg_model1)
# Calculate McFadden's R^2
log_lik_full <- logLik(log_reg_model)
log_lik_null <- logLik(glm(High_Rating ~ 1, data = imdb_data, family = binomial))
mcfadden_r2 <- 1 - (log_lik_full / log_lik_null)

print(paste("McFadden's R^2:", mcfadden_r2))
# Step 1: Linearity of Logits
# Calculate the fitted probabilities for the positive class (High_Rating = 1)
fitted_probs <- predict(log_reg_model, type = "prob")[,2]
fitted_probs
logits <- log(fitted_probs / (1 - fitted_probs))
logits
# Create scatterplots for the logit values against the continuous predictors
#par(mfrow = c(2, 2)) # Adjust the layout for multiple plots

plot(imdb_train$Released_Year, logits, xlab = "Released Year", ylab = "Logit", main = "Logit vs Released Year")
abline(lm(logits ~ imdb_train$Released_Year), col = "red")
plot(imdb_train$Runtime, logits, xlab = "Runtime", ylab = "Logit", main = "Logit vs Runtime")
abline(lm(logits ~ imdb_train$Runtime), col = "red")

plot(imdb_train$Meta_score, logits, xlab = "Meta score", ylab = "Logit", main = "Logit vs Meta score")
abline(lm(logits ~ imdb_train$Meta_score), col = "red")

plot(imdb_train$No_of_Votes, logits, xlab = "Number of Votes", ylab = "Logit", main = "Logit vs Number of Votes")
abline(lm(logits ~ imdb_train$No_of_Votes), col = "red")
#step 2: checking multicollinearity
library(car)
final_model <- log_reg_model$finalModel
vif_values <- vif(final_model)
print(vif_values)
# Subset numeric variables
numeric_vars <- imdb_data[, sapply(imdb_data, is.numeric) & names(imdb_data) != "Gross" & names(imdb_data) != "IMDB_Rating"]

# Compute correlation matrix
correlation_matrix <- cor(numeric_vars)

# Plot correlation heatmap using corrplot
c=corrplot(correlation_matrix, method = "color", type = "upper", 
           addCoef.col = "black", tl.col = "black", tl.srt = 45)
c
## Step 3: Outliers
#standardized_residuals <- rstandard(final_model)
#plot(standardized_residuals, main = "Standardized Residuals")
#abline(h = c(-2, 2), col = "red", lty = 2)
#cooks_distance <- cooks.distance(final_model)
#plot(cooks_distance, main = "Cook's Distance")
#abline(h = 4 / nrow(imdb_train), col = "red", lty = 2)
#leverage_values <- hatvalues(final_model)
#plot(leverage_values, main = "Leverage Values")
#abline(h = 2 * (ncol(imdb_train) + 1) / nrow(imdb_train), col = "red", lty = 2)

library(pROC)
# Predict probabilities on test data
probabilities <- predict(log_reg_model, newdata = imdb_test, type = "prob")
probabilities[,2]
# Extract probabilities of the positive class
predictions <- probabilities[, "1"]
predictions #predicted success probabilities by the trained LR model on the test set 
# Set the default threshold
threshold <- 0.4
# Convert predicted probabilities to class labels based on the threshold
class_predictions <- ifelse(predictions >= threshold, 1, 0)

# Print class predictions to verify
print(class_predictions)
print(imdb_test$High_Rating)
library(caret)

# Compute confusion matrix
conf_matrix1 <- confusionMatrix(factor(class_predictions, levels = c( 1,0)), factor(imdb_test$High_Rating, levels = c(1,0)))
# Print the confusion matrix
print(conf_matrix1)
# Initialize empty vectors to store results
Thresholds <- seq(0.2, 0.65, by = 0.01)
sensitivity_values <- numeric(length(Thresholds))
specificity_values <- numeric(length(Thresholds))

# Calculate sensitivity and specificity for each threshold
for (i in seq_along(Thresholds)) {
  Threshold <- Thresholds[i]
  
  # Convert predicted probabilities to class labels based on the threshold
  class_predictions <- ifelse(predictions >= Threshold, 1, 0)
  
  # Compute confusion matrix
  conf_matrix <- confusionMatrix(factor(class_predictions, levels = c(1, 0)), factor(imdb_test$High_Rating, levels = c(1, 0)))
  
  # Extract sensitivity and specificity from confusion matrix
  sensitivity <- conf_matrix$byClass["Sensitivity"]  # sensitivity for class '1' (high rating)
  specificity <- conf_matrix$byClass["Specificity"]  # specificity for class '0' (not high rating)
  
  # Store values
  sensitivity_values[i] <- sensitivity
  specificity_values[i] <- specificity
}

# Create a data frame to store the results
Threshold_results <- data.frame(Threshold = Thresholds, Sensitivity = sensitivity_values, Specificity = specificity_values)

# Print or view the results
print(Threshold_results)


# Create ROC curve object
roc_obj <- roc(imdb_test$High_Rating, predictions)
imdb_test$High_Rating

# Plot the ROC curve
plot(roc_obj, main = "ROC Curve", col = "blue", cex.main = 1.5, cex.lab = 1.5, cex.axis = 1.5) 
auc_value <- auc(roc_obj)
print(paste("AUC:", auc_value))
polygon(c(roc_obj$specificities, rev(roc_obj$specificities)),
        c(roc_obj$sensitivities, rep(0, length(roc_obj$sensitivities))),
        col = rgb(0, 0, 1, alpha = 0.2))
# Add a vertical line at specificity = 0.90
abline(v = 1.0, col = "black", lty = 2)
# Add a horizontal line at y = 1.0
abline(h = 1.0, col = "black", lty = 2)
# Add the ROC curve legend slightly lower than the top right
legend("topright", inset = c(0, 0.03), legend = "ROC Curve",
       col = "blue", lty = 1, lwd = 2, border = NA, bty = "n", cex = 1.5, text.col = "blue")

# Add the AUC value legend at the bottom right
legend("bottomright", legend = paste("AUC =", round(auc_value, 4)), 
       col = rgb(0, 0, 1, alpha = 0.2), pch = 15, pt.cex = 1.5, border = NA, bty = "n", cex = 1.5)

sensitivities=roc_obj$sensitivities
specificities=roc_obj$specificities
thresholds=roc_obj$thresholds
thresholds
# Find thresholds where specificity is at least 0.90
valid_thresholds <- thresholds[specificities >= 0.90]
valid_thresholds
valid_sensitivities <- sensitivities[specificities >= 0.90]
valid_sensitivities
# Find the threshold that maximizes sensitivity within the valid thresholds
optimal_index1 <- which.max(valid_sensitivities)
optimal_index1
optimal_threshold1 <- valid_thresholds[optimal_index1]
valid_sensitivities[optimal_index1]
# Print the optimal threshold
print(paste("Optimal Threshold:", optimal_threshold1))

#Print the sensitivity for optimal threshold
print(max(valid_sensitivities))

# Set the threshold
threshold1 <- optimal_threshold1
# Convert predicted probabilities to class labels based on the threshold
class_predictions1 <- ifelse(predictions >= threshold1, 1, 0)

# Print class predictions to verify
print(class_predictions1)
library(caret)

# Compute confusion matrix
conf_matrix1 <- confusionMatrix(factor(class_predictions1, levels = c(1,0)), factor(imdb_test$High_Rating, levels = c( 1,0)))

# Print the confusion matrix
print(conf_matrix1)

#Finding the threshold with maximum F1 score.
# Calculate precision at each threshold
precisions <- sensitivities / (sensitivities + (1 - specificities))
# Calculate F1 scores at each threshold
f1_scores <- 2 * (precisions * sensitivities) / (precisions + sensitivities)
f1_scores
# Find the index of the maximum F1 score
max_f1_index <- which.max(f1_scores)
# Get the threshold with the highest F1 score
optimal_threshold2 <- thresholds[max_f1_index]
max_f1_score <- f1_scores[max_f1_index]
# Print the results
print(paste("Optimal Threshold:", optimal_threshold2))
print(paste("Maximum F1 Score:", max_f1_score))
# Set the threshold
threshold2 <- optimal_threshold2
# Convert predicted probabilities to class labels based on the threshold
class_predictions2 <- ifelse(predictions >= threshold2, 1, 0)

# Print class predictions to verify
print(class_predictions2)
library(caret)

# Compute confusion matrix
conf_matrix2 <- confusionMatrix(factor(class_predictions2, levels = c(1,0)), factor(imdb_test$High_Rating, levels = c( 1,0)))

# Print the confusion matrix
print(conf_matrix2)



#k fold validation using the entire data
#log_reg_modelkfold <- train(High_Rating ~  Released_Year+Runtime + Meta_score + No_of_Votes , 
 #                      data = imdb_data, 
  #                     method = "glm", 
   #                    family = "binomial",
    #                   trControl = trainControl(method = "cv", number = 10, 
     #                  verboseIter = TRUE, savePredictions = "final"))
# Print the model to see the results, including accuracy
#print(log_reg_modelkfold)

# Access detailed resampling results
#resampling_resultskfold  <- log_reg_modelkfold$resample
#print(resampling_resultskfold )

# Access predictions for each fold
#predictionskfold  <- log_reg_modelkfold$pred
#print(predictionskfold )
# Calculate performance metrics for each fold
#library(pROC)

# Function to calculate performance metrics
#calculate_metricskfold <- function(predictionskfold) {
 # roc_objkfold <- roc(predictionskfold$obs, predictionskfold$pred)
  #auc_value <- auc(roc_objkfold)
  #confusionkfold <- confusionMatrix(as.factor(ifelse(predictionskfold$pred >= 0.5, 1, 0)), predictionskfold$obs)
  #list(Accuracy = confusionkfold$overall['Accuracy'], AUC = auc_value)
#}

# Apply the function to each fold
#fold_metricskfold <- predictionskfold %>%
  #group_by(Resample) %>%
  #summarize(
   # Accuracy = mean(ifelse(pred >= 0.5, 1, 0) == obs),  # Corrected accuracy calculation
    #AUC = mean(auc(roc(predictionskfold$obs, predictionskfold$pred)$auc))  # Corrected AUC calculation
  #)
#print(fold_metricskfold)

# Load necessary libraries

library(class)
library(caret)
library(dplyr)
library(ggplot2)

predictors <- imdb_data %>% select(Released_Year, Runtime, Meta_score, No_of_Votes)
target <- imdb_data$High_Rating

# Scale the predictors
scaled_predictors <- scale(predictors)

# Use the same train-test split as previously done for logistic regression
set.seed(120)
trainIndex <- createDataPartition(target, p = 0.85, list = FALSE, times = 1)
train_data <- scaled_predictors[trainIndex, ]
test_data <- scaled_predictors[-trainIndex, ]
train_labels <- target[trainIndex]
test_labels <- target[-trainIndex]

# Function to perform KNN with Euclidean distance and store metrics
evaluate_knn <- function(train_data, test_data, train_labels, test_labels) {
  k_values <- seq(1, 40, by = 1)
  results <- data.frame(k = k_values, accuracy = numeric(length(k_values)), 
                        specificity = numeric(length(k_values)), sensitivity = numeric(length(k_values)))
  
  for (i in 1:length(k_values)) {
    k <- k_values[i]
    knn_model <- class::knn(train_data, test_data, train_labels, k = k)
    cm <- caret::confusionMatrix(knn_model, test_labels, positive = "1")
    results[i, "accuracy"] <- cm$overall["Accuracy"]
    results[i, "specificity"] <- cm$byClass["Specificity"]
    results[i, "sensitivity"] <- cm$byClass["Sensitivity"]
    results[i, "F1"] <- cm$byClass["F1"]
  }
  
  return(results)
}
# Evaluate KNN and store metrics
results <- evaluate_knn(train_data, test_data, train_labels, test_labels)
# Print the results dataframe
print(results)
# Plot accuracy changes with k
accuracy_plot1 <- ggplot(results, aes(x = k, y = accuracy)) +
  geom_point(color = "blue", size = 3) +
  geom_line(color = "blue") +
  ggtitle("Accuracy vs. k Value") +
  xlab("k Value ") +
  ylab("Accuracy ") +
  theme_minimal()

print(accuracy_plot1)

# Find the value of k for which accuracy is maximum
optimal_k1 <- results[which.max(results$accuracy), "k"]
cat(paste("Optimal k for maximum accuracy:", optimal_k1, "\n"))

# Retrieve and print confusion matrix for the optimal k
final_knn_model1 <- knn(train_data, test_data, train_labels, k = optimal_k1)
final_cm1 <- confusionMatrix(final_knn_model1, test_labels, positive = "1")
print("Confusion Matrix for Optimal k (accuracy):")
print(final_cm1)

# Filter results where specificity >= 0.9
result1 <- results %>% filter(specificity >= 0.9) %>% select(k, specificity, sensitivity)
result1
# Plot sensitivity changes with k for all k for which specificity>=0.9
accuracy_plot2 <- ggplot(result1, aes(x = k, y = sensitivity)) +
  geom_point(color = "red", size = 3) +
  geom_line(color = "red") +
  ggtitle("Sensitivity (of k values for which Specificity >=0.9 vs. k Value") +
  xlab("k Value") +
  ylab("Sensitivity") +
  scale_x_continuous(breaks = result1$k) +  # Set x-axis ticks to match k values in result1
  theme_minimal()
print(accuracy_plot2)
# Find the value of k for which accuracy is maximum
optimal_k2 <- result1[which.max(result1$sensitivity), "k"]
cat(paste("Optimal k for maximum sensitivity (when specificity>=0.9
          :", optimal_k2, "\n"))

# Retrieve and print confusion matrix for the optimal k
final_knn_model2 <- knn(train_data, test_data, train_labels, k = optimal_k2)
final_cm2 <- confusionMatrix(final_knn_model2, test_labels, positive = "1")
print("Confusion Matrix for Optimal k (sensitivity when specificity>=0.9):")
print(final_cm2)

# Plot F1 score changes with k
accuracy_plot3 <- ggplot(results, aes(x = k, y = F1)) +
  geom_point(color = "magenta", size = 3) +
  geom_line(color = "magenta") +
  ggtitle("F1 score vs. k Value") +
  xlab("k Value") +
  ylab("F1 score") +
  theme_minimal()

print(accuracy_plot3)
# Find the value of k for which F1 score is maximum
optimal_k3 <- results[which.max(results$F1), "k"]
cat(paste("Optimal k for maximum F1 score:", optimal_k3, "\n"))

# Retrieve and print confusion matrix for the optimal k
final_knn_model3 <- knn(train_data, test_data, train_labels, k = optimal_k3)
final_cm3 <- confusionMatrix(final_knn_model3, test_labels, positive = "1")

print("Confusion Matrix for Optimal k (F1 score):")
print(final_cm3)

