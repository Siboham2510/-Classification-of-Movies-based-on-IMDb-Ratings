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
imdb_data <- subset(imdb_data, !is.na(Meta_score))


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
# Install and load necessary packages
install.packages("WDI")
library(WDI)
#num_movies_before_1960 <- sum(imdb_data$Released_Year < 1960, na.rm = TRUE)
#num_movies_before_1960 #83 movies : removing those 83 films
#imdb_data <- imdb_data[imdb_data$Released_Year >= 1960, ]
#imdb_data1=imdb_data
# Create binary target variable (1 if IMDB_Rating >= 8, else 0)
imdb_data$High_Rating <- as.factor(ifelse(imdb_data$IMDB_Rating >= 8.2, 1, 0))
# Split the data into training and testing sets
set.seed(120)
trainIndex <- createDataPartition(imdb_data$High_Rating, p = .8, 
                                  list = FALSE, 
                                  times = 1)
imdb_train <- imdb_data[trainIndex,]
imdb_test <- imdb_data[-trainIndex,]
log_reg_model <- train(High_Rating ~  Released_Year+Runtime + Meta_score + No_of_Votes , 
                       data = imdb_train, 
                       method = "glm", 
                       family = "binomial",
                       trControl = trainControl(method = "cv", number = 10))
  log_reg_probabilities <- predict(log_reg_model, newdata = imdb_test, type = "prob")
log_reg_probabilities
# Predict on test set
threshold <- 0.8
log_reg_predictions=predict(log_reg_model,newdata=imdb_test)
# Apply threshold to get class predictions
log_reg_predictions <- ifelse(log_reg_probabilities[, "1"] > threshold, 1, 0)

log_reg_predictions <- factor(log_reg_predictions, levels = c(0, 1))
log_reg_predictions
# Evaluate Logistic Regression model
log_reg_cm <- confusionMatrix(log_reg_predictions, imdb_test$High_Rating)
print(log_reg_cm)
install.packages("pROC")
library(pROC)
# Assuming you have a trained logistic regression model 'log_reg_model' and test data 'imdb_test'
# Predict probabilities on the test set
predicted_probs <- predict(log_reg_model, newdata = imdb_test, type = "prob")[,2]
predicted_probs
roc_curve <- roc(imdb_test$High_Rating, log_reg_predictions)
plot(roc_curve, col = "blue", main = "ROC Curve", xlab = "False Positive Rate (1 - Specificity)", ylab = "True Positive Rate (Sensitivity)")

plot.roc(roc_curve, col = "blue", main = "ROC Curve",
         xlab = "Specificity (1-False Positive Rate (FPR))",
         ylab = "True Positive Rate (TPR) or Sensitivity",
         auc.polygon = TRUE, # Shade under ROC curve
         print.auc = TRUE)  # Print AUC value

# Add diagonal line for reference
abline(a = 0, b = 1, lty = 2, col = "red")

