# Set working directory to where test and train data are
setwd("../Data/")

# Load data
train <- read.csv("train.csv")
test <- read.csv("test.csv")

# Load libraries
library(tidyverse)
library(tidytext)
library(qdap)
library(rpart)
library(randomForest)
library(corrplot)
library(tm)
library(caret)
library(gbm)
library(vtreat)
library(xgboost)
library(glmnet)

### Data Exploration ###

# Checking the dimensions of the data
dim(train) # 23313 rows
dim(test) # 5829 rows

# Checking which column is missing from test
"price" %in% names(train)
"price" %in% names(test) # price is not in test

# Checking if listing ids are unique
length(unique(train$id)) # all ids are unique

# Exploring the distribution of bedrooms
table(train$bedrooms)
mean(train$bedrooms) # mean no. of bedrooms is 1.16

# Exploring the distribution of property_type
table(train$property_type)
unique(train$property_type)

# Checking if there are errant values in price
mean(train$price < 0) # none of the price is < $0
sum(train$price == 0) # 20 listings have price of $0

# Plot the distribution of price
train %>%
  ggplot(aes(x = price)) +
  geom_histogram(binwidth = 5)

# Plot the distribution of accommodations
train %>%
  ggplot(aes(x = accommodates)) +
  geom_histogram(binwidth = 1)

# Plot the distribution of cleaning fees
train %>%
  ggplot(aes(x = cleaning_fee)) +
  geom_histogram(binwidth = 5)

# Plot the distribution of security deposit
train %>%
  ggplot(aes(x = security_deposit)) +
  geom_histogram(binwidth = 30)

# Plot the distribution of host listings count
train %>%
  ggplot(aes(x = host_listings_count)) +
  geom_histogram(binwidth = 1) +
  scale_x_continuous(limits = c(0, 100))

# Plot the distribution of minimum nights
train %>%
  ggplot(aes(x = minimum_nights)) +
  geom_histogram(binwidth = 1)

# Plot the distribution of availability 90 days
train %>%
  ggplot(aes(x = availability_90)) +
  geom_histogram(binwidth = 1)

# Plot the distribution of extra people
train %>%
  ggplot(aes(x = extra_people)) +
  geom_histogram(binwidth = 1)

# Plot the distribution of number of reviews
train %>%
  ggplot(aes(x = number_of_reviews)) +
  geom_histogram(binwidth = 1)

# Plot the distribution of bathrooms
train %>%
  ggplot(aes(x = bathrooms)) +
  geom_histogram(binwidth = 1)

# Plot the distribution of bedrooms
train %>%
  ggplot(aes(x = bedrooms)) +
  geom_histogram(binwidth = 1)

# Plot the distribution of minimum nights
train %>%
  ggplot(aes(x = polarity)) +
  geom_histogram(binwidth = 0.1)

# Plot distribution of property_type
barplot(prop.table(table(train_reduced$property_type)))
train %>%
  count(property_type) %>%
  ggplot(aes(x = property_type, y = n)) +
  geom_point()

# Plot correlation matrix on selected numerical features
corrplot(cor(train[,c("price", "accommodates", "cleaning_fee",
                      "host_listings_count", "availability_90",
                      "extra_people", "number_of_reviews",
                      "bathrooms", "bedrooms", "security_deposit",
                      "minimum_nights", "longitude", "latitude")]), 
         method = "square", 
         type = "lower",
         diag = F)

### Cleaning the data ###

# Remove price = $0
train <- train %>%
  filter(price > 0)

# Impute NAs in cleaning_fee to 0
train$cleaning_fee[is.na(train$cleaning_fee)] = 0
test$cleaning_fee[is.na(test$cleaning_fee)] = 0

# Impute NAs in security_deposit to 0
train$security_deposit[is.na(train$security_deposit)] = 0
test$security_deposit[is.na(test$security_deposit)] = 0

# property_type in test has "Train" -> "Vacation home
unique(test$property_type) %in% unique(train$property_type)
test$property_type[test$property_type == "Train"] = "Vacation home"

# Adding another variable, "cozy"
cozy_rows <- grep(pattern="cozy", x=tolower(train$description))
length(cozy_rows) # 4219 rows
length(cozy_rows) / length(train$description) # 18% of listings have "cozy"
train$cozy <- F
train$cozy[cozy_rows] <- T
table(train$cozy)

cozy_rows <- grep(pattern="cozy", x=tolower(test$description))
test$cozy <- F
test$cozy[cozy_rows] <- T
table(test$cozy)

# Adding another variable, "spacious"
spacious_rows <- grep(pattern="spacious", x=tolower(train$description))
length(spacious_rows) # 5263 rows
length(spacious_rows) / length(train$description) # 23% of listings have "spacious"
train$spacious <- F
train$spacious[spacious_rows] <- T
table(train$spacious)

spacious_rows <- grep(pattern="spacious", x=tolower(test$description))
test$spacious <- F
test$spacious[spacious_rows] <- T
table(test$spacious)

# Adding another variable, "private"
private_rows <- grep(pattern="private", x=tolower(train$description))
length(private_rows) # 7856 rows
length(private_rows) / length(train$description) # 34% of listings have "spacious"
train$private <- F
train$private[private_rows] <- T
table(train$private)

private_rows <- grep(pattern="private", x=tolower(test$description))
test$private <- F
test$private[private_rows] <- T
table(test$private)

# Text analytics and sentiment analysis
# The polarity function combs each description and assigns a sentiment score
sentiments_train <- polarity(
  removePunctuation(
    removeNumbers(
      tolower(train$description))), train$id)

train <- inner_join(train, sentiments_train[[1]], by = "id")
train <- train %>%
  select(-pos.words, -neg.words, -wc, -text.var)

sentiments_test <- polarity(
  removePunctuation(
    removeNumbers(
      tolower(test$description))), test$id)

test <- inner_join(test, sentiments_test[[1]], by = "id")
test <- test %>%
  select(-pos.words, -neg.words, -wc, -text.var)

# Select key features for random forest and boosting for both train and test
train_reduced <- train %>%
  select(price, room_type, accommodates, cleaning_fee, host_listings_count,
         availability_90,  extra_people, number_of_reviews, bathrooms,
         bedrooms, security_deposit, neighbourhood_group_cleansed,
         minimum_nights, longitude, latitude, cozy, cancellation_policy,
         property_type, polarity, spacious, private)

test_reduced <- test %>%
  select(room_type, accommodates, cleaning_fee, host_listings_count,
         availability_90,  extra_people, number_of_reviews, bathrooms,
         bedrooms, security_deposit, neighbourhood_group_cleansed,
         minimum_nights, longitude, latitude, cozy, cancellation_policy,
         property_type, polarity, spacious, private)

# Cleaning the train_reduced data
train_reduced$room_type <- as.factor(train_reduced$room_type)
train_reduced$neighbourhood_group_cleansed <- as.factor(train_reduced$neighbourhood_group_cleansed)
train_reduced$cancellation_policy <- as.factor(train_reduced$cancellation_policy)
train_reduced$property_type <- as.factor(train_reduced$property_type)
train_reduced$cozy <- as.factor(train_reduced$cozy)
train_reduced$polarity[is.na(train_reduced$polarity)] = 0
train_reduced$spacious <- as.factor(train_reduced$spacious)
train_reduced$private <- as.factor(train_reduced$private)

# Cleaning the test_reduced data
test_reduced$room_type <- as.factor(test_reduced$room_type)
test_reduced$neighbourhood_group_cleansed <- as.factor(test_reduced$neighbourhood_group_cleansed)
test_reduced$cancellation_policy <- as.factor(test_reduced$cancellation_policy)
test_reduced$property_type <- as.factor(test_reduced$property_type)
test_reduced$cozy <- as.factor(test_reduced$cozy)
test_reduced$spacious <- as.factor(test_reduced$spacious)
test_reduced$private <- as.factor(test_reduced$private)
