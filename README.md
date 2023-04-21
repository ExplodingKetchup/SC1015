# App Rating Repository

## About

This is a Mini-Project for SC1015 (Introduction to Data Science and Artificial Intelligence) which focuses on [Google Play Store Apps](https://www.kaggle.com/datasets/lava18/google-play-store-apps). For detailed walkthrough, please view the source code in order from:

1. [Data Preprocessing and EDA](https://github.com/ExplodingKetchup/SC1015/blob/main/preprocessing_and_eda.ipynb)
2. [Random Forest](https://github.com/ExplodingKetchup/SC1015/blob/main/Random%20Forest.ipynb)
3. [eXtreme Gradient Boosting](https://github.com/ExplodingKetchup/SC1015/blob/main/xgb.ipynb)
4. [Neural Network](https://github.com/ExplodingKetchup/SC1015/blob/main/dnn.ipynb)

## Problem Definition

- Are we able to predict the rating of an app on the Google Play Store based on its features?
- Which model would be the best to predict it?

## Models Used

1. Random Forest
2. eXtreme Gradient Boosting
3. Neural Network

## Details

### Data Preprocessing and EDA

Dataset is retrieved from [Google Play Store Apps](https://www.kaggle.com/datasets/lava18/google-play-store-apps) and visualized to gain insights. In data preprocessing, duplicate data and unnecessary columns in the dataset, such as the name of the app, were dropped. The "Genres" column was split into 2, a "Primary Genre" column. In EDA, the data was visualized and analyzed to gain insights. The correlation between the variables were also inspected.

Here are some of the most important findings:
- Data consists of categorical variables and continuous numerical variables.
- Numerical variables are normally distributed, although quite skewed. They have little correlation to the response variable (`Rating`).
- Categorical variables may not correlate to app `Rating` either. Different categorical values generally do not affect the distribution of `Rating`.

**These findings highly suggest that our problem is a non-linear regression problem.**

### Random Forest

Rating of an app is predicted based on its features, the variables recorded in the dataset using random forest. Multiple deicison trees are constructed during the training phase and their outputs are combined to make a final prediction. 

We tested this model with various configurations, we will examine in details models with `max_depth=5` and `max_depth=2` (Best hyperparameters for Random Forest model).

Mean-squared error, mean absolute error, and R-squared score were used to determine the validity of the model. 

#### Evaluation

Unfortunately, a negative R-squared score was obtained, which deems it a weak model. Reducing the depth of the trees yielded a positive R-squared score only when the depth was shallow (at 2).

However, judging by Mean Absolute Error and Mean Squared Error metrics, which should be more important in evaluating a non-linear regression model, Random Forest with `max_depth=2` performs well.

Additionally, we showed that `max_depth=2` is better than higher values. Higher `max_depth` will cause overfitting issues.

### eXtreme Gradient Boosting

Rating of an app is predicted based on its features, the variables recorded in the dataset using XGBoost. A sequence of weak learners are constructed in a stage-wise manner. Each new learner is added to the model to correct errors made by previous learners. This way, new models are fitted iteratively to the residual errors of the existing model and combined to make a final prediction. 2 models were tested, the first of which is trained based on all the features, and another which is trained based on the most important features (determined by a technique called "Boruta-SHAP"). 

#### Evaluation

Feature selection led to no significant improvement in the mean squared error and mean absolute error, and the error was still high, hence XGBoost is not a recommended model.

### Neural Network

Rating of an app is predicted based on its features, the variables recorded in the dataset using Deep Neural Network. An input layer, multiple hidden layers, and an output layer make up the deep neural network. Neurons in one layer are connected to the neurons in the next layer through weighted connections, which are adjusted during the trianing process to minimize the error between the network's predictions and the actual target values. 

#### Evaluation

All configurations (number of neurons per layer, number of layers) performed similarly. Mean squared error and mean absolute error were deemed sufficiently low, significantly outperform XGBoost, and are comparable to Random Forest best implementation, hence DNN is among the better model out of the 3. It is surprising that various parameters changes do not result in an overfitted model. This could be considered an advantage over Random Forest for this specific problem.

## Conclusion

- It is possible to predict the rating of an app with acceptable amount of accuracy (~0.4/5 stars difference) using a Deep Neural Network model for regression (or Random Forest with suitable parameter tuning).

## Contributors

- @ExplodingKetchup - Data Preprocessing and EDA, Neural Network, XGBoost
- @SakaJoe - Data Preprocessing and EDA, Random Forest
- @PopChork - Data Preprocessing and EDA, Random Forest

## References

- <https://www.kaggle.com/datasets/lava18/google-play-store-apps>
- <https://www.tensorflow.org/tutorials/keras/regression>
- <https://towardsdatascience.com/random-forest-regression-5f605132d19d>
- <https://www.geeksforgeeks.org/random-forest-regression-in-python/>
- <https://www.kaggle.com/code/carlmcbrideellis/an-introduction-to-xgboost-regression>
- <https://towardsdatascience.com/a-brief-introduction-to-xgboost-3eaee2e3e5d6>
- <https://www.kaggle.com/code/carlmcbrideellis/feature-selection-using-the-boruta-shap-package/notebook>

