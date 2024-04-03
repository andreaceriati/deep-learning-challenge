# deep-learning-challenge

# Report: Performance Analysis of Deep Learning Model for Alphabet Soup

## Overview of the Analysis:

The purpose of this analysis is to create a binary classifier using deep learning techniques to predict whether organizations funded by Alphabet Soup will be successful in their ventures. By leveraging machine learning algorithms, we aim to assist Alphabet Soup in selecting applicants with the highest chance of success, thereby maximizing the impact of their funding.

## Results:

Data Preprocessing:
- Target Variable(s):
    - The target variable for our model is IS_SUCCESSFUL, which indicates whether the funding was used effectively (1) or not (0).

- Feature Variable(s):
    - Features used for the model include APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, and ASK_AMT.

- Variables Removed:
    - We removed the EIN and NAME columns as they are identification columns and are not relevant for predictive modeling.

- Data Exploration:
    - We determined the number of unique values for each column.
    - For columns with more than 10 unique values, we inspected the distribution to identify rare categories.
    - Rare categorical variables were binned together in a new category labeled "Other" to prevent overfitting.

- Encoding:
    - We encoded categorical variables using one-hot encoding via pd.get_dummies() to convert them into numerical format suitable for machine learning algorithms.

- Data Splitting:
    - The preprocessed data was split into features (X) and target (y) arrays.
    - We used train_test_split to split the data into training and testing datasets.

- Scaling:
    - The training and testing feature datasets were scaled using StandardScaler to normalize the data and improve model performance.

Compiling, Training, and Evaluating the Model:

- Neural Network Model:
    - We created a total of 4 neural networks models.
    - For our best performing neural network model, we selected 43 input features corresponding to the number of features in the dataset.
    - The model consists of two hidden layers with 80 and 30 neurons, respectively, and ReLU activation functions for the first and second hidden layers, and Sigmoid as the output layer. We chose this architecture to strike a balance between model complexity and computational efficiency.

- Model Performance:
    - The target model performance was set to achieve an accuracy higher than 75%.
    - After training the models, we evaluated their performance on the testing dataset to calculate the loss and accuracy.

- Steps for Improvement:
    - Despite our efforts, we were unable to achieve the target model performance.
    - To improve model performance, we attempted various optimization techniques, including:
        - Increasing or decreasing the number of values for each bin.
        - Adding more neurons
        - Adding more hidden layers.
        - Using different activation functions for hidden layers.
        - Modifying the number of epochs during training.

- Model Sructure, Loss, and Accuracy for the Best Performing Model

<p align='center'> <img src='Deep Learning/Images/alphabet_soup_charity.PNG' width='500' height='300'></p>

<p align='center'> <img src='Deep Learning/Images/alphabet_soup_charity_loss_accuracy.PNG' width='500' height='50'></p>

## Summary:
Overall, the deep learning model showed promising results but fell short of achieving the target accuracy threshold. Despite optimization attempts, further refinements may be necessary to enhance model performance.

## Recommendation for a Different Model:

Considering the challenges faced in achieving the desired accuracy threshold with the deep learning model, it might be beneficial to explore alternative machine learning algorithms. Here are some recommendations for different models:

- Random Forest Classifier:

    Random Forest is an ensemble learning method that operates by constructing multiple decision trees during training and outputting the mode of the classes (classification) of the individual trees.
    It's robust against overfitting, works well with both categorical and numerical data, and requires minimal preprocessing.
    Random Forest can handle large datasets with high dimensionality, making it suitable for this classification task.

- Logistic Regression:

    Despite its name, logistic regression is a linear model for binary classification.
    It's simple, interpretable, and works well for problems with linear decision boundaries.
    Logistic regression can serve as a baseline model for comparison with more complex algorithms.
