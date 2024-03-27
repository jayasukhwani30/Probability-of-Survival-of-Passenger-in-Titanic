**Titanic Survival Prediction**

This project involves analyzing the Titanic dataset to predict the survival of passengers. It uses a Random Forest Classifier, a robust machine learning model, to make these predictions based on features like age, sex, and passenger class.

**Project Overview**

The Titanic dataset is a classic dataset used in data science to predict survival rates using machine learning. This project preprocesses the dataset by handling missing values, converting categorical variables, and dropping unnecessary columns. It then trains a Random Forest Classifier to predict survival, evaluates the model's accuracy, and examines the importance of different features.

**Getting Started**

**Prerequisites**

- Python 3.6 or higher
- Pandas
- Scikit-learn
- Matplotlib
- Numpy

**Installation**

1. Ensure that Python and pip are installed.
2. Install the required packages using pip:

```sh
pip install pandas scikit-learn matplotlib numpy
```


**Running the Analysis**
To run this analysis:

Ensure the Titanic dataset (Titanic-Dataset.csv) is downloaded to your local machine and update the file path in the script accordingly.
Run the Python script. The script will load the data, preprocess it, train the model, and output the accuracy of the model along with a classification report.
The script will also generate visualizations of the top 10 important features determined by the model and the predicted probabilities of survival.

**Files and Folders**
Titanic Survival Prediction.ipynb: Jupyter Notebook containing the analysis and model training.

**Model Evaluation**
The script outputs the model's accuracy and a detailed classification report including precision, recall, and F1-score for both survived and not survived predictions. Additionally, it visualizes the top 10 most important features for predicting survival.

**Feature Importance Visualization**
Feature importance scores are visualized to show which features are most influential in predicting survival. This helps in understanding the model's decision-making process.

**Predicted Probabilities**
The script calculates and displays the predicted probabilities for each passenger in the test set, providing insights into the model's confidence in its predictions.

**Conclusion**
This project demonstrates the application of a Random Forest Classifier to a real-world dataset. Through preprocessing, model training, and evaluation, we gain insights into the factors that influenced survival on the Titanic.
