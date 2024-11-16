# Datathon-2024

## 

## Catboost

We have chosen to utilize CatBoost (short for Categorical Boosting), one of the most advanced and precise machine learning models, to tackle this challenge. Developed by Yandex, CatBoost is a state-of-the-art gradient boosting library designed to handle both numerical and categorical data with exceptional efficiency. It excels in supervised machine learning tasks such as classification, regression, and ranking.

The decision to adopt CatBoost over other models stems from the nature of this problem, which involves datasets with a significant number of categorical features and demands high accuracy and computational efficiency. CatBoost’s ability to achieve state-of-the-art performance in various real-world applications is a testament to its advanced implementation of gradient boosting algorithms.

Moreover, CatBoost optimizes predictions by seamlessly handling categorical variables without extensive preprocessing, such as one-hot encoding, which can be both time-consuming and prone to error. Its robust scalability across both CPU and GPU environments ensures efficient training and inference, making it a versatile choice for academic research and production-scale projects alike.

By leveraging CatBoost’s unique features, we aim to enhance the precision and reliability of our predictions, meeting the challenges of this task with the most suitable and powerful machine learning solution available.

## Data filtering and cleaning

To predict building prices with a two-month outlook, based on a comprehensive dataset spanning nearly a year of property data from Illinois, USA, we chose to leverage a machine learning model integrated with catboost technology. To ensure the model’s optimal performance, it was essential to meticulously filter and organize the dataset for analysis.

The first step involved eliminating columns irrelevant to the prediction task, such as those containing street address formats, purchase or sale dates, and other non-essential details. Additionally, we applied specific criteria to enhance both the model’s efficiency and accuracy. One critical criterion was analyzing the number of unique values in each categorical column, as excessive variability can undermine the model's predictive power.

This rigorous preprocessing was vital for improving the chatbot’s reliability and the precision of its predictions. As part of this process, one-hot encoding was employed—a widely used data preprocessing technique in machine learning and data analysis. One-hot encoding transforms categorical variables into a machine-readable numerical format by converting each category into a binary column (0 or 1). This approach preserves the categorical nature of the data without introducing artificial ordering among the categories.

By preparing the data in this manner, we ensured that the categorical variables were structured effectively, enabling the predictive model to perform with greater accuracy and reliability.





