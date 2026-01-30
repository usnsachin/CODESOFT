Movie Rating Prediction

This project is completed as Task 2 for the CODSOFT Data Science Internship. The objective of this project is to predict the IMDb rating of movies using machine learning techniques based on features such as year of release, duration, genre, number of votes, director, and lead actors.
The dataset contains movie details including Year, Duration, Genre, Votes, Director, Actor 1, Actor 2, Actor 3, and the target variable Rating. Data preprocessing was performed to clean and convert string values into numerical form. Missing values were handled and categorical features were encoded to prepare the data for model training.
A Random Forest Regressor was used to train the model after splitting the dataset into training and testing sets. The model was evaluated using R² Score and Root Mean Squared Error (RMSE). Graphs were generated to visualize actual versus predicted ratings and to show feature importance.
The model achieved an R² score of approximately 0.34 with an RMSE of about 1.1, which is acceptable for this dataset due to the high variability and complexity of movie rating patterns.
Technologies used: Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn.

Files included:
movies.csv – Dataset
movie_rating.py – Python script
movie_rating_model.pkl – Trained model
scaler.pkl – Feature scaler
label_encoders.pkl – Encoders for categorical data
Generated plots for analysis

Author: Sachin Tiwari
CODSOFT Data Science Internship — Task 2
