import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
import datetime as dt

# Load in the csv file
games = pd.read_csv('C:\\Users\\seth.hertzog\\OneDrive - Mark One Electric Co., Inc\\Desktop\\Random Documents\\ML\\Data Files\\best-selling video games of all time.csv')

# Display the first few rows
#print(games.head())

# Display the data types of each column
#print(games.dtypes)

# Check for missing values
#print(games.isnull().sum())

# Drop irrelevant features (Publisher(s), Developer(s))
games = games.drop(['Rank', 'Publisher(s)', 'Developer(s)', 'Series', 'Platform(s)'], axis=1)

# Drop rows with missing values
games = games.dropna()

# Create dictionary to give numerical value to the top 5 games we want to predict sales for
title_dictionary = {'Minecraft': 1, 'Grand Theft Auto V': 2, 'Tetris (EA)': 3, 'Wii Sports': 4, 'PUBG: Battlegrounds': 5}

# Transform 'Title' feature into newly created Python dictionary
games['Title'] = games['Title'].map(title_dictionary)

# Drop rows with missing values
games = games.dropna()

# Convert 'Initial release date' to the number of years since the release date
games['Initial release date'] = pd.to_datetime(games['Initial release date'])
current_year = dt.datetime.now().year
games['Years since release'] = current_year - games['Initial release date'].dt.year
games = games.drop('Initial release date', axis=1)

# Split the data into a training set and a test set
X = games.drop('Sales', axis=1)
y = games['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict future sales (in this example, we predict sales for the same games but after 10 years from 2023)
future_years = 10
future_X = X.copy()
future_X['Years since release'] += future_years
future_sales = model.predict(future_X)

# Prints total number of sales for each game
print(games[['Title', 'Sales']])

# Print the future sales predictions (2023-2033)
for title, sales in zip(future_X['Title'], future_sales):
    print(f"Title {title}: {sales} sales predicted in {current_year + future_years}")