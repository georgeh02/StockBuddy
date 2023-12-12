# StockBuddy

StockBuddy is a machine learning model that aims to make investing easier by predicting stock prices based on insider trading data.
Users can discover daily stock picks with StockBuddy featuring 12-month price projections. Trained on over 15 years of insider trading reports,
Stock buddy uses the power of historical trends to make its predictions.

Please note StockBuddy is extremely inaccurate and should not be used for investment advice. StockBuddy is not a financial advisor.

Created by George Harrison

## Setup Instructions

StockBuddy runs in terminal.

Step 1

- Download the "StockBuddy" folder from the github

Step 2

- From terminal, create a virtual environment within the StockBuddy folder
  - <code>python -m venv path\to\StockBuddy</code>
- Activate the virtual environment
  - (mac / linux) <code>source activate path\to\StockBuddy</code>
  - (windows) <code>activate path\to\StockBuddy</code>
- Install relevant python libraries
  - <code>pip install pandas iexfinance kaggle zipfile numpy tqdm joblib tabulate</code>

Step 3

- While still in virtual environment, run predict script
  - <code>python3 predict.py</code>

StockBuddy will print the top 3 and bottom 3 predictions for stock prices based on today's dataset, as well as save them to a text file.
