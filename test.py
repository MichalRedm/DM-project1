"""
Script responsible for testing how well do sample machine learning
algorithms perform on the dataset - before and after processing.
"""

import sys
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def main() -> None:
    lr = LinearRegression()
    arg1 = sys.argv[1]
    df = pd.read_csv(arg1)
    X, y = df.drop("price", axis=1), df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    # TODO

if __name__ == "__main__":
    main()
