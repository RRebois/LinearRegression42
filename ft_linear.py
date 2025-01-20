import pandas as pd
from pandas import DataFrame
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive use
import matplotlib.pyplot as plt
import pickle


def load(path: str) -> DataFrame | None:
    """
    takes a path as argument and returns the data set.
    :param path: str
    :return: DataFrame or None
    """
    try:
        file = pd.read_csv(path)
        ext = path.split(".")
        assert ext[len(ext) - 1].upper() == "CSV", "Wrong file format"

        file = pd.DataFrame(file)
        print("Loading dataset of dimensions", file.shape)
        return file
    except FileNotFoundError:
        raise FileNotFoundError(path)


def lr_algo(df: DataFrame) -> None:
    """
    Train a linear regression model.
    :param df: dataframe
    :return: None
    """
    learning_rate = 0.05
    n_iters = 500
    m = df.shape[0]

    X = df.loc[:, 'km'].values
    y = df.loc[:, 'price'].values

    # Standardized X data
    X_mean = X.mean()
    X_std = X.std()
    if X_std == 0:
        raise ValueError("Standard deviation is zero. "
                         "Cannot standardize data.")
    X = (X - X.mean()) / X.std()

    # theta1
    theta1 = 0 # if more than 1 feature create array. slope
    theta0 = 0 # intercept

    for iteration in range(n_iters):
        df['price_prediction'] = np.dot(X, theta1) + theta0

        dw = (1/m) * np.dot(X, (df['price_prediction'] - y))
        db = (1/m) * np.sum(df['price_prediction'] - y)

        loss = (1 / (2 * m)) * np.sum((df['price_prediction'] - y) ** 2)
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Loss: {loss}")

        theta1 -= learning_rate * dw
        theta0 -= learning_rate * db

    # Adjust theta1 and theta0 to match not scaled data
    theta1 = theta1 / X_std
    theta0 = theta0 - (theta1 * X_mean)

    print("theta1", theta1, "\ntheta0", theta0)
    values = {'theta0': theta0, 'theta1': theta1}
    try:
        with open("thetas.pkl", "wb") as f:
            pickle.dump(values, f)
    except Exception:
        raise Exception("Could not save theta file.")

    sns.scatterplot(x="km", y="price", data=df, label="data")
    sns.lineplot(x="km", y="price_prediction", data=df,
                 color='red', label='Linear Regression')
    plt.legend()
    plt.title("Linear Regression")
    plt.savefig("plot.png")


def check_float(x: any) -> float | None:
    """Check int value"""
    try:
        x = float(x)
        return x
    except Exception:
        return pd.NA


def check_data(df: DataFrame) -> DataFrame:
    """
    Checks that the data is correct.
    :param df: DataFrame
    :return: clean DataFrame
    """
    # Check column names
    if 'km' not in df.columns:
        df['km'] = 0
    if 'price' not in df.columns:
        df['price'] = 0

    df2 = pd.DataFrame()
    df2['km'] = df['km']
    df2['price'] = df['price']

    # Check each cell's value
    assert df2.shape[0] > 0 or df2.shape[1] >= 2, "Invalid dataframe"
    df2[df2.columns[0]] = df2[df2.columns[0]].map(check_float)
    df2[df2.columns[1]] = df2[df2.columns[1]].map(check_float)

    df2 = df2.dropna()
    return df2


def main():
    try:
        df = load("./data.csv")
        df = check_data(df)
        # For best lr fit, needs Mean Squared Error (MSE) -> use gradient
        # descent (GD)
        # GD for each value, calculate the error and search to minimize it
        lr_algo(df)
    except Exception as e:
        print("Error:", e)

if __name__ == '__main__':
    main()