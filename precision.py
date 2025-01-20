import pandas as pd
from pandas import DataFrame
import pickle
import numpy as np


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


def check_npy_file() -> dict:
    """
    Checks if the npy file exists to get thetas values
    :return: dict of thetas values or 0
    """
    try:
        with open("thetas.pkl", "rb") as f:
            thetas = pickle.load(f)
    except FileNotFoundError:
        return {'theta0': 0, 'theta1': 0}
    return thetas


def mean_squared_error(df: DataFrame, thetas: dict) -> float:
    m = df.shape[0]
    df['estimated'] = df.loc[:, 'km'] * thetas['theta1'] + thetas['theta0']

    mse = np.sum((df.loc[:, 'price'] - df.loc[:, 'estimated']) ** 2) / m
    return mse


def r_squared(df: DataFrame, mse: float) -> None:
    variance = np.var(df.loc[:, 'price'])
    r = 1 - mse / variance
    print("R-squared: ", "{:.2f}".format(r))


def main():
    try:
        df = load("./data.csv")
        thetas = check_npy_file()
        mse = mean_squared_error(df, thetas)
        r_squared(df, mse)
    except Exception as e:
        print("Error:", e)

if __name__ == '__main__':
    main()