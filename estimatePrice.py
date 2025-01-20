import pickle


def prompt_miles() -> float:
    """
    Propmt for a car mileage, checks for input validity
    and returns the car mileage
    :return: float
    """
    miles = ""
    while not miles.isdigit():
        miles = input("Enter car miles: ")
        if not miles.isdigit():
            print("Invalid car miles: input must contain only digits and be positive.")
    return float(miles)


def check_npy_file() -> tuple:
    """
    Checks if the npy file exists to get thetas values
    :return: tuple of thetas values or 0
    """
    try:
        with open("thetas.pkl", "rb") as f:
            thetas = pickle.load(f)
    except FileNotFoundError:
        return 0, 0
    return thetas['theta0'], thetas['theta1']


def main():
    try:
        theta0, theta1 = check_npy_file()

        miles = prompt_miles()
        price = theta0 + theta1 * miles
        if price <= 0:
            price = 0
        print("According to the miles provided, your car price estimation "
            "is: ", "$", "{:.2f}".format(price), sep="")
    except Exception as e:
        print("Error:", e)

if __name__ == '__main__':
    main()