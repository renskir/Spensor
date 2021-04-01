from create_model import create_model
from get_data import get_data
import numpy as np


def main():
    X, y = get_data()
    model = create_model(X.shape[1:], y.shape[-1])
    model.fit(X, y, epochs=30)
    print(model.predict(X))
    model.save('data/model/model')


if __name__ == '__main__':
    main()