from create_model import create_model
from get_data import get_data
import numpy as np
from tensorflow.keras.models import load_model


def main():
    X, y = get_data()
    # model = create_model(X.shape[1:], y.shape[-1])
    model = load_model('data/model/model')
    model.fit(X, y, epochs=200)
    print(model.predict(X))
    model.save('data/model/model')


if __name__ == '__main__':
    main()