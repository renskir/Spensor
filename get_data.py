def get_data():
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder

    X = np.load('data/npy/X.npy', allow_pickle=True) / 255
    y = np.load('data/npy/labels.npy', allow_pickle=True)

    y = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1, 1))

    return X, y
