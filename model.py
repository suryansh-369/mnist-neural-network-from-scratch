from tensorflow.keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

import pandas as pd

data = pd.DataFrame(X_train.reshape(60000, 784))

print(data.head())

X_train_full = X_train
Y_train_full = Y_train

X_val = X_train_full[:10000].T
Y_val = Y_train_full[:10000].T

Y_train = Y_train.T

X_train = X_train_full[10000:].T
Y_train = Y_train_full[10000:].T

X_train = X_train.reshape(X_train.shape[0], 784)
print(X_train.shape)
X_val = X_val.reshape(X_val.shape[0], 784)
print(X_val.shape)
# Y_train = Y_train.reshape(Y_train.shape[0], 784)
print(X_train_full.shape)


