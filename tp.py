import numpy as np

x = np.array([1,2, 3,4,5,6,7,8,9,10])
y = np.array([3, 3.5, 4.2, 5, 5.8, 6.5, 7.3, 8.1, 9, 10.2])
# w = np.array([0.0]*10)
# b = np.array([0.0]*10)
w = 0
b = 0
epochs = 100
for i in range(epochs):
    y_pred = w*x +b
    loss = np.sum((y-y_pred)**2)
    mse = loss/10
    print("Looosss:", loss)
    w = w + 0.03*(2/10) * np.sum((y - y_pred)*x)
    b = b + 0.02*(2/10) * np.sum(y-y_pred)
    print("Updated Weights:", w, "and bias:", b)


    # plot loss vs w graph

y_pred = w*x +b
print(y_pred)
