from NeuralNet import *
import matplotlib.pyplot as plt
a = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
X = np.reshape(a, (4, 2, 1))
print(a.size)
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))
layers = [
    Dense(2,3),
    Tanh(),
    Dense(3,1),
    Tanh()
]

'''net = NeuralNetwork(layers)
net.train(X,Y,epochs=10000,learning_rate=0.1)

y = net.errors
plt.plot(y)
plt.show()'''