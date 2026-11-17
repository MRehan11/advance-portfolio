
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
# import math
# #                           this is the acTIVATION FUNCTIONS EHICH PERFORM THE LINEAR AND NONO LINEAR REALTIONSHIP BETWEEN THE INPUT AND OUTPUT OF THE NEURAL NETWORK
# #its return answer into decimal format if the value is positive it retruns the value accurate to the input and if value is negative it returns 0
# def relu(x):
#     return max(0, x)
# #its return in float point value according to the formula 
# def sigmoid(x):
#     return 1 / (1 + math.exp(-x))
# #its also return the ans in the float number 
# def tanh_activation(x):
#     return math.tanh(x)

# x = -2

# print("ReLU:", relu(x))
# print("Sigmoid:", sigmoid(x))
# print("Tanh:", tanh_activation(x))



model = MLPRegressor(
    hidden_layer_sizes=(3,),
    max_iter=2000,
    random_state=42
)

# Four examples, one feature per example
X = [[2], [4], [6], [8]]
y = [20, 40, 60, 80]

model.fit(X, y)

prediction = model.predict([[5]])
print("Predicted marks:", round(prediction[0], 2))

print("Input neurons:", 1)
print("Hidden neurons:", 3)
print("Output neurons:", 1)