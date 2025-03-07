# source = https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin.html

# hier teste ich eine funktion mit welcher ich die funktionswerte berechnen kann die es braucht umd den funktionswert zu minimieren!

import numpy as np
from scipy.optimize import fmin

# Sigmoid-Funktion
def activation_sigmoid(x):
    return 1 / (1 + np.exp(-x))

# bsp mit 3 neuronen im prev-layer
prev_activations = [0.4, 0.2, 0.8]
w_and_b = [0.1, 0.3, 0.4, 0.9] # das sind einfach irgendwelche werte die optimiert werden müssen, sie werden als die variablen der funktion selber gesehen 

# diese funktion berechent den error eines neurons mit der berücksichtigung all seiner gewichte und dem bias
def error(w_and_b, prev_activations, a_target):
    weighted_prev_activations = []
    for i in range(len(w_and_b[:-1])):
        weighted_prev_activations.append(w_and_b[:-1][i] * prev_activations[i])
    a = activation_sigmoid(sum(weighted_prev_activations)+w_and_b[-1])
    error = a_target - a
    return error
    
# das ist der desired output! um diesen wert zu erhalten muss man die gewichte anpassen
target_a = 1

# diese funktion ermittelt die optimalen gewichte und erstezt diese mit den alten gewichten eines neurons
def backpropagation():
    # die optimalen werte ermittteln
    optimal_w_and_b = fmin(error, w_and_b, args=(prev_activations, 1))
    print(optimal_w_and_b)

    # die alten werte mit den optimalen ersetzen
    
backpropagation()
