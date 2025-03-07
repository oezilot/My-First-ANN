'''
in diesem skript kommen alle teile zusammen. das netz wird gebaut und trainiert!
'''

import random
import math
import numpy as np
from scipy.optimize import fmin
from training_data import generate_train_data
from netz import init_network


#========== TRAININGSDATEN ==========

training_data = generate_train_data(10, 3) # 10 pixelbilder generieren welche 3X3 pixel gross sind

training_data_sample = training_data[0]
training_data_sample_target = [1, 0] if training_data[0][1] == 1 else [0, 1]
training_data_sample_image = training_data[0][0]


#========== NETZWERK INITIALISIEREN ==========

network_dimension = [9, 5, 5, 2]

network1 = init_network(network_dimension) # netzerkt nach dem n-1 ten durchlauf

# funktion um das netzwerk schöner darzustellen
def print_array_structure(array):
    for i, layer in enumerate(array):
        print(f"🔹 Ebene {i}:")
        for j, element in enumerate(layer):
            print(f"  ▪ Element {j}: {{'weights': {element["weights"]}, 'bias': {element['bias']}, 'activation': {element['activation']}, 'target_activation': {element['activation']}}}")
        print("\n")

print(f"\n------------------- Netzwerk (v.1) vor der Forward-Propagation mit initialisierten b, w -------------------\n")
print_array_structure(network1)


#========== FORWARD PROPAGATION ==========

# aktivierungsfunktionen
def activation_relu(x):
    return max(0, x)

def activation_sigmoid(x):
    return 1 / (1 + math.exp(-x))

# diese funktion wendet die weigths und biases auf einkommenden input an und berechnet den sematnischen wert
def forward_propagation(pixel_bild, netzwerk): # im pronzip füllt diese fuktion das feld "activation" des dictinaries!
    print(f"Input Pixelbild: {pixel_bild}")

    new_network = netzwerk

    # den inputwert des inputlayers als activation setzen
    for neuron in range(network_dimension[0]):
        new_network[0][neuron]["activation"] = pixel_bild[neuron] 
        print(f"{neuron}. Inputneuron: {pixel_bild[neuron]}")

    # die informationen n-ten layers werden ans n+1-ten layer weitergegeben (für das letzte layer muss eine sigmoid-funktin verwendet werden damit man die klassifiezierung so durchführen kann dass  näher bei 1 oder näher bei 0 aufteilen kann)
    for n in range(1, len(new_network)):
        for neuron in range(network_dimension[n]):
            prev_activations = [new_network[n-1][i]["activation"] for i in range(network_dimension[n-1])] # liste mit activations des vorherigen layers
            print(f"activations vom vorherigen layer: {prev_activations}")
            akt_weights = new_network[n][neuron]["weights"] # liste mit den weights eines neurons des 2ten layers
            print(f"weights eines neurons des aktuellen layers: {akt_weights}")
            akt_bias = new_network[n][neuron]["bias"]
            print(f"Bias des neurons: {akt_bias}")
            if n == len(new_network):
                pixel_updated = activation_sigmoid(sum([prev_activations[x] * akt_weights[x] for x in range(len(prev_activations))]) + akt_bias)
            else:
                pixel_updated = activation_relu(sum([prev_activations[x] * akt_weights[x] for x in range(len(prev_activations))]) + akt_bias)
            print(f"Funktionsoutput des neurons: {pixel_updated}")
            new_network[n][neuron]["activation"] = pixel_updated # summe aller activations aus dem letzen layer  

            output = [node["activation"] for node in new_network[-1]]
    return new_network, output # mit pixelbild ist hier der semantsche vektor des letzen layers gemeint, also der activations des letzen layers

network_output = forward_propagation(training_data_sample[0], network1)
network_updated = forward_propagation(training_data_sample[0], network1)[0]


#========== PREDICTION ==========

# diese funktion evaluiert die prediction
def predict(netzerkt_output):

    output_network = netzerkt_output[0]
    output_output = netzerkt_output[1]

    print_array_structure(output_network)
    print(f"Der netzwerk-Output:\n{output_output}")

    if output_output[0] > output_output[1]:
        prediction = "horizontal" # 1
    else:
        prediction = "vertikal" # 0

    return prediction

print(f"Die Prediction für das Pixelbild ist '{predict(network_output)}'!")


#========== ERROR ==========
# für jedes output neuron muss man die abweichung zum desired berechnen und die backprpagation durchführen. der desired output hat auch so viele activstions wie das outputlayer!

# error funktion, diese funktion berechent die abweichung des desired und des tatsächlichen werts
def error(datenpunkt, netzwerk_output):
    if datenpunkt[1] == 0:
        desired_output_layer = [0, 1]
    else:
        desired_output_layer = [1, 0]
    actual_output_layer = netzwerk_output[1]

    total_error = 0 

    for i in range(len(desired_output_layer)):
        error = desired_output_layer[i]-actual_output_layer[i]
        total_error = total_error + error
    
    mean_error = total_error / len(desired_output_layer)

    return mean_error

#print(f"Die Abweichung zum erwartetenden wert beträgt: {error(training_data_sample, network_output)}")


#========== BACKPROPAGATION ========== 
'''
error-funktion optimieren und optimierte werte mit alten ersetzen
'''

# Sigmoid-Funktion
def activation_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def activation(weights, bias, prev_activations):
    weighted_prev_activations = []
    for i in range(len(weights)):
        weighted_prev_activations.append(weights[i] * prev_activations[i])
    a = activation_sigmoid(sum(weighted_prev_activations)+bias)
    return a

# diese funktion berechent den error eines neurons mit der berücksichtigung all seiner gewichte
def error(w_and_b, prev_activations, a_target):
    weighted_prev_activations = []
    for i in range(len(w_and_b[:-1])):
        weighted_prev_activations.append(w_and_b[:-1][i] * prev_activations[i])
    a = activation_sigmoid(sum(weighted_prev_activations)+w_and_b[-1])
    error = a_target - a
    return error
    
# das ist eine liste mit den desired putput des last layer, ihre länge entspricht der anzahl neuronen im output layer. target_a = 1

# diese funktion ermittelt die optimalen gewichte und erstezt diese mit den alten gewichten eines neurons, das netz welches man als input verwendent muss weights, biases und activations haben sonst geht das ganze nicht!!!
# das target muss ich der form winer liste kommen
def backpropagation(netzwerk, target):

    print(f"\n------------------- Netzwerk (v.2) nach der Forward-Propagation mit a, b, w -------------------\n")
    print(netzwerk)
    print("\n")
    
    network3 = netzwerk

    # rückwärts durch das netz iterieren

    # OUTPUT-LAYER
    for index1, layer in enumerate(reversed(network3[-1:])): # nur das letzte layer!
        for index2, neuron in enumerate(layer):
            w_and_b = neuron["weights"]
            w_and_b.append(neuron["bias"]) # eine liste mit allen aktuellen weights und dem bias des aktuellen neurons
            prev_activations = [neuron["activation"] for neuron in network3[-(index1+2)]]
            target_a = target[index2]

            # für jedes neuron muss die error funktion angewendet werden und dan ihr minimum (optimalen weightd um das minimum zu erreichen) berechnet werden!
            optimal_w_and_b = fmin(error, w_and_b, args=(prev_activations, target_a))
        
            # die neuen optimierten weights und biases mit edn alten weights und biases umtauschen, ausserdem muss eine neue liste kreiert werden mit den prev-activation die es braucth um diese neue activation zu erreichen
            neuron["weights"]=optimal_w_and_b[:-1] # der letzte wert is der bias
            new_weights = neuron["weights"]

            neuron["bias"]=optimal_w_and_b[-1] # der letzte wert is der bias
            new_bias = neuron["bias"]

            neuron["activation"]= activation(w_and_b[:-1], w_and_b[-1], prev_activations) # activations neu berechenen
            new_activation = neuron["activation"]

            # Funktion zur Berechnung des Fehlers zwischen Aktivierung und Zielwert
            def error2(activations, weights, bias, a_target):
                weighted_prev_activations = []
                for i in range(len(w_and_b[:-1])):
                    weighted_prev_activations.append(activations[i] * weights[i])
                a = activation_sigmoid(sum(weighted_prev_activations)+bias)
                error = a_target - a
                return error
           
            # eine neue liste von prev_activations generieren die es braucht um den neu kreierten actiavtion-wert (new_activation) zu erreichen
            optimal_prev_activations = fmin(error2, prev_activations, args=(new_weights, new_bias, new_activation))

            # Wende objective() auf alle gegebenen Werte an und finde den besten
            optimal_prev_activation = sum(optimal_prev_activations)/len(optimal_prev_activations) # der schnitt aller optimierten activations

            for i in range(len(network3[-(index1+2)])):
                network3[-(index1+2)][i]["target_activation"] = optimal_prev_activation
                print("Target Activation added")

            print(f"\nDas 2-letzte layer mit der target_activation: {network3[-(index1+2)]}\n") # her pronte ich das 2t-etzte layer aus, also dort wo es nun target_activations hat

            print("\n---- Das Netzwerk nach der ersten schicht der backpropagation ----\n")
            print_array_structure(network3)


    # HIDDEN-LAYERS UND INPUT-LAYER
    for index1, layer in enumerate(reversed(network3[:-1])): # beim letzten hiddenlayer starten
        for index2, neuron in enumerate(layer):
            w_and_b = neuron["weights"]
            w_and_b.append(neuron["bias"]) # eine liste mit allen aktuellen weights und dem bias des aktuellen neurons
            prev_activations = [neuron["activation"] for neuron in network3[-(index1+2)]]
            target_a = neuron["target_activation"]

            # für jedes neuron muss die error funktion angewendet werden und dan ihr minimum (optimalen weightd um das minimum zu erreichen) berechnet werden!
            optimal_w_and_b = fmin(error, w_and_b, args=(prev_activations, target_a))
        
            # die neuen optimierten weights und biases mit edn alten weights und biases umtauschen, ausserdem muss eine neue liste kreiert werden mit den prev-activation die es braucth um diese neue activation zu erreichen
            neuron["weights"]=optimal_w_and_b[:-1] # der letzte wert is der bias
            new_weights = neuron["weights"]

            neuron["bias"]=optimal_w_and_b[-1] # der letzte wert is der bias
            new_bias = neuron["bias"]

            neuron["activation"]= activation(w_and_b[:-1], w_and_b[-1], prev_activations) # activations neu berechenen
            new_activation = neuron["activation"]

            # Funktion zur Berechnung des Fehlers zwischen Aktivierung und Zielwert
            def error2(activations, weights, bias, a_target):
                weighted_prev_activations = []
                for i in range(len(w_and_b[:-1])):
                    weighted_prev_activations.append(activations[i] * weights[i])
                a = activation_sigmoid(sum(weighted_prev_activations)+bias)
                error = a_target - a
                return error
           
            # eine neue liste von prev_activations generieren die es braucht um den neu kreierten actiavtion-wert (new_activation) zu erreichen
            optimal_prev_activations = fmin(error2, prev_activations, args=(new_weights, new_bias, new_activation))

            # Wende objective() auf alle gegebenen Werte an und finde den besten
            optimal_prev_activation = sum(optimal_prev_activations)/len(optimal_prev_activations) # der schnitt aller optimierten activations

            network3[-(index1+2)][index2]["target_activation"] = optimal_prev_activation
            print_array_structure(network3)

    print(f"\n------------------- Netzwerk (v.3) nach der Backward-Propagation mit veränderten w , b, a -------------------\n")
    print(network3)
    print("\n")

    return network3

backpropagation(network_updated, training_data_sample_target)

'''
fragen: 
- beo den target-activations die generiert werden...woher weiss das neuron welches dieser targets es preferieren soll? einfach alle durchprobieren

todos: 
- kommentare
- immer das gleiche random nehmen
- refactor
- chatgpt-rückmeldung
- target in einer liste von werten speichern statt mit nur 0 oder 1
- das targetvalue des outputlayers noch ins dictionary speichern!
'''