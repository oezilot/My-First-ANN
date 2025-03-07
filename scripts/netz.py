'''
dieses skript macht eine liste von listen von dictionaroes und füllt diese mit initialen biases und weights
'''


import random

# funktion für fake weights und fake biases eines neurons
def init_bias():
    bias = random.uniform(-0.1, 0.1)
    return bias
#print(init_bias()))

def init_weights(anz_weights):
    weights = [random.uniform(-1, 1) for _ in range(anz_weights)]
    return weights
#print(init_weights(4))

def init_network(dimension):

    network = [] # eine liste von listen von dictionaries (jedes neuron wird von einem dicrionary repräsentiert)
    
    # ----- Input layer ----- (hat keine biases oder weights!)
    # für jedes layer eine liste machen und diese mit den dictionaries füllen
    for layer in dimension[:1]: # nur für die input layers
        network.append([
            {
                "weights":None, 
                "bias":None, 
                "activation":None
            } for _ in range(layer)
        ]) # liste mit leeren dictionaries hinzufügen für jedes neuron des inputlayers

    # ----- Hidden layers -----
    # Für jedes Hidden-Layer eine Liste mit Dictionaries hinzufügen, diese haben weights und biases
    for index, layer in enumerate(dimension[1:-1], start=1):  # i startet bei 1, weil wir ab der 2. Schicht zählen
        network.append([
        {
            "weights": init_weights(dimension[index - 1]),  # Anzahl Gewichte = Anz. Neuronen im vorherigen Layer
            "bias": init_bias(),
            "activation": None
        } for _ in range(layer)  # Anzahl Neuronen in der aktuellen Schicht
    ])
        
    # ----- Output layer -----
    # Für das otput Layer eine Liste mit Dictionaries hinzufügen, diese haben weights und biases
    for index, layer in enumerate(dimension[-1:], start=-1):  # i startet bei 1, weil wir ab der 2. Schicht zählen
        network.append([
        {
            "weights": init_weights(dimension[index - 1]),  # Anzahl Gewichte = Anz. Neuronen im vorherigen Layer
            "bias": init_bias(),
            "activation": None
        } for _ in range(layer)  # Anzahl Neuronen in der aktuellen Schicht
    ])
    return network