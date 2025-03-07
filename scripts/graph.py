'''
dieses skriot macht einen graphen für ein existierende netzwerk
- labels an den edges
- farbe der edges und der nodes
'''

from graphviz import Digraph
from netz import init_network
import matplotlib.pyplot as plt 
from matplotlib import colors 
import numpy as np

def number_to_color(number, base_color=(1, 0, 0)):  # Standardfarbe: Rot (1, 0, 0)
    """Passt nur die Helligkeit einer gegebenen Farbe an, basierend auf einer Zahl."""
    if number == None:
        return "white"
    # Sigmoid-Funktion für Normalisierung (Werte zwischen 0 und 1)
    brightness = 1 / (1 + np.exp(-number))

    # RGB-Werte der Basisfarbe skalieren (nur Helligkeit verändern)
    adjusted_rgb = [brightness * c for c in base_color]

    # In Hex-Farbe umwandeln
    hex_color = colors.to_hex(adjusted_rgb)

    return hex_color
print(number_to_color(0.8))


dims = [9, 5, 5, 2]
network = init_network([9, 5, 5, 2])

# instantiating object 
dot = Digraph(comment='A NN') 


# aus dem netzwerk einen graphen erstellen
def generate_graph(netz):
    for index1, layer in enumerate(netz):
        prev_layer = netz[index1-1] # ein layer vorher um dann die actvations von von den vorherigen neuronen zu holen
        for index2, neuron in enumerate(layer):
            prev_neurons = [neuron for neuron in prev_layer]
            color = number_to_color(neuron["activation"])
            dot.node(f"{index1}.{index2}", f"{index1}.{index2}", style="filled", fillcolor=color) # für jedes neuron einen node machen
            if neuron["weights"]:
                print(len(neuron["weights"]))
                for index3, edge in enumerate(neuron["weights"]): # nicht für das inputlayer!
                    weight = edge
                    prev_activation = prev_neurons[index3]["activation"]
                    edge_label = f"{prev_activation}*{edge}"
                    print(edge_label)
                    color = number_to_color(weight)
                    dot.edge(f"{index1-1}.{index3}", f"{index1}.{index2}", color=color)
                    print(f"{index1-1}.{index3} --> {index1}.{index2}")
generate_graph(network)

dot.render('A NN', view=True)  # Speichert & öffnet das Bil