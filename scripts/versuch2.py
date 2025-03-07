import numpy as np

# training data (input) = koordinates (x, y)
# output = 0 oder 1 (fals die summe von x+y < 0 -> 0, sonst 1)

# create the training data. eine liste mit allen x-werten und eine lste mit allen y-werten
def genrate_training_data(anzahl_trainingsdaten):
    np.random.seed(1)  # Ensures we always get the same data
    trainings_daten_x = np.random.rand(2, anzahl_trainingsdaten) * 2 -1
    print(f"Trainingsdaten_x:\n{trainings_daten_x}")

    # expected predictions hinzufügen
    trainings_daten_y = [1 if (trainings_daten_x[0][i] + trainings_daten_x[1][i] > 0) else 0 for i in range(len(trainings_daten_x[0]))] 
    print(f"Trainingsdaten_y:\n{trainings_daten_y}")
    
    return trainings_daten_x, trainings_daten_y # beide seperat zurückgeben aber in einem tuple
print(f"Trainingsdaten (X, Y):\n{genrate_training_data(10)}")

# netz initialisieren mit gegebenen dimensionen
network_dimensions = [2, 4, 4, 2]

def init_network(net_dims): # mit random weights und biases
    network = []

    for i in range(1, len())
'''
fragen:
- in welcher form müssen trainingsdate sein?
'''