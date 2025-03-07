'''
in diesem skript generiere ich die trainingsdaten
'''

import random


'''
ein datenpunkt sieht folgendermassen aus:

picture = [
[0, 1, 0],
[1, 1, 0],
[0, 0, 0],
]

die gesamten daten werden so gefasst:

data = [(image, label), (image, label)]
'''

def generate_train_data(anz_datenpunkte, resolution):

    data = []
    
    for _ in range(anz_datenpunkte):
        picture = [[0] * resolution for _ in range(resolution)]  # Leeres Bild mit Nullen
        
        # Entscheide zufällig, ob es eine horizontale oder vertikale Linie wird
        label = random.choice([0, 1])  # 0 = vertikal, 1 = horizontal
        
        if label == 0:  # Vertikale Linie
            col = random.randint(0, resolution - 1)  # Zufällige Spalte wählen
            for row in range(resolution): # die gewählte spalte mit 1 ersetzen
                picture[row][col] = 1
        
        else:  # Horizontale Linie
            row = random.randint(0, resolution - 1)  # Zufällige Zeile wählen
            for col in range(resolution):
                picture[row][col] = 1

        # das picture zu einer 1D-liste umwandeln
        picture = [pixel[i] for pixel in picture for i in range(resolution)]
        
        data.append((picture, label))
    
    return data