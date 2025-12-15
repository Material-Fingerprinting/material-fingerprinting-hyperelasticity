import matplotlib.pyplot as plt
import pickle

# stanford colors
cardinal = [157/255, 34/255, 53/255]
palo = [0/255, 106/255, 82/255]
bay = [111/255, 162/255, 135/255]
poppy = [233/255, 131/255, 0]
plum = [98/255, 0, 89/255]
illuminating = [254/255, 197/255, 29/255]

# model colors extracted from the tab20 colormap
Demiray = [44/255, 160/255, 44/255]
Gent = [255/255, 125/255, 11/255]
Holzapfel = [31/255, 119/255, 180/255]
MooneyRivlin = [214/255, 39/255, 40/255]
Ogden = [148/255, 103/255, 189/255]
BlatzKo = [140/255, 86/255, 75/255]
NeoHooke = [188/255, 189/255, 34/255]

colors = {'cardinal': cardinal,
        'palo': palo,
        'bay': bay,
        'poppy': poppy,
        'plum': plum,
        'illuminating': illuminating,
        'Demiray': Demiray,
        'Gent': Gent,
        'Holzapfel': Holzapfel,
        'MooneyRivlin': MooneyRivlin,
        'Mooney-Rivlin': MooneyRivlin,
        'Ogden': Ogden,
        'BlatzKo': BlatzKo,
        'Blatz-Ko': BlatzKo,
        'NeoHooke': NeoHooke,
        'Neo-Hooke': NeoHooke
        }

with open('plots/colors.pickle', 'wb') as handle:
    pickle.dump(colors, handle, protocol=pickle.HIGHEST_PROTOCOL)