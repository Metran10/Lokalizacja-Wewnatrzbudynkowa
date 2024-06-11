import matplotlib.pyplot as plt
import numpy as np


def make_default_schema(ax):

    # Ustawienie rozmiaru wykresu
    #fig, ax = plt.subplots(figsize=(10, 10))

    # Dodanie kotwic
    anchors = [(0, 0, "FE"), (10, 0, "E4"), (10, 10, "FC"), (0, 10, "F7")]
    anchors_cord = [(0, 0), (10, 0), (10, 10), (0, 10)]
    not_measured = [(1,1), (2,1), (3,1), (4,1), (7,1), (8,1),
                    (1,3), (2,3), (3,3), (4,3), (7,3), (8,3),
                    (1,6), (2,6), (3,6), (4,6)]
    for i, (x, y, name) in enumerate(anchors):
        ax.scatter(x, y, color='red', marker='v', s=200)
        ax.text(x +0.2, y+0.35, f'Kotwica {name}: ({x}, {y})', fontsize=12, ha='left' if x == 0 else 'right', va='top' if y == 0 else 'bottom')

    # Dodanie punktów
    x_points = np.arange(0, 11, 1)  # punkty co 1 metr
    y_points = np.arange(0, 11, 1)
    for x in x_points:
        for y in y_points:
            if (x,y) not in anchors_cord:
                if (x,y) not in not_measured:
                    ax.scatter(x, y, color='blue', s=20)

    # for (x, y) in not_measured:
    #     ax.scatter(x, y, color='black', s=50)

    # Dodanie prostokątów (np. budynków)
    buildings = [
        (1,1.2,4,0.4), #szafa
        (7,1.2,2,0.3), #szafa
        (1,3,4,0.75),
        (7,3.3,2,0.75),
        (1,6,4,0.75),
        (7,6.6,2,0.75),

    ]
    for (x, y, width, height) in buildings:
        rect = plt.Rectangle((x-0.5, y-0.5), width, height, linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(rect)

    # Dodanie punktów (np. ludzi)
    # people_x = np.random.uniform(0, 10, 50)
    # people_y = np.random.uniform(0, 10, 50)
    # ax.scatter(people_x, people_y, color='black', s=50)

    # Ustawienie zakresów osi
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)

    # Ustawienie etykiet osi
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')

    # Pokaż wykres
    plt.savefig("DIAGEAM_EKP")



