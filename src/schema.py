import matplotlib.pyplot as plt
import numpy as np


def make_default_schema(ax):

    # Ustawienie rozmiaru wykresu
    #fig, ax = plt.subplots(figsize=(10, 10))

    # Dodanie kotwic
    anchors = [(0, 0, "1"), (10, 0, "2"), (10, 10, "3"), (0, 10, "4")]
    anchors_cord = [(0, 0), (10, 0), (10, 10), (0, 10)]
    not_measured = [(1, 1), (2, 1), (3, 1), (4, 1), (7, 1), (8, 1),
                    (1, 3), (2, 3), (3, 3), (4, 3), (7, 3), (8, 3),
                    (1, 6), (2, 6), (3, 6), (4, 6)]
    for i, (x, y, name) in enumerate(anchors):
        ax.scatter(x, y, color='red', marker='v', s=200)
        # ax.text(x + 0.2, y + 0.45, f'Anchor {name}: ({x}, {y})', fontsize=16, ha='left' if x == 0 else 'right',
        #         va='top' if y == 0 else 'bottom')
        ax.text(x + 0.2, y - 0.7 if y == 0 else y + 0.7, f'Anchor {name}: ({x}, {y})', fontsize=16,
                ha='left' if x == 0 else 'right',
                va='top' if y == 10 else 'bottom')

    # Dodanie punktów
    x_points = np.arange(0, 11, 1)  # punkty co 1 metr
    y_points = np.arange(0, 11, 1)
    for x in x_points:
        for y in y_points:
            if (x, y) not in anchors_cord:
                if (x, y) not in not_measured:
                    ax.scatter(x, y, color='blue', s=20)

    # for (x, y) in not_measured:
    #     ax.scatter(x, y, color='black', s=50)

    # Dodanie prostokątów (np. budynków)
    buildings = [
        (1, 1.2, 4, 0.4),  # szafa
        (7, 1.2, 2, 0.3),  # szafa
        (1, 3, 4, 0.75),
        (7, 3.3, 2, 0.75),
        (1, 6, 4, 0.75),
        (7, 6.6, 2, 0.75),

    ]
    for (x, y, width, height) in buildings:
        rect = plt.Rectangle((x - 0.5, y - 0.5), width, height, linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(rect)

    # Ustawienie zakresów osi
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)

    ax.tick_params(axis='both', which='major', labelsize=16)

    # Ustawienie etykiet osi
    ax.set_xlabel('X [m]', fontsize=16)
    ax.set_ylabel('Y [m]', fontsize=16)

    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

    # Pokaż wykres
    plt.savefig("DIAGEAM_EKP")

def create_default_schema():
    # Ustawienie rozmiaru wykresu
    fig, ax = plt.subplots(figsize=(10, 10))

    # Dodanie kotwic
    anchors = [(0, 0, "1"), (10, 0, "2"), (10, 10, "3"), (0, 10, "4")]
    anchors_cord = [(0, 0), (10, 0), (10, 10), (0, 10)]
    not_measured = [(1, 1), (2, 1), (3, 1), (4, 1), (7, 1), (8, 1),
                    (1, 3), (2, 3), (3, 3), (4, 3), (7, 3), (8, 3),
                    (1, 6), (2, 6), (3, 6), (4, 6)]
    for i, (x, y, name) in enumerate(anchors):
        ax.scatter(x, y, color='red', marker='v', s=200)
        # ax.text(x + 0.2, y + 0.45, f'Anchor {name}: ({x}, {y})', fontsize=16, ha='left' if x == 0 else 'right',
        #         va='top' if y == 0 else 'bottom')
        ax.text(x + 0.2, y - 0.7 if y==0 else y+0.7, f'Anchor {name}: ({x}, {y})', fontsize=16, ha='left' if x == 0 else 'right',
                va='top' if y == 10 else 'bottom')

    # Dodanie punktów
    x_points = np.arange(0, 11, 1)  # punkty co 1 metr
    y_points = np.arange(0, 11, 1)
    for x in x_points:
        for y in y_points:
            if (x, y) not in anchors_cord:
                if (x, y) not in not_measured:
                    ax.scatter(x, y, color='blue', s=20)

    # for (x, y) in not_measured:
    #     ax.scatter(x, y, color='black', s=50)

    # Dodanie prostokątów (np. budynków)
    buildings = [
        (1, 1.2, 4, 0.4),  # szafa
        (7, 1.2, 2, 0.3),  # szafa
        (1, 3, 4, 0.75),
        (7, 3.3, 2, 0.75),
        (1, 6, 4, 0.75),
        (7, 6.6, 2, 0.75),

    ]
    for (x, y, width, height) in buildings:
        rect = plt.Rectangle((x - 0.5, y - 0.5), width, height, linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(rect)

    # Ustawienie zakresów osi
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)

    ax.tick_params(axis='both', which='major', labelsize=16)

    # Ustawienie etykiet osi
    ax.set_xlabel('X [m]', fontsize=16)
    ax.set_ylabel('Y [m]', fontsize=16)

    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

    # Pokaż wykres
    plt.savefig("DIAGEAM_EKP")
    plt.show()

    pass


