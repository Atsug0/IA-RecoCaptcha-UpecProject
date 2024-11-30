import tkinter as tk
from tkinter import PhotoImage
import random
import os
from PIL import Image, ImageTk

# Chemins vers les répertoires d'images
CAR_DIR = "data/car/"
NOT_CAR_DIR = "data/not_car/"

# Configuration des probabilités
CAR_PROB = 0.4
NOT_CAR_PROB = 0.6

# Fonction de l'IA simulée (remplacez-la par votre fonction IA réelle)
def mock_ia(image_path):
    """
    Simule une IA qui prédit si une image est une voiture.
    Remplacez cette fonction par votre propre modèle IA.
    """
    return "car" in image_path

# Classe principale de l'application
class ImageGridApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Grid with IA")
        
        # Variables
        self.grid_size = 3
        self.images = []  # Stocke les chemins des images affichées
        self.image_widgets = []  # Stocke les widgets pour mise à jour facile
        
        # Création de la grille d'images
        self.grid_frame = tk.Frame(root)
        self.grid_frame.pack()
        
        # Modifier les dimensions des cellules de la grille et de l'image
        self.cell_width = 500 # Largeur de chaque cellule (x 5 par rapport à avant)
        self.cell_height = 500  # Hauteur de chaque cellule (x 5 par rapport à avant)
        
        # Créer une grille de 3x3
        for i in range(self.grid_size):
            row_widgets = []
            for j in range(self.grid_size):
                label = tk.Label(self.grid_frame, text="", width=150, height=150, borderwidth=2, relief="solid")
                label.grid(row=i, column=j, padx=10, pady=10)
                row_widgets.append(label)
            self.image_widgets.append(row_widgets)
        
        # Création des boutons
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(pady=20)
        
        self.reload_button = tk.Button(self.button_frame, text="Reload", command=self.reload_images)
        self.reload_button.pack(side=tk.LEFT, padx=20)
        
        self.start_ia_button = tk.Button(self.button_frame, text="Start IA", command=self.start_ia)
        self.start_ia_button.pack(side=tk.RIGHT, padx=20)
        
        # Chargement initial
        self.reload_images()

    def get_random_image(self):
        """Sélectionne une image aléatoire avec une probabilité donnée."""
        if random.random() < CAR_PROB:
            folder = CAR_DIR
        else:
            folder = NOT_CAR_DIR
        
        image_files = os.listdir(folder)
        if not image_files:
            raise ValueError(f"Le dossier {folder} est vide.")
        
        return os.path.join(folder, random.choice(image_files))

    def reload_images(self):
        """Recharge les images aléatoires dans la grille."""
        self.images = [self.get_random_image() for _ in range(self.grid_size ** 2)]
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                img_path = self.images[i * self.grid_size + j]
                img = Image.open(img_path).resize((self.cell_width, self.cell_height))  # Augmenter la taille des images
                img_tk = ImageTk.PhotoImage(img)
                
                label = self.image_widgets[i][j]
                label.config(image=img_tk, text="", width=150, height=150)
                
                # Empêche la suppression de l'objet PhotoImage
                label.photo = img_tk
                
                # Réinitialiser la bordure à la couleur par défaut (ici, noire ou aucune couleur)
                label.config(borderwidth=2, relief="solid", highlightbackground="black", highlightcolor="black", highlightthickness=0)


    def start_ia(self):
        """Utilise l'IA pour prédire les images et change la couleur des bordures."""
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                img_path = self.images[i * self.grid_size + j]
                result = mock_ia(img_path)
                # Change la couleur de bordure selon le résultat
                label = self.image_widgets[i][j]
                if result:
                    label.config(text="", width=150, height=150, borderwidth=2,highlightthickness=4, highlightbackground="green")
                else:
                    label.config(text="", width=150, height=150, borderwidth=2,highlightthickness=4, highlightbackground="red")

# Création de la fenêtre principale
if __name__ == "__main__":
    # Vérification des dossiers
    if not os.path.exists(CAR_DIR) or not os.path.exists(NOT_CAR_DIR):
        print("Les dossiers 'data/car/' et 'data/not_car/' doivent exister et contenir des images.")
    else:
        root = tk.Tk()
        root.resizable(False, False)
        # Modifier la taille de la fenêtre principale pour l'adapter aux images plus grandes
        window_width = 540  # largeur de la fenêtre (x 5 par rapport à la version précédente)
        window_height = 620  # hauteur de la fenêtre (x 5 par rapport à la version précédente)
        root.geometry(f"{window_width}x{window_height}")

        app = ImageGridApp(root)
        root.mainloop()
