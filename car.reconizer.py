import wx
import random
import os
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import wx.lib.agw.pygauge as pg
from ia_car import predict_car

# Load the model
model = load_model("cnn_multipleReconnection_model.h5")
categories = ['car', 'not_car']

# Image directory paths
CAR_DIR = "data/valid/car/"
NOT_CAR_DIR = "data/valid/not_car/"

# Probabilities configuration
CAR_PROB = 0.4  # Probability for car category
NOT_CAR_PROB = 0.6  # Probability for not_car category


# Function for IA prediction
def get_ia_response(image_path):
    """ 
        Appelle le modèle IA pour prédire si l'image contient une voiture. 
        Args: image_path (str): Chemin de l'image à analyser. 
        Returns: 
        bool: True si une voiture est détectée, False sinon. 
    """
    return predict_car(image_path)


class ImageGridPanel(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)

        # Grid size
        self.grid_size = 3
        self.images = []
        self.image_controls = []

        # Main sizer
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Image grid with sizer
        grid_sizer = wx.GridSizer(self.grid_size, self.grid_size, 10, 10)
        self.image_grid = grid_sizer
        main_sizer.Add(grid_sizer, 1, wx.EXPAND | wx.ALL, 10)

        # Button sizer
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Reload button
        reload_button = wx.Button(self, label="Reload")
        reload_button.Bind(wx.EVT_BUTTON, self.reload_images)
        button_sizer.Add(reload_button, 0, wx.ALL, 5)

        # Start IA button
        start_ia_button = wx.Button(self, label="Start IA")
        start_ia_button.Bind(wx.EVT_BUTTON, self.start_ia)
        button_sizer.Add(start_ia_button, 0, wx.ALL, 5)

        # Progress gauge
        self.progress_gauge = pg.PyGauge(self, range=self.grid_size**2)
        self.progress_gauge.SetValue(0)
        button_sizer.Add(self.progress_gauge, 1, wx.ALL, 5)

        main_sizer.Add(button_sizer, 0, wx.ALIGN_CENTER)

        # Set main sizer
        self.SetSizer(main_sizer)

        # Initial loading
        self.reload_images()

    def get_random_image(self):
        """
        Selects a random image based on probabilities.
        """
        if random.random() < CAR_PROB:
            folder = CAR_DIR
        else:
            folder = NOT_CAR_DIR

        image_files = os.listdir(folder)
        if not image_files:
            raise ValueError(f"The folder '{folder}' is empty.")

        return os.path.join(folder, random.choice(image_files))

    def reload_images(self, event=None):
        """
        Reloads random images into the grid.
        """
        self.images = [self.get_random_image() for _ in range(self.grid_size ** 2)]

        # Clear the image grid
        for control in self.image_controls:
            control.Destroy()
        self.image_controls.clear()

        # Add new images to the grid
        for i, img_path in enumerate(self.images):
            image = wx.Image(img_path, wx.BITMAP_TYPE_ANY)
            image = image.Scale(150, 150, wx.IMAGE_QUALITY_HIGH)
            bitmap = wx.StaticBitmap(self, -1, wx.Bitmap(image))
            self.image_grid.Add(bitmap, 1, wx.EXPAND | wx.ALL, 5)
            self.image_controls.append(bitmap)

        self.Layout()  # Update the layout

    def start_ia(self, event=None):
        """
        Uses the IA to predict images and change their appearance.
        """
        self.progress_gauge.SetValue(0)
        for i, img_path in enumerate(self.images):
            result = get_ia_response(img_path)

            # Update image appearance based on prediction
            image_control = self.image_controls[i]
            if result:
                # Add a green border or other visual indicator
                image_control.SetBackgroundColour(wx.Colour(0, 255, 0, 50))  # Semi-transparent green
            else:
                # Add a red border or other visual indicator
                image_control.SetBackgroundColour(wx.Colour(255, 0, 0, 50))  # Semi-transparent red

            # Add prediction text as a tooltip
            image_control.SetToolTip(f"Prediction: {'car' if result else 'not_car'}")

            # Update progress gauge
            self.progress_gauge.SetValue(i + 1)

            self.Layout()  # Update the layout


class MyApp(wx.App):
    def OnInit(self):
        frame = wx.Frame(None, title="Image Grid with IA", size=(540, 620))
        panel = ImageGridPanel(frame)
        frame.Show()
        return True


if __name__ == '__main__':
    app = MyApp()
    app.MainLoop()
