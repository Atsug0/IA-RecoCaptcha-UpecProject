import wx
import random
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import wx.lib.agw.pygauge as pg

# Load the model
model = load_model("cnn_multipleReconnection_model.h5")
categories = ['car', 'cats', 'dogs', 'flowers', 'none']

# Image directory paths
CATEGORIES_DIR = {
    "car": "dataset/valid/car/",
    "cats": "dataset/valid/cats/",
    "dogs": "dataset/valid/dogs/",
    "flowers": "dataset/valid/flowers/",
    "none": "dataset/valid/none/"
}

# Probabilities configuration
CATEGORY_PROB = 0.4  # Probability for selected categories
NONE_PROB = 0.6      # Probability for "none" category


# Function for IA prediction
def get_ia_response(image_path, category):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0  # Normalize between 0 and 1
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = predictions[0][class_index]  # Confidence for predicted class
    predicted_category = categories[class_index]
    print(predicted_category)
    print(image_path)
    print(confidence)
    return (predicted_category == category) and (confidence > 0.5), predicted_category, confidence


class ImageGridPanel(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)

        # Grid size
        self.grid_size = 3
        self.selected_category = "car"  # Default selected category
        self.images = []
        self.image_controls = []

        # Main sizer
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Category selection dropdown
        category_box = wx.ComboBox(self, choices=categories, value=self.selected_category)
        category_box.Bind(wx.EVT_COMBOBOX, self.on_category_change)
        main_sizer.Add(category_box, 0, wx.ALL, 10)

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
        Selects a random image based on probabilities and chosen category.
        """
        if random.random() < CATEGORY_PROB:
            folder = CATEGORIES_DIR[self.selected_category]
        else:
            folder = CATEGORIES_DIR["none"]

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
            image = wx.StaticBitmap(self, -1, wx.Bitmap(img_path, wx.BITMAP_TYPE_ANY))
            self.image_grid.Add(image, 1, wx.EXPAND | wx.ALL, 5)
            self.image_controls.append(image)

        self.Layout()  # Update the layout

    def start_ia(self, event=None):
        """
        Uses the IA to predict images and change their appearance.
        """
        self.progress_gauge.SetValue(0)
        for i, img_path in enumerate(self.images):
            result, predicted_category, confidence = get_ia_response(img_path, self.selected_category)

            # Update image appearance based on prediction
            image_control = self.image_controls[i]
            if result:
                # Add a green border or other visual indicator
                image_control.SetBackgroundColour(wx.Colour(0, 255, 0, 50))  # Semi-transparent green
            else:
                # Add a red border or other visual indicator
                image_control.SetBackgroundColour(wx.Colour(255, 0, 0, 50))  # Semi-transparent red

            # Add prediction text as a tooltip
            image_control.SetToolTip(f"Prediction: {predicted_category}\nConfidence: {confidence:.2f}")

            # Update progress gauge
            self.progress_gauge.SetValue(i + 1)

            self.Layout()  # Update the layout

    def on_category_change(self, event):
        """
        Handles category selection change.
        """
        self.selected_category = event.GetString()
        self.reload_images()


class MyApp(wx.App):
    def OnInit(self):
        frame = wx.Frame(None, title="Image Grid with IA")
        panel = ImageGridPanel(frame)
        frame.Show()
        return True

if __name__ == '__main__':
    app = MyApp()
    app.MainLoop()
