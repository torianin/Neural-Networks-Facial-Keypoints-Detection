#!/usr/bin/env python
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.properties import ObjectProperty, StringProperty
from kivy.uix.popup import Popup
from kivy.graphics import Rectangle

import os
from lib import CSVLoader
from lib import Model

class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)

class Controller(AnchorLayout):
    text_path = ObjectProperty(None)

    def load_kaggle_data(self):
        X, y = CSVLoader.load()
        net1 = Model.getNeuralNet()
        net1.fit(X, y)
    
    def do_action(self):
        self.path.text = 'Do something'

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content, size_hint=(0.9, 0.9))
        self._popup.open()
    
    def load(self, path, filename):
        with open(os.path.join(path, filename[0])) as stream:
            #self.text_path.text = stream.read()
            self.text_path.text = os.path.join(path, filename[0])
        with self.canvas:
            Rectangle(source=os.path.join(path, filename[0]), pos=self.center, size=(self.width/2., self.height/2.))
        self.dismiss_popup()

    def dismiss_popup(self):
        self._popup.dismiss()

class ControllerApp(App):

    def build(self):
        return Controller(info='Hello world')

if __name__ == '__main__':
    ControllerApp().run()
