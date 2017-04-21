import pyautogui as gui
import random

Inputs = ["left", "right", "up", "down", "z", "x", "a", "s"]

while(True):
    gui.keyDown(random.choice(Inputs))
    gui.keyUp(random.choice(Inputs))