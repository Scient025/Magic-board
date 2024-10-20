from tkinter import *
from tkinter.colorchooser import askcolor
from tkinter import ttk
import tkinter as tk
import pytesseract
from PIL import Image, ImageGrab
import sympy as sp
import numpy as np
import cv2
import tensorflow as tf

model = tf.keras.models.load_model(r'AIES\eqn-detect-model.keras')

# Define class names (make sure they match your model)
class_names = [
    "!", "(", ")", "+", ",", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "=",
    "A", "C", "Delta", "G", "H", "M", "N", "R", "S", "T", "X", "[", "]",
    "alpha", "ascii_124", "b", "beta", "cos", "d", "div", "e", "exists", "f", "forall",
    "forward_slash", "gamma", "geq", "gt", "i", "in", "infty", "int", "j", "k", "l",
    "lambda", "ldots", "leq", "lim", "log", "lt", "mu", "neq", "o", "p", "phi", "pi",
    "pm", "prime", "q", "rightarrow", "sigma", "sin", "sqrt", "sum", "tan", "theta",
    "times", "u", "v", "w", "y", "z", "{", "}"
]

root = Tk()
root.title("Notes")
root.geometry("1050x570+150+50")
root.configure(bg="#f2f3f5")
root.resizable(False, False)

current_x = 0
current_y = 0
color = 'black'

def locate_xy(work):
    global current_x, current_y

    current_x = work.x
    current_y = work.y

def addLine(work):
    global current_x, current_y

    board.create_line((current_x, current_y, work.x, work.y), width=get_current_value(), fill=color, capstyle=ROUND, smooth=True)
    current_x, current_y = work.x, work.y

def show_color(new_color):
    global color
    color = new_color

def new_canvas():
    board.delete('all')
    display_colors()

image_icon = PhotoImage(file="AIES\logo.png")
root.iconphoto(False, image_icon)

color_box = PhotoImage(file="AIES\color_section.png")
Label(root, image=color_box, bg="#f2f3f5").place(x=10, y=20)

eraser = PhotoImage(file="AIES\eraser.png")
Button(root, image = eraser, bg="#f2f3f5", command=new_canvas).place(x=30, y=400)

colors = Canvas(root, bg="#ffffff", width=37, height=300, bd=0)
colors.place(x=30, y=60)


def display_colors():
    id = colors.create_rectangle((10,10,30,30), fill='black')
    colors.tag_bind(id, '<Button-1>', lambda x: show_color('black'))

    id = colors.create_rectangle((10,40,30,60), fill='gray')
    colors.tag_bind(id, '<Button-1>', lambda x: show_color('gray'))

    id = colors.create_rectangle((10,70,30,90), fill='brown4')
    colors.tag_bind(id, '<Button-1>', lambda x: show_color('brown4'))

    id = colors.create_rectangle((10,100,30,120), fill='red')
    colors.tag_bind(id, '<Button-1>', lambda x: show_color('red'))

    id = colors.create_rectangle((10,130,30,150), fill='orange')
    colors.tag_bind(id, '<Button-1>', lambda x: show_color('orange'))

    id = colors.create_rectangle((10,160,30,180), fill='yellow')
    colors.tag_bind(id, '<Button-1>', lambda x: show_color('yellow'))

    id = colors.create_rectangle((10,190,30,210), fill='green')
    colors.tag_bind(id, '<Button-1>', lambda x: show_color('green'))

    id = colors.create_rectangle((10,220,30,240), fill='blue')
    colors.tag_bind(id, '<Button-1>', lambda x: show_color('blue'))

    id = colors.create_rectangle((10,250,30,270), fill='purple')
    colors.tag_bind(id, '<Button-1>', lambda x: show_color('purple'))

display_colors()


board = Canvas(root, width=930, height=500, background="white", cursor="hand2")
board.place(x=100, y=10)
board.bind('<Button-1>', locate_xy)
board.bind('<B1-Motion>', addLine)

current_value = tk.DoubleVar()

def get_current_value():
    return '{: .2f}'.format(current_value.get())

def slider_changed(event):
    value_label.configure(text=get_current_value())

slider = ttk.Scale(root, from_=0, to=100, orient='horizontal', command=slider_changed, variable=current_value)
slider.place(x=30, y=530)

value_label = ttk.Label(root, text=get_current_value())
value_label.place(x=27, y=550)

def capture_canvas():
    x = root.winfo_rootx() + board.winfo_x()
    y = root.winfo_rooty() + board.winfo_y()
    x1 = x + board.winfo_width()
    y1 = y + board.winfo_height()
    img = ImageGrab.grab((x, y, x1, y1))
    img.save("equation.png")
    return img

def binarize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

def detect_contours(binary_img):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10:  # Filter out small contours
            boxes.append((x, y, w, h))
    return boxes

def recognize_equation():
    img = capture_canvas()  # Capture the canvas as an image
    binary_img = binarize(img)  # Binarize the captured image
    keep = detect_contours(binary_img)  # Detect contours in the binary image
    
    eqn_list = []

    for (x, y, w, h) in sorted(keep, key=lambda x: x[0]):
        img_cropped = binary_img[y:y+h, x:x+w]
        img_resized = cv2.resize(img_cropped, (45, 45))
        img_resized = img_resized.astype('float32') / 255.0  # Normalize
        img_resized = np.expand_dims(img_resized, axis=-1)  # Add channel dimension
        img_resized = np.expand_dims(img_resized, axis=0)  # Add batch dimension
        
        # Predict using the model
        pred_class = class_names[np.argmax(model.predict(img_resized))]
        if pred_class == "times":
            pred_class = "*"
        eqn_list.append(pred_class)

    equation_string = "".join(eqn_list)
    equation_label.config(text=equation_string)
    display_equation(equation_string)

def display_equation(equation):
    board.create_text(150, 530, text=equation, fill="black", font=("Arial", 16))

def process_equation(equation):
        try:
            sympy_exp = sp.simplify(equation)
            print("Processed equation: ", sympy_exp)
            return sympy_exp
        except sp.SympifyError:
            print("Could not process the equation.")
            return None

recognize_button = Button(root, text="Recognize Equation", bg="#f2f3f5", command=recognize_equation)
recognize_button.place(x=100, y=550)

equation_label = Label(root, text="", bg="#f2f3f5", font=("Arial", 12))
equation_label.place(x=150, y=530)

root.mainloop()