import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import Canvas, Button
import sympy

# Load the model and class mappings
model = tf.keras.models.load_model('eqn-detector-model.keras')
with open('class_names.txt', 'r') as f:
    class_mapping = [line.strip() for line in f]

def preprocess_image(img):
    img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img_bin = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return img_bin

def extract_symbols(img_bin):
    contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 50
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
    return sorted(filtered_contours, key=lambda c: cv2.boundingRect(c)[0])

def resize_and_pad(img, size, padColor=0):
    h, w = img.shape[:2]
    sh, sw = size
    aspect = w/h
    if aspect > 1:
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1:
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else:
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0
    
    scaled_img = cv2.resize(img, (new_w, new_h))
    padded_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_CONSTANT, value=padColor)
    return padded_img

def predict_symbol(symbol_img):
    symbol_img_resized = resize_and_pad(symbol_img, (45, 45), 0)
    symbol_img_resized = symbol_img_resized.reshape(1, 45, 45, 1) / 255.0
    predictions = model.predict(symbol_img_resized)
    predicted_index = np.argmax(predictions)
    if predicted_index < len(class_mapping):
        pred_class = class_mapping[predicted_index]
        return pred_class if pred_class != "times" else "*"
    return None

def recognize_equation(img):
    img_bin = preprocess_image(img)
    symbols = extract_symbols(img_bin)
    equation = []
    for contour in symbols:
        x, y, w, h = cv2.boundingRect(contour)
        symbol_img = img_bin[y:y+h, x:x+w]
        predicted_symbol = predict_symbol(symbol_img)
        if predicted_symbol:
            equation.append(predicted_symbol)
    return "".join(equation)

def evaluate_equation(equation_str):
    try:
        result = sympy.sympify(equation_str)
        return str(result.evalf())
    except:
        return "Error: Invalid equation"

class EquationGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Equation Recognizer")
        
        self.canvas = Canvas(self.master, width=400, height=200, bg='white')
        self.canvas.pack()
        
        self.canvas.bind('<B1-Motion>', self.paint)
        
        self.recognize_button = Button(self.master, text="Recognize", command=self.recognize)
        self.recognize_button.pack()
        
        self.clear_button = Button(self.master, text="Clear", command=self.clear)
        self.clear_button.pack()
        
        self.result_label = tk.Label(self.master, text="")
        self.result_label.pack()
        
    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=5)
        
    def recognize(self):
        self.canvas.update()
        img = Image.new('RGB', (400, 200), color='white')
        img.paste(ImageTk.getimage(self.canvas.postscript(colormode='color')))
        equation_str = recognize_equation(img)
        result = evaluate_equation(equation_str)
        self.result_label.config(text=f"Recognized: {equation_str}\nResult: {result}")
        
    def clear(self):
        self.canvas.delete("all")
        self.result_label.config(text="")

if __name__ == "__main__":
    root = tk.Tk()
    app = EquationGUI(root)
    root.mainloop()
