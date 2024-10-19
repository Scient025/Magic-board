import tkinter as tk
from tkinter import *
from tkinter import ttk, messagebox
import cv2
import numpy as np
from keras.models import load_model
from PIL import ImageGrab


# Load your trained model and class mapping
model_path = 'AIES/eqn-detect-model.keras'
model = load_model(model_path)

with open('AIES/class_names.txt', 'r') as f:
    class_mapping = f.read().splitlines()

# Initialize main application window
root = Tk()
root.title("Equation Predictor")
root.geometry("1050x600+150+50")
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
    board.create_line((current_x, current_y, work.x, work.y), width=get_current_value(), fill=color)
    current_x, current_y = work.x, work.y

def show_color(new_color):
    global color
    color = new_color

def new_canvas():
    board.delete('all')

def is_valid_equation(equation):
    """Check if the predicted equation is valid."""
    valid_chars = set("0123456789+-*/()= ") 
    return all(char in valid_chars for char in equation)

def predict_equation(img):
    """Predicts individual symbols from the drawn image."""
    img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    _, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Extract contours
    contours_list, _ = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    eqn_list = []
    min_contour_area = 100  # Minimum area to consider a valid symbol
    filtered_contours = [c for c in contours_list if cv2.contourArea(c) > min_contour_area]

    if not filtered_contours:
        raise ValueError("No valid contours found in the drawn image.")

    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        symbol_img = img_bin[y:y + h, x:x + w]
        
        # Resize to model input size and predict
        symbol_img_resized = cv2.resize(symbol_img, (45, 45)).reshape(1, 45, 45, -1) / 255.0
        predictions = model.predict(symbol_img_resized)
        predicted_index = np.argmax(predictions)

        if predicted_index < len(class_mapping):
            pred_class = class_mapping[predicted_index]
            if pred_class == "times":  # Replace 'times' with '*'
                pred_class = "*"
            eqn_list.append(pred_class)

    return "".join(eqn_list)

def on_predict():
    """Capture the canvas as an image and predict the equation."""
    x1 = root.winfo_rootx() + board.winfo_x()
    y1 = root.winfo_rooty() + board.winfo_y()
    x2 = x1 + board.winfo_width()
    y2 = y1 + board.winfo_height()

    img = ImageGrab.grab().crop((x1, y1, x2, y2))

    try:
        predicted_equation = predict_equation(img)

        if not predicted_equation:
            raise ValueError("No symbols detected.")

        if not is_valid_equation(predicted_equation):
            raise ValueError(f"Invalid equation detected: {predicted_equation}")

        result = eval(predicted_equation)  # Evaluate the equation result
        final_output = f"The predicted equation is: {predicted_equation} = {result}"
        result_label.config(text=final_output)  # Display the result

    except Exception as e:
        messagebox.showerror("Error", str(e))

# Color selection and drawing setup
colors_frame = Frame(root)
colors_frame.place(x=10, y=20)

colors_canvas = Canvas(colors_frame, bg="#ffffff", width=37, height=300, bd=0)
colors_canvas.pack(side=LEFT)

color_button_frame = Frame(root)
color_button_frame.place(x=30, y=400)

def display_colors():
    color_options = ['black', 'gray', 'brown4', 'red', 'orange', 'yellow', 'green', 'blue', 'purple']
    
    for index, color_name in enumerate(color_options):
        id = colors_canvas.create_rectangle((10, (10 + index * 30), (30, (30 + index * 30))), fill=color_name)
        colors_canvas.tag_bind(id, '<Button-1>', lambda x: show_color(color_name))

display_colors()

# Drawing canvas setup
board = Canvas(root, width=930, height=500, bg="white", cursor="hand2")
board.place(x=100, y=10)
board.bind('<Button-1>', locate_xy)
board.bind('<B1-Motion>', addLine)

# Slider for line thickness
current_value = tk.DoubleVar()

def get_current_value():
    return '{: .2f}'.format(current_value.get())

def slider_changed(event):
    value_label.configure(text=get_current_value())

slider = tk.Scale(root, font=("Arial", 12), from_=0.5, to=10.0, length=200, tickinterval=.5, resolution=.5,
                  orient='horizontal', command=slider_changed, var=current_value)
slider.place(x=30, y=530)

value_label = tk.Label(root, text=get_current_value())
value_label.place(x=27, y=550)

# Buttons for clear and predict actions
Button(color_button_frame, text="Clear", command=new_canvas).pack(side=LEFT)
Button(color_button_frame, text="Predict", command=on_predict).pack(side=LEFT)

# Label for displaying prediction results
result_label = tk.Label(root, text="", font=("Arial", 14), fg="blue")
result_label.place(x=100, y=520)

root.mainloop()
