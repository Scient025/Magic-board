import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
from keras.models import load_model
from PIL import ImageGrab
import pyautogui  # Add this import at the top of your file

# Load your trained model and class mapping
model_path = 'AIES/eqn-detect-model.keras'
model = load_model(model_path)

with open('AIES/class_names.txt', 'r') as f:
    class_mapping = f.read().splitlines()

# Initialize main application window
root = tk.Tk()
root.title("Equation Predictor")
root.geometry("1050x600+150+50")
root.configure(bg="#f2f3f5")
root.resizable(False, False)

# Drawing variables
current_x = 0
current_y = 0
color = 'black'

# Canvas for drawing
board = tk.Canvas(root, width=930, height=500, bg="white", cursor="cross")
board.place(x=100, y=10)

def locate_xy(event):
    global current_x, current_y
    current_x = event.x
    current_y = event.y

def add_line(event):
    global current_x, current_y
    board.create_line((current_x, current_y, event.x, event.y), width=2, fill=color)
    current_x, current_y = event.x, event.y

def clear_canvas():
    board.delete('all')

def is_valid_equation(equation):
    """Check if the predicted equation is valid."""
    valid_chars = set("0123456789+-*/()= ") 
    return all(char in valid_chars for char in equation)

def predict_equation(img):
    """Predicts individual symbols from the drawn image."""
    img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img_bin = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Save the preprocessed image for debugging
    cv2.imwrite("preprocessed_image.png", img_bin)
    print("Preprocessed image saved as 'preprocessed_image.png'")

    # Extract contours
    contours_list, _ = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    eqn_list = []
    
    # Filter out small contours based on area
    min_contour_area = 50  # Reduced threshold
    filtered_contours = [c for c in contours_list if cv2.contourArea(c) > min_contour_area]

    if not filtered_contours:
        raise ValueError("No valid contours found in the drawn image.")

    for i, contour in enumerate(sorted(filtered_contours, key=lambda c: cv2.boundingRect(c)[0])):
        x, y, w, h = cv2.boundingRect(contour)
        symbol_img = img_bin[y:y + h, x:x + w]

        # Resize and pad to 45x45
        symbol_img_resized = resize_pad(symbol_img, (45, 45), 0)
        
        # Save each symbol image for debugging
        cv2.imwrite(f"symbol_{i}.png", symbol_img_resized)
        print(f"Symbol {i} saved as 'symbol_{i}.png'")
        
        # Reshape to (1, 45, 45, 1) and normalize
        symbol_img_resized = symbol_img_resized.reshape(1, 45, 45, 1) / 255.0
        
        predictions = model.predict(symbol_img_resized)
        predicted_index = np.argmax(predictions)

        if predicted_index < len(class_mapping):
            pred_class = class_mapping[predicted_index]
            if pred_class == "times":  # Replace 'times' with '*'
                pred_class = "*"
            eqn_list.append(pred_class)
            print(f"Symbol {i}: Predicted class: {pred_class}, Confidence: {predictions[0][predicted_index]:.4f}")
        else:
            print(f"Symbol {i}: Predicted index {predicted_index} out of range")

    return "".join(eqn_list)

def resize_pad(img, size, padColor=0):
    h, w = img.shape[:2]
    sh, sw = size
    
    # Calculate scaling
    scale = min(sh/h, sw/w)
    
    # Calculate new dimensions
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize the image
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create a new image with the target size
    padded = np.full((sh, sw), padColor, dtype=np.uint8)
    
    # Compute the position to paste the resized image
    y_offset = (sh - new_h) // 2
    x_offset = (sw - new_w) // 2
    
    # Paste the resized image
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return padded

def on_predict():
    """Capture the entire GUI window as an image and predict the equation."""
    # Get the position and size of the root window
    x = root.winfo_x()
    y = root.winfo_y()
    width = root.winfo_width()
    height = root.winfo_height()

    # Capture the entire GUI window
    img = pyautogui.screenshot(region=(x, y, width, height))

    # Save the captured image for debugging
    img.save("captured_gui.png")
    print("Captured image saved as 'captured_gui.png'")

    try:
        predicted_equation = predict_equation(img)

        print(f"Predicted Equation: {predicted_equation}")  # Debugging output

        if not predicted_equation:
            raise ValueError("No symbols detected.")

        if not is_valid_equation(predicted_equation):
            raise ValueError(f"Invalid equation detected: {predicted_equation}")

        result = eval(predicted_equation)  # Evaluate the equation safely (ensure it's safe)
        final_output = f"The predicted equation is: {predicted_equation} = {result}"
        result_label.config(text=final_output)  # Update label with result

    except Exception as e:
        print(f"Error occurred: {str(e)}")  # Print error to console for debugging
        messagebox.showerror("Error", str(e))

# Bind mouse events to canvas
board.bind('<Button-1>', locate_xy)
board.bind('<B1-Motion>', add_line)

# Buttons for clear and predict actions
button_frame = tk.Frame(root)
button_frame.place(x=30, y=520)

clear_button = tk.Button(button_frame, text="Clear", command=clear_canvas)
clear_button.pack(side=tk.LEFT)

predict_button = tk.Button(button_frame, text="Predict", command=on_predict)
predict_button.pack(side=tk.LEFT)

# Label for displaying prediction results
result_label = tk.Label(root, text="", font=("Arial", 14), fg="blue")
result_label.place(x=100, y=550)

# Run the application
root.mainloop()
