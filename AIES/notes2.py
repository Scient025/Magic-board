from tkinter import *
from tkinter import ttk
import tkinter as tk
from PIL import Image, ImageGrab
import subprocess
from config import * 

root = Tk()
root.title("Notes")
root.geometry("1050x570+150+50")
root.configure(bg="#f2f3f5")
root.resizable(False, False)

# Add a border to the main window
root.configure(highlightbackground="black", highlightthickness=2)

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

def new_canvas():
    board.delete('all')
    display_colors()

# Icon
image_icon = PhotoImage(file= LOGO_PATH)
root.iconphoto(False, image_icon)

# Color options
color_box = PhotoImage(file= COLOR_SECTION_PATH)
Label(root, image=color_box, bg="#f2f3f5").place(x=10, y=20)

# Clear Canvas Button
eraser = PhotoImage(file= ERASER_PATH)
Button(root, image=eraser, bg="#f2f3f5", command=new_canvas).place(x=30, y=400)

colors = Canvas(root, bg="#ffffff", width=37, height=300, bd=0)
colors.place(x=30, y=60)

def display_colors():
    color_options = ['black', 'gray', 'brown4', 'red', 'orange', 'yellow', 'green', 'blue', 'purple']
    for i, col in enumerate(color_options):
        id = colors.create_rectangle((10, 10 + i * 30, 30, 30 + i * 30), fill=col)
        colors.tag_bind(id, '<Button-1>', lambda x, color=col: show_color(color))

def show_color(new_color):
    global color
    color = new_color

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
    # Wait a moment for the GUI to update
    root.update_idletasks()
    root.update()
    
    # Get the window position and size
    x = root.winfo_rootx()
    y = root.winfo_rooty()
    width = root.winfo_width()
    height = root.winfo_height()
    
    # Capture the entire window
    img = ImageGrab.grab(bbox=(x, y, x+width, y+height))
    img.save(EQUATION_OUTPUT_PATH)  # Save the image
    return img

def recognize_equation():
    root.after(100, _capture_and_recognize)  # Wait 100ms before capturing

def _capture_and_recognize():
    capture_canvas()  # Capture the canvas as an image
    # Run the second script for equation recognition
    subprocess.run(['python', TEST_SCRIPT_PATH])

recognize_button = Button(root, text="Capture and Recognize Equation", bg="#f2f3f5", command=recognize_equation)
recognize_button.place(x=900, y=530)

root.mainloop()
