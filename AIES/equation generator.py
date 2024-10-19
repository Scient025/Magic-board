import numpy as np
import os
import random
from skimage import io
import matplotlib.pyplot as plt

folder = "archive/data/extracted_images"  # Changed directory prefix

def equation_generator(numbers=2, max_num=100):
    assert numbers > 1, "You need at least two numbers to generate a valid equation"
    # Generate numbers and operator
    generated_eqn = []
    operators = ['+', '-', '*']
    for number in range(numbers):
        gen_num = np.random.randint(0, max_num + 1)  # Include max_num
        generated_eqn.append(str(gen_num))
        if number < numbers - 1:  # Avoid adding operator after last number
            operator_ind = np.random.randint(len(operators))
            operator = operators[operator_ind]
            generated_eqn.append(operator)
    generated_eqn_string = ''.join(generated_eqn)
    ans = eval(generated_eqn_string)
    ans_string = '=' + str(ans)
    return generated_eqn_string + ans_string

eqn = equation_generator(numbers=2, max_num=100)
print(eqn)

def random_sample_file(directory):
    # Randomly selects a file from a directory
    n = 0
    random.seed()
    for root, dirs, files in os.walk(directory):
        for name in files:
            n += 1
            if random.uniform(0, n) < 1:
                rfile = os.path.join(root, name)
    return rfile

def generate_eqn_image(folder, eqn):
    eqn_array = []
    # Pick correct file
    for char in eqn:
        if char == "*": 
            char = 'times'
        char_folder = f"{folder}/{char}"
        file = random_sample_file(char_folder)  # Randomly sample an image from the directory, each (45X45)
        img = io.imread(file)
        eqn_array.append(img)
    # Concatenate all images together into 1 giant image
    eqn = np.hstack(eqn_array)
    return eqn

output_folder = "AIES/equation_images"  # Changed directory prefix
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

EQNS = 50
NUMBERS = 2  # We generate a set for two numbers
MAX_NUM = 100
SAVE = 1
for i in range(EQNS):
    eqn = equation_generator(numbers=NUMBERS, max_num=MAX_NUM)
    print(eqn)
    eqn_array = generate_eqn_image(folder, eqn)
    filename = f"{NUMBERS}numbers_{str(i).zfill(3)}.png"
    plt.imshow(eqn_array, cmap="gray")
    plt.axis('off')
    plt.savefig(output_folder + "/" + filename)
    plt.show()

# Generate equations for three numbers
EQNS = 50
NUMBERS = 3  # We generate a set for three numbers
MAX_NUM = 100
SAVE = 1
for i in range(EQNS):
    eqn = equation_generator(numbers=NUMBERS, max_num=MAX_NUM)
    print(eqn)
    eqn_array = generate_eqn_image(folder, eqn)
    filename = f"{NUMBERS}numbers_{str(i).zfill(3)}.png"
    plt.imshow(eqn_array, cmap="gray")
    plt.axis('off')
    plt.savefig(output_folder + "/" + filename)
    plt.show()
