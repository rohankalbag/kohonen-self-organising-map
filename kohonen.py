import numpy as np
import numba
from numba_progress import ProgressBar
import sys
import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tkinter import messagebox
import scipy
import time

def blockPrint():
    sys.stdout = open(os.devnull, 'w')


def enablePrint():
    sys.stdout = sys.__stdout__


@numba.njit(nogil=True)
def norm(x, y):
    return np.sum((x - y)**2)


@numba.njit(nogil=True)
def find_best_matching_unit(x, length, breadth, kohonen_map):
    running_min = norm(kohonen_map[0, 0], x)
    bmu = (0, 0)
    for i in range(length):
        for j in range(breadth):
            if (norm(kohonen_map[i, j], x) < running_min):
                bmu = (i, j)
                running_min = norm(kohonen_map[i, j], x)

    return bmu


@numba.njit(nogil=True)
def curr_lr(learning_rate, curr_iter, max_iter):
    clr = learning_rate*(1 - (curr_iter/max_iter))
    return clr


@numba.njit(nogil=True)
def neighbourhood_func(del_x, del_y, length, breadth, curr_iter, max_iter, neighbourhood_spread):
    sigma_t = (length**2 + breadth**2)*np.exp(-curr_iter/max_iter)*neighbourhood_spread
    return np.exp(-(del_x**2 + del_y**2)/2/sigma_t)


@numba.njit(nogil=True)
def update_weights(bmu, x, length, breadth, kohonen_map, curr_iter, max_iter, learning_rate, neighbourhood_spread):
    clr = curr_lr(learning_rate, curr_iter, max_iter)
    for i in range(length):
        for j in range(breadth):
            delta_w_ij = clr*(x - kohonen_map[i, j])
            delta_w_ij = neighbourhood_func(
                abs(bmu[0] - i), abs(bmu[1] - j), length, breadth, curr_iter, max_iter, neighbourhood_spread)*delta_w_ij
            kohonen_map[i, j] += delta_w_ij


@numba.njit(nogil=True)
def fit(length, breadth, img, max_iter, learning_rate, neighbourhood_spread, progress):
    kohonen_map = np.random.rand(length, breadth, img.shape[2])*255
    for i in range(max_iter):
        curr_iter = i
        for u in range(img.shape[0]):
            for v in range(img.shape[1]):
                x = img[u, v]
                bmu = find_best_matching_unit(x, length, breadth, kohonen_map)
                update_weights(bmu, x, length, breadth, kohonen_map,
                               curr_iter, max_iter, learning_rate, neighbourhood_spread)
        progress.update(1)
    return kohonen_map


def generate_coded_image(img, kohonen_map):
    coded_img = np.zeros(shape=(img.shape[0], img.shape[1], 2))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rgb_val = img[i, j]
            closest = find_best_matching_unit(
                rgb_val, kohonen_map.shape[0], kohonen_map.shape[1], kohonen_map)
            coded_coordinates = np.array(closest)
            coded_img[i, j] = coded_coordinates
    return coded_img


def generate_image_from_coded(coded_img, kohonen_map):
    reconstructed = np.zeros(
        shape=(coded_img.shape[0], coded_img.shape[1], kohonen_map.shape[2]))
    for i in range(coded_img.shape[0]):
        for j in range(coded_img.shape[1]):
            m = np.array(coded_img[i, j], dtype=np.int32)
            reconstructed[i, j] = kohonen_map[m[0], m[1]]
    return reconstructed


def generate_kohonen(img, length, breadth, lr, maxiter, spread):

    blockPrint()
    with ProgressBar(total=1) as progress:
        fit(1, 1, img, 1, 0.1, 0.1, progress)
    enablePrint()

    with ProgressBar(total=maxiter) as progress:
        kohonen_map = fit(length, breadth, img, maxiter, lr, spread, progress)

    return kohonen_map


def retrieve_inputs():
    global image_path, output_dir, root

    try:

        length = int(entry1.get())
        breadth = int(entry2.get())
        learning_rate = float(entry3.get())
        maxiter = int(entry4.get())
        neighbourhoodfunctionspread = float(entry5.get())
    
    except:
        x = messagebox.showerror("Warning", "Incompatible Parameters Inputted, Check Readme")
        exit(0)
    
    if(length < 0 or breadth < 0 or learning_rate < 0 or maxiter < 0 or neighbourhoodfunctionspread < 0):
        x = messagebox.showerror("Warning", "Incompatible Parameters Inputted, Check Readme")
        exit(0)

    print(image_path)
    print(output_dir)
    
    img_arr = None

    if(image_path[-3:] == 'jpg' or image_path[-3:] == 'png'):
        
        plt.figure()
        img = Image.open(image_path).convert('RGB')
        img = img.resize((150, 150))
        img.save(f'{output_dir}/150x150.jpg')
        img_arr = np.asarray(img)

    elif(image_path[-3:] == 'npy'):
        
        img_arr = np.load(image_path)

    elif(image_path[-3:] == 'mat'):

        mat = scipy.io.loadmat(image_path)
        keys = list(mat.keys())
        image = mat[keys[3]]
        img_arr = np.array(image, dtype=np.int32)

    else:

        x = messagebox.showerror("Warning", "Wrong Input File Type")
        exit(0)
        

    kohonen_map = generate_kohonen(img_arr, length, breadth, learning_rate, maxiter, neighbourhoodfunctionspread)
    
    if(kohonen_map.shape[2] == 3):
        
        plt.imshow(kohonen_map/255, interpolation='nearest')
        plt.savefig(f'{output_dir}/kohonen.png')
        
    coded_image = generate_coded_image(img_arr, kohonen_map)
    print(coded_image.shape)
    np.save(f'{output_dir}/coded_image', coded_image)

    if(kohonen_map.shape[2] == 3):
        
        plt.figure()
        plt.axis('off')
        img_restored = generate_image_from_coded(coded_image, kohonen_map)
        plt.imshow(img_restored/255, interpolation='nearest')
        plt.savefig(f'{output_dir}/restored.png')

    print("Saved Files\n")
    
    print("Terminating Executable\n")
    
    exit(0)


if __name__ == "__main__":

    root = tk.Tk()
    root.geometry("500x500")
    root.withdraw()

    image_path = ''
    output_dir = ''

    while len(image_path) == 0:
        image_path = filedialog.askopenfilename(title="Select Image Filename")

    while len(output_dir) == 0:
        output_dir = filedialog.askdirectory(title="Directory To Save Outputs")

    root = tk.Tk()
    root.geometry("500x500")
    root.title("Input Parameters of Kohonen Map")

    label1 = tk.Label(root, text="Length of Kohonen Map (Nodes)")
    label1.pack()
    entry1 = tk.Entry(root)
    entry1.pack()

    label2 = tk.Label(root, text="Breath of Kohonen Map (Nodes)")
    label2.pack()
    entry2 = tk.Entry(root)
    entry2.pack()

    label3 = tk.Label(root, text="Initial Learning Rate for Training")
    label3.pack()
    entry3 = tk.Entry(root)
    entry3.pack()

    label4 = tk.Label(root, text="Maximum Number of Iterations")
    label4.pack()
    entry4 = tk.Entry(root)
    entry4.pack()

    label5 = tk.Label(root, text="Neighbourhood Function Spread Factor")
    label5.pack()
    entry5 = tk.Entry(root)
    entry5.pack()

    # Create a button to retrieve inputs
    button = tk.Button(root, text="Submit", command=retrieve_inputs)
    button.pack()

    # Start the Tkinter event loop
    root.mainloop()