import tkinter as tk
from tkinter import filedialog
import threading
from tkinter import messagebox
import sys
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import scipy
import os
import numba
from numba_progress import ProgressBar
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

class UI:
	def __init__(self):
		self.root = tk.Tk()
		self.root.option_add( "*font", "courier 10 bold" )
		self.image_path = ''
		self.output_dir = ''
		self.root.geometry("600x750")
		self.root.title("Kohonen Map and Code Book Generator")
		self.error = False
		self.ready = True

		self.length = 0
		self.neighbourhoodfunctionspread = 0
		self.breadth = 0
		self.learning_rate = 0
		self.maxiter = 0
		self.img_arr = None

		
		self.gui = [
			tk.Label(self.root, text="GNR 602 Course Project\n Kohonen Map and Code Book Generator \n Rohan Rajesh Kalbag, Durgaprasad Bhat, Siddharth Anand"),
			tk.Button(self.root, text="Select Input File", command=self.task1),
			tk.Label(self.root, text="", wraplength=300),
			tk.Button(self.root, text="Select Output Directory", command=self.task2),
			tk.Label(self.root, text="", wraplength=300),
		]

		self.inputs = [
			tk.Label(self.root, text="Length of Kohonen Map (Nodes)"),
			tk.Entry(self.root),
			tk.Label(self.root, text="Breath of Kohonen Map (Nodes)"),
			tk.Entry(self.root),
			tk.Label(self.root, text="Initial Learning Rate for Training"),
			tk.Entry(self.root),
			tk.Label(self.root, text="Maximum Number of Iterations"),
			tk.Entry(self.root),
			tk.Label(self.root, text="Neighbourhood Function Spread Factor"),
			tk.Entry(self.root),
			tk.Button(self.root, text="Validate", command=self.task3),
			tk.Label(self.root, text=""),
		]

		self.generate = tk.Button(self.root, text="Generate", command=self.computation)

	def task1(self):
		threading.Thread(target=self.get_image_file).start()

	def get_image_file(self):
		image_path = ''
		while len(image_path) == 0:
			image_path = filedialog.askopenfilename(title="Select Image Filename")
		self.gui[2]["text"] = f"Selected Image: {image_path}"
		self.image_path = image_path
	
	def task2(self):
		threading.Thread(target=self.get_out_dir).start()

	def get_out_dir(self):
		output_dir = ''
		while len(output_dir) == 0:
			output_dir = filedialog.askdirectory(title="Directory To Save Outputs")
		self.gui[4]["text"] = f"Selected Output Directory: {output_dir}"
		self.output_dir = output_dir
	
	def task3(self):
		threading.Thread(target=self.validate).start()
	
	def validate(self):
		try:
			self.length = int(self.inputs[1].get())
			self.breadth = int(self.inputs[3].get())
			self.learning_rate = float(self.inputs[5].get())
			self.maxiter = int(self.inputs[7].get())
			self.neighbourhoodfunctionspread = float(self.inputs[9].get())
		
		except:
			x = messagebox.showerror("Warning", "Incompatible Parameters Inputted, Check Readme")
			self.error = True
		
		if(self.length < 0 or self.breadth < 0 or self.learning_rate < 0 or self.maxiter < 0 or self.neighbourhoodfunctionspread < 0):
			x = messagebox.showerror("Warning", "Incompatible Parameters Inputted, Check Readme")
			self.error = True

		print(self.image_path)
		print(self.output_dir)
		
		if(self.image_path[-3:] == 'jpg' or self.image_path[-3:] == 'png'):
			img = Image.open(self.image_path).convert('RGB')
			img = img.resize((150, 150))
			img.save(f'{self.output_dir}/150x150.jpg')
			self.img_arr = np.asarray(img)

		elif(self.image_path[-3:] == 'npy'):
			self.img_arr = np.load(self.image_path)

		elif(self.image_path[-3:] == 'mat'):
			mat = scipy.io.loadmat(self.image_path)
			keys = list(mat.keys())
			image = mat[keys[3]]
			self.img_arr = np.array(image, dtype=np.int32)

		else:
			x = messagebox.showerror("Warning", "Wrong Input File Type")
			self.error = True
		
		if self.error:
			self.inputs[11]["text"] = "Fix the errors and try again"
		else:
			self.generate.pack()

	
	def computation(self):

		for u in self.inputs:
			u.pack_forget()

		self.inputs[-1]["text"] = "Processing, Please Wait\n May take a few minutes \n For Hyperspectral Images it may take 15-20 min \n Do not close this window \n Check progress on terminal behind"
		self.inputs[-1].pack()
		self.generate.pack_forget()
		self.root.update()

		kohonen_map = generate_kohonen(self.img_arr, self.length, self.breadth, self.learning_rate, self.maxiter, self.neighbourhoodfunctionspread)

		if(kohonen_map.shape[2] == 3):
			plt.imshow(kohonen_map/255, interpolation='nearest')
			plt.savefig(f'{self.output_dir}/kohonen.png')
			
		coded_image = generate_coded_image(self.img_arr, kohonen_map)

		with open(f'{self.output_dir}/coded_image.txt', 'w') as outfile:
			outfile.write('# coded image shape: {0}\n'.format(coded_image.shape))
			for x in range(coded_image.shape[0]):
				for y in range(coded_image.shape[1]):
					outfile.write(f"index of bmu for pixel [{x}, {y}]: [{int(coded_image[x][y][0])}, {int(coded_image[x][y][1])}]\n")
	
		np.save(f'{self.output_dir}/coded_image', coded_image)

		if(kohonen_map.shape[2] == 3):
			plt.figure()
			plt.axis('off')
			img_restored = generate_image_from_coded(coded_image, kohonen_map)
			plt.imshow(img_restored/255, interpolation='nearest')
			plt.savefig(f'{self.output_dir}/restored.png')
		
		self.inputs[-1]["text"] = "Thank You"
		self.root.update()
		time.sleep(2)
		sys.exit(0)
	
ui = UI()

if __name__ == "__main__":
	if ui.error:
		sys.exit(0)
	for k in ui.gui:
		k.pack(pady=10, padx=10)
	for u in ui.inputs:
		u.pack(pady=5, padx=10)
	ui.root.mainloop()