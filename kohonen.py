#imports for the libraries used
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
    # to block printing
    sys.stdout = open(os.devnull, 'w')


def enablePrint():
    # to enable printing
    sys.stdout = sys.__stdout__


@numba.njit(nogil=True)
def norm(x, y):
    # returns euclidean distance between x and y
    return np.sum((x - y)**2)


@numba.njit(nogil=True)
def find_best_matching_unit(x, length, breadth, kohonen_map): #finds the winning node in the current Kohonen map
    # variable to store minimum index
    running_min = norm(kohonen_map[0, 0], x) #current minimum 
    bmu = (0, 0) # dummy initaialization
    for i in range(length): # iterate over the map
        for j in range(breadth):
            if (norm(kohonen_map[i, j], x) < running_min): # calculate norm, if lower than running minimum then set as BMU 
                bmu = (i, j)
                running_min = norm(kohonen_map[i, j], x)  # update running minimum

    return bmu


@numba.njit(nogil=True)
def curr_lr(learning_rate, curr_iter, max_iter): # Caclulates current learning rate  
    clr = learning_rate*(1 - (curr_iter/max_iter))  # Linear function to calculate current learning rate
    return clr


@numba.njit(nogil=True)
def neighbourhood_func(del_x, del_y, length, breadth, curr_iter, max_iter, neighbourhood_spread): # Calculates the update magnitude for the specified neoghbour
    sigma_t = (length**2 + breadth**2)*np.exp(-curr_iter/max_iter)*neighbourhood_spread
    return np.exp(-(del_x**2 + del_y**2)/2/sigma_t)


@numba.njit(nogil=True)
def update_weights(bmu, x, length, breadth, kohonen_map, curr_iter, max_iter, learning_rate, neighbourhood_spread): # Updates the kohonen map
    clr = curr_lr(learning_rate, curr_iter, max_iter) # Get the current learning rate
    for i in range(length): # iterate over the Kohonen map
        for j in range(breadth):
            delta_w_ij = clr*(x - kohonen_map[i, j])   # calulate the current_learning_rate(pixel_vlaue-kohonen_cell_value) 
            delta_w_ij = neighbourhood_func(           
                abs(bmu[0] - i), abs(bmu[1] - j), length, breadth, curr_iter, max_iter, neighbourhood_spread)*delta_w_ij # Multipy by neighbourhood function
            kohonen_map[i, j] += delta_w_ij # update the Kohonen cell


@numba.njit(nogil=True)
def fit(length, breadth, img, max_iter, learning_rate, neighbourhood_spread, progress): # Function to train the Kohonen map 
    kohonen_map = np.random.rand(length, breadth, img.shape[2])*255 # initialize the map
    for i in range(max_iter): # train for specified number of iterations
        curr_iter = i 
        for u in range(img.shape[0]): # iterate over the Map 
            for v in range(img.shape[1]): 
                x = img[u, v]  
                bmu = find_best_matching_unit(x, length, breadth, kohonen_map) # find the BMU
                update_weights(bmu, x, length, breadth, kohonen_map,
                               curr_iter, max_iter, learning_rate, neighbourhood_spread) # Update the Kohonen map
        progress.update(1)
    return kohonen_map


def generate_coded_image(img, kohonen_map): # Generate coded image
    coded_img = np.zeros(shape=(img.shape[0], img.shape[1], 2)) # Initialize the image
    for i in range(img.shape[0]): # iterate over the image
        for j in range(img.shape[1]):
            rgb_val = img[i, j]                     # choose current pixel
            closest = find_best_matching_unit(
                rgb_val, kohonen_map.shape[0], kohonen_map.shape[1], kohonen_map) # find closest match in Kohonen Map
            coded_coordinates = np.array(closest)  # get co-ordinates of the BMU
            coded_img[i, j] = coded_coordinates  
    return coded_img


def generate_image_from_coded(coded_img, kohonen_map):      # Reconstruct the image from the Kohonen map
    reconstructed = np.zeros(
        shape=(coded_img.shape[0], coded_img.shape[1], kohonen_map.shape[2]))  # initialize the coded image with zeros
    for i in range(coded_img.shape[0]):   # iterate over the image 
        for j in range(coded_img.shape[1]): 
            m = np.array(coded_img[i, j], dtype=np.int32)   # get the co-ordinates of the cell to be used for reconstruction
            reconstructed[i, j] = kohonen_map[m[0], m[1]]   # replace the pixel with the BMU from the Kohonen Map
    return reconstructed


def generate_kohonen(img, length, breadth, lr, maxiter, spread):
	# dummy call to warm numba JIT
    blockPrint()
    with ProgressBar(total=1) as progress:
        fit(1, 1, img, 1, 0.1, 0.1, progress)
    enablePrint()

	# track progress on command line using progress bar
    with ProgressBar(total=maxiter) as progress:
		# generate kohonen map using fit method
        kohonen_map = fit(length, breadth, img, maxiter, lr, spread, progress)

    return kohonen_map

class UI:
	def __init__(self):
		# main GUI variable initialization
		self.root = tk.Tk()
		# choose font style
		self.root.option_add( "*font", "courier 10 bold" )
		self.image_path = ''
		self.output_dir = ''
		self.root.geometry("600x750")
		self.root.title("Kohonen Map and Code Book Generator")
		# variable to store whether error occured
		self.error = False
		self.ready = True
		
		# parameters for the kohonen map obtained from user
		self.length = 0
		self.neighbourhoodfunctionspread = 0
		self.breadth = 0
		self.learning_rate = 0
		self.maxiter = 0
		self.img_arr = None

		# widgets for gui
		self.gui = [
			tk.Label(self.root, text="GNR 602 Course Project\n Kohonen Map and Code Book Generator \n Rohan Rajesh Kalbag, Durgaprasad Bhat, Siddharth Anand"),
			tk.Button(self.root, text="Select Input File", command=self.task1),
			tk.Label(self.root, text="", wraplength=300),
			tk.Button(self.root, text="Select Output Directory", command=self.task2),
			tk.Label(self.root, text="", wraplength=300),
		]
		
		# widgets for text boxees and corresponding labels

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

		# final generate button
		self.generate = tk.Button(self.root, text="Generate", command=self.computation)

	def task1(self):
		# start thread to get image file
		threading.Thread(target=self.get_image_file).start()

	def get_image_file(self):
		# code to generate GUI to select the image file name
		image_path = ''
		while len(image_path) == 0:
			image_path = filedialog.askopenfilename(title="Select Image Filename")
		self.gui[2]["text"] = f"Selected Image: {image_path}"
		self.image_path = image_path
	
	def task2(self):
		# start thread to get output directory file name
		threading.Thread(target=self.get_out_dir).start()

	def get_out_dir(self):
		# code to generate GUI to select output directory
		output_dir = ''
		while len(output_dir) == 0:
			output_dir = filedialog.askdirectory(title="Directory To Save Outputs")
		self.gui[4]["text"] = f"Selected Output Directory: {output_dir}"
		self.output_dir = output_dir
	
	def task3(self):
		# thread to validate inputs
		threading.Thread(target=self.validate).start()
	
	def validate(self):
		try:
			# check if all parameters are valid and types are correct
			self.length = int(self.inputs[1].get())
			self.breadth = int(self.inputs[3].get())
			self.learning_rate = float(self.inputs[5].get())
			self.maxiter = int(self.inputs[7].get())
			self.neighbourhoodfunctionspread = float(self.inputs[9].get())
		
		except:
			# if not flag an error
			x = messagebox.showerror("Warning", "Incompatible Parameters Inputted, Check Readme")
			self.error = True
		
		if(self.length < 0 or self.breadth < 0 or self.learning_rate < 0 or self.maxiter < 0 or self.neighbourhoodfunctionspread < 0):
			# check if all parameters are non negative else flag error
			x = messagebox.showerror("Warning", "Incompatible Parameters Inputted, Check Readme")
			self.error = True

		# print output directory and input image path
		print(self.image_path)
		print(self.output_dir)
		
		# check if file extentions are correct else flag error
		if(self.image_path[-3:] == 'jpg' or self.image_path[-3:] == 'png'):
			img = Image.open(self.image_path).convert('RGB')
			img = img.resize((150, 150))
			# save resized image
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
		# performs main kohonen map computation

		# update gui
		for u in self.inputs:
			u.pack_forget()

		self.inputs[-1]["text"] = "Processing, Please Wait\n May take a few minutes \n For Hyperspectral Images it may take 15-20 min \n Do not close this window \n Check progress on terminal behind"
		self.inputs[-1].pack()
		self.generate.pack_forget()
		self.root.update()

		# generate kohonen for inputted images
		kohonen_map = generate_kohonen(self.img_arr, self.length, self.breadth, self.learning_rate, self.maxiter, self.neighbourhoodfunctionspread)

		# if image is RGB then visualize as image
		if(kohonen_map.shape[2] == 3):
			plt.imshow(kohonen_map/255, interpolation='nearest')
			plt.savefig(f'{self.output_dir}/kohonen.png')
		
		# generate coded image from kohonen map and image
		coded_image = generate_coded_image(self.img_arr, kohonen_map)

		# generate human readable coded image
		with open(f'{self.output_dir}/coded_image.txt', 'w') as outfile:
			outfile.write('# coded image shape: {0}\n'.format(coded_image.shape))
			for x in range(coded_image.shape[0]):
				for y in range(coded_image.shape[1]):
					outfile.write(f"index of bmu for pixel [{x}, {y}]: [{int(coded_image[x][y][0])}, {int(coded_image[x][y][1])}]\n")
	
		np.save(f'{self.output_dir}/coded_image', coded_image)

		# if image is RGB then restore and visualize
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
		# terminate program

# main gui class instance
ui = UI()

if __name__ == "__main__":
	# main thread
	# update GUI
	if ui.error:
		sys.exit(0)
	for k in ui.gui:
		k.pack(pady=10, padx=10)
	for u in ui.inputs:
		u.pack(pady=5, padx=10)
	# loop GUI continuously forever until termination
	ui.root.mainloop()