import numpy as np
import numba
from numba_progress import ProgressBar
import sys, os

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
            if(norm(kohonen_map[i,j], x) < running_min):
                bmu = (i, j)
                running_min = norm(kohonen_map[i,j], x)

    return bmu

@numba.njit(nogil=True)
def curr_lr(learning_rate, curr_iter, max_iter):
        clr = learning_rate*(1 - (curr_iter/max_iter))
        return clr

@numba.njit(nogil=True)
def neighbourhood_func(del_x, del_y, length, breadth, curr_iter, max_iter):
        sigma_t = (length**2 + breadth**2)*np.exp(-curr_iter/max_iter)/16
        return np.exp(-(del_x**2 + del_y**2)/2/sigma_t)

@numba.njit(nogil=True)
def update_weights(bmu, x, length, breadth, kohonen_map, curr_iter, max_iter, learning_rate):
    clr = curr_lr(learning_rate, curr_iter, max_iter)
    for i in range(length):
        for j in range(breadth):
            delta_w_ij = clr*(x - kohonen_map[i, j]) 
            delta_w_ij = neighbourhood_func(abs(bmu[0] - i), abs(bmu[1] - j), length, breadth, curr_iter, max_iter)*delta_w_ij
            kohonen_map[i, j] += delta_w_ij

@numba.njit(nogil=True)
def fit(length, breadth, img, max_iter, learning_rate, progress):
    kohonen_map = np.random.rand(length, breadth, img.shape[2])*255
    for i in range(max_iter):
        curr_iter = i
        for u in range(img.shape[0]):
            for v in range(img.shape[1]):
                x = img[u, v]
                bmu = find_best_matching_unit(x, length, breadth, kohonen_map)
                update_weights(bmu, x, length, breadth, kohonen_map, curr_iter, max_iter, learning_rate)
        progress.update(1)
    return kohonen_map


def generate_coded_image(img, kohonen_map):
    coded_img =  np.zeros(shape=(img.shape[0], img.shape[1], 2))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rgb_val = img[i, j]
            closest = find_best_matching_unit(rgb_val, kohonen_map.shape[0], kohonen_map.shape[0], kohonen_map)
            coded_coordinates = np.array(closest)
            coded_img[i, j] = coded_coordinates
    return coded_img


def generate_image_from_coded(coded_img, kohonen_map):
    reconstructed = np.zeros(shape=(coded_img.shape[0], coded_img.shape[1], kohonen_map.shape[2]))
    for i in range(coded_img.shape[0]):
        for j in range(coded_img.shape[1]):
            m = np.array(coded_image[i, j], dtype=np.int32)
            reconstructed[i, j] = kohonen_map[m[0],m[1]]
    return reconstructed


def generate_kohonen(img, length, breadth, lr, maxiter):
    
    blockPrint()
    with ProgressBar(total = 1) as progress:
        fit(1, 1, img, 1, 0.1, progress)
    enablePrint()
    
    with ProgressBar(total = maxiter) as progress:
        kohonen_map = fit(length, breadth, img, maxiter, lr, progress)
    
    return kohonen_map

if __name__ == "__main__":
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt

    # for init image
    plt.figure()
    img = Image.open('initimg.png').convert('RGB')
    img = img.resize((200, 200))
    img.save('initimg_200x200.png')
    img_arr = np.asarray(img)
    kohonen_map = generate_kohonen(img_arr, 10, 10, 0.1, 10)
    plt.imshow(kohonen_map/255, interpolation='nearest')
    plt.savefig('initimg_kohonen.png')
    coded_image = generate_coded_image(img_arr, kohonen_map)
    np.save('initimg_coded_image', coded_image)

    plt.figure()
    plt.axis('off')
    img_restored = generate_image_from_coded(coded_image, kohonen_map)
    plt.imshow(img_restored/255, interpolation='nearest')
    plt.savefig('initimg_restored.png')

    #for mumbai satellite image

    plt.figure()
    img = Image.open('mumbai.jpg').convert('RGB')
    img = img.resize((200, 200))
    img.save('mumbai_200x200.jpg')
    img_arr = np.asarray(img)
    kohonen_map = generate_kohonen(img_arr, 10, 20, 0.5, 100)
    plt.imshow(kohonen_map/255, interpolation='nearest')
    plt.savefig('mumbai_kohonen.png')
    coded_image = generate_coded_image(img_arr, kohonen_map)
    np.save('mumbai_coded_image', coded_image)

    plt.figure()
    plt.axis('off')
    img_restored = generate_image_from_coded(coded_image, kohonen_map)
    plt.imshow(img_restored/255, interpolation='nearest')
    plt.savefig('mumbai_restored.png')

    # generate coded image for hyperspectral image  
    import scipy.io
    import numpy as np
    mat = scipy.io.loadmat('KSC.mat')
    image = mat['KSC']
    image = np.array(image, dtype=np.int32)
    coded_image = generate_kohonen(image, 10, 10, 0.4, 50)
    np.save('KSC_coded_image', coded_image)