import numpy as np

class kohonen:
    def __init__(self, length, breadth, learning_rate, max_iter):
        self.length = length
        self.breadth = breadth
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.curr_iter = 0

    def find_best_matching_unit(self, x):
        running_min = np.linalg.norm(self.kohonen_map[0,0], x)
        bmu = (0, 0)
        for i in range(self.length):
            for j in range(self.breadth):
                if(np.linalg.norm(self.kohonen_map[i,j], x) < running_min):
                    bmu = (i, j)
                    running_min= np.linalg.norm(self.kohonen_map[i,j], x)

        return bmu
    
    def curr_lr(self):
        clr = self.learning_rate*(1 - (self.curr_iter/self.max_iter))
        return clr
        
    def neighbourhood_func(self, del_x, del_y):
        sigma_t = (self.length**2 + self.breadth**2)*np.exp(-self.curr_iter/self.max_iter)/16
        return np.exp(-(del_x**2 + del_y**2)/2/sigma_t)

    def update_weights(self, bmu):
        curr_lr = self.curr_lr(self.curr_iter)
        for i in range(self.length):
            for j in range(self.breadth):
                delta_w_ij = curr_lr*(self.kohonen_map[bmu[0], bmu[1]] - self.kohonen_map[i, j]) 
                delta_w_ij = self.neighbourhood_func(self.curr_iter, abs(bmu[0] - i), abs(bmu[1] - j))*delta_w_ij
                self.kohonen_map[i, j] += delta_w_ij
                

    def fit(self, img):
        self.kohonen_map = np.random.rand(self.length, self.breadth, img.shape[2])
        for i in range(self.max_iter):
            self.curr_iter = i
            for u in img.shape[0]:
                for v in img.shape[1]:
                    x = img[u, v]
                    bmu = self.find_best_matching_unit(x)
                    self.update_weights(bmu)
            
    def generate_coded_image(self):
        return self.kohonen_map