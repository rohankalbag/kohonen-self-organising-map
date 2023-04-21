#                                   Kohonen Self Organising Map
##                                    GNR-602 Course Project

### Instructions for running .exe

**Image formats supported**: .png, .jpg, .mat (single image in a single array in x_pixels * y_pixels * n_channels for .mat)


**Selecting Input file**: Press the ‘Select Input File’ button, use the file navigator to navigate to the required file. Select the file 


**Selecting destination folder**: Press the ‘Select Output Directory’ button, use the file navigator to navigate to the folder where the outputs are needed to be stored. Select the folder


**Input parameters**:
1. Length of Kohonen Map (nodes) : positive integer. number of nodes in the Kohonen map in x direction 
2. Breadth of Kohonen Map (nodes): positive integer. positive integer, number of nodes in the Kohonen map in y direction 
3. Initial Learning Rate for Training: positive float. Learning rate for the first iteration 
4. Maximum Number of Iterations: positive integer. The number iteration the models needs to be trained for.  
5. Neighborhood Function Spread Factor : positive float (Usually less than 1) Larger factor will increase the number of neighbors affected and the degree of effect. Controls the spread of the neighborhood function


Press the ‘Validate’ button to check if the given parameters are compatible. If the parameters are compatible, the ‘Generate’ button will appear. Press ‘Generate’ to start running the algorithm


Please wait till the training completes and the results are generated. The time taken depends on the image type and the input parameters. Do not close the window, it will automatically close once the results are generated. For checking the progress of the training, please see the console (terminal) that will popup behind the GUI.


**Outputs**: 
1. 150x150.png : A downsized version of the input image (150 x 150 pixels)
2. coded_image.npy : A numpy file containing the coded image. Users can import this into their own python code using numpy.load() 
3. coded_image.txt : A human readable version of the coded image with the best matching Kohonen cell for each pixel.
4. kohonen.png : The trained Kohonen map visualized as an image
5. restored.png : The final result. The original image reconstructed using the best matching Kohonen map cells. 
