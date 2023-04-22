#                                   Kohonen Self Organising Map
##                  GNR-602 : Advanced Methods in Satellite Image Processing
##                                       Course Project


*Abstract: In this project we implement a Numba-JIT Accelerated Pythonic Implementation of the Kohonen Self Organising Map with user-specified grid matrix size, and a multispectral image as input and we generate a coded image using the trained SOM as a code book. We also restore the image from its codebook and compare it with original image*

[Project Files](https://iitbacin-my.sharepoint.com/:u:/g/personal/20d170033_iitb_ac_in/EU_LjvJoHBhOoA7GgfJpv-0BlsK5mMnNCdfzmVr2Sm6PWw?e=bOu5I7)

### Collaborators

- Rohan Rajesh Kalbag 
- Durgaprasad Bhat
- Siddharth Anand
  
### To generate executable in Windows

```bash
./generate_executable.ps1
```

### To generate executable in Ubuntu
```bash
chmod +x generate_executable.sh
./generate_executable.sh
```

### Instructions for running `.exe`

**Image formats supported**: `.png`, `.jpg`, `.mat` (single image in a single array in `x_pixels` * `y_pixels` * `n_channels` for `.mat`)


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
1. `150x150.png` : A downsized version of the input image ($150$ x $150$ pixels)
2. `coded_image.npy` : A numpy file containing the coded image. Users can import this into their own python code using `numpy.load()` 
3. `coded_image.txt` : A human readable version of the coded image with the best matching Kohonen cell for each pixel.
4. `kohonen.png` : The trained Kohonen map visualized as an image
5. `restored.png` : The final result. The original image reconstructed using the best matching Kohonen map cells. 

## Results

- Find a detailed analysis of the code, experiments and inferences in the [Presentation](https://github.com/rohankalbag/kohonen-self-organising-map/blob/main/presentation.pdf)

## Some Cool Inferences

#### Tom and Jerry

- Input Image (was resized to $150$ x $150$)

![tj](https://user-images.githubusercontent.com/46604893/233677504-d163639f-7a09-4696-a20f-702e5210cb70.jpg)


- Parameters

```
{
  l = 12
  b = 12
  lr = 0.5
  niter = 20
  nbf = 0.03
}
```

- Kohonen Map

![kohonen](https://user-images.githubusercontent.com/46604893/233674986-dc459449-40ae-4a04-95c1-7cb4dbb7d438.png)

- Restored Image

![restored](https://user-images.githubusercontent.com/46604893/233675058-a8d8ba14-d3bf-4d19-bd50-2cfeea58b073.png)

## RGB (3 channel) Satellite Image of Mumbai

- Input Image (was resized to $200$ x $200$)

![mumbai](https://user-images.githubusercontent.com/46604893/233677679-b1872cde-d345-4878-98d6-e5494c2f3617.jpg)

- Parameters

```
{
  l = 20
  b = 35
  lr = 0.25
  niter = 100
  nbf = 0.02083
}
```

- Kohonen Map

![mumbai_kohonen_2 5](https://user-images.githubusercontent.com/46604893/233676619-895de481-ae92-41e8-8228-e67ff27a46cb.png)

- Restored Image 

![mumbai_restored_2 5](https://user-images.githubusercontent.com/46604893/233676677-4d2f8a9d-2f6c-4b35-8a6e-38bfd65fa237.png)


