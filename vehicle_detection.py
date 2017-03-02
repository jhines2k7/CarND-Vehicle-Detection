import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import cv2
import glob
from skimage.feature import hog
#from skimage import color, exposure
# images are divided up into vehicles and non-vehicles

non_vehicle_images = glob.glob('non-vehicles/**/*.png')
vehicle_images = glob.glob('vehicles/**/*.png')
cars = []
notcars = []

for image in non_vehicle_images:
    notcars.append(image)

for image in vehicle_images:
    cars.append(image)

# Define a function to return some characteristics of the dataset 
def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    example_img = mpimg.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype
    # Return data_dict
    return data_dict
    
data_info = data_look(cars, notcars)

print('Your function returned a count of', 
      data_info["n_cars"], ' cars and', 
      data_info["n_notcars"], ' non-cars')
print('of size: ',data_info["image_shape"], ' and data type:', 
      data_info["data_type"])
# Just for fun choose random car / not-car indices and plot example images   
car_ind = np.random.randint(0, len(cars))
notcar_ind = np.random.randint(0, len(notcars))
    
# Read in car / not-car images
car_image = mpimg.imread(cars[car_ind])
notcar_image = mpimg.imread(notcars[notcar_ind])

# Plot the examples
fig = plt.figure()
plt.subplot(121)
plt.imshow(car_image)
plt.title('Example Car Image')
plt.subplot(122)
plt.imshow(notcar_image)
plt.title('Example Not-car Image')
plt.savefig('output_images/my_car_not_car.png', bbox_inches='tight')

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=feature_vec)
        return features

def downsample_image(img):
    return cv2.resize(img, (32, 32))

colorspace = cv2.COLOR_RGB2HLS #cv2.COLOR_RGB2HLS cv2.COLOR_RGB2LUV cv2.COLOR_RGB2HSV cv2.COLOR_RGB2YUV cv2.COLOR_RGB2YCrCb 

# Generate a random index to look at a car image
ind = np.random.randint(0, len(cars))
# Read in the image
car_image = mpimg.imread(cars[ind])
non_car_image = mpimg.imread(notcars[ind])

#downsampled_car_image = downsample_image(car_image)
#downsampled_non_car_image = downsample_image(non_car_image)

car_image_converted = cv2.cvtColor(car_image, colorspace)
non_car_image_converted = cv2.cvtColor(non_car_image, colorspace)
# Define HOG parameters
orient = 8
pix_per_cell = 8
cell_per_block = 2
# Call our function with vis=True to see an image output
features, hog_image = get_hog_features(car_image_converted[:, :, 1], orient, 
                        pix_per_cell, cell_per_block, 
                        vis=True, feature_vec=False)

features, non_car_hog_image = get_hog_features(non_car_image_converted[:, :, 1], orient, 
                        pix_per_cell, cell_per_block, 
                        vis=True, feature_vec=False)

params = {
    'axes.labelsize': 'small',
    'axes.titlesize':'small',
    'xtick.labelsize':'x-small',
    'ytick.labelsize':'x-small'
}

pylab.rcParams.update(params)

# Plot the examples
fig = plt.figure()

plt.subplot(141)
plt.imshow(car_image_converted[:, :, 1], cmap='gray')
plt.title('Car CH-1')

plt.subplot(142)
plt.imshow(hog_image, cmap='gray')
plt.title('Car CH-1 HOG')

plt.subplot(143)
plt.imshow(non_car_image_converted[:, :, 1], cmap='gray')
plt.title('Non-car CH-1')

plt.subplot(144)
plt.imshow(non_car_hog_image, cmap='gray')
plt.title('Non-car CH-1 HOG')

plt.savefig('output_images/my_HOG_example.png', bbox_inches='tight')
