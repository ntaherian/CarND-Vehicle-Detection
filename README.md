**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction with a color transform and append binned color features on a labeled training set of images and train a classifier Linear SVM classifier

* Implement a sliding-window technique and use trained classifier to search for vehicles in images.

* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.

* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4


---
### Histogram of Oriented Gradients (HOG)

The code for this step is contained in the 4th code cell of the IPython notebook the file called `Vehicle-Detection.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images in the 2nd code cell.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Final choice of HOG parameters.

I tried various combinations of parameters and in the end I decided to use `All` HOG channels (meaning using all channels of the image to extract hog features) with Histogram features and Spatial features.   

#### 3. training a classifier using selected HOG features (and color features)

The code for this part is in the 6th cell of of the IPython notebook the file called `Vehicle-Detection.py`). The helper functions used for this project are all contained in `vehicle_detection.py`.

I trained a linear SVM by first creating an array stack of feature vectors. Then normalizing the data. Next I split up data into randomized training and test sets and then trained data using `LinearSVC()`.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for this part is contained in the 10th cell of the IPython notebook the file called `Vehicle-Detection.py`).  

The helper functions used for this part is contained in `vehicle_detection.py`.

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.

The `find_cars` only has to extract hog features once and then can be sub-sampled to get all of its overlaying windows. Each window is defined by a scaling factor where a scale of 1 would result in a window that's 8 x 8 cells then the overlap of each window is in terms of the cell distance. This means that a `cells_per_step = 2` would result in a search window overlap of 75%. Its possible to run this same function multiple times for different scale values to generate multiple-scaled search windows.

Here are some example images:

![alt text][image4]
---

### Video Implementation

Here's a [link to my video result](./project_video.mp4)

I used the method process_frame (see `continous_vehicle_detection.py`) to process each frame independently. I used a smoothing of 20 to get rid of false positives as a car keeps on being detected in consecutive frames.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]


---

### Discussion

The pipeline fails to detect cars appearing far from the car and also cars on the other side of the highway. I believe if the linear svm was trained on more data we could overcome these shortcomings as well



