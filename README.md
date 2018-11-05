# digit_recognition_opencv
Digit Recognition using OpenCV and Keras in Python with detailed description for beginner

<h3> Step 1 - Installing the required Libraries </h3>
<p>I have used Anaconda Spyder IDE for this project. Install the following libraries into your machine using Anaconda Command Prompt.</p>

<p><b>pip install opencv-python <br>
pip install tensorflow <br>
pip install keras</b></p>

<h3> Step 2 - Run model_build.py file </h3>
<p>This file creates model using <b>CNN (Convolutional Neural Network)</b>. Accuary is <b>99.86%</b>.I have worked with 30 epochs and 90 batch_size. It took approx. 1 hour for me to fit the model. You can always change the values to experiment with the accuracy of the model. At the end, I have serialize the model and converted into JSON format to save the model into local disk for easier access.</p>

<h3> Step 3 - Run webcam_digit.py file </h3>
<p>This file loads model from JSON. It is then compiled. Images are captured frame by frame from an active webcam. Captured image frames are passed through image processing. First, it is converted to GrayScale. Second, it is processed using GaussianBlur of OpenCV to remove unwanted noise from the images. Third, Images are converted to threshold images. Fourth- Images are converted into numpy array of images and then they are resized and reshaped into 28*28 shapes. And finally, labels (digit) of array of images from the model are predicted. </p>
