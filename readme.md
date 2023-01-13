# Minimal examples
There is both a minimal example for the charuco and aruco board. Both are inspired by the article at 
https://medium.com/vacatronics/3-ways-to-calibrate-your-camera-using-opencv-and-python-395528a51615
and do not produce the desired result. 
### Charuco calibration example
We hake the image provided on the medium.com article, calibrate it and expect the target.png. Instead we get the undistorted.png which is messed up around the edges. 
### Aruco calibration example
Here we have two sets of images. The first on "images_aruco_medium" the example image from the medium.com article. While looking better, 
the resulting image still does not look as good as the target image which was achieved in the article. 

The second set of images are three images from our 3D-Scanner with the provisional calibration card on it. When using those images
for calibration, the results are also very distorted around the top and bottom images. 