import os

import cv2
import numpy as np

# https://medium.com/vacatronics/3-ways-to-calibrate-your-camera-using-opencv-and-python-395528a51615

# The images in the images_charuco directory are from the tutorial on medium.com
# The resulting image at undistorted.png is clearly broken and does not match the result target.png that was achieved in the medium.com article.

def save_coefficients(mtx, dist, path):
    '''Save the camera matrix and the distortion coefficients to given path/file.'''
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write('K', mtx)
    cv_file.write('D', dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()

def load_coefficients(path):
    '''Loads camera matrix and distortion coefficients.'''
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode('K').mat()
    dist_matrix = cv_file.getNode('D').mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]


def calibrate_charuco(img_list, marker_length, square_length, nx, ny):
    '''Apply camera calibration using aruco.
    The dimensions are in cm.
    '''
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_1000)
    board = cv2.aruco.CharucoBoard_create(nx, ny, square_length, marker_length, aruco_dict)
    arucoParams = cv2.aruco.DetectorParameters_create()

    counter, corners_list, id_list = [], [], []
    first = 0
    # Find the ArUco markers inside each image
    for img in img_list:
        print(f'using image {img}')
        image = cv2.imread(img)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(img_gray, aruco_dict, parameters=arucoParams)

        resp, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(markerCorners=corners, markerIds=ids, image=img_gray, board=board)
        # If a Charuco board was found, let's collect image/corner points
        # Requiring at least 20 squares
        if resp > 20:
            # Add these corners and ids to our calibration arrays
            corners_list.append(charuco_corners)
            id_list.append(charuco_ids)

    # Actual calibration
    ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(charucoCorners=corners_list, charucoIds=id_list, board=board,
                                                                imageSize=img_gray.shape, cameraMatrix=None,distCoeffs=None)

    return ret, mtx, dist, rvecs, tvecs


MARKER_LENGTH = 2.4
SQUARE_LENGTH = 3.2
NX = 5
NY = 7

img_list = [os.path.join('resources', 'images_charuco', 'distorted.png')]

ret, mtx, dist, rvecs, tvecs = calibrate_charuco(img_list, MARKER_LENGTH, SQUARE_LENGTH, NX, NY)
# Save coefficients into a file
save_coefficients(mtx, dist, "calibration_charuco.json")

# Load coefficients
# mtx, dist = load_coefficients('calibration_charuco.json')
original = cv2.imread(os.path.join('resources', 'images_charuco', 'distorted.png'))
dst = cv2.undistort(original, mtx, dist, None, mtx)
cv2.imwrite(os.path.join('resources', 'images_charuco', 'undistorted.png'), dst)
