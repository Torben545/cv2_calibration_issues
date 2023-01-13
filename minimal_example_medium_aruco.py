import json
import os
import glob
import pathlib

import numpy as np
import cv2



def save_coefficients(mtx, dist, path):
    '''Save the camera matrix and the distortion coefficients to given path/file.'''
    cv_file = cv2.FileStorage(path + '.cv2file', cv2.FILE_STORAGE_WRITE)
    cv_file.write('K', mtx)
    cv_file.write('D', dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()
    with open(path + '.json', 'w') as f:
        json.dump({'matrix': mtx.tolist(), 'distortion': dist.tolist()}, f)

def load_coefficients(path):
    '''Loads camera matrix and distortion coefficients.'''
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path +'.cv2file', cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode('K').mat()
    dist_matrix = cv_file.getNode('D').mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]


def calibrate_aruco(images_paths, marker_length, marker_separation, nx, ny):
    '''Apply camera calibration using aruco.
    The dimensions are in cm.
    '''
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_1000)
    arucoParams = cv2.aruco.DetectorParameters_create()
    board = cv2.aruco.GridBoard_create(nx, ny, marker_length, marker_separation, aruco_dict)

    counter, corners_list, id_list = [], [], []
    first = 0
    # Find the ArUco markers inside each image
    for img in images_paths:
        image = cv2.imread(str(img))
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(
            img_gray,
            aruco_dict,
            parameters=arucoParams
        )

        if first == 0:
            corners_list = corners
            id_list = ids
        else:
            corners_list = np.vstack((corners_list, corners))
            id_list = np.vstack((id_list,ids))
        first = first + 1
        counter.append(len(ids))

    counter = np.array(counter)
    # Actual calibration
    ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraAruco(
        corners_list,
        id_list,
        counter,
        board,
        img_gray.shape,
        None,
        None
    )
    return [ret, mtx, dist, rvecs, tvecs]


IMAGE_BATCH = 'ascand'

if IMAGE_BATCH == 'medium':
    # Dimensions in cm
    MARKER_LENGTH = 3
    MARKER_SEPARATION = 0.25
    CALIB_FILE = "calibration_aruco_medium"
    IMAGES_FORMAT = 'jpg'
    NX, NY = 5, 7
    images_paths = [os.path.join('resources', 'images_aruco_medium', 'image000.jpg')]
    target_paths = [os.path.join('resources', 'images_aruco_medium', 'image0000_undist.png')]

elif IMAGE_BATCH == 'ascand':
    # Dimensions in cm
    MARKER_LENGTH = 1
    MARKER_SEPARATION = 0.2
    CALIB_FILE = "calibration_aruco_medium"
    IMAGES_FORMAT = 'jpg'
    NX, NY = 12, 6
    images_paths = [os.path.join('resources', 'images_aruco_ascand', 'frame_000137.png'),
                    os.path.join('resources', 'images_aruco_ascand', 'frame_000156.png'),
                    os.path.join('resources', 'images_aruco_ascand', 'frame_000184.png')]
    target_paths = [os.path.join('resources', 'images_aruco_ascand', 'frame_000137_undist.png'),
                    os.path.join('resources', 'images_aruco_ascand', 'frame_000156_undist.png'),
                    os.path.join('resources', 'images_aruco_ascand', 'frame_000184_undist.png')]

else:
    raise ValueError("Unknown image batch {}".format(IMAGE_BATCH))

# Calibrate
ret, mtx, dist, rvecs, tvecs = calibrate_aruco(images_paths, MARKER_LENGTH, MARKER_SEPARATION, NX, NY)
# Save coefficients into a file
save_coefficients(mtx, dist, CALIB_FILE)


# Load coefficients
mtx, dist = load_coefficients(CALIB_FILE)

for image_path_in, image_path_out in zip(images_paths, target_paths):
    original = cv2.imread(image_path_in)
    dst = cv2.undistort(original, mtx, dist, None, None)
    cv2.imwrite(image_path_out, dst)