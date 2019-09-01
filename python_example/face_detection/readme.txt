# Face detection
This is a python version of a face detection, using NumPy, OpenCV and the caffe model contains in
res10_300x300_ssd_iter_140000.caffemodel

It also uses the DNN API of OpenCV.

A picture can be passed as first argument. If no argument is pass, the webcam is used.

detect_face.py is used to detect face in a picture and has to been executed as follow:
python3 detect_faces.py --image <img>

detect_faces_video.py is used to detect face in a video using the webcam and has to been executed as follow:
python3 detect_faces_video.py


