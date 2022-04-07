from cgi import print_arguments
import cv2
import os
import numpy as np
import mediapipe as mp
from sklearn.feature_extraction import img_to_graph
mpFaceDetection = mp.solutions.face_detection

# El PATH se cambia a Test o Train segun el conjunto que se desea preprocesar
path = "./Raw/Test"
directories = os.listdir(path)
print("Lista archivos:", directories)
for directory in directories:
    dirPath = path + "/" + directory
    IMAGES = []

    for file_name in os.listdir(dirPath):
        image_path = dirPath + "/" + file_name
        # For static images:
        IMAGES.append(image_path)

    with mpFaceDetection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as faceDetection:
        for idx, file in enumerate(IMAGES):
            img = cv2.imread(file)
            
            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            results = faceDetection.process(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            # Draw face detections of each face.
            if not results.detections:
                continue
            auxImg = img.copy()
            height, width, channels = img.shape
            for detection in results.detections:
                # mp_drawing.draw_detection(aux_image, detection)
                x = int(detection.location_data.relative_bounding_box.xmin * width)
                y = int(detection.location_data.relative_bounding_box.ymin * height)
                w = int(detection.location_data.relative_bounding_box.width * width)
                h = int(
                    detection.location_data.relative_bounding_box.height * height)

                rostro = np.array([])
                rostro = auxImg[y:y+h, x:x+w]

                if(rostro.size != 0):
                    rostro = cv2.resize(
                        rostro, (75, 75), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(dirPath.replace('./Raw/', './Preprocess/') +
                                '/' + str(idx+1) + '_' + directory + '.png', rostro)
                # else:
                #     rostro = cv2.resize(auxImg, (75, 75),
                #                         interpolation=cv2.INTER_CUBIC)
                #     rostro = cv2.cvtColor(rostro, cv2.COLOR_BGR2GRAY)
                #     cv2.imwrite(dirPath.replace('./Raw/', './Preprocess/') +
                #                 '/' + str(idx+1) + '_' + directory + '.png', rostro)

path = "./Raw/Train"
directories = os.listdir(path)
print("Lista archivos:", directories)
for directory in directories:
    dirPath = path + "/" + directory
    IMAGES = []

    for file_name in os.listdir(dirPath):
        image_path = dirPath + "/" + file_name
        # For static images:
        IMAGES.append(image_path)

    with mpFaceDetection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as faceDetection:
        for idx, file in enumerate(IMAGES):
            img = cv2.imread(file)
            
            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            results = faceDetection.process(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            # Draw face detections of each face.
            if not results.detections:
                continue
            auxImg = img.copy()
            height, width, channels = img.shape
            for detection in results.detections:
                # mp_drawing.draw_detection(aux_image, detection)
                x = int(detection.location_data.relative_bounding_box.xmin * width)
                y = int(detection.location_data.relative_bounding_box.ymin * height)
                w = int(detection.location_data.relative_bounding_box.width * width)
                h = int(
                    detection.location_data.relative_bounding_box.height * height)

                rostro = np.array([])
                rostro = auxImg[y:y+h, x:x+w]

                if(rostro.size != 0):
                    rostro = cv2.resize(
                        rostro, (75, 75), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(dirPath.replace('./Raw/', './Preprocess/') +
                                '/' + str(idx+1) + '_' + directory + '.png', rostro)
