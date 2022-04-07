import cv2
import os
import numpy as np
import time

path = "./Preprocess/Train"
directories = os.listdir(path)
print("Lista archivos:", directories)
labels = []
rostros = []
label = 0
for directory in directories:
    dirPath = path + "/" + directory

    for file in os.listdir(dirPath):
        imgPath = dirPath + "/" + file
        image = cv2.imread(imgPath, 0)
        rostros.append(image)
        if('NoMask' in file):
            labels.append(1)
        else:
            labels.append(0)


print("Mask: ", np.count_nonzero(np.array(labels) == 0))
print("NoMask: ", np.count_nonzero(np.array(labels) == 1))
print()

# LBPHFaceRecognizer
inicio = time.time()
faceMaskModel = cv2.face.LBPHFaceRecognizer_create()
# Entrenamiento
print("Entrenando LBPHFaceRecognizer...")
faceMaskModel.train(rostros, np.array(labels))
# Almacenar modelo
faceMaskModel.write("./Models/LBPHmodel.xml")
fin = time.time()
print("Tiempo de entrenamiento: " + (str)(fin-inicio))
print()


# EigenFaceRecognizer
inicio = time.time()
faceMaskModel = cv2.face.EigenFaceRecognizer_create()
# Entrenamiento
print("Entrenando EigenFaceRecognizer...")
faceMaskModel.train(rostros, np.array(labels))
# Almacenar modelo
faceMaskModel.write("./Models/Eigenmodel.xml")
fin = time.time()
print("Tiempo de entrenamiento: " + (str)(fin-inicio))
print()


# FisherFaceRecognizer
inicio = time.time()
faceMaskModel = cv2.face.FisherFaceRecognizer_create()
# Entrenamiento
print("Entrenando FisherFaceRecognizer...")
faceMaskModel.train(rostros, np.array(labels))
# Almacenar modelo
faceMaskModel.write("./Models/Fishermodel.xml")
fin = time.time()
print("Tiempo de entrenamiento: " + (str)(fin-inicio))
print()


print("Modelos almacenados")
