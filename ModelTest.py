import cv2
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import time

#Abriendo modelos
inicio = time.time()
print("Abriendo LBPHFaceRecognizer...")
lbphModel = cv2.face.LBPHFaceRecognizer_create()
lbphModel.read("./Models/LBPHmodel.xml")
fin = time.time()
print("Tiempo de apertura: " + (str)(fin-inicio))
print()

inicio = time.time()
print("Abriendo EigenFaceRecognizer...")
eigenModel = cv2.face.EigenFaceRecognizer_create()
eigenModel.read("./Models/Eigenmodel.xml")
fin = time.time()
print("Tiempo de apertura: " + (str)(fin-inicio))
print()

inicio = time.time()
print("Abriendo FisherFaceRecognizer...")
fisherModel = cv2.face.FisherFaceRecognizer_create()
fisherModel.read("./Models/Fishermodel.xml")
fin = time.time()
print("Tiempo de apertura: " + (str)(fin-inicio))
print()


#Iniciamos vectores
LABELS = ["Mask", "NoMask"]
y_pred_lbph = []
y_pred_eigen = []
y_pred_fisher = []
y_true = []

#Iniciamos contadores
timeLBPH = 0
timeEigen = 0
timeFisher = 0
cont = 0

path = "./Preprocess/Test"
directories = os.listdir(path)
for directory in directories:
    dirPath = path + "/" + directory

    for file_name in os.listdir(dirPath):
        cont = cont + 1
        if('NoMask' in file_name):
            y_true.append('NoMask')
        else:
            y_true.append('Mask')

        imgPath = dirPath + "/" + file_name
        rostro = cv2.imread(imgPath)
        rostro = cv2.cvtColor(rostro, cv2.COLOR_BGR2GRAY)

        #LBPH
        inicio = time.time()
        result = lbphModel.predict(rostro)
        fin = time.time()
        timeLBPH = timeLBPH + (fin-inicio)
        if (result[0] == 0):
            y_pred_lbph.append('Mask')
        else:
            y_pred_lbph.append('NoMask')

        #Eigen
        inicio = time.time()
        result = eigenModel.predict(rostro)
        fin = time.time()
        timeEigen = timeEigen + (fin-inicio)
        if (result[0] == 0):
            y_pred_eigen.append('Mask')
        else:
            y_pred_eigen.append('NoMask')
        
        #Fisher
        inicio = time.time()
        result = fisherModel.predict(rostro)
        fin = time.time()
        timeFisher = timeFisher + (fin-inicio)
        if (result[0] == 0):
            y_pred_fisher.append('Mask')
        else:
            y_pred_fisher.append('NoMask')

#Tiempos
print("Tiempos de prediccion promedio para " + str(cont) + " imagenes: ")
print('LBPH: '+str(timeLBPH/cont))
print('EigenFace: ' +str(timeEigen/cont))
print('FisherFace: ' +str(timeFisher/cont))
print()

#LBPH
cm = confusion_matrix(y_true, y_pred_lbph, labels=LABELS)
print(classification_report(y_true, y_pred_lbph, target_names=LABELS))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
disp.plot()
plt.title('LBPH')
plt.show()

#Eigen
cm = confusion_matrix(y_true, y_pred_eigen, labels=LABELS)
print(classification_report(y_true, y_pred_eigen, target_names=LABELS))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
disp.plot()
plt.title('EigenFace')
plt.show()

#Fisher
cm = confusion_matrix(y_true, y_pred_fisher, labels=LABELS)
print(classification_report(y_true, y_pred_fisher, target_names=LABELS))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
disp.plot()
plt.title('FisherFace')
plt.show()