from unittest import result
import cv2

faceMaskModel = cv2.face.LBPHFaceRecognizer_create()
faceMaskModel.read("./Models/LBPHmodel.xml")

img = cv2.imread("1_Mask.png")
rostro = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
result = faceMaskModel.predict(rostro)
print("1_Mask.png: " + str(result))

img = cv2.imread("1_NoMask.png")
rostro = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
result = faceMaskModel.predict(rostro)
print("1_NoMask.png: " + str(result))
