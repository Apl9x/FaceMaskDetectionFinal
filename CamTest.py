import cv2
import mediapipe as mp
mpFaceDetection = mp.solutions.face_detection
camera = cv2.VideoCapture(0)
# Leer el modelos
lbphModel = cv2.face.LBPHFaceRecognizer_create()
lbphModel.read("./Models/LBPHmodel.xml")
eigenModel = cv2.face.EigenFaceRecognizer_create()
eigenModel.read("./Models/Eigenmodel.xml")
fisherModel = cv2.face.FisherFaceRecognizer_create()
fisherModel.read("./Models/Fishermodel.xml")
LABELS = ["Mask", "NoMask"]
#LBPH por defecto
modelUsed = lbphModel
modelLabel = 'LBPH'
with mpFaceDetection.FaceDetection(
        min_detection_confidence=0.5) as face_detection:
    while True:
        ret, frame = camera.read()
        if ret == False:
            break
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)
        if results.detections is not None:
            for detection in results.detections:
                xmin = int(
                    detection.location_data.relative_bounding_box.xmin * width)
                ymin = int(
                    detection.location_data.relative_bounding_box.ymin * height)
                w = int(detection.location_data.relative_bounding_box.width * width)
                h = int(detection.location_data.relative_bounding_box.height * height)
                if xmin < 0 and ymin < 0:
                    continue
                #cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), (0, 255, 0), 5)
                rostro = frame[ymin: ymin + h, xmin: xmin + w]
                if(rostro.size != 0):
                    rostroGrises = cv2.cvtColor(
                        rostro, cv2.COLOR_BGR2GRAY)
                    rostro = cv2.resize(
                        rostroGrises, (75, 75), interpolation=cv2.INTER_CUBIC)

                    result = modelUsed.predict(rostro)
                    print(result)
                    cv2.putText(frame, modelLabel, (xmin, ymin - 40), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)
                    # if result[1] < 150:
                    color = (0, 255, 0) if LABELS[result[0]] == "Mask" else (0, 0, 255)
                    cv2.putText(frame, "{}".format(
                        LABELS[result[0]]), (xmin, ymin - 15), 2, 1, color, 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (xmin, ymin),
                                    (xmin + w, ymin + h), color, 2)

        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1)
        if k == 27:
            break
        else:
            if k==49:
                modelUsed = lbphModel
                modelLabel = 'LBPH'
                print(modelUsed)
            else:
                if k==50:
                    modelUsed = eigenModel
                    modelLabel = 'EigenFace'
                    print(modelUsed)
                else:
                    if k==51:
                        modelUsed = fisherModel
                        modelLabel = 'FisherFace'
                        print(modelUsed)
camera.release()
cv2.destroyAllWindows()
