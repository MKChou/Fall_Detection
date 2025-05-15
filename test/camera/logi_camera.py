import cv2

cap = cv2.VideoCapture(0)
#jetson nano/Logi = 0


ret , frame = cap.read()

while True:
    ret , frame = cap.read()
    if ret:
        cv2.imshow('LOGI(MK)', frame)
    else:
        break

    if cv2.waitKey(10) == ord('q'):
        break

