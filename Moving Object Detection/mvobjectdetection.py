import cv2
import time
import imutils

cam=cv2.VideoCapture(0)
time.sleep(1)

first_frame=None
area = 500
c=0

while True:

    ret, img = cam.read()
    text = 'Normal'
    img = imutils.resize(img, width=800)

    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # cv2.imshow("img",grayImg)
    gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0)
            # cv2.imshow("img", gaussianImg)

    if first_frame is None:
        first_frame = gaussianImg
                # cv2.imshow("img",first_frame)
        continue
    imgDiff = cv2.absdiff(first_frame, gaussianImg)
            # cv2.imshow("img",imgDiff)
    threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1]
            # cv2.imshow("img",threshImg)
    threshImg = cv2.dilate(threshImg, None, iterations=2)
            # cv2.imshow("img",threshImg)

    cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)



    for c in cnts:
        if cv2.contourArea(c) < area:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        x = x+1
        text = f"Moving object detected{x}"

    print(text)
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


    cv2.imshow("hello",img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
