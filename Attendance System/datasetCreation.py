import cv2
import os


def create(name):
    dataset = "dataset"
    emp = "employees"

    path = os.path.join(dataset, emp)
    if not os.path.isdir(path):
        os.mkdir(path)

    (width, height) = (130, 100)
    alg = "haarcascade_frontalface_default.xml"
    haar_cascade = cv2.CascadeClassifier(alg)

    cam = cv2.VideoCapture(0)
    while True:
        ret, img = cam.read()
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = haar_cascade.detectMultiScale(grayimg, 1.3, 4)

        for (x, y, w, h) in face:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            faceonly = grayimg[y:y + h, x:x + w]
            resizeimg = cv2.resize(faceonly, (width, height))
            cv2.imwrite("%s/%s.jpg" % (path, name), resizeimg)

        cv2.imshow("Capturing face", img)
        key = cv2.waitKey(1)
        if key == "q":
            break

    cam.release()
    cv2.destroyAllWindows()
    print("Image Captured succssfully")
    print("Want to add more?")
    x = input()
    if x == "y":
        create()
    else:
        exit()


print("Welcome to employee photo dataset creation")
print("Enter the name of employee:")
name = input()
print("see in the camera")
create(name)
