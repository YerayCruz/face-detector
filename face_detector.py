import cv2

a = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

b = cv2.VideoCapture(0) ## Access to camera

while True:
    c, d = b.read() ##c references the rectangle, and d the image
    e = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    f = a.detectMultiScale(e, 1.3, 6)

    for (x1, y1, w1, h1) in f:
        cv2.rectangle(d, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 5)

    cv2.imshow("img", d)
    h = cv2.waitKey(40) & 0xff
    if h == 40:
        break

b .release()
cv2.destroyAllWindows()
