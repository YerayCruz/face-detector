import cv2

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Initialize the video capture from default camera (index 0)
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    # Convert the frame to grayscale (face detection works on grayscale)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=6)

    # Draw rectangles around detected faces
    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 5)

    # Display the resulting frame
    cv2.imshow("Face Detection", frame)
    
    # Exit condition: press 'Esc' key (ASCII 27)
    key = cv2.waitKey(40) & 0xff
    if key == 27:
        break

# Release the capture and destroy windows
video_capture.release()
cv2.destroyAllWindows()
