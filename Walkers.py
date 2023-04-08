import cv2

# Load the body classifier
body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')


# Open the video stream
cap = cv2.VideoCapture("walking.avi")

# Loop through each frame in the video stream
while True:
    # Read the frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect bodies in the frame
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)

    # Draw rectangles around the detected bodies
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    # Display the frame
    cv2.imshow('Human Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()
