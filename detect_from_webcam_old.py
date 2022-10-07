import numpy as np
import cv2
import face_recognition

# Load images
image_dir = "./input_faces/"
obama_image = cv2.imread(image_dir + 'barack-obama-1.jpg')
biden_image = cv2.imread(image_dir + 'joe-biden-1.jpg')
unkown_image = cv2.imread(image_dir + 'biden-obama-1.jpeg')

# Encode known faces
obama_image_encoding = face_recognition.face_encodings(obama_image)[0]
biden_image_encoding = face_recognition.face_encodings(biden_image)[0]

# Create arrays of known encoded faces and names
known_face_encodings = [
  obama_image_encoding, 
  biden_image_encoding
]
known_face_names = [
  "Barack Obama",
  "Joe Biden"
]

face_names = []

# Start webcam, run facial rec every n frame
vid = cv2.VideoCapture(0)
n = 1
count = 0

while(True):

  if count % n == 0:

    # Capture vid frame by frame
    ret, frame = vid.read()

    # Find location of faces in frame, encode those
    unknown_face_locations = face_recognition.face_locations(frame)
    unkown_image_encoding = face_recognition.face_encodings(frame, unknown_face_locations)

    # Iterate through encoded faces in unkown image
    for face_encoding in unkown_image_encoding:

      # Find if face is a match to known faces
      matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
      name = "unkown"

      # Find known face with shortest distance to unkown one
      face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
      best_match_index = np.argmin(face_distances)
      if matches[best_match_index]:
        name = known_face_names[best_match_index]

      # Append name of face (either found or unkown) to array
      face_names.append(name)

      # Draw name and bounding boxes onto image
      for (top, right, bottom, left), name in zip(unknown_face_locations, face_names):

        # Draw bounding box around face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw label with name below corrosponding face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display image w/ bounding boxes over faces
    cv2.imshow("image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

# After the loop release the cap object
vid.release()
cv2.destroyAllWindows()