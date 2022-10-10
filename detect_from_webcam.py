from genericpath import isfile
import numpy as np
import cv2
import face_recognition
import os

# Load trained images
trained_path = "./trained_images/"
trained_images = []
trained_images_names = []

# Path to place unknown faces
unknown_path = "./unknown_faces/"

# Loop through all image paths in trained images folder
for image_path in os.listdir(trained_path):

  # Create full input path and read then encode image
  input_path = os.path.join(trained_path, image_path)
  image = cv2.imread(input_path)
  encoded = face_recognition.face_encodings(image)[0]
  trained_images.append(encoded)

  # Add name to face
  name = os.path.splitext(image_path)[0]
  trained_images_names.append(name)


# Start webcam, run facial rec every n frame
vid = cv2.VideoCapture(0)
n = 1
frame_count = 0
unknown_count = 1

while(True):

  if frame_count % n == 0:
    face_names = []

    # Capture vid frame by frame
    ret, frame = vid.read()

    # Find location of faces in frame, encode those
    unknown_face_locations = face_recognition.face_locations(frame)
    unknown_image_encoding = face_recognition.face_encodings(frame, unknown_face_locations)

    # Iterate through encoded faces in unknown image
    for face_encoding in unknown_image_encoding:

      # Find if face is a match to known faces
      matches = face_recognition.compare_faces(trained_images, face_encoding)
      name = "unknown"

      # Find known face with shortest distance to unknown one
      face_distances = face_recognition.face_distance(trained_images, face_encoding)
      best_match_index = np.argmin(face_distances)
      if matches[best_match_index]:
        name = trained_images_names[best_match_index]

      # Append name of face (either found or unknown) to array
      face_names.append(name)

      # Draw name and bounding boxes onto image
      for (top, right, bottom, left), name in zip(unknown_face_locations, face_names):

        if name == "unknown":

          # Crop image to just face
          crop_face = frame[top: bottom, left: right]

          # Encode unknown face
          unknown_face_encoded = face_recognition.face_encodings(crop_face)

          # If high enough conf there is a face, grab first face in array
          if unknown_face_encoded:

            unknown_face_encoded = unknown_face_encoded[0]

            # Get file name of unknown face
            name = name + str(unknown_count)
            unknown_face_name = name + '.jpg'
            unknown_count += 1

            # Encode face
            trained_images.append(unknown_face_encoded)
            trained_images_names.append(unknown_face_name)

            # Save face to file system
            unknown_face_path = os.path.join(unknown_path, unknown_face_name)
            cv2.imwrite(unknown_face_path, frame)

        # Draw bounding box around face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw label with name below corrosponding face
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (0, 255, 0), 1)

    # Display image w/ bounding boxes over faces
    cv2.imshow("image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

# After the loop release the cap object
vid.release()
cv2.destroyAllWindows()