import cv2
import mediapipe as mp
import warnings

# Suppress the UserWarning related to deprecated SymbolDatabase.GetPrototype
warnings.filterwarnings("ignore", category=UserWarning, module='mediapipe')

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=5)  # Adjust max_num_faces for multiple face detection

# Initialize MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Capture video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and find the facemesh
    results = face_mesh.process(rgb_frame)

    # Check if faces are detected
    if results.multi_face_landmarks:
        # Loop through each detected face
        for idx, face_landmarks in enumerate(results.multi_face_landmarks):
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())

            # Display "Face Detected" with a label for each face
            cv2.putText(frame, f"Face {idx + 1} Detected", (50, 50 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        # Display "No Face Detected" if no face is found
        cv2.putText(frame, "No Face Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('FaceMesh', frame)

    # Break the loop with 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
