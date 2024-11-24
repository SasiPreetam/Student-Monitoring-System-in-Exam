import cv2
import mediapipe as mp

# Initialize Mediapipe face mesh and drawing utilities
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Define the indices for mouth landmarks
UPPER_LIP_TOP = 13
LOWER_LIP_BOTTOM = 14

# Function to calculate the Euclidean distance between two points
def euclidean_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

# Function to detect mouth open status
def detect_mouth_open_once(model_path=""):
    """
    Function to detect if a person's mouth is open using Mediapipe.
    Returns 1 if the mouth is open, and 0 if it is closed.
    Once an open mouth is detected, it stops further detection.
    
    Parameters:
    - model_path: Optional parameter for model loading if additional model-based processing is needed.
    
    Returns:
    - int: 1 if the mouth is open, 0 if closed.
    """
    
    # Initialize Mediapipe face mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    mouth_open_warning_sent = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the image to RGB as Mediapipe uses RGB images
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get the frame dimensions
        frame_height, frame_width, _ = frame.shape
        
        # Process the frame and detect face landmarks
        result = face_mesh.process(rgb_frame)
        
        if result.multi_face_landmarks and not mouth_open_warning_sent:
            for face_landmarks in result.multi_face_landmarks:
                # Get normalized coordinates of the mouth landmarks
                upper_lip_top = face_landmarks.landmark[UPPER_LIP_TOP]
                lower_lip_bottom = face_landmarks.landmark[LOWER_LIP_BOTTOM]
                
                # Convert to pixel coordinates
                upper_lip_top = (int(upper_lip_top.x * frame_width), int(upper_lip_top.y * frame_height))
                lower_lip_bottom = (int(lower_lip_bottom.x * frame_width), int(lower_lip_bottom.y * frame_height))
                
                # Calculate the distance between the upper and lower lip
                lip_distance = euclidean_distance(upper_lip_top, lower_lip_bottom)
                
                # Check if the lip distance exceeds the threshold
                if lip_distance > 15:  # Adjust this threshold as needed
                    print("Warning: Mouth is open")
                    mouth_open_warning_sent = True  # Send warning only once
                    cap.release()
                    cv2.destroyAllWindows()
                    return 1  # Mouth is open, return 1
                
                # Optional: Display feedback on the frame (for debugging)
                cv2.putText(frame, "No Issue", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Draw facial landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                )
        
        # Display the result for debugging
        cv2.imshow('Mouth Detection', frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0  # Mouth is closed, return 0

# Example usage
mouth_status = detect_mouth_open_once()
print("Mouth Status:", mouth_status)
