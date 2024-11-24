import csv
from datetime import datetime
from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import time
import speech_recognition as sr
import PyPDF2
import mediapipe as mp

# Object Detection Function
def detect_phone():
    np.set_printoptions(suppress=True)
    model = load_model("keras_Model.h5", compile=False)
    class_names = [line.strip() for line in open("labels.txt", "r").readlines()]
    camera = cv2.VideoCapture(0)
    start_time = time.time()

    while True:
        ret, image = camera.read()
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        image = (image / 127.5) - 1
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        if class_name == class_names[1] and confidence_score > 0.7:
            camera.release()
            cv2.destroyAllWindows()
            return 1, confidence_score  # Phone detected

        if time.time() - start_time > 10:
            camera.release()
            cv2.destroyAllWindows()
            return 0, 0  # No phone detected within 10 seconds

        if cv2.waitKey(1) == 27:  # ESC key to exit
            camera.release()
            cv2.destroyAllWindows()
            return 0, 0

# Mouth Detection Function
def detect_mouth_open_once():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
    cap = cv2.VideoCapture(0)
    UPPER_LIP_TOP, LOWER_LIP_BOTTOM = 13, 14

    def euclidean_distance(point1, point2):
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width, _ = frame.shape
        result = face_mesh.process(rgb_frame)

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                upper_lip_top = face_landmarks.landmark[UPPER_LIP_TOP]
                lower_lip_bottom = face_landmarks.landmark[LOWER_LIP_BOTTOM]
                upper_lip_top = (int(upper_lip_top.x * frame_width), int(upper_lip_top.y * frame_height))
                lower_lip_bottom = (int(lower_lip_bottom.x * frame_width), int(lower_lip_bottom.y * frame_height))
                lip_distance = euclidean_distance(upper_lip_top, lower_lip_bottom)

                if lip_distance > 15:
                    cap.release()
                    cv2.destroyAllWindows()
                    return 1  # Mouth is open

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0  # Mouth is closed

# Audio Verification Functions
def read_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages])
        return text.lower()

def record_audio_and_convert_to_text(duration=10):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Recording audio...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.record(source, duration=duration)
        print("Converting audio to text...")
        try:
            return recognizer.recognize_google(audio).lower()
        except sr.UnknownValueError:
            return "Could not understand the audio"
        except sr.RequestError:
            return "Could not request results; check your internet connection"

def check_matching_words(pdf_text, spoken_text):
    pdf_words = set(pdf_text.split())
    spoken_words = set(spoken_text.split())
    matching_words = pdf_words & spoken_words
    if matching_words:
        return f"Copying detected: Matched words: {matching_words}"
    return "No matching words found"

# CSV Functionality
def log_to_csv(data, csv_file="cheating_log.csv"):
    headers = ["Date", "Time", "Cheating Detected", "Cheating Reason", "Speech Content", "Mouth Status", "Device Detected", "Confidence Score"]
    try:
        with open(csv_file, "a", newline="") as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(headers)  # Write headers only for a new file
            writer.writerow(data)
    except Exception as e:
        print(f"Error writing to CSV: {e}")

# Main Script Execution
if __name__ == "__main__":
    phone_detection_result, confidence_score = detect_phone()
    cheating_reason = ""
    speech_content = "N/A"

    mouth_detection_result = detect_mouth_open_once()
    mouth_status = "Open" if mouth_detection_result == 1 else "Closed"

    if mouth_detection_result == 1:
        cheating_reason = "Mouth open detected"
        pdf_path = r"C:\Users\pretam\Desktop\VS Code\PYTHON\Capstone\question paper.pdf"
        pdf_text = read_pdf(pdf_path)
        speech_content = record_audio_and_convert_to_text(duration=5)
        audio_verification_result = check_matching_words(pdf_text, speech_content)
        if "Copying detected" in audio_verification_result:
            cheating_reason += f" and {audio_verification_result}"
    else:
        if phone_detection_result == 1:
            cheating_reason = "Phone detected"
            mouth_status = "N/A"


    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    cheating_detected = "Yes" if cheating_reason else "No"
    data_row = [date, time_str, cheating_detected, cheating_reason, speech_content, mouth_status, "Yes" if phone_detection_result == 1 else "No", confidence_score]
    log_to_csv(data_row)
    print("Detection complete. Results logged to CSV.")
