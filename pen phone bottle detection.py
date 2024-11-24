from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import time

def detect_phone():
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("keras_Model.h5", compile=False)

    # Load the labels
    class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

    # CAMERA can be 0 or 1 based on default camera of your computer
    camera = cv2.VideoCapture(0)

    # Start a timer to limit detection duration to 10 seconds
    start_time = time.time()

    while True:
        # Grab the webcam image
        ret, image = camera.read()

        # Resize the raw image to (224, 224) pixels
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

        # Show the image in a window
        cv2.imshow("Webcam Image", image)

        # Prepare the image as a numpy array and reshape it for the model's input shape
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image array
        image = (image / 127.5) - 1

        # Make a prediction
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Check if the detected object is a phone and confidence is greater than 0.5
        if class_name == class_names[1] and confidence_score > 0.5:
            camera.release()
            cv2.destroyAllWindows()
            return 1  # Return 1 if phone is detected

        # Check if 10 seconds have passed since starting the loop
        elapsed_time = time.time() - start_time
        if elapsed_time > 10:
            camera.release()
            cv2.destroyAllWindows()
            return 0  # Return 0 if no phone was detected within 10 seconds

        # Listen to the keyboard for key presses
        keyboard_input = cv2.waitKey(1)

        # 27 is the ASCII code for the esc key on your keyboard
        if keyboard_input == 27:
            print("Program exited by user.")
            camera.release()
            cv2.destroyAllWindows()
            return 0  # Return 0 if user exits with ESC key

# Example of how to use the function
result = detect_phone()
# Now you can use `result` as 1 if a phone is found or 0 if not, without printing.
print(result)