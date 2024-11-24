import speech_recognition as sr
import PyPDF2

# Function to read text from PDF
def read_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        # Iterate over all pages in the PDF
        for page in reader.pages:
            text += page.extract_text()
        return text.lower()  # Return the text in lowercase for easy comparison

# Function to record audio and convert it to text
def record_audio_and_convert_to_text(duration=10):
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Recording for 10 seconds...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.record(source, duration=duration)
        print("Recording complete. Converting audio to text...")

        try:
            text = recognizer.recognize_google(audio)
            return text.lower()  # Convert to lowercase for easy comparison
        except sr.UnknownValueError:
            return "Could not understand the audio"
        except sr.RequestError:
            return "Could not request results; check your internet connection"

# Function to check if any word in the PDF matches the spoken words
def check_matching_words(pdf_text, spoken_text):
    # Split the PDF text and spoken text into words
    pdf_words = set(pdf_text.split())
    spoken_words = set(spoken_text.split())

    # Check for any matching words
    matching_words = pdf_words & spoken_words

    if matching_words:
        print("Copying: Words matched:", matching_words)
    else:
        print("Okay: No matching words found")

# Main function
def main(pdf_path):
    # Step 1: Read the text from the PDF
    pdf_text = read_pdf(pdf_path)

    # Step 2: Capture audio and convert it to text
    spoken_text = record_audio_and_convert_to_text()

    # Step 3: Check if any words from PDF match the spoken words
    check_matching_words(pdf_text, spoken_text)

# Example usage
pdf_path = "question paper.pdf"  # Replace with the path to your PDF
main(pdf_path)
