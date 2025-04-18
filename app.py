import streamlit as st
import numpy as np
import cv2
import pandas as pd
import math
import easyocr
import re
from ultralytics import YOLO
import playsound
import time
import threading

last_sound_time = 0

detected_log = {}

# Load YOLO models
model_clothes = YOLO(r"model\best.pt")
model_idcard = YOLO(r"model\id_card.pt")

# Load OCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Define the regular expression pattern for extracting text
pattern = re.compile("[0-9][0-9][A-Z]{3}[0-9]{4}")

# Functions for detecting objects
def id_card_detect(frame, box):
    x1, y1, x2, y2 = box.xyxy[0]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

    # Add black rectangle for text background
    text = "ID Card"
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 0, 0), -1)

    # Add white text
    cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def clothes_detect(frame, box, cls):
    x1, y1, x2, y2 = box.xyxy[0]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Determine the class name ("Tshirt" or "Shirt")
    class_name = "Shirt" if cls == 0 else "Tshirt"

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Add black rectangle for text background
    (text_width, text_height), baseline = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 0, 0), -1)

    # Add white text
    cv2.putText(frame, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def load_student_ids(csv_path='student_ids.csv'):
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        # Convert the 'ID' column to a list
        return df['ID'].tolist()
    except FileNotFoundError:
        st.error("Student ID database file not found.")
        return []
    except Exception as e:
        st.error(f"An error occurred while loading the database: {e}")
        return []

def ocr_operation(frame):
    """
    Perform OCR on the frame and check if the extracted registration number is in the database.
    Returns True if the registration number is found in the database, otherwise False.
    """
    global detected_log
    student_ids = load_student_ids()  # Load student IDs dynamically
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    results = reader.readtext(np.array(gray_frame))  # Perform OCR

    current_time = time.time()
    for result in results:
        text = result[1]  # Extract the OCR text
        match = pattern.search(text)  # Search for the pattern
        if match:
            registration_number = match.group()
            # Skip if the registration number is already in the log
            if registration_number in detected_log:
                continue

            # Add the registration number to the log
            detected_log[registration_number] = current_time

            # Check if the registration number is in the database
            if registration_number in student_ids:
                st.success(f"Registration Number {registration_number} is in the Database")
                return True  # Compliant
            else:
                st.error(f"Registration Number {registration_number} is NOT in the Database")
                return False  # Non-compliant

    return False  # No valid registration number detected

def play_sound_with_cooldown(sound_file, cooldown=5):
    """
    Play a sound file with a cooldown period to avoid repeated playback.
    """
    global last_sound_time
    current_time = time.time()
    if current_time - last_sound_time >= cooldown:
        threading.Thread(target=playsound.playsound, args=(sound_file,), daemon=True).start()
        last_sound_time = current_time

def check_compliance(frame, id_card_detected):
    """
    Check compliance and display the result on the frame.
    Play a sound based on compliance status with a cooldown for non-compliance.
    """
    if id_card_detected:
        # Uniform is in compliance
        compliance_message = "Uniform is in Compliance"
        color = (0, 255, 0)  # Green
        play_sound_with_cooldown('compliance.mp3', cooldown=5)  # Play compliance sound immediately
    else:
        # Non-compliance detected
        compliance_message = "Non-compliant Uniform Detected"
        color = (0, 0, 255)  # Red
        play_sound_with_cooldown('noncompliance.mp3', cooldown=5)  # Play non-compliance sound with cooldown

    # Display the compliance message on the frame
    (text_width, text_height), baseline = cv2.getTextSize(compliance_message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_x = 10  # Position the text at the top-left corner
    text_y = 50
    cv2.rectangle(frame, (text_x - 5, text_y - text_height - 5), (text_x + text_width + 5, text_y + 5), (0, 0, 0), -1)
    cv2.putText(frame, compliance_message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

def clean_detected_log():
    """
    Remove registration numbers from the log if they were detected more than 5 seconds ago.
    """
    global detected_log
    current_time = time.time()
    keys_to_remove = [key for key, timestamp in detected_log.items() if current_time - timestamp > 5]
    for key in keys_to_remove:
        del detected_log[key]

# Streamlit UI
st.title('Real-time Object Detection and Text Extraction')

streaming = st.empty()

# Button to start webcam stream
if streaming.button('Start Webcam Stream', key='start_stream'):
    stop_stream = st.button('Stop Stream', key='stop_stream_1')
    try:
        cap = cv2.VideoCapture(0)
        while True:
            if stop_stream:
                break
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame.")
                break

            # Perform detection for clothes
            results_clothes = model_clothes(frame)

            # Perform detection for ID cards
            results_id_card = model_idcard(frame)

            # Run OCR operation and check compliance
            ocr_compliant = ocr_operation(frame)

            # Process results for clothes
            for result in results_clothes:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls in [0, 1]:  # Assuming 0 = Tshirt, 1 = Shirt
                        clothes_detect(frame, box, cls)

            # Process results for ID card
            id_card_detected = False
            for result in results_id_card:
                boxes = result.boxes
                for box in boxes:
                    id_card_detected = True  # Set flag to True if an ID card is detected
                    id_card_detect(frame, box)

            # Final compliance check: ID card detected OR registration number is in the database
            is_compliant = id_card_detected or ocr_compliant

            # Check compliance and display the result on the frame
            check_compliance(frame, is_compliant)

            # Clean the detected log to remove entries older than 5 seconds
            clean_detected_log()

            # Display the frame
            streaming.image(frame, channels="BGR")

            # Limit frame rate to reduce CPU/GPU usage
            time.sleep(0.03)

    except Exception as e:
        st.error(f"An error occurred: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()