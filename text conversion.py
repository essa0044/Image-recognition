# Very important!!! Use tensorFlow Version 2.12.

import cv2
import easyocr
import numpy as np
import matplotlib.pyplot as plt
from CurrencyConverter import CurrencyConverter
import re

# Function to convert detected string to float (handles errors)
def convert_string_to_float(text: str):
    # Remove all non-numeric characters except for commas and dots
    pattern = re.compile(r'[^0-9.,]')
    text = pattern.sub('', text)

    # Handle case when the resulting text is empty or invalid
    if not text:
        raise ValueError("No numeric value found in the detected text")

    text = text.replace(",", ".")  # Replace commas with dots for float conversion

    try:
        amount = float(text)  # Convert cleaned text to float
    except ValueError:
        raise ValueError(f"Unable to convert '{text}' to float")

    return amount


# Initialize the webcam (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press Enter to capture an image or 'q' to quit.")

while True:
    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Display the frame in a window
    cv2.imshow('Camera', frame)

    # Check if the user presses "Enter" (13 is the Enter key) or 'q' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == 13:  # ASCII code for Enter key
        print("Image captured!")
        break
    elif key == ord('q'):
        print("Exiting without capturing.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

# Convert the captured image to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Apply thresholding
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Optionally apply some noise reduction
thresh = cv2.medianBlur(thresh, 3)

# Save the preprocessed image (optional)
cv2.imwrite('preprocessed_image.jpg', thresh)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # Specify the language(s) you want to recognize

# Perform OCR using EasyOCR
results = reader.readtext(thresh)

# Display the text detected by EasyOCR
print("Detected text:")
detected_text = ""  # Initialize a variable to store the detected text
for (bbox, text, prob) in results:
    print(f'{text} (Confidence: {prob:.2f})')
    detected_text += text + " "  # Concatenate all detected text

# Display the detected text before conversion
print(f"Detected text before conversion: {detected_text}")

# Attempt to convert detected text to float
try:
    amount = convert_string_to_float(detected_text)
except ValueError as e:
    print(f"Error: {e}")
    exit()

# Now continue with currency conversion
currency1 = 'eur'
currency2 = 'jpy'

cc = CurrencyConverter(currency1, currency2)
currency = cc.amount_in_foreign_currency(amount)

print(f"Detected: {amount} {currency1.upper()}")
print(f"Converted: {currency:.2f} {currency2.upper()}")

# Display only the preprocessed image (thresh) using matplotlib
plt.figure(figsize=(8, 6))

plt.imshow(thresh, cmap='gray')
plt.title('Preprocessed Image')
plt.axis('off')

# Add detected and converted currency information as separate strings below the image
plt.text(60, thresh.shape[0] + 50, f"Detected: {amount} {currency1.upper()}",
         horizontalalignment='left', verticalalignment='center', fontsize=36, color='black')
plt.text(60, thresh.shape[0] + 120, f"Converted: {currency:.2f} {currency2.upper()}",
         horizontalalignment='left', verticalalignment='center', fontsize=36, color='black')

plt.tight_layout()
plt.show()
