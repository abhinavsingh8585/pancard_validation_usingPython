import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import pytesseract
import re
from difflib import SequenceMatcher
from PIL import Image
from pyzbar.pyzbar import decode

# Load the trained H5 model

def validation(pan_card_image_path):
    model = load_model('validation_model.h5')

    # Define the input image size expected by the model
    input_size = (100, 100)  # Adjust to match your model's expected input size

    # Load and preprocess the PAN card image
    img = image.load_img(pan_card_image_path, target_size=input_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values to [0, 1]

    # Make a prediction using the loaded model
    prediction = model.predict(img_array)

    # Extract the prediction value from the array
    predicted_value = prediction[0][0]

    # Adjust the threshold based on your model's output
    threshold = 0.1

    # Interpret the prediction
    if predicted_value > threshold:
        result = "Valid PAN Card"
    else:
        result = "Invalid PAN Card"

    print("Prediction:", result)



def imageToWord(pan_card_image_path):
    # Load the image using OpenCV
    image = cv2.imread(pan_card_image_path)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    qcd = cv2.QRCodeDetector()

    retval, decoded_info, points, straight_qrcode = qcd.detectAndDecodeMulti(gray_image)
    print(retval, decoded_info)

    _, threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    extracted_text = pytesseract.image_to_string(threshold_image)
    # print(extracted_text) 

    # Regular expression pattern to match a valid PAN Card number
    pan_pattern = re.compile(r"[A-Z]{5}[0-9]{4}[A-Z]{1}")
    
    # Regular expression pattern to match the date of birth in the format DD/MM/YYYY
    dob_pattern = re.compile(r"\d{2}/\d{2}/\d{4}")
    
    pan_number = pan_pattern.search(extracted_text)
    print('pan number', pan_number)
    dob = dob_pattern.search(extracted_text)
    # print('dob', dob)
    if pan_number and dob:
        # if similarity_ratio >= similarity_threshold:
            return pan_number.group(0), dob.group(0)
    
    return "Invalid alert"

if __name__ == "__main__":
    
    pan_card_image_path = './images/real/test.png'
    validation(pan_card_image_path)
    # for image_path in image_paths:
    result = imageToWord(pan_card_image_path)
    print(f"Image: {pan_card_image_path}\nResult: {result}\n")




