import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
import whisper
import pyaudio
import numpy as np
import soundfile as sf
import tempfile
import json
from datetime import datetime
import spacy
from spacy.matcher import Matcher
import pyttsx3
import cv2
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
import re
import shutil
import requests


os.environ["PATH"] += os.pathsep + "C:\\Users\\priya\\Downloads\\ffmpeg-2024-08-07-git-94165d1b79-essentials_build\\bin"

# Load the Whisper model
model = whisper.load_model("small")

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Audio recording configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 1.5

def detect_silence(audio_data):
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    return np.mean(np.abs(audio_array)) < SILENCE_THRESHOLD

def record_audio():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Recording... Speak now.")
    frames = []
    silence_counter = 0
    start_time = datetime.now()
    
    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        if detect_silence(data):
            silence_counter += 1
        else:
            silence_counter = 0
        
        if silence_counter * (CHUNK / RATE) >= SILENCE_DURATION:
            break
        
        if (datetime.now() - start_time).total_seconds() > 10:
            break
    
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
    temp_filename = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    sf.write(temp_filename, audio_data, RATE)
    
    return temp_filename

def transcribe_audio(filename):
    result = model.transcribe(filename)
    os.remove(filename)
    return result["text"].strip()

def upload_json_to_pastebin(json_data, api_dev_key):
    # PasteBin.com API endpoint
    api_url = 'https://pastebin.com/api/api_post.php'

    # Prepare the data for the API request
    data = {
        'api_dev_key': api_dev_key,
        'api_option': 'paste',
        'api_paste_code': json.dumps(json_data, indent=4),
        'api_paste_format': 'json',
        'api_paste_name': 'Inspection Data',
        'api_paste_private': '0',  # 0 = public, 1 = unlisted, 2 = private
        'api_paste_expire_date': '1W'  # Expires in 1 week
    }

    # Send the POST request to PasteBin.com
    response = requests.post(api_url, data=data)

    # Check if the request was successful
    if response.status_code == 200:
        # The response text contains the URL of the new paste
        paste_url = response.text
        print(f"JSON data uploaded successfully. URL: {paste_url}")
        return paste_url
    else:
        print(f"Failed to upload JSON data. Status code: {response.status_code}")
        print(f"Response: {response.text}")
        return None

def speak(text):
    engine.say(text)
    engine.runAndWait()

def get_voice_input(prompt):
    speak(prompt)
    print(prompt)
    audio_file = record_audio()
    response = transcribe_audio(audio_file)
    print(f"You said: {response}")
    return response

def extract_tire_info(text):
    doc = nlp(text.lower())
    
    condition_keywords = ["good", "ok", "okay", "fine", "bad", "worn", "replace", "replacement"]
    condition = next((word for word in condition_keywords if word in text.lower()), None)
    
    pressure_match = re.search(r'\d+(?:\.\d+)?\s*psi', text.lower())
    pressure = pressure_match.group() if pressure_match else None
    
    return condition, pressure

import geocoder


def gather_pre_inspection_data():
    pre_inspection_data = {}
    
    # Automatically get date and geolocation
    pre_inspection_data['Date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    g = geocoder.ip('me')
    pre_inspection_data['Geolocation'] = f"{g.latlng[0]}, {g.latlng[1]}" if g.latlng else "Unable to determine"
    
    print("Before we begin the inspection, please provide some initial information.")
    
    fields = [
        ('Vehicle Make', "What is the make of the vehicle?"),
        ('Vehicle Model', "What is the model of the vehicle?"),
        ('Vehicle Year', "What is the year of the vehicle?"),
        ('VIN', "Please provide the Vehicle Identification Number (VIN)."),
        ('Mileage', "What is the current mileage of the vehicle?"),
        ('Customer Name', "What is the customer's name?"),
        ('Inspector Name', "What is your name (the inspector)?")
    ]
    
    for key, prompt in fields:
        while True:
            print(prompt)
            value = input(f"{speak(prompt)} ")
            if value.strip():  # Ensure the input is not empty
                pre_inspection_data[key] = value
                break
            else:
                print("This field cannot be empty. Please try again.")
    
    return pre_inspection_data

def process_pre_inspection(inspection_data):
    pre_inspection_data = gather_pre_inspection_data()
    inspection_data["PRE_INSPECTION"] = pre_inspection_data

def get_tire_info(position):
    condition, pressure = None, None
    while condition is None or pressure is None:
        prompt = f"Describe the tire condition and pressure for the {position} tire:"
        if condition is None and pressure is None:
            prompt += " (e.g., 'good condition, 32 psi')"
        elif condition is None:
            prompt += " (Please specify the condition)"
        elif pressure is None:
            prompt += " (Please specify the pressure in PSI)"
        
        response = get_voice_input(prompt)
        new_condition, new_pressure = extract_tire_info(response)
        
        if new_condition:
            condition = new_condition
        if new_pressure:
            pressure = new_pressure
        
        if condition is None:
            speak("I couldn't understand the tire condition. Please try again.")
        if pressure is None:
            speak("I couldn't understand the tire pressure. Please try again.")
    
    return condition, pressure

def capture_image(position, imgbb_api_key):
    while True:
        confirm = get_voice_input(f"Are you ready to capture the image for the {position} tire? Say 'yes' when ready.")
        if 'yes' in confirm.lower():
            break
        else:
            speak("I didn't hear 'yes'. Let's try again.")

    print("Opening camera...")
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open camera.")
        return "Image not captured"

    cv2.namedWindow(f"Capture Image for {position}")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow(f"Capture Image for {position}", frame)

        k = cv2.waitKey(1)
        if k % 256 == 113:  # 'q' key to quit
            img_name = f"{position}tire{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(img_name, frame)
            print(f"Picture taken: {img_name}")
            break

    cam.release()
    cv2.destroyAllWindows()

    # Upload image to ImgBB
    with open(img_name, "rb") as file:
        url = "https://api.imgbb.com/1/upload"
        payload = {
            "key": imgbb_api_key,
        }
        files = {
            "image": file,
        }
        response = requests.post(url, data=payload, files=files)
    
    if response.status_code == 200:
        img_url = response.json()['data']['url']
        print(f"Image uploaded successfully: {img_url}")
        return img_url
    else:
        print(f"Failed to upload image. Status Code: {response.status_code}")
        return "Image upload failed"



def process_tire_data(inspection_data):
    positions = ["front left", "front right", "rear left", "rear right"]
    for position in positions:
        condition, pressure = get_tire_info(position)
        
        inspection_data["TIRES"][f"Tire Condition for {position.title()}"] = condition.title()
        inspection_data["TIRES"][f"Tire Pressure for {position.title()}"] = pressure
        
        img_name = capture_image(position.title(),"3347f7da087a34b5b41cba3c171d2b41")
        inspection_data["TIRES"][f"Tire Image for {position.title()}"] = img_name
    
    summary = get_voice_input("Please provide an overall Tire Summary:")
    inspection_data["TIRES"]["Overall Tire Summary"] = summary

def extract_battery_info(text):
    doc = nlp(text.lower())
    
    make = next((ent.text for ent in doc.ents if ent.label_ == "ORG"), None)
    date = next((ent.text for ent in doc.ents if ent.label_ == "DATE"), None)
    voltage_match = re.search(r'\d+(?:\.\d+)?\s*v', text.lower())
    voltage = voltage_match.group() if voltage_match else None
    water_level = next((token.text for token in doc if token.text in ["good", "ok", "low"]), None)
    condition = "damaged" if any(token.text in ["damage", "damaged","bad"] for token in doc) else "good"
    leak_rust = "yes" if any(token.text in ["leak", "rust"] for token in doc) else "no"
    
    return make, date, voltage, water_level, condition, leak_rust

def get_battery_info():
    make, date, voltage, water_level, condition, leak_rust = None, None, None, None, None, None
    while any(info is None for info in [make, date, voltage, water_level, condition, leak_rust]):
        prompt = "Describe the battery condition, including make, replacement date, voltage, water level, overall condition, and any leaks or rust:"
        if make is None:
            prompt += " (Please specify the make)"
        if date is None:
            prompt += " (Please specify the replacement date)"
        if voltage is None:
            prompt += " (Please specify the voltage in volts)"
        if water_level is None:
            prompt += " (Please specify the water level: good, ok, or low)"
        if condition is None:
            prompt += " (Please specify the overall condition: good or damaged)"
        if leak_rust is None:
            prompt += " (Please specify if there are any leaks or rust)"
        
        response = get_voice_input(prompt)
        new_make, new_date, new_voltage, new_water_level, new_condition, new_leak_rust = extract_battery_info(response)
        
        make = new_make if new_make else make
        date = new_date if new_date else date
        voltage = new_voltage if new_voltage else voltage
        water_level = new_water_level if new_water_level else water_level
        condition = new_condition if new_condition else condition
        leak_rust = new_leak_rust if new_leak_rust else leak_rust
        
        if make is None:
            speak("I couldn't understand the battery make. Please try again.")
        if date is None:
            speak("I couldn't understand the replacement date. Please try again.")
        if voltage is None:
            speak("I couldn't understand the voltage. Please try again.")
        if water_level is None:
            speak("I couldn't understand the water level. Please try again.")
        if condition is None:
            speak("I couldn't understand the overall condition. Please try again.")
        if leak_rust is None:
            speak("I couldn't understand if there are any leaks or rust. Please try again.")
    
    return make, date, voltage, water_level, condition, leak_rust

def process_battery_data(inspection_data):
    make, date, voltage, water_level, condition, leak_rust = get_battery_info()
    
    inspection_data["BATTERY"]["Battery Make"] = make
    inspection_data["BATTERY"]["Battery replacement date"] = date
    inspection_data["BATTERY"]["Battery Voltage"] = voltage
    inspection_data["BATTERY"]["Battery Water level"] = water_level.title()
    inspection_data["BATTERY"]["Condition of Battery"] = "Y" if condition == "damaged" else "N"
    inspection_data["BATTERY"]["Any Leak / Rust in battery"] = leak_rust.upper()
    
    summary = get_voice_input("Please provide a Battery overall Summary:")
    inspection_data["BATTERY"]["Battery overall Summary"] = summary

def extract_exterior_info(text):
    doc = nlp(text.lower())
    
    damage = "Y" if any(token.text in ["rust", "dent", "damage", "damaged"] for token in doc) else "N"
    oil_leak = "Y" if "oil" in text.lower() and "leak" in text.lower() else "N"
    
    return damage, oil_leak

def get_exterior_info():
    damage, oil_leak = None, None
    while damage is None or oil_leak is None:
        prompt = "Describe the exterior condition of the vehicle, including any damage or oil leaks:"
        if damage is None:
            prompt += " (Please specify if there is any rust, dent, or damage)"
        if oil_leak is None:
            prompt += " (Please specify if there is any oil leak)"
        
        response = get_voice_input(prompt)
        new_damage, new_oil_leak = extract_exterior_info(response)
        
        damage = new_damage if new_damage else damage
        oil_leak = new_oil_leak if new_oil_leak else oil_leak
        
        if damage is None:
            speak("I couldn't understand if there is any exterior damage. Please try again.")
        if oil_leak is None:
            speak("I couldn't understand if there is any oil leak. Please try again.")
    
    return damage, oil_leak

def process_exterior_data(inspection_data):
    damage, oil_leak = get_exterior_info()
    
    inspection_data["EXTERIOR"]["Rust, Dent or Damage to Exterior"] = damage
    inspection_data["EXTERIOR"]["Oil leak in Suspension"] = oil_leak
    
    summary = get_voice_input("Please provide an Exterior Overall Summary:")
    inspection_data["EXTERIOR"]["Overall Summary"] = summary

def extract_brake_info(text):
    doc = nlp(text.lower())
    
    fluid_level = next((token.text for token in doc if token.text in ["good", "ok", "low"] and "fluid" in [t.text for t in token.children]), None)
    front_condition = next((token.text for token in doc if token.text in ["good", "ok", "replace"] and "front" in [t.text for t in token.children]), None)
    rear_condition = next((token.text for token in doc if token.text in ["good", "ok", "replace"] and "rear" in [t.text for t in token.children]), None)
    emergency_brake = next((token.text for token in doc if token.text in ["good", "ok", "low"] and "emergency" in [t.text for t in token.children]), None)
    
    return fluid_level, front_condition, rear_condition, emergency_brake

def get_brake_info():
    fluid_level, front_condition, rear_condition, emergency_brake = None, None, None, None
    while any(info is None for info in [fluid_level, front_condition, rear_condition, emergency_brake]):
        prompt = "Describe the condition of the brakes, including fluid level, front and rear brake condition, and emergency brake:"
        if fluid_level is None:
            prompt += " (Please specify the fluid level: good, ok, or low)"
        if front_condition is None:
            prompt += " (Please specify the front brake condition: good, ok, or replace)"
        if rear_condition is None:
            prompt += " (Please specify the rear brake condition: good, ok, or replace)"
        if emergency_brake is None:
            prompt += " (Please specify the emergency brake condition: good, ok, or low)"
        
        response = get_voice_input(prompt)
        new_fluid_level, new_front_condition, new_rear_condition, new_emergency_brake = extract_brake_info(response)
        
        fluid_level = new_fluid_level if new_fluid_level else fluid_level
        front_condition = new_front_condition if new_front_condition else front_condition
        rear_condition = new_rear_condition if new_rear_condition else rear_condition
        emergency_brake = new_emergency_brake if new_emergency_brake else emergency_brake
        
        if fluid_level is None:
            speak("I couldn't understand the brake fluid level. Please try again.")
        if front_condition is None:
            speak("I couldn't understand the front brake condition. Please try again.")
        if rear_condition is None:
            speak("I couldn't understand the rear brake condition. Please try again.")
        if emergency_brake is None:
            speak("I couldn't understand the emergency brake condition. Please try again.")
    
    return fluid_level, front_condition, rear_condition, emergency_brake

def process_brakes_data(inspection_data):
    fluid_level, front_condition, rear_condition, emergency_brake = get_brake_info()
    
    inspection_data["BRAKES"]["Brake Fluid level"] = fluid_level.title()
    inspection_data["BRAKES"]["Brake Condition for Front"] = front_condition.title()
    inspection_data["BRAKES"]["Brake Condition for Rear"] = rear_condition.title()
    inspection_data["BRAKES"]["Emergency Brake"] = emergency_brake.title()
    
    summary = get_voice_input("Please provide a Brake Overall Summary:")
    inspection_data["BRAKES"]["Brake Overall Summary"] = summary

def extract_engine_info(text):
    doc = nlp(text.lower())
    
    damage = "Y" if any(token.text in ["rust", "dent", "damage", "damaged"] for token in doc) else "N"
    
    oil_condition = next((token.text for token in doc if token.text in ["good", "bad"] and "oil" in [t.text for t in token.children]), None)
    if oil_condition is None:
        oil_condition = input("Please specify the oil condition (good/bad): ")
    
    oil_color = next((token.text for token in doc if token.text in ["clean", "brown", "black"] and "oil" in [t.text for t in token.children]), None)
    if oil_color is None:
        oil_color = input("Please specify the oil color (clean/brown/black): ")
    
    brake_fluid_condition = next((token.text for token in doc if token.text in ["good", "bad"] and "brake" in [t.text for t in token.children] and "fluid" in [t.text for t in token.children]), None)
    if brake_fluid_condition is None:
        brake_fluid_condition = input("Please specify the brake fluid condition (good/bad): ")
    
    brake_fluid_color = next((token.text for token in doc if token.text in ["clean", "brown", "black"] and "brake" in [t.text for t in token.children] and "fluid" in [t.text for t in token.children]), None)
    if brake_fluid_color is None:
        brake_fluid_color = input("Please specify the brake fluid color (clean/brown/black): ")
    
    oil_leak = "Y" if "oil" in text.lower() and "leak" in text.lower() else "N"
    
    return damage, oil_condition, oil_color, brake_fluid_condition, brake_fluid_color, oil_leak

def process_engine_data(inspection_data):
    response = get_voice_input("Describe the engine condition, including any damage, oil condition and color, brake fluid condition and color, and any oil leaks:")
    damage, oil_condition, oil_color, brake_fluid_condition, brake_fluid_color, oil_leak = extract_engine_info(response)
    
    inspection_data["ENGINE"]["Rust, Dents or Damage in Engine"] = damage
    if damage == "Y":
        speak("You mentioned there is some engine damage. Please describe it in more detail.")
        damage_details = get_voice_input("Describe the engine damage in detail:")
        inspection_data["ENGINE"]["Detailed Engine Damage"] = damage_details
    else:
        inspection_data["ENGINE"]["Detailed Engine Damage"] = "No engine damage reported"

    inspection_data["ENGINE"]["Engine Oil Condition"] = oil_condition.title()
    inspection_data["ENGINE"]["Engine Oil Color"] = oil_color.title()
    inspection_data["ENGINE"]["Brake Fluid Condition"] = brake_fluid_condition.title()
    inspection_data["ENGINE"]["Brake Fluid Color"] = brake_fluid_color.title()
    inspection_data["ENGINE"]["Any oil leak in Engine"] = oil_leak
    
    summary = get_voice_input("Please provide an Engine Overall Summary:")
    inspection_data["ENGINE"]["Overall Summary"] = summary

def process_customer_feedback(inspection_data):
    feedback = get_voice_input("Please provide any feedback from the Customer:")
    inspection_data["Voice of Customer"]["Customer Feedback"] = feedback

def process_section(section_name, process_func, inspection_data):
    process_func(inspection_data)
    while True:
        data_summary = json.dumps(inspection_data[section_name.upper()], indent=4)
        print(f"Collected data for {section_name}:\n{data_summary}")

        response = get_voice_input(f"Would you like to go to the next step, retry, pause, or stop?").lower()
        if "next" in response or "continue" in response:
            return True
        elif "retry" in response:
            speak(f"Let's go over the {section_name} section again.")
            process_func(inspection_data)
        elif "pause" in response:
            speak("Inspection paused. Press Enter to continue.")
            input("Inspection paused. Press Enter to continue.")
        elif "stop" in response:
            return False
        else:
            speak("I didn't understand. Please say 'next', 'retry', 'pause', or 'stop'.")

def create_pdf(data, filename):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica", 12)
    y = height - 40

    def add_section(title, content):
        nonlocal y
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, y, title)
        y -= 20
        c.setFont("Helvetica", 12)
        for key, value in content.items():
            if key.endswith("Image"):
                try:
                    img = ImageReader(value)
                    img_width, img_height = img.getSize()
                    aspect = img_width / float(img_height)
                    img_width = 3 * inch
                    img_height = img_width / aspect
                    c.drawImage(img, 40, y - img_height, width=img_width, height=img_height)
                    y -= img_height + 10
                except Exception as e:
                    print(f"Error adding image {value}: {e}")
            else:
                c.drawString(40, y, f"{key}: {value}")
                y -= 20
            if y < 50:
                c.showPage()
                y = height - 40

    for section, content in data.items():
        add_section(section, content)
        y -= 20

    c.save()

def create_inspection_folder(data, pdf_filename):
    # Create a folder with a timestamp
    folder_name = f"inspection_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(folder_name, exist_ok=True)

    # Copy the PDF to the folder
    pdf_path = os.path.join(folder_name, os.path.basename(pdf_filename))
    shutil.copy(pdf_filename, pdf_path)

    # Create and save the JSON file in the folder
    json_filename = 'inspection_data.json'
    json_path = os.path.join(folder_name, json_filename)
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

    # Copy all image files to the folder
    for section in data.values():
        for key, value in section.items():
            if key.endswith("Image") and os.path.isfile(value):
                image_path = os.path.join(folder_name, os.path.basename(value))
                shutil.copy(value, image_path)

    return folder_name

def main():
    pre_inspection_data = gather_pre_inspection_data()
    
    inspection_data = {
        "PRE_INSPECTION": pre_inspection_data,
        "TIRES": {},
        "BATTERY": {},
        "EXTERIOR": {},
        "BRAKES": {},
        "ENGINE": {},
        "Voice of Customer": {}
    }

    sections = [
        ("Tires", process_tire_data),
        ("Battery", process_battery_data),
        ("Exterior", process_exterior_data),
        ("Brakes", process_brakes_data),
        ("Engine", process_engine_data),
        ("Customer Feedback", process_customer_feedback)
    ]

    for section_name, process_func in sections:
        speak(f"Let's start the {section_name} inspection.")
        if not process_section(section_name, process_func, inspection_data):
            speak("Inspection stopped. Saving collected data.")
            break

    # Generate PDF report
    pdf_filename = f"inspection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    create_pdf(inspection_data, pdf_filename)
    print(f"PDF report generated: {pdf_filename}")

    # Create folder with all inspection data
    folder_name = create_inspection_folder(inspection_data, pdf_filename)
    print(f"Inspection package created: {folder_name}")

    # Clean up individual files
    os.remove(pdf_filename)

    # Move image files to the created folder instead of deleting them
    for section in inspection_data.values():
        for key, value in section.items():
            if key.endswith("Image") and os.path.isfile(value):
                new_image_path = os.path.join(folder_name, os.path.basename(value))
                shutil.move(value, new_image_path)
                # Update the path in the inspection_data
                section[key] = new_image_path

    # Update the JSON file with the new image paths
    json_filename = 'inspection_data.json'
    json_path = os.path.join(folder_name, json_filename)
    with open(json_path, 'w') as f:
        json.dump(inspection_data, f, indent=4)

    # Upload JSON data to PasteBin.com
    api_dev_key = '5wzQMjOVCkDDDWzQj564ZippIEuRj2at'  # Replace with your actual API key
    paste_url = upload_json_to_pastebin(inspection_data, api_dev_key)
    
    if paste_url:
        print(f"JSON data uploaded to PasteBin.com. URL: {paste_url}")
        # Optionally, you can save the URL to a file in the inspection folder
        with open(os.path.join(folder_name, 'pastebin_url.txt'), 'w') as f:
            f.write(paste_url)
    else:
        print("Failed to upload JSON data to PasteBin.com")

    speak("Inspection completed. All data has been saved in a folder and uploaded to PasteBin.com.")

if _name_ == "_main_":
    main()
