import cv2
import math
import os
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from pygame import mixer
import base64
import threading

# Email Configuration
EMAIL = "Namanchopra156@gmail.com"
PASSWORD = "spdd abjt nvvo fvuj"
TO_EMAIL = "Namanchopra156@gmail.com"    

# Flask app setup
app = Flask(__name__)
socketio = SocketIO(app)

# Load YOLO model
model = YOLO("yolov8n.pt")

# Directory for saving snapshots
if not os.path.exists("snapshot"):
    os.makedirs("snapshot")

# Initialize pygame mixer
mixer.init()

# Global variable to control alarm state
alarm_active = False

def send_email(snapshot_path, detection_time):
    try:
        subject = "Human Detected in Security Camera Feed"
        body = f"A human was detected at {detection_time}. See the attached snapshot."

        # Create the email
        msg = MIMEMultipart()
        msg['From'] = EMAIL
        msg['To'] = TO_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Attach snapshot
        with open(snapshot_path, 'rb') as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(snapshot_path)}')
            msg.attach(part)

        # Send email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL, PASSWORD)
        server.sendmail(EMAIL, TO_EMAIL, msg.as_string())
        server.quit()
        print(f"Email sent to {TO_EMAIL}.")
    except Exception as e:
        print("Error sending email:", e)

def play_alarm():
    global alarm_active
    while alarm_active:
        mixer.music.load("static/alert.mp3")
        mixer.music.play()
        # Wait for the sound to finish playing or until alarm is stopped
        while mixer.music.get_busy() and alarm_active:
            mixer.music.set_volume(1.0)  # Ensure full volume
            socketio.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

def video_stream():
    global alarm_active
    cap = cv2.VideoCapture(0)  # Webcam feed
    last_email_time = None
    alarm_thread = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Object detection
        results = model(frame, stream=True)
        human_detected = False
        detection_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])

                if cls == 0:  # Person class
                    human_detected = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Human Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Handle detection alerts
        if human_detected:
            # Start alarm if not already active
            if not alarm_active:
                alarm_active = True
                alarm_thread = socketio.start_background_task(play_alarm)

            # Send email at most every 5 minutes
            if last_email_time is None or (datetime.now() - last_email_time).total_seconds() > 300:
                last_email_time = datetime.now()
                snapshot_path = f"snapshot/snapshot_{detection_time.replace(':', '_').replace(' ', '_')}.jpg"
                cv2.imwrite(snapshot_path, frame)
                send_email(snapshot_path, detection_time)
        else:
            # Stop alarm if no human detected
            if alarm_active:
                alarm_active = False
                # Wait for the alarm thread to terminate
                if alarm_thread:
                    alarm_thread.join(timeout=1)
                # Stop the music
                mixer.music.stop()

        # Encode and send frame to frontend
        _, buffer = cv2.imencode('.jpg', frame)
        frame_data = base64.b64encode(buffer).decode('utf-8')
        socketio.emit('video_frame', {'frame': frame_data})

    cap.release()

@socketio.on('start_video')
def start_video():
    socketio.start_background_task(video_stream)

if __name__ == "__main__":
    socketio.run(app, debug=True)