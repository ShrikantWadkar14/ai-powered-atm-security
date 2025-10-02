# AI-Powered ATM Security System

A prototype solution to revolutionize ATM security using computer vision and deep learning. This project detects suspicious activities, weapons, and tampering in real time, sending instant alerts to security teams. The system features a modern web dashboard, live video streaming, and multi-channel notifications.
![atm github](https://github.com/user-attachments/assets/a8481625-9d94-4425-82d8-7c9044412992)

## 🚀 Features

- **Live Video Stream:** Real-time display of ATM camera feed with detection overlays.
- **Suspicious Activity Detection:** Identifies weapons, tampering, and dangerous actions using YOLOv8 and custom models.
- **Alert System:** Sends instant alerts via SMS, email, and automated phone calls (Twilio).
- **30s Alert Cooldown:** Prevents duplicate alerts within 30 seconds.
- **Web UI:** Upload video files or connect RTSP streams, view alerts, and monitor status.
- **Snapshot Storage:** Saves annotated frames for each alert.
- **Modular Design:** Easily extend detection logic and alert channels.
- **Prototype:** Working demo with ongoing development for scalability and robustness.

## 🛠️ Technologies Used

- Python, Flask, OpenCV
- YOLOv8, TensorFlow Lite
- Bootstrap 5, jQuery
- Twilio (SMS, call), SMTP (email)

## ⚡ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shrikantwadkar14/ai-powered-atm-security.git
   cd ai-powered-atm-security
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv atm-env
   atm-env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download model files:**
   - Place `yolov8n.pt` and `best.pt` in the project root.

5. **Configure Twilio and Email:**
   - Edit `app.py` with your Twilio and SMTP credentials.

## 💻 Usage

1. **Start the Flask server:**
   ```bash
   python app.py
   ```

2. **Open the web UI:**
   - Go to `http://localhost:5000` in your browser.

3. **Configure video source:**
   - Upload a video file or enter an RTSP stream URL.
   - Click "Start Detection" to begin live monitoring.

4. **View live stream and alerts:**
   - The UI displays the annotated video and recent alerts.
   - Alerts are sent via SMS, email, and phone call (if configured).

## 🔔 Alert & Call Feature

- When a high-level threat (e.g., weapon detected) is identified:
  - The system sends an SMS and email to configured contacts.
  - **Automated phone call** is triggered using Twilio, delivering a voice alert.
  - Alert snapshots are saved for review.
- Alerts are managed with a 30-second cooldown to avoid spamming.

## 📁 File Structure

```
├── app.py                # Main Flask backend
├── templates/
│   └── index.html        # Web UI
├── detectors.py          # Detection logic
├── action_detector.py    # Action analysis
├── tamper.py             # Tamper detection
├── decision.py           # Decision engine
├── alerts.py             # Alert manager (SMS, email, call)
├── requirements.txt      # Python dependencies
├── yolov8n.pt, best.pt   # Model files
├── snapshots/            # Saved alert images
├── atm-env/              # Python virtual environment
```

## 🧩 Troubleshooting

- **No video stream in browser:** Try Firefox, check backend logs for errors, ensure detection is started.
- **Alerts not sent:** Verify Twilio and SMTP credentials in `app.py`.
- **Model errors:** Ensure model files are present and paths are correct.

## 🏗️ Ongoing Work

- Improving scalability and robustness for real-world deployment.
- Adding more detection features and alert channels.

## 🤝 Contributing

Pull requests are welcome! Please open issues for feature requests or bugs.

## 📄 License

MIT License

---

**Note:** Replace `yourusername` in the clone URL with your GitHub username. Update Twilio and email credentials in `app.py` before deployment.

For more about my work, visit: [www.shrikantwadkar.com](https://www.shrikantwadkar.com)
