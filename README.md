# 🎯 Lock-In Tracker

**Real-time focus monitoring application using AI-powered face tracking and gaze detection.**

This application uses your webcam alongside AI-powered face landmark detection to monitor your focus levels. It tracks how long you maintain focus, logs distraction points, and builds productivity metrics over time, presented in a modern graphical dashboard.

*Note: This application utilizes the Windows-native `winsound` library and is currently designed for Windows environments.*

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Database & Analytics](#database--analytics)
- [Troubleshooting](#troubleshooting)

---

## 🚀 Overview

**Lock-In Tracker** gamifies productivity by using real-time face tracking to determine if you are focused on your screen or distracted.

### What Problem Does It Solve?

Modern workers struggle with distractions. This app provides:

* **Gamified tracking:** Build focus streaks instead of counting distraction time.
* **Real-time feedback:** Audio alerts when you lose focus.
* **Quantified metrics:** Track your session focus score.
* **Historical analytics:** Monitor productivity trends over time.
* **Live calibration:** Adjust detection sensitivity on the fly to fit your environment.

---

## ✨ Features

### 🎮 Gamified Focus Tracking

* **Focus Streak**: A timer that counts up as long as you maintain focus.
* **Distraction Points**: If focus is lost for >2.0 seconds, your streak breaks, a point is added, and an audio alert triggers.
* **Focus Score**: A dynamically calculated percentage of the total session time you spent focused.

### 🎥 Real-Time Face Detection

* **Live Video Feed**: See yourself with a focus state detection overlay (green bounding box = focused, red = distracted).
* **Head Pose Estimation**: Tracks yaw (left/right) and pitch (up/down) rotation.
* **Gaze Ratio Detection**: Monitors horizontal iris position relative to eye corners to ensure your eyes are on the screen.

### 🎚️ Live Calibration

* **Dynamic Threshold Sliders**: Adjust horizontal (Yaw) and vertical (Pitch) head tilt sensitivity directly from the dashboard without restarting the application.

### 📊 Advanced Analytics Dashboard

* **Lifetime Stats Panel**: Tracks total hours focused, total distraction events, and your all-time average focus score across all historical sessions.
* **Distraction Trends Chart**: An auto-updating Matplotlib graph showing your historical focus patterns.

---

## 💻 Tech Stack

* **Language**: Python 3.8+
* **Core Architecture**:
    * `mediapipe` – AI-powered face mesh and landmark detection.
    * `opencv-python (cv2)` – Video capture, image processing, and PnP matrix math.
    * `customtkinter` – Modern dark-themed GUI framework with beveled edges.
    * `matplotlib` – Analytics visualization and UI embedding.
    * `sqlite3` – Local session data persistence.
    * `numpy` – Numerical computations and Euclidean distance formulas.
    * `threading` – Asynchronous audio alerts and camera buffering to prevent GUI lag.

---

## 🛠️ Quick Start

### Prerequisites

* Windows OS
* Webcam (built-in or USB)
* Python 3.8+ installed

### Setup

```bash
# 1. Clone the repository
git clone [https://github.com/yourusername/lock-in-tracker.git](https://github.com/yourusername/lock-in-tracker.git)
cd lock-in-tracker

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python focus_app.py

### Setup

"No camera found": Ensure Windows Privacy Settings allow apps to access the camera, and no other applications (like Zoom or Teams) are currently using it.

"Face not detected": Ensure adequate lighting. Extreme angles or heavy glare on glasses may interfere with iris tracking.

"Performance is slow": The app automatically downscales frames >640px wide. If lag persists, ensure your device is plugged into power to prevent CPU throttling.
