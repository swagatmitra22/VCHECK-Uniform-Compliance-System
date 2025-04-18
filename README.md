# VCHECK: Uniform Compliance Monitoring System

**VCHECK** is a real-time system that monitors uniform compliance by detecting ID cards and classifying clothing (e.g., shirts or t-shirts) using deep learning and computer vision. It ensures adherence to dress code policies through automated detection, OCR-based validation, and instant feedback via a user-friendly Streamlit interface.

---

## 🚀 Features

- **🎯 ID Card Detection**  
  Real-time detection using a custom-trained YOLOv8 model with high accuracy.

- **👕 Clothing Classification**  
  Classifies clothing types (Shirt/T-shirt) using a lightweight YOLOv8 model.

- **🔍 OCR for Registration Numbers**  
  Extracts registration numbers from ID cards using EasyOCR with Levenshtein correction.

- **✅ Compliance Verification**  
  Validates registration numbers against a local database (SQLite/CSV) to confirm compliance.

- **📢 Real-Time Alerts**  
  Provides audio and visual alerts on non-compliance, with cooldown timers.

- **🗂 Logging and Audit Trail**  
  Automatically logs violations with timestamps and registration numbers. Exportable in CSV format.

- **💻 Interactive Streamlit UI**  
  Intuitive web dashboard for real-time video display, detection overlays, and controls.

---

## 🧠 System Architecture

### 1. **Input Layer**
- Logitech C920 HD Pro or compatible webcam.
- Captures video at 30 FPS over USB 3.0.

### 2. **Processing Layer**
- Frame sampling at 2–3 FPS using OpenCV.
- **YOLOv8n** for ID card detection (mAP50: 0.995).
- **YOLOv8s** for clothing classification (mAP50: 0.983).
- ROI extraction, adaptive thresholding, and grayscale processing.
- EasyOCR (ResNet-34 + LSTM) for reading ID numbers.
- Regex-based format checking and Levenshtein distance correction.
- Registration validation via Pandas queries on `student_ids.csv`.

### 3. **Control Flow & Optimization**
- Multithreading: separate threads for webcam, inference, OCR, and alert logic.
- TensorRT acceleration with FP16 quantization via ONNX Runtime.
- Adaptive resource allocation: dynamic batching, warm-up, CUDA stream pipelining.

### 4. **Output Layer**
- Streamlit dashboard with live video feed and bounding boxes:
  - Magenta (ID card)
  - Green (Clothing)
- Audio alerts via `noncompliance.mp3`.
- Logging system for registration audits and violation records.

---

## 🛠 Installation

1. **Clone the repository:**
   ```
   git clone https://github.com/your-username/VCHECK-Uniform-Compliance-System.git
   cd VCHECK-Uniform-Compliance-System
   ```

1. **Run the App:**
   ```
   streamlit run app.py
   ```