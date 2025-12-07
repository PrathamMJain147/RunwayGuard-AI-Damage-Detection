# RunwayGuard: AI Airport Runway Damage Detection ✈️

RunwayGuard is a computer vision system designed to detect and classify damage on airport runways. Utilizing the power of **YOLOv8** and **Streamlit**, this tool helps identify hazards like cracks and holes to ensure aircraft safety and maintenance efficiency.

## 🚀 Features

* **Real-time Detection:** Identifies 5 specific types of runway damage.
* **Web Interface:** User-friendly dashboard built with Streamlit for uploading and analyzing images.
* **State-of-the-Art Model:** Powered by Ultralytics YOLOv8n for fast and accurate inference.
* **Visual Feedback:** Draws bounding boxes and confidence scores on detected hazards.

## 📂 Classes Detected

The model is trained to recognize the following surface anomalies:
1.  **L1_Hole** (Minor holes)
2.  **L2_Hole** (Medium holes)
3.  **L3_Hole** (Major holes)
4.  **Mild Cracks**
5.  **Severe Cracks**

## 🛠️ Tech Stack

* [Python 3.9+](https://www.python.org/)
* [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics) - Object Detection
* [Streamlit](https://streamlit.io/) - Web UI
* [OpenCV](https://opencv.org/) - Image Processing

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/RunwayGuard.git](https://github.com/YOUR_USERNAME/RunwayGuard.git)
    cd RunwayGuard
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Weights:**
    Ensure your trained model weights (`best.pt`) are placed in a folder named `weights/` or update the path in `app.py`.

## 🖥️ Usage

### Running the Web Application
To start the interactive interface:

```bash
streamlit run app.py
