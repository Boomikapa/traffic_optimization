1. Streamlit

What is it?
Streamlit is a Python framework used to create web applications for data science and machine-learning projects.

In your project:
It creates the traffic dashboard, file upload, buttons, metrics, charts, sidebar, and alerts.

Example:

st.title("AI-Driven Traffic Optimization Framework")

---------------------------------------------------------------------

OpenCV

What is it?
OpenCV is a computer vision library used to process and analyze images and videos.

In your project:
You use OpenCV to:

Decode uploaded images
Draw bounding boxes
Display vehicle labels
Convert BGR images to RGB

-------------------------------------------

YOLOv8

YOLO means You Only Look Once.

It is an object-detection algorithm that can identify objects in an image and determine where they are.

Your project loads:

self.model = YOLO("yolov8n.pt")

and performs detection using:

results = self.model(image_cv, verbose=False)

What does YOLO give you?

For every detected object, it can provide:

Object class
Confidence score
Bounding-box coordinates

For example:

Car → 0.92 confidence
Motorcycle → 0.87 confidence
Bus → 0.91 confidence

Your project identifies:

Car
Motorcycle
Bus
Truck

--------------------------------------------------------------

Complete Project Flow

The most important thing to remember for your interview is this:

        Road Image
             ↓
        OpenCV Processing
             ↓
          YOLOv8
             ↓
      Vehicle Detection
             ↓
   Vehicle Classification
             ↓
       Vehicle Count
             ↓
      Density Calculation
             ↓
     Weather Adjustment
             ↓
    Traffic Status
             ↓
   ┌─────────┴─────────┐
   ↓                   ↓
Signal Decision      Alerts
   ↓
Route Diversion
   ↓
Dashboard
