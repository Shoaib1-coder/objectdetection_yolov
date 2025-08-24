import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time

# Streamlit page configuration
st.set_page_config(page_title="Object Detection", layout="wide")

# Title and instructions
st.title("Object Detection — Image & Video")
st.write("Upload an image or video and click **Detect** for object detection.")
# Fixed YOLO model settings
MODEL_PATH = "yolov8n.pt"   # This will auto-download from Ultralytics if not present
CONF_THRESH = 0.25   # Confidence threshold
IOU_THRESH = 0.45   # IoU threshold for NMS
MAX_DET = 300    # Maximum number of detections per image
SHOW_LABELS = True  # Whether to show labels on boxes

# Load YOLO model with caching
@st.cache_resource(show_spinner=False)
def load_model():
    return YOLO(MODEL_PATH)

with st.spinner(f"Loading YOLO model {MODEL_PATH} ..."):
    model = load_model()


# File uploader (accepts images or videos)
uploaded_file = st.file_uploader("Upload image or video", 
                                 type=["jpg","jpeg","png","mp4","mov","avi","mkv"])

# If no file uploaded, stop app
if uploaded_file is None:
    st.info("Upload an image (.jpg/.png) or video (.mp4/.avi) to begin.")
    st.stop()



# YOLO prediction parameters
predict_kwargs = {
    "conf": CONF_THRESH,     # Confidence threshold
    "iou": IOU_THRESH,       # IoU threshold
    "max_det": MAX_DET,      # Max detections
    "verbose": False,        # Suppress extra logs
}

# Button to trigger detection
detect_button = st.button("Detect")

# Function to annotate images with bounding boxes + labels
def annotate_image_cv2(img_bgr, boxes, scores, class_ids, names, show_labels=True):
    """Annotate cv2 BGR image inplace"""
    for (x1, y1, x2, y2), score, cls in zip(boxes, scores, class_ids):
        # Convert coordinates to integers
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        color = (0, 255, 0)  # Green box color
        # Draw rectangle
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        if show_labels:
            # Prepare label text (class name + confidence)
            label = f"{names[int(cls)]} {score:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # Background rectangle for label
            cv2.rectangle(img_bgr, (x1, y1 - 18), (x1 + w, y1), color, -1)
            # Put label text
            cv2.putText(img_bgr, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0,0,0), 1, cv2.LINE_AA)

# If "Detect" button is clicked
if detect_button:
    t0 = time.time()  # Start timer
    filename = uploaded_file.name.lower()

    # Check if file is image or video
    is_image = any(filename.endswith(ext) for ext in (".jpg", ".jpeg", ".png"))
    is_video = any(filename.endswith(ext) for ext in (".mp4", ".mov", ".avi", ".mkv"))

    # ========== IMAGE PROCESSING ==========
    if is_image:
        # Read image with PIL and convert to numpy
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)

        # Run YOLO detection
        with st.spinner("Running detection on image..."):
            results = model.predict(img_np, **predict_kwargs)

        # Extract results (boxes, scores, classes)
        res = results[0]
        boxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.array([])
        scores = res.boxes.conf.cpu().numpy() if len(res.boxes) else np.array([])
        class_ids = res.boxes.cls.cpu().numpy() if len(res.boxes) else np.array([])
        names = model.model.names if hasattr(model, "model") else model.names

        # Convert RGB → BGR for cv2 annotation
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        if boxes.size:  # Annotate if detections exist
            annotate_image_cv2(img_bgr, boxes, scores, class_ids, names, show_labels=SHOW_LABELS)

        # Convert back BGR → RGB for display
        annotated_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Show annotated image in Streamlit
        st.image(annotated_rgb, caption="Detected", use_column_width=True)
        st.success(f"Done — detected {len(boxes)} objects in {time.time()-t0:.2f}s")

        # Save annotated image temporarily for download
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        Image.fromarray(annotated_rgb).save(tmp.name)
        st.download_button("Download annotated image", 
                           data=open(tmp.name, "rb").read(), 
                           file_name=f"annotated_{uploaded_file.name}")

    # ========== VIDEO PROCESSING ==========
    elif is_video:
        # Save uploaded video temporarily
        tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
        tfile_in.write(uploaded_file.getbuffer())
        tfile_in.flush()

        # Open video with cv2
        cap = cv2.VideoCapture(tfile_in.name)

        if not cap.isOpened():
            st.error("Could not open uploaded video.")
        else:
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Define output video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            tfile_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            out = cv2.VideoWriter(tfile_out.name, fourcc, fps, (width, height))

            # Frame counter for progress bar
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            progress_bar = st.progress(0)
            frame_idx = 0

            # Process video frame by frame
            with st.spinner("Processing video..."):
                while True:
                    ret, frame = cap.read()
                    if not ret:  # End of video
                        break
                    # Convert BGR → RGB
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Run detection
                    results = model.predict(img_rgb, **predict_kwargs)
                    res = results[0]
                    boxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.array([])
                    scores = res.boxes.conf.cpu().numpy() if len(res.boxes) else np.array([])
                    class_ids = res.boxes.cls.cpu().numpy() if len(res.boxes) else np.array([])
                    names = model.model.names if hasattr(model, "model") else model.names
                    # Annotate frame
                    if boxes.size:
                        annotate_image_cv2(frame, boxes, scores, class_ids, names, show_labels=SHOW_LABELS)
                    # Write annotated frame to output video
                    out.write(frame)
                    frame_idx += 1
                    # Update progress bar
                    if frame_count:
                        progress_bar.progress(min(1.0, frame_idx / frame_count))

                # Release resources
                cap.release()
                out.release()
                progress_bar.empty()

            # Show processed video
            st.success(f"Video processed in {time.time()-t0:.2f}s — result below:")
            video_bytes = open(tfile_out.name, "rb").read()
            st.video(video_bytes)

            # Provide download button
            st.download_button("Download annotated video", 
                               data=video_bytes, 
                               file_name=f"annotated_{uploaded_file.name}")

    # If unsupported file type
    else:
        st.error("Unsupported file type.")