import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import tempfile
import os
import glob
import time
import torch


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pothole Detection & Tracking",
    page_icon="🚧",
    layout="wide"
)

st.title("🚧 Pothole Detection & Tracking")


# ============================================================
# DEVICE DETECTION
# ============================================================

if torch.cuda.is_available():
    DEVICE = 0
    DEVICE_NAME = "NVIDIA GPU"
else:
    DEVICE = "cpu"
    DEVICE_NAME = "CPU"


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.header("⚙️ Model Controls")

confidence_threshold = st.sidebar.slider(
    "Detection Confidence Threshold",
    0.05,
    1.0,
    0.25,
    0.05
)

iou_threshold = st.sidebar.slider(
    "NMS IoU Threshold",
    0.1,
    0.9,
    0.45,
    0.05
)

enable_tta = st.sidebar.checkbox(
    "Enable TTA for Images",
    value=False,
    help="Improves image accuracy but is slower."
)


# ============================================================
# VIDEO SETTINGS
# ============================================================

st.sidebar.header("🚀 Video Performance")

frame_skip = st.sidebar.selectbox(
    "Process Every Nth Frame",
    options=[1, 2, 3, 4],
    index=1,
    help="1 = every frame, 2 = every second frame, etc."
)

video_imgsz = st.sidebar.selectbox(
    "Video Inference Size",
    options=[320, 416, 512, 640],
    index=2
)

max_video_width = st.sidebar.selectbox(
    "Maximum Video Width",
    options=[640, 800, 960, 1280],
    index=2
)


# ============================================================
# HARDWARE INFORMATION
# ============================================================

st.sidebar.header("💻 Hardware")

if torch.cuda.is_available():

    st.sidebar.success(
        f"🚀 GPU detected: {torch.cuda.get_device_name(0)}"
    )

else:

    st.sidebar.info(
        "💻 CUDA GPU not detected. Using CPU."
    )


# ============================================================
# LOAD CUSTOM POTHOLE MODEL
# ============================================================

@st.cache_resource
def load_pothole_model():

    # --------------------------------------------------------
    # OPTION 1:
    # best.pt in same directory as app.py
    # --------------------------------------------------------

    if os.path.exists("best.pt"):
        model_path = "best.pt"

        return YOLO(model_path)


    # --------------------------------------------------------
    # OPTION 2:
    # Search inside runs folders
    # --------------------------------------------------------

    found_weights = glob.glob(
        "runs/**/best.pt",
        recursive=True
    )

    if found_weights:

        model_path = found_weights[0]

        return YOLO(model_path)


    # --------------------------------------------------------
    # DO NOT FALL BACK TO YOLOV8N
    # --------------------------------------------------------

    raise FileNotFoundError(
        "Custom pothole model 'best.pt' was not found. "
        "Please upload best.pt to your GitHub repository."
    )


# ============================================================
# INITIALIZE MODEL
# ============================================================

try:

    model = load_pothole_model()

    st.sidebar.success(
        "✅ Custom Pothole Model Loaded"
    )

    st.sidebar.write(
        "### 🔍 Model Classes"
    )

    st.sidebar.write(
        model.names
    )

except Exception as e:

    st.error(
        f"❌ Could not load custom pothole model: {e}"
    )

    st.info(
        "Make sure your trained best.pt file is present "
        "in the same GitHub folder as app.py."
    )

    st.stop()


# ============================================================
# MODEL VALIDATION
# ============================================================

# Display warning if model doesn't appear to contain pothole
class_names = list(model.names.values())

pothole_found = any(
    "pothole" in str(name).lower()
    for name in class_names
)

if pothole_found:

    st.sidebar.success(
        "🕳️ Pothole class detected in model!"
    )

else:

    st.sidebar.warning(
        "⚠️ 'pothole' class was not found in model.names."
    )

    st.sidebar.write(
        "Your best.pt may not be the trained pothole model."
    )


# ============================================================
# INPUT MODE
# ============================================================

mode = st.radio(
    "Select Input Source:",
    ("Image", "Video"),
    horizontal=True
)


# ============================================================
# IMAGE DETECTION
# ============================================================

if mode == "Image":

    st.header("🖼️ Image Pothole Detection")

    uploaded_file = st.file_uploader(
        "Upload Road Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        )

        col1, col2 = st.columns(2)

        # ----------------------------------------------------
        # ORIGINAL IMAGE
        # ----------------------------------------------------

        with col1:

            st.subheader(
                "Original Input"
            )

            st.image(
                image,
                width="stretch"
            )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        with col2:

            st.subheader(
                "🔍 Pothole Detection"
            )

            with st.spinner(
                "Analyzing image..."
            ):

                results = model.predict(

                    source=image,

                    conf=confidence_threshold,

                    iou=iou_threshold,

                    imgsz=640,

                    augment=enable_tta,

                    device=DEVICE,

                    verbose=False
                )

            result = results[0]

            # ------------------------------------------------
            # DRAW RESULTS
            # ------------------------------------------------

            annotated_image = result.plot()

            st.image(
                annotated_image,
                caption="Detected Potholes",
                width="stretch"
            )

            # ------------------------------------------------
            # NUMBER OF DETECTIONS
            # ------------------------------------------------

            if result.boxes is not None:

                num_detected = len(
                    result.boxes
                )

            else:

                num_detected = 0

            st.metric(
                "🕳️ Potholes Detected",
                num_detected
            )

            # ------------------------------------------------
            # DETECTION DETAILS
            # ------------------------------------------------

            if num_detected > 0:

                st.success(
                    f"Detected {num_detected} pothole(s)."
                )

                st.subheader(
                    "Detection Details"
                )

                for i, box in enumerate(
                    result.boxes
                ):

                    class_id = int(
                        box.cls[0].item()
                    )

                    confidence = float(
                        box.conf[0].item()
                    )

                    class_name = model.names[
                        class_id
                    ]

                    st.write(
                        f"**{i + 1}. {class_name}** "
                        f"— Confidence: "
                        f"{confidence:.2%}"
                    )

            else:

                st.warning(
                    "No potholes detected."
                )


# ============================================================
# VIDEO DETECTION
# ============================================================

elif mode == "Video":

    st.header(
        "🎥 Video Pothole Detection & Tracking"
    )

    uploaded_video = st.file_uploader(
        "Upload Road Video",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv"
        ]
    )

    if uploaded_video is not None:

        temp_video_path = None

        try:

            # ------------------------------------------------
            # SAVE TEMPORARY VIDEO
            # ------------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as tfile:

                tfile.write(
                    uploaded_video.read()
                )

                temp_video_path = tfile.name


            # ------------------------------------------------
            # OPEN VIDEO
            # ------------------------------------------------

            cap = cv2.VideoCapture(
                temp_video_path
            )

            if not cap.isOpened():

                st.error(
                    "❌ Error opening video file."
                )

                st.stop()


            # ------------------------------------------------
            # VIDEO INFORMATION
            # ------------------------------------------------

            total_frames = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            original_fps = cap.get(
                cv2.CAP_PROP_FPS
            )

            if original_fps <= 0:

                original_fps = 30.0


            video_width = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )

            video_height = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )

            if total_frames > 0:

                video_duration = (
                    total_frames
                    / original_fps
                )

            else:

                video_duration = 0


            # ------------------------------------------------
            # VIDEO INFORMATION DISPLAY
            # ------------------------------------------------

            info1, info2, info3, info4 = (
                st.columns(4)
            )

            info1.metric(
                "Video Duration",
                f"{video_duration:.1f}s"
            )

            info2.metric(
                "FPS",
                f"{original_fps:.1f}"
            )

            info3.metric(
                "Resolution",
                f"{video_width}×{video_height}"
            )

            info4.metric(
                "Total Frames",
                total_frames
            )


            st.divider()


            # ------------------------------------------------
            # PROCESSING AREA
            # ------------------------------------------------

            st.subheader(
                "🚀 Processing Video"
            )

            st.write(
                f"Running inference on: "
                f"**{DEVICE_NAME}**"
            )

            if DEVICE == "cpu":

                st.caption(
                    "CPU mode detected. "
                    "Frame skipping and reduced "
                    "resolution are enabled."
                )


            # ------------------------------------------------
            # DISPLAY AREA
            # ------------------------------------------------

            st_frame = st.empty()


            # ------------------------------------------------
            # PROGRESS BAR
            # ------------------------------------------------

            progress_bar = st.progress(
                0,
                text="Starting..."
            )


            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            metric1, metric2, metric3 = (
                st.columns(3)
            )

            current_metric = metric1.empty()

            total_metric = metric2.empty()

            speed_metric = metric3.empty()


            # ------------------------------------------------
            # TRACKING VARIABLES
            # ------------------------------------------------

            unique_track_ids = set()

            frame_number = 0

            processed_frames = 0

            processing_start = time.time()


            # ------------------------------------------------
            # VIDEO LOOP
            # ------------------------------------------------

            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:
                    break

                frame_number += 1


                # --------------------------------------------
                # FRAME SKIPPING
                # --------------------------------------------

                if (
                    frame_number
                    % frame_skip
                    != 0
                ):

                    continue


                processed_frames += 1


                # --------------------------------------------
                # RESIZE VIDEO
                # --------------------------------------------

                height, width = (
                    frame.shape[:2]
                )

                if width > max_video_width:

                    scale = (
                        max_video_width
                        / width
                    )

                    new_width = int(
                        width * scale
                    )

                    new_height = int(
                        height * scale
                    )

                    frame = cv2.resize(
                        frame,
                        (
                            new_width,
                            new_height
                        ),
                        interpolation=cv2.INTER_AREA
                    )


                # --------------------------------------------
                # YOLO TRACKING
                # --------------------------------------------

                results = model.track(

                    source=frame,

                    conf=confidence_threshold,

                    iou=iou_threshold,

                    imgsz=video_imgsz,

                    persist=True,

                    tracker="botsort.yaml",

                    device=DEVICE,

                    verbose=False
                )


                result = results[0]


                # --------------------------------------------
                # TRACKING INFORMATION
                # --------------------------------------------

                frame_potholes = 0


                if (
                    result.boxes is not None
                    and result.boxes.id is not None
                ):

                    track_ids = (
                        result
                        .boxes
                        .id
                        .int()
                        .cpu()
                        .tolist()
                    )


                    for track_id in track_ids:

                        unique_track_ids.add(
                            track_id
                        )


                    frame_potholes = len(
                        track_ids
                    )


                elif result.boxes is not None:

                    frame_potholes = len(
                        result.boxes
                    )


                # --------------------------------------------
                # DRAW DETECTIONS
                # --------------------------------------------

                annotated_frame = (
                    result.plot()
                )


                # --------------------------------------------
                # BGR → RGB
                # --------------------------------------------

                rgb_frame = cv2.cvtColor(
                    annotated_frame,
                    cv2.COLOR_BGR2RGB
                )


                # --------------------------------------------
                # DISPLAY FRAME
                # --------------------------------------------

                st_frame.image(
                    rgb_frame,
                    width="stretch"
                )


                # --------------------------------------------
                # UPDATE METRICS
                # --------------------------------------------

                current_metric.metric(
                    "🕳️ Potholes in Current Frame",
                    frame_potholes
                )


                total_metric.metric(
                    "🎯 Unique Track IDs",
                    len(unique_track_ids)
                )


                # --------------------------------------------
                # PROCESSING SPEED
                # --------------------------------------------

                elapsed = (
                    time.time()
                    - processing_start
                )


                if elapsed > 0:

                    processing_fps = (
                        processed_frames
                        / elapsed
                    )

                else:

                    processing_fps = 0


                speed_metric.metric(
                    "⚡ Processing FPS",
                    f"{processing_fps:.1f}"
                )


                # --------------------------------------------
                # PROGRESS
                # --------------------------------------------

                if total_frames > 0:

                    progress = min(
                        frame_number
                        / total_frames,
                        1.0
                    )

                else:

                    progress = 0


                processed_video_time = (
                    frame_number
                    / original_fps
                )


                progress_bar.progress(

                    progress,

                    text=(
                        f"Processing "
                        f"{progress * 100:.1f}% | "
                        f"Video: "
                        f"{processed_video_time:.1f}s / "
                        f"{video_duration:.1f}s"
                    )
                )


            # =================================================
            # FINISHED
            # =================================================

            cap.release()


            processing_time = (
                time.time()
                - processing_start
            )


            st.divider()


            st.success(
                "✅ Video processing completed!"
            )


            # ------------------------------------------------
            # FINAL RESULTS
            # ------------------------------------------------

            result1, result2, result3 = (
                st.columns(3)
            )


            result1.metric(
                "🕳️ Unique Track IDs",
                len(unique_track_ids)
            )


            result2.metric(
                "⏱️ Processing Time",
                f"{processing_time:.1f}s"
            )


            if processing_time > 0:

                average_fps = (
                    processed_frames
                    / processing_time
                )

            else:

                average_fps = 0


            result3.metric(
                "⚡ Average FPS",
                f"{average_fps:.1f}"
            )


            # ------------------------------------------------
            # SPEED COMPARISON
            # ------------------------------------------------

            if video_duration > 0:

                if processing_time < video_duration:

                    realtime_multiplier = (
                        video_duration
                        / processing_time
                    )

                    st.success(
                        f"🚀 Processing was "
                        f"{realtime_multiplier:.2f}× "
                        f"faster than the video duration."
                    )

                else:

                    slowdown = (
                        processing_time
                        / video_duration
                    )

                    st.info(
                        f"Processing took "
                        f"{slowdown:.2f}× "
                        f"the video duration."
                    )


        except Exception as e:

            st.error(
                f"❌ Error processing video: {e}"
            )


        finally:

            # ------------------------------------------------
            # CLEANUP
            # ------------------------------------------------

            if (
                temp_video_path is not None
                and os.path.exists(
                    temp_video_path
                )
            ):

                try:

                    os.remove(
                        temp_video_path
                    )

                except Exception:

                    pass
