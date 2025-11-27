import cv2
# import insightface
# from insightface.app import FaceAnalysis
import numpy as np


#app = FaceAnalysis(name="buffalo_l")
#app.prepare(ctx_id=0, det_size=(640, 640))

import os


# app = FaceAnalysis(name="buffalo_l")
# app.prepare(ctx_id=-1, det_size=(640, 640))



IMG_SIZE = 299

import cv2

def robust_face_detection(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    detectors = [
        #("insightface", lambda: insightface_detector(img_rgb)),
        ("opencv_haar", lambda: opencv_haar_detector(img_rgb)),
        ("dlib_hog", lambda: dlib_hog_detector(img_rgb)),
        #("center_crop", lambda: center_crop_fallback(img_rgb)),
        #("multiple_scales", lambda: multi_scale_detection(img_rgb))
    ]

    for name, detector in detectors:
        face = detector()
        if face is not None:
            return face

    # ✅ Explicit failure AFTER all 5 strategies
    return None





# def insightface_detector(img_rgb):
#     faces = app.get(img_rgb)
#     if len(faces) > 0:
#         face = faces[0]
#         x1, y1, x2, y2 = face.bbox.astype(int)
#         return safe_crop(img_rgb, x1, y1, x2, y2)
#     return None


def opencv_haar_detector(img_rgb):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    profile_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_profileface.xml'
    )

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return safe_crop(img_rgb, x, y, x + w, y + h)

    faces = profile_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return safe_crop(img_rgb, x, y, x + w, y + h)

    return None


def dlib_hog_detector(img_rgb):
    try:
        import dlib
        detector = dlib.get_frontal_face_detector()
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        faces = detector(gray, 1)

        if len(faces) > 0:
            f = faces[0]
            return safe_crop(img_rgb, f.left(), f.top(), f.right(), f.bottom())
    except ImportError:
        pass
    return None


def center_crop_fallback(img_rgb):
    h, w = img_rgb.shape[:2]
    crop_size = int(min(h, w) * 0.8)
    y1 = (h - crop_size) // 2
    x1 = (w - crop_size) // 2
    return safe_crop(img_rgb, x1, y1, x1 + crop_size, y1 + crop_size)


def multi_scale_detection(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    for scale in [1.0, 1.2, 1.5, 0.8]:
        scaled = cv2.resize(gray, None, fx=scale, fy=scale)
        faces = face_cascade.detectMultiScale(scaled, 1.1, 4)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            x, y, w, h = int(x / scale), int(y / scale), int(w / scale), int(h / scale)
            return safe_crop(img_rgb, x, y, x + w, y + h)

    return None


def safe_crop(img, x1, y1, x2, y2):
    """Ensure crop coordinates are within image bounds"""
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return cv2.resize(crop, (299, 299))




def robust_face_detection_from_array(img_bgr):
    """Same as robust_face_detection but accepts a numpy BGR image instead of a file path."""
    if img_bgr is None:
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    detectors = [
        #("insightface", lambda: insightface_detector(img_rgb)),
        ("opencv_haar", lambda: opencv_haar_detector(img_rgb)),
        ("dlib_hog", lambda: dlib_hog_detector(img_rgb)),
        #("center_crop", lambda: center_crop_fallback(img_rgb)),
        ("multiple_scales", lambda: multi_scale_detection(img_rgb)),
    ]

    for name, detector in detectors:
        face = detector()
        if face is not None:
            return face

    return None



