import cv2
import os
import glob
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import deque
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report
import math

# ---------------- CONFIGURATION ----------------
MODEL_DET_PATH = "runs/detect/yolo_drowsy_person_split_v2/weights/best.pt"
MODEL_POSE_PATH = "yolo11x-pose.pt"
DATASET_ROOT = "data-slayer-3/test"

# Evaluator paths
TEST_FOLDER = "data-slayer-3/test"
CSV_PATH = "data-slayer-3/kunjaw.csv"
OUTPUT_FILE = "submission_evaluation_final_v7.csv"

# Logic thresholds
CONF_THRESHOLD = 0.4
HEAD_DROP_THRESHOLD = 110
MOVEMENT_THRESHOLD = 50.0  # threshold untuk activity (std dev)
WRIST_CONF_THRESHOLD = 0.3  # threshold minimal confidence untuk mempercayai wrist keypoint
WINDOW_SIZE = 10
WINDOW_THRESHOLD = 6
HAND_STREAK_REQUIRED = 2  # harus 2 frame berturut-turut untuk konfirmasi hand-mouth dari salah satu sumber

# UI output (unused in evaluator except for potential debug)
OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720
OUTPUT_SIZE = (OUTPUT_WIDTH, OUTPUT_HEIGHT)

# Pose selection / behavior
FORCE_POSE_FROM_DET = False  # jika True -> crop person terbesar via detector lalu jalankan pose hanya pada crop
POSE_SELECT_METHOD = "largest_bbox"  # largest_bbox | closest_center | highest_conf_sum

# Palm estimation / geometric helpers (keputusan dari visualizer)
EXTENSION_RATIO = 0.6
PALM_BEND_ANGLE = 0

# ---------------- LOAD MODELS ----------------
print("Loading models...")
try:
    model_det = YOLO(MODEL_DET_PATH)
    model_pose = YOLO(MODEL_POSE_PATH)
    print("✅ Models loaded")
except Exception as e:
    print("❌ Error loading models:", e)
    raise SystemExit(1)

# ---------------- Helpers ----------------
def crop_to_bbox(frame, bbox, pad=20):
    x1,y1,x2,y2 = map(int, bbox)
    h,w = frame.shape[:2]
    x1 = max(0, x1-pad); y1 = max(0, y1-pad)
    x2 = min(w-1, x2+pad); y2 = min(h-1, y2+pad)
    if x2 <= x1 or y2 <= y1:
        return frame.copy(), (0,0,w-1,h-1)
    return frame[y1:y2, x1:x2], (x1,y1,x2,y2)

def rotate_vector(vx, vy, angle_degrees):
    rad = math.radians(angle_degrees)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    new_vx = vx * cos_a - vy * sin_a
    new_vy = vx * sin_a + vy * cos_a
    return new_vx, new_vy

def select_primary_person_from_pose_result(pose_result, frame_shape, method="largest_bbox"):
    h, w = frame_shape[:2]
    centers = []
    bboxes = []
    conf_sums = []
    try:
        if hasattr(pose_result.keypoints, "data") and len(pose_result.keypoints.data) > 0:
            all_kpts = [kp.cpu().numpy() for kp in pose_result.keypoints.data]
            for k in all_kpts:
                xs = k[:,0]; ys = k[:,1]
                valid = (xs > 0) & (ys > 0)
                if valid.sum() == 0:
                    bboxes.append((0,0,0,0)); centers.append((w/2, h/2)); conf_sums.append(0.0); continue
                x1, x2 = xs[valid].min(), xs[valid].max()
                y1, y2 = ys[valid].min(), ys[valid].max()
                bboxes.append((x1,y1,x2,y2))
                centers.append(((x1+x2)/2.0, (y1+y2)/2.0))
                conf_sums.append(float(np.nansum(k[:,2])))
        else:
            all_xy = pose_result.keypoints.xy.cpu().numpy()
            for k in all_xy:
                xs = k[:,0]; ys = k[:,1]
                valid = (xs > 0) & (ys > 0)
                if valid.sum() == 0:
                    bboxes.append((0,0,0,0)); centers.append((w/2, h/2)); conf_sums.append(0.0); continue
                x1, x2 = xs[valid].min(), xs[valid].max()
                y1, y2 = ys[valid].min(), ys[valid].max()
                bboxes.append((x1,y1,x2,y2))
                centers.append(((x1+x2)/2.0, (y1+y2)/2.0))
                conf_sums.append(0.0)
    except Exception:
        return None

    if not bboxes:
        return None

    if method == "largest_bbox":
        areas = [max(0,(x2-x1)) * max(0,(y2-y1)) for (x1,y1,x2,y2) in bboxes]
        return int(np.argmax(areas))
    if method == "closest_center":
        img_cx, img_cy = w/2.0, h/2.0
        dists = [((cx - img_cx)**2 + (cy - img_cy)**2) for (cx,cy) in centers]
        return int(np.argmin(dists))
    if method == "highest_conf_sum":
        return int(np.argmax(conf_sums))
    return 0

def is_point_in_box(point, box):
    if point is None or box is None:
        return False
    px, py = point[:2] if len(point) >= 2 else (None, None)
    if px is None or py is None:
        return False
    x1, y1, x2, y2 = box
    return x1 < px < x2 and y1 < py < y2

# ---------------- Pose extraction (single person focus) ----------------
def get_pose_features(frame, pose_select_method=POSE_SELECT_METHOD):
    annotated_frame = frame.copy()
    is_head_drop = False
    is_hand_near_mouth = False
    debug_text = []

    l_wrist_pt = None
    r_wrist_pt = None
    face_box = None
    l_wrist_conf = None
    r_wrist_conf = None

    H, W = frame.shape[:2]
    kpts_data_local = None

    # Option A: force crop by detector -> run pose on crop (more robust when many people)
    if FORCE_POSE_FROM_DET:
        try:
            res_person = model_det.predict(frame, conf=0.3, verbose=False)[0]
            boxes = res_person.boxes.xyxy.cpu().numpy() if hasattr(res_person.boxes, "xyxy") else []
            classes = res_person.boxes.cls.cpu().numpy().astype(int) if hasattr(res_person.boxes, "cls") else []
            labels = [model_det.names[int(c)].lower() for c in classes] if len(classes)>0 else []
            person_boxes = []
            for b, lbl in zip(boxes, labels):
                if "person" in lbl:
                    person_boxes.append(b)
            if person_boxes:
                areas = [(b[2]-b[0])*(b[3]-b[1]) for b in person_boxes]
                sel_idx = int(np.argmax(areas))
                crop_box = person_boxes[sel_idx]
                crop, (cx1,cy1,cx2,cy2) = crop_to_bbox(frame, crop_box, pad=20)
                pres = model_pose.predict(crop, conf=0.5, verbose=False)[0]
                try:
                    if hasattr(pres.keypoints, "data") and len(pres.keypoints.data) > 0:
                        kd = pres.keypoints.data[0].cpu().numpy().copy()
                        kd[:,0] += cx1; kd[:,1] += cy1
                        kpts_data_local = kd
                    else:
                        kxy = pres.keypoints.xy.cpu().numpy()[0].copy()
                        kpts_data_local = np.hstack([kxy, np.zeros((kxy.shape[0],1))])
                    annotated_frame = pres.plot() if hasattr(pres, "plot") else frame.copy()
                except Exception:
                    pass
        except Exception:
            pass

    # Option B: full frame pose and pick primary person
    if kpts_data_local is None:
        try:
            res = model_pose.predict(frame, conf=0.5, verbose=False)
            pose_result = res[0]
            annotated_frame = pose_result.plot() if hasattr(pose_result, "plot") else frame.copy()
            primary_idx = select_primary_person_from_pose_result(pose_result, frame.shape, method=pose_select_method)
            try:
                if hasattr(pose_result.keypoints, "data") and len(pose_result.keypoints.data) > 0:
                    arrs = [kp.cpu().numpy() for kp in pose_result.keypoints.data]
                    if primary_idx is not None and primary_idx < len(arrs):
                        kpts_data_local = arrs[primary_idx]
                    else:
                        kpts_data_local = arrs[0]
                else:
                    all_xy = pose_result.keypoints.xy.cpu().numpy()
                    if primary_idx is not None and primary_idx < len(all_xy):
                        kxy = all_xy[primary_idx]
                    else:
                        kxy = all_xy[0]
                    kpts_data_local = np.hstack([kxy, np.zeros((kxy.shape[0],1))])
            except Exception:
                kpts_data_local = None
        except Exception:
            kpts_data_local = None

    # parse keypoints for chosen person
    try:
        if kpts_data_local is not None and kpts_data_local.shape[0] >= 11:
            nose = kpts_data_local[0][:2]; l_eye = kpts_data_local[1][:2]; r_eye = kpts_data_local[2][:2]; l_ear = kpts_data_local[3][:2]; r_ear = kpts_data_local[4][:2]
            l_shoulder = kpts_data_local[5][:2]; r_shoulder = kpts_data_local[6][:2]
            l_elbow = kpts_data_local[7][:2]; r_elbow = kpts_data_local[8][:2]
            l_wrist = kpts_data_local[9][:2]; r_wrist = kpts_data_local[10][:2]

            l_wrist_conf = float(kpts_data_local[9][2]) if kpts_data_local.shape[1] > 2 else None
            r_wrist_conf = float(kpts_data_local[10][2]) if kpts_data_local.shape[1] > 2 else None

            # --- ROTASI & PERPANJANGAN (palm estimation) ---
            try:
                if l_wrist[0] != 0 and l_elbow[0] != 0:
                    vx = l_wrist[0] - l_elbow[0]
                    vy = l_wrist[1] - l_elbow[1]
                    rot_vx, rot_vy = rotate_vector(vx, vy, PALM_BEND_ANGLE)
                    l_wrist[0] += rot_vx * EXTENSION_RATIO
                    l_wrist[1] += rot_vy * EXTENSION_RATIO
                if r_wrist[0] != 0 and r_elbow[0] != 0:
                    vx = r_wrist[0] - r_elbow[0]
                    vy = r_wrist[1] - r_elbow[1]
                    rot_vx, rot_vy = rotate_vector(vx, vy, -PALM_BEND_ANGLE)
                    r_wrist[0] += rot_vx * EXTENSION_RATIO
                    r_wrist[1] += rot_vy * EXTENSION_RATIO
            except Exception:
                pass

            # HEAD DROP
            try:
                if nose[0] != 0 and (l_shoulder[0] != 0 or r_shoulder[0] != 0):
                    if l_shoulder[0] != 0 and r_shoulder[0] != 0:
                        shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
                    elif l_shoulder[0] != 0:
                        shoulder_y = l_shoulder[1]
                    else:
                        shoulder_y = r_shoulder[1]
                    distance_head = shoulder_y - nose[1]
                    if distance_head < HEAD_DROP_THRESHOLD:
                        is_head_drop = True
                        debug_text.append("HEAD DROP")
            except Exception:
                pass

            # FACE BOX (improved boxed used in visualizer)
            try:
                face_points = [p for p in [nose, l_eye, r_eye, l_ear, r_ear] if (p is not None and p[0] != 0)]
                if face_points:
                    xs = [p[0] for p in face_points]; ys = [p[1] for p in face_points]
                    pad = 0
                    fx1 = max(0, min(xs)-pad); fy1 = max(0, min(ys)-pad)
                    fx2 = min(W-1, max(xs)+pad); fy2 = min(H-1, max(ys)+pad+200)
                    face_box = [fx1, fy1, fx2, fy2]
            except Exception:
                pass

            # WRIST -> check proximity to face_box (using confidence if available)
            try:
                if l_wrist is not None and l_wrist[0] != 0:
                    lx, ly = float(l_wrist[0]), float(l_wrist[1])
                    lc = float(l_wrist_conf) if l_wrist_conf is not None else 0.0
                    l_wrist_pt = (lx, ly, lc)
                    cond = (lc >= WRIST_CONF_THRESHOLD) if l_wrist_conf is not None else False
                    if face_box is not None and cond and is_point_in_box((lx, ly), face_box):
                        is_hand_near_mouth = True
                        debug_text.append("L-HAND ON FACE")
            except Exception:
                pass
            try:
                if r_wrist is not None and r_wrist[0] != 0:
                    rx, ry = float(r_wrist[0]), float(r_wrist[1])
                    rc = float(r_wrist_conf) if r_wrist_conf is not None else 0.0
                    r_wrist_pt = (rx, ry, rc)
                    cond = (rc >= WRIST_CONF_THRESHOLD) if r_wrist_conf is not None else False
                    if face_box is not None and cond and is_point_in_box((rx, ry), face_box):
                        is_hand_near_mouth = True
                        debug_text.append("R-HAND ON FACE")
            except Exception:
                pass

    except Exception:
        pass

    pose_status_str = " | ".join(debug_text) if debug_text else "Pose: Normal"
    return is_head_drop, is_hand_near_mouth, annotated_frame, pose_status_str, l_wrist_pt, r_wrist_pt, face_box, (l_wrist_conf, r_wrist_conf)

# ---------------- Detector Hand_Mouth overlap check ----------------
def detector_hand_mouth_near_face(det_result, face_box, conf_thr=0.25, min_overlap_pixels=1):
    if face_box is None:
        return False
    try:
        boxes = det_result.boxes.xyxy.cpu().numpy()
        classes = det_result.boxes.cls.cpu().numpy().astype(int)
        confs = det_result.boxes.conf.cpu().numpy() if hasattr(det_result.boxes, "conf") else np.ones(len(classes))
        fx1, fy1, fx2, fy2 = map(float, face_box)
        face_area = max(0.0, fx2 - fx1) * max(0.0, fy2 - fy1)
        if face_area <= 0:
            return False
        for b, c, conf in zip(boxes, classes, confs):
            label = model_det.names[int(c)]
            if label != "Hand_Mouth":
                continue
            if conf < conf_thr:
                continue
            x1, y1, x2, y2 = map(float, b)
            ix1 = max(fx1, x1); iy1 = max(fy1, y1)
            ix2 = min(fx2, x2); iy2 = min(fy2, y2)
            inter_w = max(0.0, ix2 - ix1); inter_h = max(0.0, iy2 - iy1)
            inter_area = inter_w * inter_h
            if inter_area >= min_overlap_pixels:
                return True
    except Exception:
        pass
    return False

# ---------------- Combined logic ----------------
def analyze_combined_logic(det_result, is_head_drop, is_hand_near_mouth, face_box):
    try:
        class_indices = det_result.boxes.cls.cpu().numpy().astype(int)
        labels = set([model_det.names[idx] for idx in class_indices])
    except Exception:
        labels = set()

    det_hand_mouth_near = detector_hand_mouth_near_face(det_result, face_box, conf_thr=0.25)
    hand_mouth_detected = det_hand_mouth_near or is_hand_near_mouth

    if hand_mouth_detected:
        return True, True

    if "Sunglasses" in labels:
        if "Open_Mouth" in labels:
            return True, False
        return is_head_drop, False

    if "Closed_Eyes" in labels:
        if "Mask" in labels:
            return True, False
        if "Closed_Mouth" in labels:
            return True, False
        if "Open_Mouth" in labels:
            return True, False
        return True, False

    eyes_missing = "Closed_Eyes" not in labels and "Open_Eyes" not in labels
    if eyes_missing:
        return is_head_drop, False

    return False, False

# ---------------- Activity (movement) calculation ----------------
def analyze_activity_level(wrist_history):
    """
    - expects wrist_history to contain items like (x, y, conf) or None
    - requires at least 5 valid wrist points with conf >= WRIST_CONF_THRESHOLD
    - compute std dev of x and y across qualified points, average them
    - if average std > MOVEMENT_THRESHOLD -> considered ACTIVE
    """
    if len(wrist_history) < 5:
        return False
    # filter points that are not None and have confidence >= threshold
    qualified = []
    for p in wrist_history:
        if p is None:
            continue
        # allow both formats just in case: (x,y) or (x,y,conf)
        if len(p) >= 3:
            x, y, c = p[0], p[1], p[2]
        else:
            x, y, c = p[0], p[1], 0.0
        if c is None:
            continue
        if c >= WRIST_CONF_THRESHOLD:
            qualified.append((x, y))
    if len(qualified) < 5:
        return False
    points = np.array(qualified)
    std_x = np.std(points[:,0])
    std_y = np.std(points[:,1])
    return ((std_x + std_y) / 2.0) > MOVEMENT_THRESHOLD

# ---------------- Evaluator: predict sequence using full logic ----------------
def predict_sequence_full_scan(folder_path):
    image_paths = sorted(glob.glob(os.path.join(folder_path, "*.*")))
    if not image_paths:
        return 0

    history = deque(maxlen=WINDOW_SIZE)     # sliding window for drowsy-only frames (hand-mouth excluded)
    l_wrist_hist = deque(maxlen=WINDOW_SIZE)
    r_wrist_hist = deque(maxlen=WINDOW_SIZE)

    detected_drowsy_by_window = False
    detected_hand_mouth = False
    detected_high_activity = False

    # streaks for confirmation (pose/det)
    pose_hand_streak = 0
    det_hand_streak = 0
    confirmed_hand_mouth = False

    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        # detection
        res_det = model_det.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]

        # pose
        is_head_drop, is_hand_in_box, pose_frame, pose_info, lw, rw, face_box, wrist_confs = get_pose_features(frame)

        # update streaks (confirm hand-mouth only after streak)
        if is_hand_in_box:
            pose_hand_streak += 1
        else:
            pose_hand_streak = 0

        det_near = detector_hand_mouth_near_face(res_det, face_box, conf_thr=0.25)
        if det_near:
            det_hand_streak += 1
        else:
            det_hand_streak = 0

        if (pose_hand_streak >= HAND_STREAK_REQUIRED) or (det_hand_streak >= HAND_STREAK_REQUIRED):
            confirmed_hand_mouth = True
            detected_hand_mouth = True

        # activity - append wrist tuples (x,y,conf) or None
        # get_pose_features returns lw/rw as (x,y,conf) or None in our implementation
        l_wrist_hist.append(lw)
        r_wrist_hist.append(rw)
        is_active = analyze_activity_level(l_wrist_hist) or analyze_activity_level(r_wrist_hist)
        if is_active:
            detected_high_activity = True

        # combined frame logic (instant)
        is_drowsy_frame, hand_mouth_instant = analyze_combined_logic(res_det, is_head_drop, is_hand_in_box, face_box)
        # but we only honor hand_mouth if confirmed via streak
        hand_mouth_detected = confirmed_hand_mouth

        # ---------- Sliding window update (IMPORTANT) ----------
        # We DO NOT let confirmed hand-mouth contribute +1 to the drowsy sliding window.
        # This prevents short/partial clips with repeated hand-mouth from inflating window.
        if is_active:
            # active movement: do not count as drowsy regardless; preserve UI semantics
            history.append(0)
        else:
            if hand_mouth_detected:
                # flagged, but do NOT add to drowsy-window count
                history.append(0)
            else:
                # only add 1 if frame is drowsy (non-hand-mouth)
                history.append(1 if is_drowsy_frame else 0)

        if len(history) == WINDOW_SIZE and sum(history) >= WINDOW_THRESHOLD:
            detected_drowsy_by_window = True

    # final decision priority preserved:
    if detected_hand_mouth:
        return 1
    if detected_high_activity:
        return 0
    if detected_drowsy_by_window:
        return 1
    return 0

# ---------------- Batch evaluator (CSV) ----------------
def run_evaluator_from_csv():
    if not os.path.exists(CSV_PATH):
        print(f"Error: file {CSV_PATH} not found.")
        return

    df_truth = pd.read_csv(CSV_PATH)
    # normalize columns if user provided id/label differently
    if 'id' not in df_truth.columns or 'label' not in df_truth.columns:
        df_truth.columns = ['id', 'label']

    print(f"Total sequences: {len(df_truth)}")
    y_true = []
    y_pred = []
    submission_data = []
    wrong_list = []

    for index, row in tqdm(df_truth.iterrows(), total=len(df_truth)):
        folder_name = str(row['id']).strip()
        gt = int(row['label'])
        folder_path = os.path.join(TEST_FOLDER, folder_name)
        if not os.path.exists(folder_path):
            pred = 0
        else:
            pred = predict_sequence_full_scan(folder_path)
        y_true.append(gt)
        y_pred.append(pred)
        submission_data.append({"id": folder_name, "label": pred})
        if pred != gt:
            wrong_list.append(f"{folder_name} (Pred:{pred} | GT:{gt})")

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    print("\n=== EVALUATION RESULTS ===")
    print(f"Accuracy: {acc:.2%}")
    print(f"F1-macro: {f1_macro:.4f}")
    print(classification_report(y_true, y_pred, target_names=['Alert (0)', 'Drowsy (1)']))

    if wrong_list:
        print(f"\nWrong ({len(wrong_list)}):")
        for item in wrong_list[:20]:
            print(" -", item)
        if len(wrong_list) > 20:
            print(" ... and", len(wrong_list)-20, "more")

    df_result = pd.DataFrame(submission_data)
    df_result.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved predictions to {OUTPUT_FILE}")

# ---------------- Main entry (ONLY RUN EVALUATOR) ----------------
if __name__ == "__main__":
    print("Running batch evaluator from CSV...")
    run_evaluator_from_csv()
