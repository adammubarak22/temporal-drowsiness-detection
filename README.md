# temporal-drowsiness-detection

A driver drowsiness detection system that combines object detection, pose estimation, and temporal reasoning to reduce false positives from single-frame predictions.

## Problem Statement
Detecting driver drowsiness is challenging because visual cues such as closed eyes or hand-to-mouth gestures are often ambiguous when observed in a single frame. Occlusions, short gestures, and temporary movements frequently lead to false positives.

## High-Level Approach
This project addresses the problem by combining:
- YOLO-based object detection for facial and hand-related cues
- Pose estimation for head drop and wrist position analysis
- Temporal logic using streak confirmation and sliding window analysis

Instead of relying on per-frame classification, decisions are made at the sequence level.

## Key Design Decisions
- Hand-to-mouth gestures are only confirmed after consecutive frame streaks to reduce noise
- Hand-to-mouth events are excluded from the sliding drowsiness window to prevent short gesture inflation
- Wrist movement variance is used to detect high activity and override drowsiness predictions
- Final predictions follow a priority-based decision logic rather than majority voting

## Status
This project is part of an ongoing exploration of temporal reasoning in computer vision systems. Further refactoring and documentation improvements are planned.
