# Temporal Drowsiness Detection

A real-time driver drowsiness detection system that combines pose estimation, facial landmark analysis, and temporal reasoning to reduce false positives from single-frame predictions.

---

## Overview

Detecting driver drowsiness using computer vision is challenging because many visual cues are inherently ambiguous when observed in isolation. Short eye closures, brief yawns, head movements, or hand gestures can easily trigger false alarms if treated as frame-level events.

This project addresses the problem by **shifting from per-frame classification to sequence-level reasoning**, where drowsiness is inferred from sustained behavioral patterns over time rather than instantaneous visual signals.

---

## Key Idea

> **Drowsiness is a temporal behavior, not a single-frame visual state.**

Instead of asking:
- “Is the driver drowsy in this frame?”

The system evaluates:
- “Is there a consistent pattern of fatigue-related behavior across time?”

---

## System Architecture

The system consists of three main layers:

### 1. Visual Feature Extraction

**Pose Estimation (YOLOv8 Pose)**
- Head drop detection using body-relative ratios (camera-agnostic)
- Wrist tracking for activity analysis

**Facial Analysis (MediaPipe FaceMesh)**
- Eye closure detection using Eye Aspect Ratio (EAR)
- Yawning detection using Mouth Aspect Ratio (MAR)

---

### 2. Temporal Reasoning Engine

Instead of binary flags, each cue contributes to a **continuous fatigue score** that evolves over time.

Key mechanisms:
- Streak-based confirmation for eyes closed and yawning
- Event-based handling for hand gestures (burst detection)
- Activity override using wrist movement variance
- Adaptive score decay to allow recovery from drowsy states

---

### 3. State Machine

The final driver state is determined using a priority-based state machine:

- `ALERT`
- `FATIGUED`
- `DROWSY`

State transitions depend on accumulated temporal evidence rather than majority voting or single-frame thresholds.

---

## Design Decisions

- **Temporal streaks over sliding windows**  
  Prevents short, noisy gestures from inflating drowsiness scores.

- **Hand activity treated as an event, not a fatigue signal**  
  Reduces false positives caused by face-touching or brief gestures.

- **Activity-based override**  
  High wrist movement variance indicates alertness and accelerates recovery.

- **Interpretable scoring instead of end-to-end deep learning**  
  Enables debugging, reasoning, and explainability for each decision path.

---

## Scoring Logic (High-Level)

- Head drop + facial fatigue cues increase score
- Sustained eye closure and yawning amplify fatigue
- Physical activity reduces or resets score
- Adaptive decay speeds up recovery from false drowsy states

---

## Project Status

This project is part of an ongoing exploration of **temporal reasoning in computer vision systems**.  
Future work includes:
- Code refactoring and modularization
- Visualization of intermediate signals
- Quantitative evaluation on recorded driving sessions

---

## Technologies Used

- Python
- OpenCV
- Ultralytics YOLOv8 (Pose Estimation)
- MediaPipe FaceMesh
- NumPy

---

## Author

This project was developed as an individual exploration of real-time vision systems and temporal decision-making, with a focus on robustness, interpretability, and practical deployment considerations.
