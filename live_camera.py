"""
Live Camera Hair Type Detection — real-time classification using your webcam.

How it works:
    1. Press R to start a scan — the model collects frames for ~2 seconds.
    2. Predictions are averaged across all sampled frames to produce a
       stable result.
    3. The result stays locked on screen until you press R again to re-scan.

Usage:
    python live_camera.py
    python live_camera.py --checkpoint checkpoints/best_model.pth --camera 0

Controls:
    R      — start a new scan / re-evaluate
    S      — save a screenshot
    Q/ESC  — quit
"""

import argparse
import time
import os
from datetime import datetime

import cv2
import torch
import numpy as np
from PIL import Image

from dataset import get_val_transforms, CLASS_NAMES, TYPE_LABELS
from model import create_model


# ── Colors & styling ─────────────────────────────────────────────────────────

# BGR colors for OpenCV
COLORS = {
    "Straight": (255, 180, 50),   # light blue
    "Wavy":     (50, 220, 255),   # yellow-orange
    "Curly":    (100, 255, 100),  # green
    "Kinky":    (180, 100, 255),  # purple
}

SCAN_COLOR = (0, 255, 255)   # cyan border while scanning
BORDER_COLOR = (200, 200, 0) # teal border showing analysis region

BG_COLOR   = (30, 30, 30)
TEXT_WHITE = (255, 255, 255)
TEXT_GRAY  = (180, 180, 180)
BAR_BG     = (60, 60, 60)


# ── Model loading ────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = ckpt.get("class_names", CLASS_NAMES)
    model = create_model(num_classes=len(class_names), pretrained=False, freeze_backbone=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


# ── Preprocessing ────────────────────────────────────────────────────────────

def get_analysis_roi(frame):
    """Return the center crop ROI coordinates (square) used for prediction."""
    h, w = frame.shape[:2]
    size = min(h, w) * 3 // 4  # 75% of the smaller dimension
    cx, cy = w // 2, h // 2
    x1 = cx - size // 2
    y1 = cy - size // 2
    x2 = x1 + size
    y2 = y1 + size
    return x1, y1, x2, y2


def preprocess_frame(frame, device):
    """Crop the analysis ROI and convert to a model-ready tensor."""
    x1, y1, x2, y2 = get_analysis_roi(frame)
    roi = frame[y1:y2, x1:x2]
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    transform = get_val_transforms()
    tensor = transform(pil_img).unsqueeze(0).to(device)
    return tensor


# ── Drawing helpers ──────────────────────────────────────────────────────────

def draw_rounded_rect(img, pt1, pt2, color, radius=15, thickness=-1, alpha=0.7):
    """Draw a semi-transparent rounded rectangle overlay."""
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_analysis_border(frame, scanning=False):
    """Draw a rectangle border showing the region the model analyzes."""
    x1, y1, x2, y2 = get_analysis_roi(frame)
    color = SCAN_COLOR if scanning else BORDER_COLOR
    thickness = 3 if scanning else 2

    # Animated dashed corners when scanning
    if scanning:
        # Pulsing effect — alternate thickness based on time
        pulse = int(time.time() * 4) % 2
        thickness = 3 + pulse

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Corner accents (short lines at each corner for emphasis)
    corner_len = 25
    # Top-left
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness + 1)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness + 1)
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness + 1)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness + 1)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness + 1)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness + 1)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness + 1)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness + 1)

    # Label
    label = "SCANNING..." if scanning else "Analysis Region"
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_results_panel(frame, pred_class, confidence, all_probs):
    """Draw a results overlay on the left side of the frame."""
    h, w = frame.shape[:2]

    # ── Panel dimensions — calculate from content ────────────────────────
    pad = 15          # inner padding
    bar_h = 16        # height of each probability bar
    bar_gap = 32      # vertical space between bar rows
    label_w = 75      # width reserved for class-name labels
    pct_w = 40        # width reserved for percentage text

    panel_w = 340
    # Header area: 95px, plus 4 bars with gaps, plus bottom padding
    panel_h = 100 + len(CLASS_NAMES) * bar_gap + pad

    margin = 15
    x1, y1 = margin, margin
    x2, y2 = x1 + panel_w, y1 + panel_h

    # Semi-transparent dark background
    draw_rounded_rect(frame, (x1, y1), (x2, y2), BG_COLOR, radius=12, alpha=0.85)

    # ── Title ────────────────────────────────────────────────────────────
    color = COLORS.get(pred_class, TEXT_WHITE)
    type_num = CLASS_NAMES.index(pred_class) + 1
    title = f"Type {type_num} - {pred_class}"

    cv2.putText(frame, "HAIR TYPE DETECTED", (x1 + pad, y1 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_GRAY, 1, cv2.LINE_AA)

    cv2.putText(frame, title, (x1 + pad, y1 + 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)

    # ── Confidence ───────────────────────────────────────────────────────
    conf_text = f"Confidence: {confidence:.1%}"
    cv2.putText(frame, conf_text, (x1 + pad, y1 + 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_WHITE, 1, cv2.LINE_AA)

    # ── Probability bars ─────────────────────────────────────────────────
    bar_start_y = y1 + 110
    bar_x = x1 + pad + label_w
    bar_max_w = panel_w - pad * 2 - label_w - pct_w - 5  # ensure bars + pct fit

    for i, cls in enumerate(CLASS_NAMES):
        prob = all_probs[cls]
        cy = bar_start_y + i * bar_gap

        # Class label
        cv2.putText(frame, cls, (x1 + pad, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_GRAY, 1, cv2.LINE_AA)

        # Background bar
        cv2.rectangle(frame,
                      (bar_x, cy - bar_h // 2),
                      (bar_x + bar_max_w, cy + bar_h // 2),
                      BAR_BG, -1)

        # Filled bar
        fill_w = max(0, int(prob * bar_max_w))
        bar_color = COLORS.get(cls, TEXT_WHITE)
        if fill_w > 0:
            cv2.rectangle(frame,
                          (bar_x, cy - bar_h // 2),
                          (bar_x + fill_w, cy + bar_h // 2),
                          bar_color, -1)

        # Percentage text (right-aligned, inside the panel)
        pct_text = f"{prob:.0%}"
        cv2.putText(frame, pct_text, (bar_x + bar_max_w + 6, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_WHITE, 1, cv2.LINE_AA)


def draw_scan_progress(frame, elapsed, duration):
    """Draw a progress bar while scanning is in progress."""
    h, w = frame.shape[:2]
    bar_w = 300
    bar_h = 8
    bx = (w - bar_w) // 2
    by = h - 60

    progress = min(elapsed / duration, 1.0)

    # Background
    cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), BAR_BG, -1)
    # Fill
    fill = int(progress * bar_w)
    if fill > 0:
        cv2.rectangle(frame, (bx, by), (bx + fill, by + bar_h), SCAN_COLOR, -1)

    # Label
    remaining = max(0, duration - elapsed)
    label = f"Scanning... {remaining:.1f}s remaining"
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.putText(frame, label, ((w - text_size[0]) // 2, by - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, SCAN_COLOR, 1, cv2.LINE_AA)


def draw_status_bar(frame, fps, state_label="IDLE"):
    """Draw a bottom status bar with FPS and controls help."""
    h, w = frame.shape[:2]
    bar_h = 35
    cv2.rectangle(frame, (0, h - bar_h), (w, h), BG_COLOR, -1)

    if state_label == "SCANNING":
        status_color = SCAN_COLOR
    elif state_label == "RESULT":
        status_color = (0, 255, 100)
    else:
        status_color = TEXT_GRAY

    cv2.circle(frame, (20, h - bar_h // 2), 6, status_color, -1)
    cv2.putText(frame, state_label, (35, h - bar_h // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)

    cv2.putText(frame, f"FPS: {fps:.0f}", (140, h - bar_h // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_GRAY, 1, cv2.LINE_AA)

    help_text = "R: scan/re-scan  |  S: screenshot  |  Q: quit"
    text_size = cv2.getTextSize(help_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
    cv2.putText(frame, help_text, (w - text_size[0] - 15, h - bar_h // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_GRAY, 1, cv2.LINE_AA)


def draw_idle_prompt(frame):
    """Draw a centered prompt when no scan has been done yet."""
    h, w = frame.shape[:2]
    msg = "Press R to scan your hair type"
    text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    tx = (w - text_size[0]) // 2
    ty = h // 2 + text_size[1] // 2

    # Dark backdrop
    bx1, by1 = tx - 20, ty - text_size[1] - 15
    bx2, by2 = tx + text_size[0] + 20, ty + 15
    draw_rounded_rect(frame, (bx1, by1), (bx2, by2), BG_COLOR, radius=10, alpha=0.75)

    cv2.putText(frame, msg, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_WHITE, 2, cv2.LINE_AA)


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Live camera hair type detection")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pth")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--width", type=int, default=1280, help="Camera width")
    parser.add_argument("--height", type=int, default=720, help="Camera height")
    parser.add_argument("--scan_duration", type=float, default=2.0,
                        help="How many seconds to collect frames during a scan")
    parser.add_argument("--sample_interval", type=float, default=0.2,
                        help="Seconds between frame samples during a scan")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    print(f"  Loading model from {args.checkpoint} …")

    model = load_model(args.checkpoint, device)
    print("  Model loaded ✓")

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print("  ❌ Could not open camera. Check your camera index (--camera 0, 1, 2, …)")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Camera opened ✓  ({actual_w}×{actual_h})")
    print(f"  Scan duration: {args.scan_duration}s")
    print(f"\n  Press R to scan, Q to quit.\n")

    # ── State machine ────────────────────────────────────────────────────
    # States: "idle" → "scanning" → "result" → (R pressed) → "scanning" …
    state = "idle"

    # Scan state
    scan_start = 0.0
    scan_probs_buffer = []       # list of probability dicts collected during scan
    last_sample_time = 0.0

    # Result state (locked after scan completes)
    result_class = None
    result_confidence = 0.0
    result_probs = {c: 0.0 for c in CLASS_NAMES}

    # FPS tracking
    fps = 0.0
    frame_count = 0
    fps_start = time.time()

    os.makedirs("screenshots", exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("  ❌ Failed to read from camera")
            break

        now = time.time()
        display = frame.copy()

        # ── State: SCANNING ──────────────────────────────────────────────
        if state == "scanning":
            elapsed = now - scan_start

            # Sample a frame at intervals
            if now - last_sample_time >= args.sample_interval:
                last_sample_time = now
                with torch.no_grad():
                    tensor = preprocess_frame(frame, device)
                    outputs = model(tensor)
                    probs = torch.softmax(outputs, dim=1).squeeze()
                    prob_dict = {CLASS_NAMES[i]: probs[i].item() for i in range(len(CLASS_NAMES))}
                    scan_probs_buffer.append(prob_dict)

            # Draw scanning UI
            draw_analysis_border(display, scanning=True)
            draw_scan_progress(display, elapsed, args.scan_duration)

            # Check if scan is done
            if elapsed >= args.scan_duration:
                # Average all collected probability dicts
                avg_probs = {c: 0.0 for c in CLASS_NAMES}
                for p in scan_probs_buffer:
                    for c in CLASS_NAMES:
                        avg_probs[c] += p[c]
                n = len(scan_probs_buffer)
                if n > 0:
                    avg_probs = {c: v / n for c, v in avg_probs.items()}

                result_class = max(avg_probs, key=avg_probs.get)
                result_confidence = avg_probs[result_class]
                result_probs = avg_probs

                state = "result"
                print(f"  ✓ Scan complete — {result_class} ({result_confidence:.1%}) "
                      f"[averaged {n} frames]")

        # ── State: RESULT (locked display) ───────────────────────────────
        elif state == "result":
            draw_analysis_border(display, scanning=False)
            draw_results_panel(display, result_class, result_confidence, result_probs)

        # ── State: IDLE ──────────────────────────────────────────────────
        else:
            draw_analysis_border(display, scanning=False)
            draw_idle_prompt(display)

        # ── FPS counter ──────────────────────────────────────────────────
        frame_count += 1
        elapsed_fps = now - fps_start
        if elapsed_fps >= 1.0:
            fps = frame_count / elapsed_fps
            frame_count = 0
            fps_start = now

        state_label = state.upper()
        draw_status_bar(display, fps, state_label)

        cv2.imshow("Hair Type Detector", display)

        # ── Key handling ─────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):  # Q or ESC
            break
        elif key in (ord("r"), ord("R")):    # R — start / re-scan
            state = "scanning"
            scan_start = time.time()
            scan_probs_buffer = []
            last_sample_time = 0.0
            print("  🔄 Scanning started…")
        elif key in (ord("s"), ord("S")):    # S — screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshots/hair_scan_{timestamp}.png"
            cv2.imwrite(filename, display)
            print(f"  📸  Screenshot saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n  Camera closed. Goodbye!\n")


if __name__ == "__main__":
    main()
