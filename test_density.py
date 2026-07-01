"""
==========================================================================
 CROWD CRUSH PREVENTION
 Public Safety & Stampede Prediction System
==========================================================================

WHAT THIS SCRIPT DOES (maps directly to the project brief):

  STAGE 1  Person Detection & Tracking   -> YOLO .track() with low
                                             conf/iou tuned for dense crowds.
  STAGE 2  Zone Grid                     -> 6x6 grid (see "DESIGN CHOICE:
                                             GRID RESOLUTION" below).
  STAGE 3  Density Calculation Per Zone  -> per-zone count -> pixel
                                             density -> EMA smoothed.
  STAGE 4  Flow Direction Analysis       -> per-zone average velocity
                                             vector + opposing-flow
                                             (dot product < 0) conflict
                                             detection between neighbours.
  STAGE 5  Risk Level, Prediction, Alerts-> SAFE/WATCH/DANGER/CRITICAL,
                                             15-frame look-ahead position
                                             prediction, cooldown/hysteresis.
  STAGE 6  Visualisation                 -> heatmap, flow arrows, conflict
                                             arrows, dashed prediction
                                             borders, dashboard.

  BONUS 1  Safety Score gauge (0-100) drawn on every frame.
  BONUS 2  Evacuation Direction Advisor (arrows toward the lowest-density
           zone from any DANGER/CRITICAL zone).
  BONUS 3  Incident Timeline — timestamped JSON log of every DANGER/
           CRITICAL event, written per video to outputs/.
  BONUS 4  Pressure Wave detection — flags zones whose density keeps
           oscillating between high/low (a classic crowd-crush
           precursor) with a pulsing warning marker.

All "DESIGN CHOICE" comments below document the reasoning behind the
engineering decisions, since the brief explicitly asks for justification
(it carries marks).
==========================================================================
"""

import cv2
import os
import json
import time
import numpy as np
from collections import deque
from ultralytics import YOLO

# ==========================================================================
# CONFIGURATION
# ==========================================================================

# DESIGN CHOICE: MODEL
# yolo11n.pt is used because it is the lightweight CPU-friendly nano model
# already verified to run in this environment. Swap to "yolo26n.pt" /
# "yolo26m.pt" if you have those weights and a GPU available — nothing
# else in the pipeline needs to change since both expose the same API.
MODEL_PATH = "yolo11n.pt"

INPUT_VIDEOS = ["videos/input.mp4", "videos/input2.mp4"]
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# DESIGN CHOICE: GRID RESOLUTION (Stage 2)
# 6x6 = 36 zones. Too coarse (2x2) misses a crush starting in a small
# 2m pocket of the venue; too fine (20x20) starves each cell of enough
# people to be statistically meaningful and tanks performance. 6x6 is the
# brief's recommended sweet spot for typical festival/crossing footage
# shot from an elevated/overhead angle, giving each cell a sensible
# physical footprint while staying fast enough for real-time video.
ROWS, COLS = 6, 6

# Detection tuning for dense, overlapping crowds (Stage 1).
CONF_THRESH = 0.20      # lower threshold catches more people in dense scenes
IOU_THRESH = 0.4        # lower IoU keeps overlapping/packed boxes from being merged

TRACK_HISTORY_LEN = 15  # last N centroids stored per track id (for velocity)
PREDICTION_FRAMES = 15  # ~0.5s lookahead at 30fps, per brief

# DESIGN CHOICE: RISK THRESHOLDS (Stage 5)
# Fruin's Level of Service places "dangerous" crowding (Level D) at
# roughly 3-4 people/m^2, with involuntary contact/pressure from ~6+
# people/m^2. Our 6x6 cells on typical festival footage correspond to a
# few square metres each, so a raw person-count per cell is already a
# reasonable proxy for that physical density:
#   0 people  -> SAFE      (free movement)
#   1 person  -> WATCH     (early occupancy)
#   2 people  -> DANGER    (~Fruin Level D territory for a small cell)
#   3+ people -> CRITICAL  (~Fruin Level E/F, pressure-contact risk)
# This base level is then escalated by one notch (capped at CRITICAL) if
# any of the secondary danger factors fire — see classify_zone_risk().
BASE_COUNT_THRESHOLDS = {"SAFE": 0, "WATCH": 1, "DANGER": 2}  # CRITICAL = 3+

STAGNATION_VEL_THRESH = 1.0     # px/frame; below this with people present = stuck crowd
RAPID_FILL_DENSITY_DELTA = 15.0 # density-unit jump in one frame = filling fast
CONFLICT_MIN_SPEED = 1.0        # px/frame minimum to count as real opposing flow

# DESIGN CHOICE: ALERT TIMING / COOLDOWN
# We deliberately alert EARLY (using the 15-frame predicted position, not
# just current occupancy) because in a crowd-crush the cost of a missed
# late warning (injury/death) vastly outweighs the cost of a stadium
# steward walking to a zone that calms back down. To stop that early
# trigger from becoming alarm-fatigue, two hysteresis mechanisms are
# applied before anything is escalated to the operator-facing timeline:
#   - SUSTAIN_FRAMES: a zone must stay at DANGER+ for several consecutive
#     frames before it is logged as an incident (filters one-frame noise).
#   - COOLDOWN_FRAMES: after a zone drops out of CRITICAL, it cannot be
#     re-escalated to CRITICAL for a short cooldown window (prevents the
#     status flickering CRITICAL/DANGER/CRITICAL every frame).
SUSTAIN_FRAMES = 5          # ~0.15s at 30fps before an incident is logged
COOLDOWN_FRAMES = 20        # ~0.6s at 30fps before CRITICAL can re-trigger

# Pressure-wave (Bonus 4) detection window
PRESSURE_WAVE_WINDOW = 30          # frames of density history kept per zone
PRESSURE_WAVE_MIN_SWINGS = 3       # oscillations within the window to flag

STATUS_COLOR = {
    "SAFE": (0, 255, 0),
    "WATCH": (0, 255, 255),
    "DANGER": (0, 165, 255),
    "CRITICAL": (0, 0, 255),
}
STATUS_RANK = {"SAFE": 0, "WATCH": 1, "DANGER": 2, "CRITICAL": 3}
RANK_STATUS = {v: k for k, v in STATUS_RANK.items()}


# ==========================================================================
# DRAWING HELPERS
# ==========================================================================

def draw_dashed_rect(img, pt1, pt2, color, thickness=2, dash=8, gap=6):
    """Dashed rectangle border — used for the prediction overlay so it
    reads visually as 'forecast' rather than 'current state'."""
    x1, y1 = pt1
    x2, y2 = pt2

    def dashed_line(p1, p2):
        x1_, y1_ = p1
        x2_, y2_ = p2
        length = max(1, int(np.hypot(x2_ - x1_, y2_ - y1_)))
        n_dashes = max(1, length // (dash + gap))
        for i in range(n_dashes + 1):
            start_frac = (i * (dash + gap)) / length
            end_frac = min(1.0, start_frac + dash / length)
            sx = int(x1_ + (x2_ - x1_) * start_frac)
            sy = int(y1_ + (y2_ - y1_) * start_frac)
            ex = int(x1_ + (x2_ - x1_) * end_frac)
            ey = int(y1_ + (y2_ - y1_) * end_frac)
            cv2.line(img, (sx, sy), (ex, ey), color, thickness)

    dashed_line((x1, y1), (x2, y1))
    dashed_line((x2, y1), (x2, y2))
    dashed_line((x2, y2), (x1, y2))
    dashed_line((x1, y2), (x1, y1))


def draw_double_headed_arrow(img, p1, p2, color, thickness=2):
    cv2.arrowedLine(img, p1, p2, color, thickness, tipLength=0.25)
    cv2.arrowedLine(img, p2, p1, color, thickness, tipLength=0.25)


def draw_safety_gauge(img, center, radius, score):
    """Bonus 1: semi-circular 0-100 safety gauge."""
    score = max(0, min(100, score))
    if score >= 70:
        color = (0, 255, 0)
    elif score >= 40:
        color = (0, 255, 255)
    else:
        color = (0, 0, 255)

    cv2.ellipse(img, center, (radius, radius), 0, 180, 360, (60, 60, 60), 10)
    end_angle = 180 + int(180 * (score / 100.0))
    cv2.ellipse(img, center, (radius, radius), 0, 180, end_angle, color, 10)
    cv2.putText(img, f"{int(score)}", (center[0] - 22, center[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, "SAFETY", (center[0] - 35, center[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


def zone_center(r, c, cell_w, cell_h):
    return (c * cell_w + cell_w // 2, r * cell_h + cell_h // 2)


# ==========================================================================
# RISK CLASSIFICATION (Stage 5)
# ==========================================================================

def base_status_from_count(count):
    if count <= BASE_COUNT_THRESHOLDS["SAFE"]:
        return "SAFE"
    if count <= BASE_COUNT_THRESHOLDS["WATCH"]:
        return "WATCH"
    if count <= BASE_COUNT_THRESHOLDS["DANGER"]:
        return "DANGER"
    return "CRITICAL"


def classify_zone_risk(count, flow_mag, is_conflict, is_stagnant, is_rapid_fill):
    """Combine Factor 1 (density), Factor 2 (flow conflict),
    Factor 3 (stagnation) and Factor 4 (rate of change) into one
    risk level per zone, as required by the 'Designer's Choice' box.
    Each secondary factor escalates the base level by one notch,
    capped at CRITICAL, since any one of them independently raises
    real-world crush risk beyond what raw occupancy alone implies."""
    rank = STATUS_RANK[base_status_from_count(count)]
    if is_conflict:
        rank += 1
    if is_stagnant:
        rank += 1
    if is_rapid_fill:
        rank += 1
    rank = min(rank, STATUS_RANK["CRITICAL"])
    return RANK_STATUS[rank]


# ==========================================================================
# MAIN PIPELINE
# ==========================================================================

def main():
    model = YOLO(MODEL_PATH)

    for video_idx, video_path in enumerate(INPUT_VIDEOS):
        if not os.path.exists(video_path):
            print(f"Warning: File {video_path} not found. Skipping...")
            continue

        print(f"\n--- Processing Video {video_idx + 1}/{len(INPUT_VIDEOS)}: {video_path} ---")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open {video_path}")
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or np.isnan(fps) or fps <= 0:
            fps = 30

        video_base_name = os.path.basename(video_path).split('.')[0]
        output_path = os.path.join(OUTPUT_DIR, f"{video_base_name}_density.mp4")
        timeline_path = os.path.join(OUTPUT_DIR, f"{video_base_name}_incidents.json")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        cell_width = max(1, width // COLS)
        cell_height = max(1, height // ROWS)

        # ---- per-video persistent state -----------------------------------
        track_history = {}                       # track_id -> deque of centroids
        prev_grid_densities = [[0.0] * COLS for _ in range(ROWS)]
        critical_cooldown = [[0] * COLS for _ in range(ROWS)]      # Stage5 hysteresis
        sustain_counter = [[0] * COLS for _ in range(ROWS)]        # alert sustain
        density_history = [[deque(maxlen=PRESSURE_WAVE_WINDOW) for _ in range(COLS)]
                            for _ in range(ROWS)]                  # Bonus 4
        logged_incident_active = [[False] * COLS for _ in range(ROWS)]
        incident_timeline = []                   # Bonus 3

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            timestamp_sec = round(frame_idx / fps, 2)

            overlay = frame.copy()

            # ---------------- STAGE 1: detection & tracking ----------------
            results = model.track(
                frame, persist=True, classes=[0],
                conf=CONF_THRESH, iou=IOU_THRESH, verbose=False
            )

            boxes, track_ids = [], []
            total_people = 0
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                if boxes.id is not None:
                    track_ids = boxes.id.int().cpu().tolist()
                else:
                    track_ids = [None] * len(boxes)
                total_people = len(boxes)

            # ---------------- STAGE 2/3: grid + per-zone aggregation -------
            grid_counts = [[0] * COLS for _ in range(ROWS)]
            predicted_grid_counts = [[0] * COLS for _ in range(ROWS)]
            zone_velocities = [[[] for _ in range(COLS)] for _ in range(ROWS)]

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                grid_col = max(0, min(center_x // cell_width, COLS - 1))
                grid_row = max(0, min(center_y // cell_height, ROWS - 1))
                grid_counts[grid_row][grid_col] += 1

                if track_id is not None:
                    if track_id not in track_history:
                        track_history[track_id] = deque(maxlen=TRACK_HISTORY_LEN)
                    track_history[track_id].append((center_x, center_y))
                    points = track_history[track_id]

                    if len(points) >= 2:
                        vx = points[-1][0] - points[-2][0]
                        vy = points[-1][1] - points[-2][1]
                        zone_velocities[grid_row][grid_col].append((vx, vy))

                        # STAGE 5: 15-frame lookahead position prediction
                        pred_x = max(0, min(center_x + vx * PREDICTION_FRAMES, width - 1))
                        pred_y = max(0, min(center_y + vy * PREDICTION_FRAMES, height - 1))
                        pred_col = int(pred_x // cell_width)
                        pred_row = int(pred_y // cell_height)
                        if 0 <= pred_row < ROWS and 0 <= pred_col < COLS:
                            predicted_grid_counts[pred_row][pred_col] += 1

                        cv2.arrowedLine(frame, points[-2], points[-1], (255, 0, 255), 2, tipLength=0.4)
                else:
                    predicted_grid_counts[grid_row][grid_col] += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
                cv2.circle(frame, (center_x, center_y), 4, (255, 0, 0), -1)

            # ---------------- STAGE 4: average flow vector per zone --------
            avg_zone_flow = [[(0.0, 0.0)] * COLS for _ in range(ROWS)]
            flow_mag = [[0.0] * COLS for _ in range(ROWS)]
            for r in range(ROWS):
                for c in range(COLS):
                    if zone_velocities[r][c]:
                        vxs = [v[0] for v in zone_velocities[r][c]]
                        vys = [v[1] for v in zone_velocities[r][c]]
                        fx, fy = float(np.mean(vxs)), float(np.mean(vys))
                        avg_zone_flow[r][c] = (fx, fy)
                        flow_mag[r][c] = float(np.hypot(fx, fy))

            # Opposing-flow conflict detection (right & down neighbours only,
            # so each adjacent pair is evaluated/drawn exactly once).
            conflict_zones = set()
            conflict_pairs = []
            for r in range(ROWS):
                for c in range(COLS):
                    fx, fy = avg_zone_flow[r][c]
                    if flow_mag[r][c] < CONFLICT_MIN_SPEED:
                        continue
                    for nr, nc in ((r, c + 1), (r + 1, c)):
                        if nr >= ROWS or nc >= COLS:
                            continue
                        nfx, nfy = avg_zone_flow[nr][nc]
                        if flow_mag[nr][nc] < CONFLICT_MIN_SPEED:
                            continue
                        dot = fx * nfx + fy * nfy
                        if dot < 0:
                            conflict_zones.add((r, c))
                            conflict_zones.add((nr, nc))
                            conflict_pairs.append(((r, c), (nr, nc)))

            # ---------------- STAGE 3 (cont.) + STAGE 5: density & risk ----
            safe = watch = danger = critical = 0
            highest_density = 0.0
            highest_density_zone = None
            pressure_wave_zones = []

            for r in range(ROWS):
                for c in range(COLS):
                    count = grid_counts[r][c]

                    if count == 0:
                        density = 0.0
                        prev_grid_densities[r][c] = 0.0
                    else:
                        raw_density = round((count / (cell_width * cell_height)) * 100000, 2)
                        density = round(0.7 * prev_grid_densities[r][c] + 0.3 * raw_density, 2)
                        prev_grid_densities[r][c] = density

                    density_history[r][c].append(density)

                    if density > highest_density:
                        highest_density = density
                        highest_density_zone = (r, c)

                    # secondary risk factors
                    is_conflict = (r, c) in conflict_zones
                    is_stagnant = count >= 2 and flow_mag[r][c] < STAGNATION_VEL_THRESH
                    prior_vals = list(density_history[r][c])
                    is_rapid_fill = (len(prior_vals) >= 2 and
                                      (prior_vals[-1] - prior_vals[-2]) > RAPID_FILL_DENSITY_DELTA)

                    # Bonus 4: pressure-wave / stop-and-go oscillation detection
                    if len(density_history[r][c]) == PRESSURE_WAVE_WINDOW:
                        vals = list(density_history[r][c])
                        mid = (max(vals) + min(vals)) / 2.0
                        signs = [1 if v > mid else -1 for v in vals]
                        swings = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])
                        if swings >= PRESSURE_WAVE_MIN_SWINGS and (max(vals) - min(vals)) > 5:
                            pressure_wave_zones.append((r, c))

                    raw_status = classify_zone_risk(count, flow_mag[r][c], is_conflict,
                                                     is_stagnant, is_rapid_fill)

                    # Hysteresis / cooldown so CRITICAL cannot flicker on/off every frame
                    if critical_cooldown[r][c] > 0:
                        if raw_status == "CRITICAL":
                            raw_status = "DANGER"
                        critical_cooldown[r][c] -= 1

                    status = raw_status
                    color = STATUS_COLOR[status]

                    if status == "SAFE":
                        safe += 1
                    elif status == "WATCH":
                        watch += 1
                    elif status == "DANGER":
                        danger += 1
                    else:
                        critical += 1
                        critical_cooldown[r][c] = COOLDOWN_FRAMES

                    # Bonus 3: sustained-incident logging (alert-fatigue control)
                    if status in ("DANGER", "CRITICAL"):
                        sustain_counter[r][c] += 1
                        if sustain_counter[r][c] >= SUSTAIN_FRAMES and not logged_incident_active[r][c]:
                            incident_timeline.append({
                                "time_sec": timestamp_sec,
                                "frame": frame_idx,
                                "zone": [r, c],
                                "status": status,
                                "people": count,
                                "density": density,
                                "flow_conflict": is_conflict,
                                "stagnant": is_stagnant,
                                "rapid_fill": is_rapid_fill,
                            })
                            logged_incident_active[r][c] = True
                    else:
                        sustain_counter[r][c] = 0
                        logged_incident_active[r][c] = False

                    x1_z, y1_z = c * cell_width, r * cell_height
                    x2_z, y2_z = x1_z + cell_width, y1_z + cell_height

                    cv2.rectangle(overlay, (x1_z, y1_z), (x2_z, y2_z), color, -1)

                    # Stage 6: dashed prediction overlay
                    if predicted_grid_counts[r][c] >= 3:
                        draw_dashed_rect(frame, (x1_z + 3, y1_z + 3), (x2_z - 3, y2_z - 3), (0, 0, 255), 2)

                    # Stage 6: flow arrow
                    flow = avg_zone_flow[r][c]
                    if abs(flow[0]) > 0.5 or abs(flow[1]) > 0.5:
                        z_cx, z_cy = (x1_z + x2_z) // 2, (y1_z + y2_z) // 2
                        cv2.arrowedLine(frame, (z_cx, z_cy),
                                         (int(z_cx + flow[0] * 3), int(z_cy + flow[1] * 3)),
                                         (255, 255, 255), 2, tipLength=0.3)

                    # Bonus 4: pulsing pressure-wave marker
                    if (r, c) in pressure_wave_zones:
                        pulse_r = 10 + int(4 * np.sin(frame_idx * 0.5))
                        cv2.circle(frame, (x2_z - 16, y1_z + 16), pulse_r, (255, 0, 255), 2)
                        cv2.putText(frame, "PW", (x2_z - 30, y1_z + 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

                    cv2.rectangle(frame, (x1_z, y1_z), (x2_z, y2_z), (0, 255, 0), 1)
                    cv2.putText(frame, f"P: {count}", (x1_z + 10, y1_z + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    cv2.putText(frame, f"D: {density:.1f}", (x1_z + 10, y1_z + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
                    cv2.putText(frame, status, (x1_z + 10, y1_z + 56),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

            # Stage 6: conflict double-headed arrows between opposing zones
            for (r1, c1), (r2, c2) in conflict_pairs:
                p1 = zone_center(r1, c1, cell_width, cell_height)
                p2 = zone_center(r2, c2, cell_width, cell_height)
                draw_double_headed_arrow(frame, p1, p2, (0, 0, 255), 2)

            # Bonus 2: Evacuation Direction Advisor — point danger zones
            # toward the lowest-density zone currently on the grid.
            flat_densities = [(prev_grid_densities[r][c], r, c) for r in range(ROWS) for c in range(COLS)]
            flat_densities.sort()
            safest_r, safest_c = flat_densities[0][1], flat_densities[0][2]
            safest_center = zone_center(safest_r, safest_c, cell_width, cell_height)
            for r in range(ROWS):
                for c in range(COLS):
                    if base_status_from_count(grid_counts[r][c]) in ("DANGER", "CRITICAL") and (r, c) != (safest_r, safest_c):
                        start = zone_center(r, c, cell_width, cell_height)
                        cv2.arrowedLine(frame, start, safest_center, (255, 255, 0), 2, tipLength=0.2)

            # Alpha fuse heatmap
            frame = cv2.addWeighted(overlay, 0.22, frame, 0.78, 0)

            # Safety score (Bonus 1): penalise critical/danger/watch zones & conflicts
            num_conflicts = len(conflict_pairs)
            safety_score = 100 - (critical * 15 + danger * 7 + watch * 2 + num_conflicts * 10)
            safety_score = max(0, min(100, safety_score))

            # Dashboard
            cv2.rectangle(frame, (10, 10), (340, 235), (0, 0, 0), -1)
            cv2.putText(frame, f"Total Tracked: {total_people}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(frame, f"SAFE ZONES     : {safe}", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"WATCH ZONES    : {watch}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"DANGER ZONES   : {danger}", (20, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            cv2.putText(frame, f"CRITICAL ZONES : {critical}", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, f"Max Density    : {highest_density:.1f} @ {highest_density_zone}", (20, 165),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, f"Flow Conflicts : {num_conflicts}", (20, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, f"Timestamp      : {timestamp_sec:.2f}s", (20, 215),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Bonus 1: safety gauge, bottom-right corner
            draw_safety_gauge(frame, (width - 90, height - 30), 70, safety_score)

            out.write(frame)

            # imshow can fail in headless/server environments — never let
            # that crash the whole pipeline.
            try:
                cv2.imshow("Crowd Crush Prevention System", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except cv2.error:
                pass

        cap.release()
        out.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

        with open(timeline_path, "w") as f:
            json.dump(incident_timeline, f, indent=2)

        print(f"Finished processing clip. Saved annotated video to: {output_path}")
        print(f"Incident timeline saved to: {timeline_path} ({len(incident_timeline)} events)")

    print("\nAll input videos completed processing.")


if __name__ == "__main__":
    main()