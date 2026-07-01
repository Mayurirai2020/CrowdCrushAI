# CrowdCrushAI

## Overview
CrowdCrushAI is a crowd density analysis system developed using Python, OpenCV, and YOLOv11. It detects people in videos, divides each frame into grid cells, and calculates crowd density for monitoring and analysis.

## Features
- Person detection using YOLOv11
- Grid-based crowd density calculation
- Processes multiple input videos
- Saves processed output videos

## Technologies Used
- Python
- OpenCV
- Ultralytics YOLOv11
- NumPy

## How to Run
1. Install dependencies:
   pip install ultralytics opencv-python numpy

2. Run:
   python3 test_density.py

## Project Structure
- detect.py
- grid.py
- test.py
- test_density.py
- videos/
- outputs/

## Author
Mayuri Rai