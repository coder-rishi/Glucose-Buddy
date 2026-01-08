# Glucose-Buddy

Glucose Buddy is a small hardware–software project I built to explore whether short-term glucose trends can be predicted on a constrained device, without confusing predictions with real measurements.
Most continuous glucose monitors are good at showing recent glucose values, but they don’t really explore what might happen next. I wanted to see if a small system could make simple short-window predictions locally, and how those predictions should be presented so they aren’t mistaken for actual readings.
This project is experimental and educational. It is not a medical device and is not intended for diagnosis, treatment, or clinical use.

# What this demo does
1. Synthetic glucose data is sent to a Nightscout dashboard, simulating what a CGM might upload.
2. An ESP32 fetches a short window of recent glucose values from Nightscout.
3. A TensorFlow Lite Micro model runs locally on the ESP32.
4. The predicted glucose value is posted back to Nightscout as a separate, clearly labeled point, without replacing or altering real data.
The main goal is to keep predictions and measurements visibly separate.

# Repository structure
firmware/
  GlucoseBuddy_NS_Predict_V1.ino
  glucose.h

scripts/
  nightscout_push_synthetic.example.py

model/
  CGM_Glucose_Prediction_Buddy_Colab.ipynb
  glucose.tflite
  normalization.json

- firmware/
ESP32 Arduino code that fetches data, runs inference, and posts predictions.

- scripts/
Python script (example only) that uploads synthetic glucose data to Nightscout.
Real API secrets are intentionally not included.

- model/
Training notebook and exported model artifacts used for on-device inference.

# How to run the demo (high level)
1. Deploy a Nightscout instance (used only for visualization and data flow).
2. Use the Python script (with your own credentials) to upload synthetic glucose data.
3. Flash the ESP32 with the firmware in firmware/.
4. Trigger the ESP32 prediction (ns command in Serial).
5. View the predicted point appearing on the Nightscout graph as a separate entry.

# Design constraints
- Limited RAM and flash on the ESP32
- Fixed input window size
- On-device inference only (no cloud model execution)
- Clear separation between predicted values and measured values
Many early ideas (like increasing model size) failed because of these limits and had to be redesigned.

# Data and model notes
- All glucose data used for testing is synthetic.
- The model predicts short-term trends based on a small recent window.
- Predictions can be wrong, especially during sudden changes (meals, stress, missing data).
- The project focuses more on system behavior and presentation than prediction accuracy.

# Inspiration and attribution
- Initial inspiration came from a KNIME blog post on glucose prediction with LSTM models:
https://www.knime.com/blog/predict-blood-glucose-lstm-cgm-data
This helped frame glucose as a time-series problem but did not cover embedded deployment.
- Nightscout is used as an open-source platform for glucose data visualization.
- TensorFlow Lite for Microcontrollers is used for on-device inference.
- ChatGPT was used as a pair-programming assistant to help generate scaffolding, debug iterations, and explore alternatives. All architectural decisions, testing choices, and integration steps were made by me.

# Status
This project is a working prototype.
Some parts are complete (data flow, on-device inference, dashboard integration), while others are still experimental (model robustness, handling real behavioral changes).

