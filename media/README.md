# Demo Visuals

## 1. Default view of Nighscout Dashboard
Each grey dot on the dashboard represents a blood glucose reading sent by Continuous Glucose Monitor (CGM). For the purposes of this demo, we are using a Python feeder script to send this blood glucose reading data

## 2. Synthetic CGM Data being sent by our Python script (/scripts/nightscout_push_synthetic.py)
This python script sends CGM data every 60 seconds to Nighscout dashboard

## 3. Trigger from ESP32
This screenshot shows the firmware running on ESP32 on Arduino IDE. A trigger from the "Serial Monitor" tab makes the ESP32 fetch a short window of glucose values from Nightscout, normalize, pass them into the on-device TensorFlow LiteMicro model. This model predicts the glucose value that is sent back to Nightscout

## 4. Predicted value posted as a separate point
The larger grey dot on the dashboard represents a predicted glucose value posted by the ESP32. It is labelled separately with the tag, "Entered By: ESP32-PRED" 