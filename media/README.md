# Demo Visuals

## 1. Default view of Nighscout Dashboard
Each grey dot on the dashboard represents a blood glucose reading sent by Continuous Glucose Monitor (CGM). For the purposes of this demo, we are using a Python feeder script to send this blood glucose reading data

![Nightscout Dashboard](https://github.com/user-attachments/assets/fd3199c3-b4f5-4a4e-84bc-2f9c3ff7e3ab)

## 2. Synthetic CGM Data being sent by our Python script (/scripts/nightscout_push_synthetic.py)
This python script sends CGM data every 60 seconds to Nighscout dashboard

![Synthetic CGM Data from Python](https://github.com/user-attachments/assets/dfd6b4c1-7be5-41b0-b742-f0d94f52e152)

## 3. Trigger from ESP32 (/firmware/GlucoseBuddy_NS_Predict_V1.ino)
This screenshot shows the firmware running on ESP32 on Arduino IDE. A trigger from the "Serial Monitor" tab makes the ESP32 fetch a short window of glucose values from Nightscout, normalize, pass them into the on-device TensorFlow LiteMicro model. This model predicts the glucose value that is sent back to Nightscout

![Trigger from ESP32](https://github.com/user-attachments/assets/0923c4a3-f660-455e-9527-af9061e719f2)

## 4. Predicted value posted as a separate point
The larger grey dot on the dashboard represents a predicted glucose value posted by the ESP32. It is labelled separately with the tag, "Entered By: ESP32-PRED" 


![Prediction on dashboard](https://github.com/user-attachments/assets/9393ade8-6114-4e30-8744-250385e07ee5)
