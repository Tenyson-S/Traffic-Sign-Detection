🚦 Traffic Sign Detection using Deep Learning
A deep learning-based system to recognize and classify traffic signs to improve road safety.
Built using Python, TensorFlow/Keras, and the GTSRB dataset.

📌 Features
Detects and classifies 43 types of traffic signs.
Achieves 95%+ accuracy on test data.
Uses Convolutional Neural Networks (CNN) for high accuracy.
Script for training and testing the model.
Supports image-based testing (can be extended to real-time video).

📂 Project Structure

├── train.py         # Train the CNN model
├── test.py          # Test the trained model on images
├── myModel.h5       # Saved trained model
├── labels.csv       # Mapping of class indices to traffic sign names
├── Test/            # Folder containing test images
├── README.md        # Project documentation

🛠️ Installation & Setup
Clone the repository

git clone https://github.com/your-username/traffic-sign-detection.git

cd traffic-sign-detection

Install dependencies
pip install -r requirements.txt

Run Training
python train.py

Run Testing
python test.py

📊 Dataset
We used the German Traffic Sign Recognition Benchmark (GTSRB) dataset.
📥 Download here: GTSRB Dataset

🚗 Applications in Road Safety
Driver Assistance Systems – Alerts drivers about detected traffic signs.
Autonomous Vehicles – Ensures compliance with road rules.
Road Monitoring Systems – Detects missing/damaged signs.
Driver Training Simulators – Teaches sign recognition to learners.

📷 Example Output
Detected Sign: "Speed Limit 60 km/h"
Confidence: 98%

(You can add sample output images here)

📈 Results
Metric	Value
Training Accuracy	97%
Test Accuracy	95%
Classes Detected	43

🧠 Model Architecture (CNN)
Conv2D + ReLU Activation – Feature extraction
MaxPooling2D – Downsampling
Flatten Layer – Converts 2D features to 1D
Dense Layers – Classification
Softmax Output – Probability distribution



🤝 Acknowledgements
Dataset: GTSRB Benchmark (http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
Frameworks: TensorFlow (https://www.tensorflow.org/), Keras (https://keras.io/)


