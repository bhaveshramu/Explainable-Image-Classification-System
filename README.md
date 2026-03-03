Explainable Image Classification System
## 🚀 Key Highlights
- ResNet18 Transfer Learning
- Grad-CAM Explainability
- Confidence Scoring
- Interactive Streamlit Web App
- Real-time Image Prediction

1. Project Overview
The Explainable Image Classification System is a deep learning–based web application that classifies images using a pretrained ResNet18 model and provides visual explanations using Grad-CAM.
Unlike traditional image classifiers that only provide predictions, this system enhances transparency by highlighting the regions of the image that influenced the model’s decision.
This improves interpretability, reliability, and trust in AI systems.

2. Problem Statement
Traditional deep learning models operate as black-box systems, providing predictions without explaining the reasoning behind them.
In critical applications such as healthcare, security, and automated decision-making, understanding why a model made a prediction is equally important as the prediction itself.
This project addresses the need for explainable AI by combining image classification with visual interpretation techniques.

3. Objectives
The main objectives of this project are:

-To implement an image classification system using a pretrained ResNet18 model.
-To apply transfer learning for efficient training.
-To integrate Grad-CAM for visual explanation of predictions.
-To deploy the model as an interactive web application using Streamlit.
-To provide both prediction confidence and human-readable explanation.

4. System Architecture
The system follows this pipeline:
Image Input
→ Image Preprocessing
→ ResNet18 Model
→ Prediction & Confidence Score
→ Grad-CAM Heatmap Generation
→ Explanation Summary
→ Web Interface Display


5. Technologies Used
-Python
-PyTorch
-Torchvision
-Grad-CAM
-Streamlit
-OpenCV
-NumPy
-Pillow

6. Key Features
Binary Image Classification (Cats vs Dogs)
Transfer Learning using ResNet18
Grad-CAM Explainability
Confidence Score Display
Interactive Web Interface
Clean Two-Column Layout (Image + Heatmap)
Human-readable Explanation Summary

7. Installation & Setup
Step 1: Clone or Download the Project
    Download the project folder and open it in VS Code.

Step 2: Create Virtual Environment
    python -m venv venv

Activate it:
    venv\Scripts\activate

Step 3: Install Dependencies
    python -m pip install torch torchvision matplotlib pillow numpy streamlit grad-cam opencv-python

Step 4: Prepare Dataset Structure
dataset/
   train/
      cats/
      dogs/
   test/
      cats/
      dogs/

Each folder must contain corresponding image files.

Step 5: Train the Model
    From the src directory:
    python train.py

This will save the trained model inside:
    models/model.pth

Step 6: Run the Web Application
    From the app directory:
    streamlit run app.py
The browser will automatically open the web interface.

8. Project Structure
AI_Image_Explainability/
│
├── dataset/
├── models/
├── outputs/
├── src/
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── explain.py
│
├── app/
│   └── app.py
│
├── venv/
└── README.md


9. Working Mechanism
User uploads an image through the web interface.
The image is resized and converted into tensor format.
The ResNet18 model performs classification.
Softmax is applied to calculate confidence scores.
Grad-CAM generates a heatmap showing important regions.

The system displays:
Predicted Class
Confidence Score
Grad-CAM Visualization
Explanation Summary


10. Results
The system successfully:
Classifies images into defined categories.
Highlights influential image regions.
Provides confidence metrics.
Enhances interpretability of deep learning predictions.


11. Limitations
Currently supports only binary classification (Cats vs Dogs).
Performance depends on dataset size and quality.
Small dataset may reduce generalization accuracy.


12. Future Enhancements
Multi-class classification support.
AI-generated vs Real image detection.
Accuracy evaluation dashboard.
Model performance metrics display.
Online deployment for public access.
Downloadable prediction reports.


13. Conclusion

The Explainable Image Classification System demonstrates how deep learning models can be made transparent and interpretable using Grad-CAM. By combining transfer learning with visual explanation techniques, the project bridges the gap between prediction accuracy and model interpretability.

This system provides a practical implementation of explainable AI and serves as a foundation for more advanced applications in healthcare, security, and digital media analysis.
