# GlaucoGuard- 'An AI Powered Diagnostic System'
GlaucoGuard is a full-stack Computer Vision application designed to automate the diagnosis of glaucoma from fundus images. This system provides a comprehensive diagnostic solution, from image analysis to report generation, and includes a chatbot for user assistance. The core of this project is a Convolutional Neural Network (CNN) model that achieves 97% accuracy in classifying images. A key feature of this system is its focus on interpretability; it integrates Grad-CAM to visualize the regions of the image the model is focusing on, ensuring that the diagnoses are based on correct biomarkers and not spurious correlations. This project was built to tackle the challenges of real-world medical AI, including handling imbalanced datasets and the critical need for model transparency.

# 🚀 Features
High-Accuracy CNN Model: Achieves 97% accuracy in Glaucoma detection using a deep learning model (e.g., CNN, ResNet, VGG16) trained on fundus images.

Model Interpretability (XAI): Implements Grad-CAM to produce heatmaps that visualize why the model made a specific prediction, enhancing trust and verifiability.

Interactive Web Application: Built with Streamlit to provide a user-friendly interface for medical professionals.

Multi-Image Upload & PDF Reporting: Allows users to upload multiple images at once and automatically generates a diagnostic PDF report.

AI-Powered Chatbot: Integrates an AI chatbot (using Google Gemini API) to assist users with queries related to glaucoma.

# 🛠️ Tech Stack
Machine Learning / CV:

Python

TensorFlow / Keras

OpenCV

Scikit-learn

Pillow (PIL)

Web Application & Deployment:

Streamlit

Docker (for containerization)

Chatbot:

Google Gemini API

# 📦 Installation & Usage
To run this application locally, clone the repository, install the dependencies, and run the Streamlit app.

# Clone the repository
git clone https://github.com/adilraz321/Glaucoguard--AI-Powered-Diagnostic-system.git
cd Glaucoguard--AI-Powered-Diagnostic-system

# Install the required libraries
pip install -r requirements.txt

# Run the Streamlit application
streamlit run app.py

📈 Model & Interpretability
The primary challenge in this project was building a model that was not only accurate but also trustworthy.

Diagnosis: The CNN model processes a given fundus image and classifies it as 'Glaucoma' or 'Normal' with a 97% accuracy rate.

Interpretability (Grad-CAM): To validate the model's prediction, a Grad-CAM heatmap is generated. This visualization highlights the specific areas of the optic disc and cup that the model used to make its decision, confirming that its diagnosis is based on genuine medical biomarkers.


