import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from fpdf import FPDF
from google import genai

# Initialize Google Gemini API
API_KEY = ''
client = genai.Client(api_key=API_KEY)

def get_gemini_response(user_query):
    try:
        response = client.models.generate_content(model="gemini-2.0-flash", contents=user_query)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI Setup
st.set_page_config(page_title="Automated Glaucoma Detector", layout="wide")
st.title("🔍 Glaucoma Detection System")
st.markdown(""" ### Upload Fundus Images Below 👇 
Our AI-powered system will analyze your eye images and provide a prediction along with a confidence score.
""")

# Load pre-trained model
model = tf.keras.models.load_model("my_model2.h5")

def preprocess_image(image_data):
    image = ImageOps.fit(image_data, (100, 100), Image.ANTIALIAS)
    image = image.convert('RGB')
    image = np.asarray(image) / 255.0
    return image[np.newaxis, ...]

def grad_cam(model, image_array):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer('conv2d_4').output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)[0]
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return np.array(heatmap)

def generate_pdf(predictions, images, heatmaps):
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font('DejaVu', '', 'ttf/DejaVuSans.ttf', uni=True)  
    pdf.set_font('DejaVu', size=12)
    pdf.cell(200, 10, "Glaucoma Detection Report", ln=True, align='C')
    pdf.ln(10)
    
    for idx, (pred, img, heatmap) in enumerate(zip(predictions, images, heatmaps)):
        result = "Healthy" if pred > 0.5 else "Glaucoma Detected"
        pdf.cell(200, 10, f"Image {idx + 1}: {result} (Confidence: {pred:.2f})", ln=True)
        pdf.ln(5)
        
        fundus_path = f"fundus_{idx}.jpg"
        img.save(fundus_path, format="JPEG")
        pdf.image(fundus_path, x=10, y=None, w=80)
        pdf.ln(5)
        
        heatmap_path = f"heatmap_{idx}.jpg"
        plt.imsave(heatmap_path, heatmap, format="jpeg", cmap="jet")
        pdf.image(heatmap_path, x=100, y=None, w=80)
        pdf.ln(10)
    
    pdf.cell(200, 10, "Medical Advice:", ln=True)
    pdf.ln(5)
    if all(p > 0.5 for p in predictions):
        pdf.multi_cell(0, 10, "✅ Your eyes appear healthy. Maintain regular checkups.")
    else:
        pdf.multi_cell(0, 10, "⚠️ Glaucoma detected. Consult an ophthalmologist immediately.")
    
    report_path = "glaucoma_report.pdf"
    pdf.output(report_path)
    return report_path

uploaded_files = st.file_uploader("Upload fundus images", type=["jpg", "png"], accept_multiple_files=True)
if uploaded_files:
    
    predictions = []
    images = []
    heatmaps = []
    
    with st.spinner("Processing images..."):
        cols = st.columns(len(uploaded_files))
        for idx, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file)
            images.append(image)
            image_array = preprocess_image(image)
            prediction = model.predict(image_array)[0][0]
            predictions.append(prediction)
            
            heatmap = grad_cam(model, image_array)
            heatmaps.append(heatmap)
            
            result_text = "✅ Healthy" if prediction > 0.5 else "⚠️ Glaucoma Detected"
            result_color = "green" if prediction > 0.5 else "red"
            
            cols[idx].image(image, caption=f"{result_text} (Confidence: {prediction:.2f})", use_container_width=True)
            cols[idx].markdown(f"<h3 style='color:{result_color};'>{result_text}</h3>", unsafe_allow_html=True)
            
            fig, ax = plt.subplots()
            ax.imshow(cv2.resize(heatmap, (100, 100)), cmap='jet', alpha=0.5)
            cols[idx].pyplot(fig)
    
    report_path = generate_pdf(predictions, images, heatmaps)
    with open(report_path, "rb") as file:
        st.download_button(label="📄 Download Report", data=file, file_name="glaucoma_report.pdf", mime="application/pdf")

# Sidebar - Glaucoma Chatbot 
st.sidebar.header("💬 Glaucoma Chatbot")
user_query = st.sidebar.text_input("Ask me anything about glaucoma:")
if user_query:
    chatbot_reply = get_gemini_response(user_query)
    st.sidebar.write(f"🤖 {chatbot_reply}")








