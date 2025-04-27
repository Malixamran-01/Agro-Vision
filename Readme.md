# Plant Diseases Detection

## Project Overview

This project is a deep learning-based web application for plant disease recognition. It uses a Convolutional Neural Network (CNN) trained on an augmented dataset of plant leaf images to classify diseases across 38 classes. The app is built with Streamlit and allows users to upload an image of a plant leaf, predicts the disease, and provides the disease name in both English and Hindi (using the Gemini API for translation).

---

## Features

- **Image Upload & Prediction:** Upload a plant leaf image and get instant disease classification.
- **Multilingual Support:** Disease names are translated to Hindi for broader accessibility.
- **Model:** Custom CNN trained on a large, augmented dataset of plant diseases.
- **Dataset:** 38 classes, including healthy and diseased leaves for crops like Apple, Tomato, Potato, Corn, Grape, etc.
- **Web App:** User-friendly interface built with Streamlit.

---

## Directory Structure

- `test-plant-diseases/`
  - `main.py` – Streamlit app for prediction and translation.
  - `test-plant-diseases.ipynb` – Notebook for model testing and evaluation.
  - `train-plant-diseases.ipynb` – Notebook for model training and data preprocessing.
  - `plant-diseases-model.keras` – Trained Keras model.
  - `training_history.json` – Model training accuracy history.
  - `.gitignore` – Files and folders to ignore in version control.
  - `README.md` – Project documentation (this file).
- `New Plant Diseases Dataset(Augmented)/` – Augmented dataset with `train/` and `valid/` folders, each containing 38 class subfolders.
- `test/` – Contains test images for various diseases.

---

## Model Architecture

- Multiple Conv2D and MaxPooling2D layers for feature extraction.
- Dropout layers for regularization.
- Dense layers for classification.
- Output layer: 38-way softmax for disease class prediction.
- Trained for 10 epochs, achieving high accuracy (up to ~98% on validation).

---

## How to Run

1. **Install Requirements:**
   - Python 3.8+
   - TensorFlow, Streamlit, Pillow, numpy, python-dotenv, google-generativeai

2. **Set up Gemini API:**
   - Create a `.env` file in `test-plant-diseases/` with your Gemini API key:
     ```
     GEMINI_API_KEY=your_api_key_here
     ```

3. **Run the App:**
   ```
   cd test-plant-diseases
   streamlit run main.py
   ```

4. **Usage:**
   - Open the Streamlit web interface.
   - Upload a plant leaf image (jpg, jpeg, png).
   - Click "Predict" to see the disease name in English and Hindi.

---

## Dataset

- Sourced from the "New Plant Diseases Dataset (Augmented)".
- 38 classes, including healthy and various diseased leaves.
- Images are preprocessed to 128x128 RGB.

---

## Results

- **Training Accuracy:** Up to 99.3%
- **Validation Accuracy:** Up to 96.8%
- **Sample Training History:** See `training_history.json` for epoch-wise accuracy.

---

## Acknowledgements

- Dataset: [Kaggle Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- Gemini API for translation.

