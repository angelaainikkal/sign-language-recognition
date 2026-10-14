# Sign Language Alphabet Recognition

## About the Project

Sign Language Alphabet Recognition is a mini-project that uses computer vision and machine learning to recognize static sign-language alphabet gestures from images.

The system processes an input image, extracts useful image features, and uses a trained K-Nearest Neighbors (KNN) classifier to predict the corresponding alphabet.

## Features

- Upload an image of a sign-language alphabet.
- Process the image using OpenCV.
- Extract image features using Histogram of Oriented Gradients (HOG).
- Predict the alphabet using a trained KNN model.
- Display the prediction through a simple Streamlit web interface.

## Technologies Used

- **Python** – Main programming language
- **OpenCV** – Image processing
- **NumPy** – Array and dataset operations
- **scikit-image** – HOG feature extraction
- **scikit-learn** – KNN classification and evaluation
- **Joblib** – Saving and loading the trained model
- **Streamlit** – Web application interface
- **Git and GitHub** – Version control and collaboration

## Project Structure

```text
sign-language-recognition/
├── .streamlit/
│   └── config.toml
├── data/
│   ├── dataset_report.json
│   ├── labels.npy
│   └── test_sign_A.png
├── models/
│   └── knn_results.json
├── src/
│   ├── create_test_images.py
│   ├── evaluate_knn.py
│   ├── make_features.py
│   ├── predict_sign.py
│   ├── train_knn.py
│   └── verify_dataset.py
├── app.py
├── .gitignore
└── README.md
```

The trained model and full image dataset may need to be obtained or generated separately because they are not included in the GitHub repository.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/angelaainikkal/sign-language-recognition.git
cd sign-language-recognition
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install the required libraries

```bash
python -m pip install numpy opencv-python scikit-image scikit-learn joblib streamlit
```

### 4. Prepare the dataset and model

Obtain the project dataset separately and place the required dataset files in the `data/` directory.

Train the KNN model using:

```bash
python src/train_knn.py
```

Ensure the trained model is saved at:

```text
models/knn_hog_alphabet.joblib
```

If the model has already been trained and shared separately, place it in the `models/` directory instead.

### 5. Run the application

```bash
python -m streamlit run app.py
```

The application will open in your web browser. Upload a sign-language alphabet image to view the prediction.

## How It Works

1. **Image Input:** The user uploads an image containing a static sign-language gesture.
2. **Preprocessing:** OpenCV reads the image in grayscale and resizes it to the required dimensions.
3. **Feature Extraction:** HOG extracts features representing the shapes and edges in the image.
4. **Classification:** The trained KNN model predicts the alphabet from the extracted features.
5. **Output:** The Streamlit interface displays the predicted alphabet.

## Project Scope

The current implementation focuses on recognizing static alphabet gestures from images. It does not perform continuous sign-language sentence translation or speech translation.

## Team Project

This project is developed as part of the AIML Lab mini-project.

**Repository:** [Sign Language Recognition](https://github.com/angelaainikkal/sign-language-recognition)

## Live Demo

Try the deployed application here:

**[Launch Signify – Sign Language Alphabet Recognition](https://signify-alphabet-recognition.streamlit.app/)