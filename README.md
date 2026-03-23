# SkinSense

A Streamlit web app for skin lesion classification using a trained EfficientNet model.

## Project Structure

```
SkinSense/
├── app.py                  # Streamlit app entry point
├── predictor.py            # Model loading, preprocessing, and prediction logic
├── best_modelnew.h5        # Trained Keras model (already placed here)
├── label_encoder.pkl       # ⚠️ YOU NEED TO ADD THIS (see below)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Add `label_encoder.pkl` to the project root.
   This is the scikit-learn `LabelEncoder` object saved during model training.
   Example of how it was likely saved:
   ```python
   import pickle
   with open("label_encoder.pkl", "wb") as f:
       pickle.dump(label_encoder, f)
   ```
   If you don't have this file, the app will still work but will show
   numeric class indices (Class_0, Class_1, ...) instead of real names.

## Run the App

```bash
streamlit run app.py
```

## How Prediction Works

1. User uploads an image via the browser
2. Image is resized to 224x224 and normalized to [0, 1]
3. `model.predict()` is called — no rules, no hardcoding
4. The class with the highest score is returned along with confidence %
