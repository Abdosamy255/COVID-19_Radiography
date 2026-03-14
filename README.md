# Chest X-Ray Disease Classification App

A Streamlit application for classifying chest X-ray images into one of four categories:

- COVID
- Lung_Opacity
- Normal
- Viral Pneumonia

The app uses a pre-trained TensorFlow/Keras model (`model.keras`) and provides:

- Fast model loading with caching
- Image preprocessing compatible with DenseNet input requirements
- Predicted class with confidence score
- Full class probability breakdown

## Project Structure

- `app.py`: Streamlit application entrypoint
- `model.keras`: Trained Keras model
- `COVID-19_Radiography_Dataset/`: Dataset folder used for model training
- `chest_de.ipynb`: Notebook for experimentation/training workflow

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`).

## Notes

- Keep `model.keras` in the project root next to `app.py`.
- Supported input formats: JPG, JPEG, PNG.
- This tool is for educational/research use and does not replace clinical diagnosis.
