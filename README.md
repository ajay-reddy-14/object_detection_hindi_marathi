# Multilingual Object Detection & Translation System

An AI-powered application that detects objects in images and provides instant translations of the detected labels in **Hindi** and **Marathi**.

## 🚀 Features
- **Object Detection**: Identifies 26+ classes of everyday objects (e.g., Car, Tree, Person, Bus).
- **Multilingual Support**: Instantly translates English labels to Hindi (हिंदी) and Marathi (मराठी).
- **Multiple Models**: Trained and evaluated on MobileNetV2, ResNet50, VGG16, and more.
- **User-Friendly UI**: Built with Streamlit for easy image uploading and visualization.
- **High Accuracy**: The primary model (MobileNetV2) achieves **~89.5% accuracy**.

## 🛠️ Tech Stack
- **Language**: Python 3.x
- **Deep Learning**: TensorFlow, Keras
- **Web Framework**: Streamlit
- **Computer Vision**: OpenCV, Pillow
- **Dataset**: Open Images V7

## 📂 Project Structure
```
├── app.py                 # Main Streamlit application
├── model_factory.py       # Factory to load different model architectures
├── train_all_models.py    # Script to train all models sequentially
├── translate.py           # Translation logic (English -> Hindi/Marathi)
├── requirements.txt       # Python dependencies
├── experiments/           # Stores trained models and logs
└── new_dataset/           # Dataset directory (Train/Valid/Test)
```

## ⚙️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/ajay-reddy-14/object_detection_hindi_marathi.git
    cd object_detection_hindi_marathi
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♂️ Usage

### Run the Web App
To start the application interface:
```bash
streamlit run app.py
```
Upload an image, and the app will display the detected object along with its translations.

### Train Models
To train all models (MobileNetV2, ResNet50, etc.) from scratch:
```bash
python train_all_models.py
```
Results will be saved in the `experiments/` folder.

## 📊 Model Performance
We evaluated multiple architectures. **MobileNetV2** was selected for deployment due to its balance of speed and accuracy.

| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- |
| **MobileNetV2** | **89.54%** | **0.899** | **0.895** | **0.895** |
| VGG16 | 76.38% | 0.774 | 0.764 | 0.763 |
| ResNet50 | 41.06% | 0.425 | 0.411 | 0.357 |

## 🤝 Contributing
Feel free to open issues or submit pull requests to improve the project!
