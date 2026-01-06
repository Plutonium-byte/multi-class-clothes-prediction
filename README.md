### Mutli Class Clothes Classification

This project leverages **Transfer Learning** to perform multi-class image classification on a variety of clothing items. By utilizing a pre-trained **Xception** architecture, the model effectively identifies different garment types, bridging the gap between raw pixel data and meaningful labels.

---

## 🚀 Project Overview

The core of this project is built on taking a powerful model trained on a massive dataset (ImageNet) and adapting it to a specific task: clothing recognition.

The workflow involves:

1. **Data Preparation**: Reorganizing raw images into structured sets and applying augmentation.
2. **Model Adaptation**: Using the Xception base model (minus its top layer) and adding custom dense layers.
3. **Optimization**: Tuning hyperparameters to maximize classification accuracy.
4. **Deployment**: Offering both a local CLI-based workflow and a web-based interface.

---

## 🛠 Tech Stack

* **Frameworks**: [TensorFlow](https://www.tensorflow.org/), [Keras](https://keras.io/)
* **Data Handling**: [NumPy](https://numpy.org/), [split-folders](https://pypi.org/project/split-folders/)
* **UI/Deployment**: [Gradio](https://gradio.app/), [Hugging Face Spaces](https://huggingface.co/spaces)
* **Image Processing**: [Pillow (PIL)](https://www.google.com/search?q=https://python-pillow.org/)
* **Environment**: [Google Colab](https://colab.research.google.com/)

---

## 📊 Dataset & Preprocessing

The model is trained using the **Clothes Dataset** sourced from Kaggle.

* **Source**: [Ryan Badai's Clothes Dataset](https://www.kaggle.com/datasets/ryanbadai/clothes-dataset)
* **Data Augmentation**: To improve the model's robustness and overall score, data augmentation techniques (such as rotations, zooms, and flips) were implemented during the training phase.

---

## 🏗 Model Architecture & Tuning

The project utilizes the **Xception** model as a feature extractor.

### Transfer Learning Strategy

1. **Base Model**: Xception (pre-trained on ImageNet) with the top classification layer removed.
2. **Custom Layers**: A custom head was added, consisting of Global Average Pooling and specialized Dense layers.
3. **Hyperparameter Tuning**: To achieve optimal performance, three specific parameters were tuned:
* **Learning Rate**: Adjusted to ensure stable convergence during fine-tuning.
* **Dense Layer Size**: Optimized the number of units in the second dense layer.
* **Dropout**: Implemented to reduce overfitting and improve generalization on the test set.



---

## 💻 Usage

### Local Setup

The project is organized into three main Python scripts:

Run the below command to install dependencies (`requirements.txt` is inside the Docker folder):
```
pip install -r requirements.txt
```

Executes the below 3 scripts to test the project.

1. **`prepare_data.py`**: Uses `split-folders` to organize raw images into train, validation, and test directories.
2. **`train.py`**: Configures the Xception base, applies data augmentation, and executes the training loop with the tuned hyperparameters.
3. **`test.py`**: Loads the saved model to perform inference and generate predictions on new images.

Notes: 
1. You could optionally use the Dockerfile if you prefer that but executing `train.py` is mandatory as it generates the model and saves it. The name of this saved model should be mentioned in the `Dockerfile` by replacing `model.h5` with the actual name.
2. Add the respective paths in `train_data`, `val_data` inside `train.py`, `path` inside `test.py`
3. Add image path in `service.py` if you're planning to use Docker. 
   
### Cloud / Web Interface

The project is also hosted on **Hugging Face Spaces**. This version uses **Gradio** to provide a user-friendly drag-and-drop interface for image classification.
<img width="1820" height="904" alt="Screenshot (5)" src="https://github.com/user-attachments/assets/f05d3ae1-7698-4207-9390-09aa3202f115" />


---
