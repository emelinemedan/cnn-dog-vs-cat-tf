# Dog vs Cat Image Classification

This deep learning project aims to classify images as either dogs or cats using a Convolutional Neural Network (CNN). It was implemented with TensorFlow/Keras in Python.

---

## Objective
To build an image classification model capable of accurately distinguishing between cats and dogs.

---

## Dataset
The dataset comes from Kaggle: [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data). It includes:
- 12,500 images of **cats**
- 12,500 images of **dogs**

---

## Folder Structure
```
Dogs_vs_cats/
│
├── train/               # Training images
│   ├── cats/            # Cat images
│   └── dogs/            # Dog images
│
└── test1/               # Test images (unlabeled, for final predictions)
```

You will need to manually organize the `train/` folder into two subfolders: `cats/` and `dogs/`.

---

## Model Architecture
A CNN was built using Keras:
- Conv2D + MaxPooling2D layers
- Dropout for regularization
- Flatten and Dense layers

Early stopping and learning rate reduction were used to improve training.

---

## Performance
- **Training Accuracy**: Up to ~89%
- **Validation Accuracy**: Up to ~90%
- Plotted learning curves for loss and accuracy show steady learning.

---

## How to Run
1. Install requirements:
```bash
pip install tensorflow matplotlib numpy
```

2. Run the training script to train the CNN.
3. The trained model will be saved as `dog_cat_classifier.h5`.

---

## 🔹 Interactive Prediction
A script allows the user to:
- Input an image name from the `test1/` folder
- View the image
- Get a prediction: "Dog" or "Cat"

Example prompt:
```text
Enter image number (e.g., 150):
```

---

## 🔧 Skills Applied
- CNNs (Convolutional Neural Networks)
- Data preprocessing & augmentation
- Training optimization (early stopping, learning rate scheduling)
- Model evaluation (accuracy, classification report, confusion matrix)
- Interactive user input with prediction rendering

---

## Author
**Emeline Medan**  
GitHub: [@emelinemedan](https://github.com/emelinemedan)

---

## Acknowledgements
Dataset: [Kaggle Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data)
