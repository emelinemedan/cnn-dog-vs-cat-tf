# cnn-dog-vs-cat-tf
Binary image classifier using a CNN to distinguish between dogs and cats, trained on TensorFlow with custom evaluation and Grad-CAM visualization.

# Dog vs Cat Classifier

Ce projet a pour objectif de construire un modèle de classification binaire (chien vs chat) à partir d'images, en utilisant un réseau de neurones convolutif (CNN) entraîné avec TensorFlow/Keras.

---

## Structure du projet

```
Dog_vs_Cats/
├── train/
│   ├── cats/           # Images de chats pour l'entraînement
│   └── dogs/           # Images de chiens pour l'entraînement
├── test1/                  # Images pour tester le modèle (sans labels)
├── model/                  # Contiendra le fichier .h5 du modèle sauvegardé
└── script.py               # Code Python
```

Les données proviennent de [Kaggle - Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data).
Après téléchargement, placez les images dans `train/`, puis séparez-les manuellement ou par script dans `cats/` et `dogs/`. Placez les images non labellisées dans `test1/`.

---

## Modèle utilisé

Le modèle est un CNN composé de 4 blocs convolution + max pooling, suivi d'un flatten, d'un dropout, d'une couche dense ReLU, puis d'une sortie sigmoid :

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

Le modèle est compilé avec :
```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

---

## Entraînement

- Images normalisées via `ImageDataGenerator`
- Data augmentation : rotation, zoom, flip horizontal
- Split automatique 80/20 pour validation
- Callbacks :
  - `EarlyStopping` pour éviter le surapprentissage
  - `ReduceLROnPlateau` pour ajuster le learning rate

---

## Évaluation

Après l'entraînement, le modèle atteint une précision de **89,5 %** sur le jeu de validation.

```text
=== Accuracy globale : 89.52% ===
```

Une matrice de confusion et un rapport de classification sont générés pour visualiser les performances par classe (chat vs chien).

---

## Interprétabilité

La dernière couche de convolution est identifiée automatiquement. Des outils comme Grad-CAM peuvent ensuite être utilisés pour visualiser les zones influentes dans la prédiction (non implémenté ici, mais prêt à être ajouté).

---

## Prédiction interactive

Un système interactif permet de choisir une image dans le dossier `test1/` (ex. image 450), d'afficher cette image et de laisser l'IA prédire :

```python
Prédiction : Dog (score : 1.00)
```

Cela permet de valider visuellement la performance du modèle sur de nouvelles images.

---

## Installation & requirements

Installez les librairies nécessaires :
```bash
pip install tensorflow opencv-python matplotlib seaborn scikit-learn
```

Assurez-vous que votre structure de données respecte le format ci-dessus (`train/cats`, `train/dogs`, `test1/`).

---

## Remarques

- Le modèle est sauvegardé en `.h5` à la fin de l'entraînement.
- Possibilité d'ajouter des visualisations Grad-CAM ou SHAP pour une interprétation plus fine.
- Test possible sur n'importe quelle image externe (pas uniquement celles de test1).

---

## Auteur
**Émeline Medan**  
GitHub: [@emelinemedan](https://github.com/emelinemedan)
