# Image Captioning From Scratch

> Générer automatiquement des descriptions en langage naturel à partir d'images — sans aucun poids pré-entraîné.

<p align="center">
  <code>Image 224×224</code> → <b>CNN Encoder</b> (6 blocs) → <code>vecteur 512-d</code> → <b>LSTM Decoder</b> (2 couches) → <code>"a dog running in the park"</code>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.4.1-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Dataset-MS%20COCO%202017-blue" alt="COCO">
  <img src="https://img.shields.io/badge/GPU-Kaggle%20T4%2FP100-76B900?logo=nvidia&logoColor=white" alt="GPU">
  <img src="https://img.shields.io/badge/From%20Scratch-100%25-orange" alt="From Scratch">
</p>

---

## Sommaire

- [Objectif](#-objectif)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Techniques clés](#-techniques-clés)
- [Hyperparamètres](#-hyperparamètres)
- [Résultats](#-résultats)
- [Lancer le projet](#-lancer-le-projet)
- [Structure du projet](#-structure-du-projet)

---

## Objectif

Concevoir et entraîner un modèle **encoder-decoder** capable de générer une phrase en anglais décrivant le contenu d'une image.

**Contrainte principale** : tout le modèle est construit **from scratch** — aucun transfert learning, aucun poids pré-entraîné (pas de ResNet, pas de features ImageNet). Le CNN et le LSTM sont tous deux entraînés à partir de zéro.

---

## Architecture

```
┌─────────────┐     ┌───────────────────┐     ┌──────────┐     ┌──────────────────┐     ┌──────────┐
│   Image     │────▶│   CNN Encoder     │────▶│  Vecteur  │────▶│   LSTM Decoder   │────▶│  Caption  │
│  224×224×3  │     │  6 blocs VGG-like │     │  512-d    │     │   2 couches      │     │  texte   │
└─────────────┘     └───────────────────┘     └──────────┘     └──────────────────┘     └──────────┘
```

### CNN Encoder

CNN de style **VGG** avec **6 blocs convolutionnels** et **12 couches Conv2d** :

| Bloc | Canaux | Résolution |
|------|--------|------------|
| Bloc 1 | 3 → 32 | 224 → 112 |
| Bloc 2 | 32 → 64 | 112 → 56 |
| Bloc 3 | 64 → 128 | 56 → 28 |
| Bloc 4 | 128 → 256 | 28 → 14 |
| Bloc 5 | 256 → 512 | 14 → 7 |
| Bloc 6 | 512 → 512 | 7 → 3 |

Chaque bloc : `Conv2d(3×3)` → `BatchNorm` → `ReLU` → `Conv2d(3×3)` → `BatchNorm` → `ReLU` → `MaxPool2d(2×2)`

Suivi de : `AdaptiveAvgPool2d(1,1)` → `Dropout(0.3)` → `Linear(512, 512)` → `BatchNorm1d`

### LSTM Decoder

- **Embedding** : vocab_size → 256 dimensions (padding_idx=0)
- **LSTM** : 2 couches empilées, hidden_dim=512, dropout=0.35
- **Projection** : Linear(512 → vocab_size) → logits
- **Initialisation** : le vecteur de features de l'image initialise h₀ et c₀ via deux projections linéaires

---

## Dataset

**MS COCO 2017** — le dataset de référence pour l'image captioning.

| | Valeur |
|---|---|
| Images train | ~118 000 |
| Images val | ~5 000 |
| Captions utilisées / image | 2 (sur 5 disponibles) |
| Captions pour évaluation | 5 (toutes les références) |
| Taille du vocabulaire | ~9 000 mots |
| Seuil de fréquence min | 5 occurrences |

Le split validation est séparé 50/50 **par image** (pas par caption) pour créer les sets de validation et de test.

---

## Techniques clés

### Pack Padded Sequences
Le LSTM ignore les tokens `<pad>` grâce à `pack_padded_sequence`, ce qui évite d'apprendre à prédire du padding et améliore la qualité des gradients.

### Scheduled Teacher Forcing
Le ratio de teacher forcing décroît linéairement de **1.0 → 0.6** sur 30 epochs. Le modèle apprend progressivement à gérer ses propres erreurs de prédiction, réduisant l'exposure bias.

### Label Smoothing (ε = 0.1)
Les targets ne sont plus one-hot mais lissées : 0.9 pour le vrai mot, 0.1 réparti sur le reste du vocabulaire. Empêche le modèle d'être trop confiant et améliore la généralisation.

### Data Augmentation
Pipeline complet à l'entraînement (essentiel car le CNN est from scratch) :
- `Resize(240)` → `RandomCrop(224)` → `RandomHorizontalFlip(0.5)` → `RandomRotation(±10°)` → `ColorJitter(±20%)` → `Normalize`

### Beam Search
Inférence avec beam search (k=3 ou k=5) et normalisation du score par la longueur pour éviter de favoriser les captions courtes.

### Évaluation BLEU multi-références
Score BLEU-4 implémenté from scratch, évalué en prenant le meilleur score parmi les 5 captions de référence de chaque image.

---

## Hyperparamètres

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `IMG_SIZE` | 224 | Features visuelles plus riches |
| `BATCH_SIZE` | 32 | Compromis mémoire GPU / stabilité gradients |
| `LEARNING_RATE` | 3e-4 | "Karpathy constant" — excellent défaut pour Adam |
| `NUM_LSTM_LAYERS` | 2 | Plus de capacité pour le vocabulaire COCO |
| `EMBED_DIM` | 256 | Dimension des word embeddings |
| `HIDDEN_DIM` | 512 | Dimension de l'état caché du LSTM |
| `DROPOUT` | 0.3 / 0.35 | Encoder / Decoder |
| `LABEL_SMOOTHING` | 0.1 | Régularisation de la loss |
| `TF_RATIO` | 1.0 → 0.6 | Teacher forcing décroissant |
| `GRAD_CLIP` | 5.0 | Anti-explosion gradients LSTM |
| `NUM_EPOCHS` | 30 | Avec ReduceLROnPlateau (factor=0.5, patience=3) |

### Initialisation des poids

| Couche | Méthode |
|--------|---------|
| Conv2d | Kaiming Normal (fan_out, ReLU) |
| BatchNorm | γ=1, β=0 |
| Embedding | Uniform [-0.1, 0.1] |
| LSTM input-hidden | Xavier Uniform |
| LSTM hidden-hidden | Orthogonal |
| Forget gate bias | 1.0 |
| fc_out | Xavier Uniform |

---

## Résultats

Le modèle est évalué sur 200 images du test set avec beam search (k=3).

Les métriques BLEU-4 sont calculées en single-reference et multi-reference (max sur les 5 captions de chaque image).

Exemples de captions générées :

```
Image: [photo d'un chien dans un parc]
Greedy  : "a dog is standing in the grass"
Beam(3) : "a dog is running through the grass in a park"
Beam(5) : "a brown dog running through a green field"
```

---

## Lancer le projet

### Prérequis

- Python 3.8+
- PyTorch 2.4.1
- torchvision 0.19.1
- GPU recommandé (entraîné sur Kaggle T4/P100)

### Sur Kaggle (recommandé)

1. Créer un nouveau notebook sur [Kaggle](https://www.kaggle.com/)
2. Ajouter le dataset **`awsaf49/coco-2017-dataset`** via le bouton "Add Data"
3. Activer le GPU (Settings → Accelerator → GPU)
4. Importer le notebook et exécuter toutes les cellules

### En local

```bash
# Cloner le repo
git clone https://github.com/<votre-username>/image-captioning-from-scratch.git
cd image-captioning-from-scratch

# Installer les dépendances
pip install torch==2.4.1 torchvision==0.19.1

# Télécharger MS COCO 2017
# - Images : https://cocodataset.org/#download
# - Annotations : captions_train2017.json, captions_val2017.json

# Modifier les chemins dans la classe Config puis lancer le notebook
```

### Reprise d'entraînement

Le système de checkpoint sauvegarde automatiquement le meilleur modèle (selon la val_loss). Pour reprendre un entraînement interrompu, il suffit de relancer : le script détecte le checkpoint et reprend là où il s'est arrêté.

---

## Structure du projet

```
.
├── notebook.ipynb            # Notebook principal (entraînement + évaluation)
├── README.md
├── checkpoint_coco/
│   └── best_model.pth        # Checkpoint du meilleur modèle
└── assets/                   # Images pour le README (optionnel)
```

---

## Résumé des composants

| Composant | Choix | Détail |
|-----------|-------|--------|
| **Dataset** | MS COCO 2017 | ~118k images, 2 captions/image pour le train |
| **Encoder** | CNN 6 blocs (12 conv) | VGG-like, from scratch, BatchNorm + ReLU |
| **Decoder** | LSTM 2 couches | Pack padded, scheduled teacher forcing |
| **Loss** | CrossEntropy | label_smoothing=0.1, ignore_index=0 |
| **Optimiseur** | Adam | lr=3e-4 + ReduceLROnPlateau |
| **Inférence** | Beam Search | k=3 et k=5, score normalisé par longueur |
| **Métrique** | BLEU-4 | Multi-références, implémenté from scratch |
| **Plateforme** | Kaggle GPU | T4/P100, checkpoint resume |

---

<p align="center">
  <i>Projet 2A — École d'Ingénieur — Deep Learning</i>
</p>
