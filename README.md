# Facial Expression Recognition in Classroom using Hybrid MobileNetV3-ViT with Token Downsampling

This repository contains the implementation of my undergraduate thesis project on **Facial Expression Recognition (FER)** for classroom environments.  
The proposed model combines **MobileNetV3** (for efficient feature extraction) with a **Vision Transformer (ViT)** (for global context modeling), enhanced by **token downsampling** for improved efficiency.

Publication: https://doi.org/10.47709/brilliance.v5i1.6323

Dataset: https://huggingface.co/datasets/MoKhaa/FER2013

Model: https://huggingface.co/MoKhaa/Hybrid_MobileNetV3_ViT

## Research Motivation
In large classroom settings, it is often difficult for educators to continuously monitor each student's engagement and emotional state.  
Facial expressions provide valuable cues for understanding students’ participation and emotional conditions.  
This research proposes a **robust and efficient FER system** suitable for real-world classroom scenarios with limited device resources.

## Dataset Overview
<img width="741" height="380" alt="image" src="https://github.com/user-attachments/assets/707dbd85-a840-421f-a336-188c442726e7" />
<img width="740" height="181" alt="image" src="https://github.com/user-attachments/assets/8b2f68ca-655a-4fee-b25e-4c98e866cce0" />

The model is trained and evaluated on the **FER-2013** dataset (preprocessed version: `fer2013v2_clean.csv`).  
Classes include:
- Angry  
- Disgust  
- Fear  
- Happy  
- Sad  
- Surprise  
- Neutral

## Model Architecture
<img width="742" height="315" alt="image" src="https://github.com/user-attachments/assets/bd9a8f08-fb91-472e-af1f-5522458553f1" />

1. **Feature Extractor (MobileNetV3)**: Used to efficiently extract key features from facial images. MobileNetV3 was chosen for its lightweight design, making it suitable for real-time applications.
2. **Classifier (Vision Transformer)**: The extracted features are then processed by a Vision Transformer (ViT) for expression classification. ViT can learn contextual relationships between facial features, thereby improving accuracy.
3. **Token Downsampling**: This technique is implemented within the ViT to reduce the number of non-informative tokens, speeding up the training and inference processes without a significant loss in accuracy.

## Installation and Usage
To run this project in your local environment, follow these steps:
1. Clone the repository
```bash
git clone https://github.com/Khaairi/Facial_Expression_Recognition_Hybrid_ViT_Mobilenetv3.git
cd Facial_Expression_Recognition_Hybrid_ViT_Mobilenetv3
```
2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Place dataset (`fer2013v2_clean.csv`) inside `data/` folder and model checkpoint (`hybrid_mobilenet_vit_pooling_SAM_best.pt`) inside `checkpoints/` folder for model evaluation.
5. Run training & evaluation model
```bash
python src/train.py # For model training
python src/evaluate.py # For model evaluatio
```
6. Checkpoints and results (plots, best model) will be saved in the `results/` folder.

## Experimental Results
Confusion matrix and training curves are available in the `results/` directory.
- Best Test Accuracy: 71.24%
- Best Test F1-Score: 70.81%

Comparison of Model Performance in This Study with Previous Works:
|                   Model                  | Test Accuracy on FER2013 |
|:----------------------------------------:|:------------------------:|
| ViT+HOG (Rajae et al., 2025)             |            58%           |
| MobileViT (Jiang et al., 2024)           |          62.20%          |
| 5-layer CNN (Mukhopadhyay et al., 2020)  |            65%           |
| ResNet50 (Ping, 2024)                    |          65.31%          |
| VGG16 (Ping, 2024)                       |          67.01%          |
| ViT (Ping, 2024)                         |          69.60%          |
| EmoNeXt-Base (El Boudouri & Bohi, 2023)  |          74.91%          |
| Ensamble 7 CNN (Lawpanom et al., 2024)   |          75.15%          |
| EmoNeXt-Large (El Boudouri & Bohi, 2023) |          75.57%          |
| ResEmoteNet (Roy et al., 2025)           |          79.79%          |
| Hybrid MobileNetV3-ViT (This research)   |          71.24%          |
