# Image-Classification-with-Transfer-Learning
Overview

This project implements an image classification system using transfer learning on modern convolutional neural networks. The solution fine-tunes a pre-trained EfficientNetB0 model on a custom dataset and includes a complete training pipeline, inference script, evaluation utilities, and Grad-CAM visualizations for model interpretability.

The repository is designed to be production-ready, modular, and easy to extend for real-world ML engineering tasks.

⸻

Key Features
	•	Transfer learning using EfficientNetB0 (swappable to ResNet50 or MobileNet).
	•	Two-phase training strategy: head training followed by fine-tuning.
	•	Automated dataset loading and preprocessing with tf.data.
	•	Data augmentation pipeline for robust model generalization.
	•	Training callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau.
	•	SavedModel export for deployment.
	•	Prediction script for single-image inference.
	•	Grad-CAM visualization for understanding model decisions.
	•	Config-driven architecture for hyperparameters and paths.

⸻

Project Structure

image-classification-transfer-learning/
│
├── train.py               # Training pipeline
├── inference.py           # Single image prediction
├── gradcam.py             # Grad-CAM visualization utility
├── utils.py               # Data split + plotting helpers
├── configs.py             # Configuration values
├── requirements.txt
├── README.md
└── experiments/           # Models, logs, outputs (auto-created)


⸻

Dataset Structure

The dataset must follow the directory structure below:

data/
  train/
    class_1/
    class_2/
    ...
  val/
    class_1/
    class_2/
  test/
    class_1/
    class_2/

If you only have a single folder of class images, you may auto-split it with:

from utils import create_train_val_test_split
create_train_val_test_split("source_data", "data", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)


⸻

Installation
	1.	Create a virtual environment:

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

	2.	Install dependencies:

pip install -r requirements.txt


⸻

Training

Run the main training script:

python train.py

This will:
	•	Load the dataset.
	•	Train the classification head.
	•	Fine-tune the top layers of the base model.
	•	Save:
	•	Best weights: experiments/best_model.h5
	•	Exported model: experiments/saved_model/
	•	Training curve: experiments/training_history.png
	•	Class names file (if saved): experiments/class_names.json

⸻

Inference

To predict the class of a single image:

python inference.py --model experiments/saved_model --image path/to/image.jpg

The script outputs the top predicted classes with probabilities.

⸻

Grad-CAM Visualization

To generate an attention heatmap for understanding model decisions:

python gradcam.py --model experiments/saved_model --image path/to/image.jpg

This creates a Grad-CAM output image (gradcam.jpg) showing activated regions responsible for the prediction.

⸻

Configuration

Hyperparameters and paths are controlled via configs.py:

Examples:

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
BASE_MODEL = "efficientnet"
INITIAL_EPOCHS = 8
FINE_TUNE_EPOCHS = 10
FINE_TUNE_AT = 100

These can be modified to tune training performance.

⸻

Model Architecture

The default model uses the following pipeline:
	•	EfficientNetB0 base (ImageNet weights, convolutional layers frozen initially)
	•	Global Average Pooling
	•	Dropout layer for regularization
	•	Dense classification head with softmax activation
	•	Fine-tuning enabled from the top N convolutional layers after initial training

The model is trained using Adam optimizer and cross-entropy loss.

⸻

Best Practices
	•	Increase augmentation if the dataset is small.
	•	Adjust FINE_TUNE_AT to control how many layers are unfrozen during fine-tuning.
	•	Use mixed-precision training if supported by GPU.
	•	Save class names for consistent inference.
	•	Monitor validation metrics to prevent overfitting.

⸻

Extending This Project

Possible enhancements include:
	•	Switching to EfficientNetV2, ResNet50, or MobileNetV3.
	•	Exporting to TFLite or ONNX for mobile and edge deployment.
	•	Adding FastAPI or Flask serving API.
	•	Implementing k-fold cross-validation.
	•	Adding metric logging with TensorBoard or Weights & Biases.

⸻
