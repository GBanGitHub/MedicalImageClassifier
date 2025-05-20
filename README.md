# Medical Image Classifier

This project is a deep learning-based medical image classification system designed to identify various medical conditions from medical images such as X-rays and CT scans. It uses state-of-the-art deep learning models and is optimized for GPU acceleration.

## Project Development

The project was developed through the following steps:

1. **Data Collection & Preparation**: We collected publicly available medical image datasets (like chest X-rays and CT scans). Images were organized into labeled folders, and metadata was tracked in CSV files (e.g., `metadata.csv`) to record labels, splits (train/val/test), and file paths. We applied preprocessing techniques such as resizing, normalization, and data augmentation (random rotations, flips, and intensity changes) to improve model robustness.

2. **Model Selection & Training**: We explored several deep learning architectures, focusing on convolutional neural networks (CNNs) because they work well for image analysis. Models were implemented using popular frameworks (like PyTorch or TensorFlow). We tuned hyperparameters (learning rate, batch size, optimizer) using validation performance. Training was accelerated using GPU (CUDA) to handle large datasets and deep models efficiently.

3. **Evaluation**: We assessed model performance using metrics like accuracy (overall correctness), precision (correct positive predictions), recall (coverage of actual positives), and F1-score (balance between precision and recall). Cross-validation and separate test sets ensured that our results were reliable and not overfitted.

4. **Codebase Organization**: The repository is structured for clarity and scalability, separating data, model definitions, utility functions, and scripts. This modular approach makes it easy to extend or modify components as needed.

## Models Used

- **ResNet (Residual Network)**: Uses skip connections to allow gradients to flow through deeper networks, making it easier to train very deep models. ResNet is widely used as a strong baseline for image classification tasks.

- **DenseNet (Densely Connected Network)**: Connects each layer to every other layer in a feed-forward fashion, improving information and gradient flow. DenseNet often achieves high accuracy with fewer parameters and is effective for medical images where subtle features matter.

- **EfficientNet**: Scales network width, depth, and resolution in a balanced way, achieving high accuracy with fewer resources. EfficientNet is particularly useful when computational efficiency is important, such as in real-time or resource-constrained environments.

All models were trained and evaluated using GPU acceleration to enable rapid experimentation and support large-scale datasets.

## Project Structure

```
MedicalImageClassifier/
├── data/                  # Contains raw images, processed datasets, and metadata (e.g., metadata.csv)
├── models/                # Model architecture definitions and saved weights/checkpoints
├── utils/                 # Utility functions for data loading, preprocessing, augmentation, and evaluation
├── train.py               # Main script for training models; configurable via command-line arguments
├── predict.py             # Script for running inference on new images using trained models
├── notebooks/             # Jupyter notebooks for exploratory data analysis, prototyping, and visualization
└── README.md              # Project documentation (this file)
```

- **data/**: Stores all datasets and metadata. The `metadata.csv` file describes each image, its label, and which split (train/val/test) it belongs to.
- **models/**: Contains Python files defining model architectures (e.g., ResNet, DenseNet, EfficientNet) and saved model weights after training.
- **utils/**: Includes helper scripts for tasks like data preprocessing, augmentation, metric calculation, and visualization.
- **train.py**: The main entry point for training. Accepts arguments for dataset path, model type, hyperparameters, etc.
- **predict.py**: Used to classify new images with a trained model. Outputs predicted labels and confidence scores.
- **notebooks/**: Interactive Jupyter notebooks for data exploration, visualization, and rapid prototyping.

## Workflow Overview

1. **Prepare Data**: Organize images and metadata in the `data/` directory. Use provided utilities to preprocess and augment data.
2. **Train Model**: Run `train.py` with desired settings to train a model. Model checkpoints are saved in `models/`.
3. **Evaluate**: Use built-in metrics and validation scripts to assess model performance.
4. **Inference**: Use `predict.py` to classify new images or batches of images.
5. **Explore & Analyze**: Use Jupyter notebooks in `notebooks/` for further analysis, visualization, or experimentation.

---

For more details on the models or structure, see the respective directories and scripts in the repository. If you are new to deep learning or medical imaging, reviewing the notebooks and utility scripts is a good place to start. 