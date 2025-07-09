# VisionScope: Advanced Object Detection for Chess Analysis

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/Exill18/VisionScope-Demo)


## Table of Contents

* [Introduction](#introduction)
* [Features](#features)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Usage](#usage)
  * [Training a Model](#training-a-model)
  * [Running Inference](#running-inference)
* [Hugging Face Demo](#hugging-face-demo)
* [Dataset Information](#dataset-information)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)

## Introduction

**VisionScope** is a robust computer vision project focused on advanced object detection, with an emphasis on chessboard analysis and piece recognition. Utilizing the state-of-the-art YOLOv5 model, VisionScope provides tools for dataset management, model training, real-time inference, and result visualization. Although designed for chess, the framework is flexible and can be extended to support general object detection tasks.

The project includes a dedicated `ChessVision` submodule tailored for in-depth chess analysis and a Hugging Face demo for quick deployment and demonstration.

👉 **Try the Live Demo on Hugging Face**: [VisionScope-Demo](https://huggingface.co/spaces/Exill18/VisionScope-Demo)


## Features

* **YOLOv5 Integration**: Leverages the efficiency and accuracy of YOLOv5 for object detection.
* **Chess Piece Recognition**: Custom-trained models for identifying individual chess pieces on standard boards.
* **Flexible Dataset Management**: Supports YOLO format with utilities for generating and organizing label files.
* **Real-time Inference**: Detects objects in both images and video streams.
* **Custom Drawing Utilities**: Provides easy-to-use functions for visualizing detection outputs.
* **Training Scripts**: Includes scripts for training YOLOv5 models, with optional Weights & Biases (W\&B) integration.
* **Hugging Face Demo**: A plug-and-play Gradio/Streamlit app for showcasing capabilities via a web UI.
* **CUDA Support**: GPU acceleration for faster training and inference.

## Project Structure

```
VisionScope/
├── ChessVision/                      # Specialized module for chess vision tasks
│   ├── NotUsing/                     # Deprecated or experimental scripts
│   │   └── getPieces.py
│   ├── data_generated/labels/        # Generated label files for chess games
│   │   └── hikaru_gameX_moveY.txt
│   ├── main.py                       # Main script for chess vision tasks
│   └── requirements.txt              # Dependencies for ChessVision
├── HugginFaceco/VisionScope-Demo/    # Hugging Face Space demo
│   ├── app.py                        # Gradio/Streamlit app
│   ├── README.md                     # Demo-specific documentation
│   └── requirements.txt              # Demo dependencies
├── data/                             # General-purpose dataset folder
│   └── labels/
│       ├── train/                    # Training labels (e.g., UI elements)
│       └── val/                      # Validation labels
├── data_generated/labels/            # Auto-labeled chess game frames
│   └── hikaru_gameX_moveY.txt
├── runs/                             # Output from training runs
│   └── detect/train/
│       ├── args.yaml
│       └── results.csv
├── chess_dataset.yaml                # Dataset config for YOLOv5
├── config.yaml                       # General project configuration
├── cuda.py                           # GPU detection utilities
├── drawer.py                         # Drawing utilities for bounding boxes
├── inference.py                      # Inference script
├── labeled_images.json               # Metadata for labeled images
├── requirements.txt                  # Main dependencies
├── script.py                         # Main training script
└── wandbAttempeted.py                # Training with Weights & Biases
```

## Installation

1. **Clone the repository:**

```bash
git clone <repository_url>
cd VisionScope
```

2. **Create a virtual environment (recommended):**

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
pip install -r ChessVision/requirements.txt        # For ChessVision
pip install -r HugginFaceco/VisionScope-Demo/requirements.txt  # For demo
```

## Usage

### Training a Model

1. **Prepare your dataset** following the YOLO format, and update `chess_dataset.yaml` accordingly.
2. **Edit training parameters** in `config.yaml`, `script.py`, or `wandbAttempeted.py` (e.g., epochs, batch size).
3. **Start training:**

```bash
python script.py # Used to train the model for the chess
# Or with W&B integration
python wandbAttempeted.py
```

4. Training artifacts will be saved in `runs/detect/train/`.

### Running Inference

1. Ensure you have a trained weights file (e.g., `best.pt`).
2. Run inference:

```bash
python inference.py --source <image_path> --weights <path_to_weights.pt>
```

3. Explore additional flags in `inference.py` for fine-tuning (e.g., confidence threshold).

## Hugging Face Demo

To run the demo locally:

```bash
cd HugginFaceco/VisionScope-Demo
pip install -r requirements.txt
python app.py
```

Open your browser and navigate to the provided localhost address.

## Dataset Information

* **`data_generated/labels/`**: Auto-generated labels for chess frames. Each `.txt` aligns with an image, listing objects in YOLO format.
* **`data/labels/train/` and `val/`**: Labels for general object/UI detection. Demonstrates the broader applicability of VisionScope beyond chess.

## Contributing

Contributions are encouraged! To contribute:

* Fork the repo
* Create a feature branch
* Submit a pull request

Feel free to open issues or suggest enhancements.

## License

This project is licensed under the [MIT License](LICENSE).


