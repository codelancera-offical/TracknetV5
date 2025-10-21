# GEMINI.md - TracknetV5 Project

## Project Overview

This project is a deep learning-based object tracking system named TracknetV5. It is specifically designed to track a tennis ball in video footage. The core of the project is the UTrackNetV1 model, a U-Net-like architecture tailored for tracking. The entire framework is built using Python and the PyTorch deep learning library.

The project follows a modular, factory-based design pattern, which makes it highly configurable and extensible. Key components like the model, dataset, loss function, and optimizer are all built dynamically from configuration files.

### Core Technologies

*   **Programming Language:** Python
*   **Deep Learning Framework:** PyTorch
*   **Image/Video Processing:** OpenCV
*   **Data Handling:** pandas, NumPy

### Architecture

The project is structured into several key directories:

*   `configs/`: Contains experiment configuration files. These files define all the parameters for a training run.
*   `datasets_factory/`: Manages data loading and preprocessing.
*   `models_factory/`: Defines the neural network architectures. The main model is `UTrackNetV1`, which is composed of a backbone, a neck, and a head.
*   `engine/`: Contains the core training and validation logic, encapsulated in the `Runner` class.
*   `losses_factory/`, `optimizers_factory/`, `metrics_factory/`: Factories for creating loss functions, optimizers, and evaluation metrics.
*   `train.py`: The main script for initiating model training.
*   `utracknetv1_mvat_wbce_inference_contour_method.py`: A script for running inference on videos.

## Building and Running

### Setup

1.  **Install Dependencies:** The required Python packages are listed in `requirements.txt`. You can install them using pip:

    ```bash
    pip install -r requirements.txt
    ```

### Training

The training process is driven by the `train.py` script, which takes a configuration file as an argument.

1.  **Configure Your Experiment:** Create or modify a configuration file in `configs/experiments/`. For example, you can use `utracknetv1_mvat_tennis_b2e500_wbce.py` as a template. In the configuration file, you need to specify the dataset paths, model parameters, and training settings.

2.  **Run Training:** Execute the `train.py` script with the path to your configuration file.

    ```bash
    python train.py
    ```

    *Note: The `train.py` script currently has a hardcoded configuration path. You may need to modify the script to accept a command-line argument for the configuration file.*

### Inference

The `Readme.md` file provides the command for running inference on a video.

1.  **Run Inference:** Use the `utracknetv1_mvat_wbce_inference_contour_method.py` script, providing the path to the video folder and the model weights.

    ```bash
    python .\utracknetv1_mvat_wbce_inference_contour_method.py <video_folder_path> <weights_path>
    ```

## Development Conventions

*   **Configuration over Code:** The project emphasizes using configuration files to define experiments, rather than hardcoding parameters in the scripts. This makes it easy to reproduce experiments and try new ideas.
*   **Factory Pattern:** The use of factories for building components (models, datasets, etc.) allows for easy extension. To add a new model, for example, you would define it in the `models_factory/models` directory and register it with the `MODELS` builder.
*   **Modular Design:** The code is organized into logical modules (data, models, engine, etc.), which promotes code reuse and maintainability.
