# MCSurface: Revisiting Mode Connectivity in Neural Networks with Bezier Surface

**Official Implementation for the paper "Revisiting Mode Connectivity in Neural Networks with Bezier Surface"**

## Overview

This project implements a method for exploring mode connectivity in the parameter space of neural networks using Bezier surfaces. It provides tools to find different modes (local minima) and connect them via Bezier surfaces, thus helping to visualize and understand the loss landscape of neural networks more comprehensively. All scripts support configuration via YAML files, with command-line arguments to override specific settings.

## Project Structure

- `find_modes.py`: Script to find local minima (modes) of the loss surface for specified neural network architectures and datasets.
- `surfaces.py`: Script to train a Bezier surface given four control points (modes) that define the corners of the surface in the loss landscape.
- `eval.py`: Script to evaluate a trained Bezier surface or the initial linear interpolation of modes, and visualize the loss/accuracy landscape.
- `configs/`: Directory containing example YAML configuration files for each script.

## Requirements

- Python 3.8 or above
- PyTorch
- torchvision
- NumPy
- Matplotlib (for visualization)
- timm (for ViT models)
- PyYAML (for configuration files)
- requests (for TinyImageNet download)
- gdown (for downloading from Google Drive, optional for pre-trained model)
- wandb (optional, for experiment tracking)

## Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/TIML-Group/MCSurface.git
cd MCSurface
pip install -r requirements.txt # Create and use a requirements.txt file
```

## Usage

All scripts can be configured using YAML files located in the `configs/` directory. Command-line arguments can be used to override settings in the YAML files.

The primary workflow example will use the **VGG model on the CIFAR10 dataset**.

### 1. Finding Modes

Use `find_modes.py` to train models and save checkpoints that represent different modes (local minima). These checkpoints will serve as the corner points for your Bezier surface.

**Supported Models**: VGG, ResNet, ViT, MobileNet
**Supported Datasets**: CIFAR10, CIFAR100, TinyImageNet

**Example using YAML configuration (VGG on CIFAR10):**
Create a configuration file, for instance, `configs/find_modes/vgg_cifar10_baseline.yaml`. This file will specify the model (`Vgg`), dataset (`CIFAR10`), learning rate, number of epochs, checkpoint folder, etc.

Example content for `configs/find_modes/vgg_cifar10_baseline.yaml`:

```yaml
# configs/find_modes/vgg_cifar10_baseline.yaml
dataset: CIFAR10
model: Vgg
lr: 0.001 # Adjust as needed
weight_decay: 2e-3 # Adjust as needed
num_epochs: 160 # Adjust as needed
batch_size: 128
checkpoint_folder: './checkpoints_vgg_cifar10' # Checkpoints will be saved here
scheduler_type: MultiStepLR
scheduler_milestones: [100, 140]
scheduler_gamma: 0.2
wandb_project: "find_modes_vgg_cifar10"
wandb_name_prefix: "vgg_cifar10_run"
```

Run the script using this configuration:

```bash
python find_modes.py --config configs/find_modes/vgg_cifar10_baseline.yaml
```

To override a parameter from the YAML file, add it as a command-line argument:

```bash
python find_modes.py --config configs/find_modes/vgg_cifar10_baseline.yaml --lr 0.0005 --num_epochs 180
```

This script will typically be run multiple times (e.g., 4 times with slight variations or for different random seeds) to generate the distinct checkpoints needed for the Bezier surface corners. Ensure the checkpoints are saved to distinct paths or are uniquely named if you intend to use them together.

### 2. Training Bezier Surface

Use `surfaces.py` to train a Bezier surface connecting four pre-trained model checkpoints (modes) obtained from the previous step.

**Example using YAML configuration (VGG on CIFAR10):**
Reference the configuration file `configs/surfaces/original_vgg_cifar10_surface.yaml`. This file specifies the model (`Vgg`), dataset (`CIFAR10`), hyperparameters for surface training, and crucial paths to the four checkpoint files generated by `find_modes.py`.

Example content for `configs/surfaces/original_vgg_cifar10_surface.yaml`:

```yaml
# configs/surfaces/original_vgg_cifar10_surface.yaml
model_type: Vgg
dataset: CIFAR10
num_bends: 2
learning_rate: 0.005
weight_decay: 4e-4
num_samples: 20
batch_size: 128
init_epochs: 6
total_epochs: 26
checkpoint_paths:
  - './checkpoints_vgg_cifar10/run1/model_epoch_XXX.pth' # Replace XXX and paths
  - './checkpoints_vgg_cifar10/run2/model_epoch_YYY.pth' # with your actual
  - './checkpoints_vgg_cifar10/run3/model_epoch_ZZZ.pth' # checkpoint paths from step 1
  - './checkpoints_vgg_cifar10/run4/model_epoch_AAA.pth'
output_surface_model_path_template: 'saved_models/Surface_{model_type}_{dataset}_b{num_bends}.pth'
```

Run the script:

```bash
python surfaces.py --config configs/surfaces/original_vgg_cifar10_surface.yaml
```

This will train the Bezier surface and save the trained `SurfaceNet` model to a path like `saved_models/Surface_Vgg_CIFAR10_b2.pth` (based on the template).

### 3. Evaluating Bezier Surface & Loss Landscape

Use `eval.py` to evaluate the trained Bezier surface or to explore the landscape defined by the initial linear interpolation of the corner modes.

**Using the Pre-trained VGG/CIFAR10 SurfaceNet Model:**

To quickly get started with evaluation for the VGG/CIFAR10 example, you can download a pre-trained `SurfaceNet` model:

- **Download Link**: [VGG/CIFAR10 SurfaceNet Model](https://drive.google.com/file/d/1aqdP4dHaTMMJ4DsftA_9JFs8SwaoyMzx/view?usp=sharing)

Download this file and save it, for example, as `saved_models/Surface_Vgg_CIFAR10_b2.pth`. You can then use this pre-trained model with the evaluation script.

**a) Evaluating a Trained Bezier Surface (VGG on CIFAR10):**
Create or use a YAML configuration for evaluation. This file should specify the `model_type` (`Vgg`), `dataset` (`CIFAR10`), `num_bends`, `checkpoint_paths` (the same four used for training the surface, or placeholders if using the pre-trained model primarily for its surface parameters), and the `surface_model_path` (pointing to your downloaded or self-trained model).

**Example `configs/eval/eval_vgg_cifar10_surface.yaml` (create this file):**

```yaml
# configs/eval/eval_vgg_cifar10_surface.yaml
model_type: Vgg
dataset: CIFAR10
num_bends: 2 # Should match the num_bends of the pre-trained/trained surface

# Checkpoints for initializing SurfaceNet's structure.
# These should ideally match the corners of the provided/trained surface.
# If using the pre-trained model primarily for its surface parameters, 
# these specific paths are mainly for structural consistency during SurfaceNet object creation.
checkpoint_paths:
  - './checkpoints_vgg_cifar10/run1/model_epoch_XXX.pth' # Replace with your actual/placeholder paths
  - './checkpoints_vgg_cifar10/run2/model_epoch_YYY.pth'
  - './checkpoints_vgg_cifar10/run3/model_epoch_ZZZ.pth'
  - './checkpoints_vgg_cifar10/run4/model_epoch_AAA.pth'

# Path to the SurfaceNet model (downloaded pre-trained or self-trained)
surface_model_path: 'saved_models/Surface_Vgg_CIFAR10_b2.pth' 

batch_size: 256
```

Run evaluation:

```bash
python eval.py --config configs/eval/eval_vgg_cifar10_surface.yaml
```

This will load the `SurfaceNet` model, evaluate it on a grid of (u,v) points, and save loss and accuracy surface plots to a `plots/` directory.

**b) Evaluating with Linear Interpolation (VGG on CIFAR10, Skipping Trained Surface Model Load):**
To evaluate the loss landscape formed by the linear interpolation of the control points (i.e., without loading a trained `SurfaceNet` model, using only the `checkpoint_paths` to define the corners), use the `--skip_load_model` flag. The `surface_model_path` in the config will be ignored if this flag is present.

```bash
python eval.py --config configs/eval/eval_vgg_cifar10_surface.yaml --skip_load_model
```

This is useful for visualizing the initial landscape before training the Bezier surface or comparing it to the trained surface.

## Configuration

Detailed configurations for each script can be found in their respective YAML files within the `configs/` directory (e.g., `configs/find_modes/`, `configs/surfaces/`, `configs/eval/`). These files provide templates and examples for various parameters. The general approach is:

1. Copy an example YAML file or create a new one (e.g., `vgg_cifar10_baseline.yaml` for finding modes, `original_vgg_cifar10_surface.yaml` for surfaces, `eval_vgg_cifar10_surface.yaml` for evaluation).
2. Modify parameters as needed for your experiment, ensuring checkpoint paths are correct and correspond to your VGG/CIFAR10 runs.
3. Run the corresponding Python script with the `--config path/to/your_config.yaml` argument.
4. Optionally, override specific parameters via additional command-line arguments.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgements

Thanks to all the contributors who have helped with developing algorithms, testing, and documentation.
