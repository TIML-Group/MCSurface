# MCSurface

Bezier Surface Mode Connectivity

## Overview
This project implements a method for exploring mode connectivity in the parameter space of neural networks using Bezier surfaces. It provides tools to find different modes (local minima) and connect them via Bezier surfaces, thus helping to visualize and understand the loss landscape of neural networks more comprehensively.

## Project Structure

- `find_modes.py`: Script to find local minima (modes) of the loss surface for specified neural network architectures and datasets.
- `surface.py`: Script to train a Bezier surface given four control points that connect modes in the loss landscape.

## Requirements

- Python 3.8 or above
- PyTorch
- torchvision
- NumPy
- Matplotlib (for visualization)

## Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/your-repository/mode-connectivity-bezier.git
cd mode-connectivity-bezier
```

Usage
Finding Modes
To find different modes for the neural network, use the find_modes.py script. You can specify the model, dataset, and other parameters.
```bash
python find_modes.py --model GCN --dataset PROTEINS --epochs 100 --lr 0.01
```

support vgg, resnet, vit model working on CIFAR10/100 ImageNet dataset


Training Bezier Surface
To train a Bezier surface given the four control points, use the surface.py script. You must specify the control points, model, and dataset.
```bash
python surface.py --model GCN --dataset PROTEINS --control-points path/to/control_points.npy
```

Configuration
You can adjust various parameters for both mode finding and surface training by editing the corresponding sections in the provided scripts or by specifying additional command-line arguments.

Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
Thanks to all the contributors who have helped with developing algorithms, testing, and documentation.
