# Readme for Adversarial Regularizer Project in Autoencoders

## Project Overview
This project explores the use of Adversarial Constrained Autoencoder Interpolation (ACAI) in enhancing the quality of latent space for 3D human poses representation. Using the h36m dataset for 3D human poses, we experiment with different autoencoder architectures to assess the impact of ACAI on latent space structuring, smoothness of interpolation, and dataset coverage.

## Dataset and Autoencoders
The h36m dataset, consisting of 3D poses represented as 17x3 tensors, is used. Augmentations such as rotation and horizontal flip are applied. Autoencoders with symmetrical bottleneck fully connected layers are employed, with varying latent-space sizes.

## Implementation Details
- **Autoencoders:** Vanilla autoencoders with different latent-space dimensions.
- **Interpolations:** Spherical interpolation between latent codes of pairs of poses.
- **ACAI Regularization:** Applied to encourage interpolated datapoints to be indistinguishable from real data reconstructions.
- **Metrics:** Mean Distance, Smoothness Ratios, Average Normalized Change over Interpolations (ANCI), and Dataset Coverage (Recall and Precision Distances).

## Results
Experiments show the influence of ACAI on latent space, highlighting changes in latent space structure, interpolation smoothness, and dataset coverage. The results suggest that ACAI can potentially enhance latent space quality, leading to more diverse and meaningful pose generation.

## Key Files and Scripts
- `main.py`: Main script for running experiments and generating results.
- `trainer.py`: Training script including model initialization, dataset handling, and execution of training procedures.
- `config_utils.py`, `models_common.py`, `losses_common.py`, etc.: Utility scripts for configuration, model definitions, loss functions, etc.
- `plot_generation.py`: Script for generating plots and visualizations of the results.

## Requirements
- Python 3.x
- PyTorch
- NumPy
- Matplotlib

## How to Run
1. **Prepare the Dataset:** Ensure the h36m dataset is properly set up and pre-processed.
2. **Configure the Experiment:** Modify the configuration files to set up the desired autoencoder architecture and ACAI parameters.
3. **Train the Model:** Run `main.py` to start the training process.
4. **Evaluate Results:** Use `plot_generation.py` to visualize the results and understand the impact of ACAI on latent space.

## Conclusion
This study provides insights into the effects of ACAI on autoencoder-based latent spaces for 3D human poses. Future work could explore wider hyperparameter ranges, different architectures, and enhanced visualization tools for better latent space analysis.

## References
- D. Berthelot, C. Raffel, A. Roy, and I. Goodfellow, "Understanding and improving interpolation in autoencoders via an adversarial regularizer," 2018.
- A. Davydov, P. Fua, "Adversarial parametric pose prior," 2021.

## Contact
- **Researcher:** Vincent Naayem
- **Supervisor:** Dr. Andrey Davydov
- **Professor:** Professor Pascal Fua

## Acknowledgements
Special thanks to all who contributed to the research, especially for the support and guidance provided by Dr. Andrey Davydov and Professor Pascal Fua.



# neural-network-trainer

## The core of the project is a custom yet functional approach to neural networks training.

Aims at making neural networks training on pytorch more handy and clear.

It comprises of experiment description (.yaml dictionary files), starting point file (`main.py`) and the `deep_cvlab` directory. The latter includes:

- `core`: The main class `Trainer` that parses experiment description and initializes the experiment. Its instance is shared between all training steps needed.
- `datasets`: Initialization datasets and dataloaders, `datasets_common.py` file. All files with dataset classes must lie in this folder. 
- `functional`: Additional functional, e.g. custom NN layers.
- `losses`: Initialization of losses functions, `losses_common.py` file. All specific criterion classes must lie in the `loss.py`.
- `models`: Initialization of NN models, `models_common.py` file. All files with models architectures must lie in this folder. 
- `optimizers`: Initialization of optimizers and schedulers (if needed), `optimizers_common.py` file. 
- `procedures`: Initialization of training procedures, `procedures_common.py` file. All files with procedures must lie in the folder `procedures/procedures`. Every `<proc>.py` file must include `train()` and `valid()`functions that describe the training behavior for every training and validation epoch. 
- `utils`: Additional utilities, e.g. visualization or augmentation transformations. Includes the `metrics.py:AvgMeter` class that is able to keep track of any measurements necessary for training.
- `models_pretrained`: State dicts for pretrained models. 
- `run`: `main.py` file is a starting point of the training. The training starts with the following command (from the ROOT folder): `python run/main.py --cfg experiments/<exp>.yaml`
