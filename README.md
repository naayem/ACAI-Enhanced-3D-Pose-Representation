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


# pose_interpolation
