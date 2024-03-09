# README: Arguments Explanation

This document provides an explanation of the various command-line arguments used in the program along with their default values and possible options.

### Boolean Arguments:
- `--wandb`: If specified, enables the use of WandB. Default: `True`.
- `--test`: If specified, enables test mode. Default: `False`.
- `--pc`: If specified, enables the use of PC. Default: `True`.
- `--Transfer_model`: If specified, enables transfer learning for the model. Default: `False`.
- `--Transfer_encoder`: If specified, enables transfer learning for the encoder. Default: `False`.
- `--add_stat_info`: If specified, adds statistical information. Default: `True`.
- `--add_station_altitude`: If specified, adds station altitude information. Default: `True`.
- `--gtnorm`: If specified, enables ground truth normalization. Default: `False`.
- `--SP`: If specified, performs signal processing. Default: `False`.
- `--augmentation_flag`: If specified, enables data augmentation. Default: `True`.
- `--freqtime`: If specified, considers frequency over time. Default: `False`.

### Numeric Arguments:
- `--fno`: Number of features. Default: `1`.
- `--fsiz`: Feature size. Default: `4`.
- `--batchsize`: Batch size. Default: `64`.
- `--n_epochs`: Number of epochs. Default: `100`.
- `--step_size`: Step size. Default: `20`.
- `--gamma`: Gamma value. Default: `0.9`.
- `--FC_size`: Fully connected layer size. Default: `256`.
- `--radius`: Radius value. Default: `300`.
- `--magnitude`: Magnitude value. Default: `3.5`.
- `--depth`: Depth value. Default: `10000`.
- `--stat_dist`: Station distance. Default: `120`.
- `--train_percentage`: Training percentage. Default: `80`.
- `--augmentation_parameter`: Augmentation parameter. Default: `1`.
- `--fs`: Frequency. Default: `100`.
- `--window_size`: Window size. Default: `1`.
- `--signaltime`: Signal time. Default: `60`.
- `--lr`: Learning rate. Default: `0.0001`.
- `--dropout`: Dropout rate. Default: `0.1`.

### String Arguments:
- `--gt_select`: Ground truth selection. Default: `["Distance"]`.
- `--dataset`: Dataset name. Default: `"STEAD"`.
- `--crossvalidation_type`: Cross-validation type. Default: `"Chronological"`.
- `--loss_function`: Loss function type. Default: `"MAE"`.
- `--network`: Network architecture. Default: `"TCN"`.

### Floating-point Arguments:
- `--signal_aug_rate`: Signal augmentation rate. Default: `0.3`.

### Location Arguments:
- `--lat`: Latitude. Default: `36.77`.
- `--lon`: Longitude. Default: `-119.41`.

### Notes:
- Use appropriate flags to enable or disable specific features according to your requirements.
- Adjust numeric values as needed based on the specifics of your dataset and model architecture.

Please refer to the documentation or source code for further details on each argument's usage and functionality.
