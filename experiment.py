from __future__ import print_function
import os
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from datasetTorch import structureData
from trainModel import train_model
from logs import logfile
from inferenceExperiment import inference_experiment
from datasetLoader import datasetCreator
from model_select import model_select
import torch.nn as nn

# filter warnings
import warnings
import argparse

warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import timedelta
from Plots import PlotExperiment
from utils import determine_logfile_columns, format_data
import time
import pickle

parser = argparse.ArgumentParser(
    prog='DeepQuake',
    description='Finds earthquakes',
    epilog='Interface: mert.caglar@metu.edu.tr')

parser.add_argument('--wandb', default=True, action="store_false")
parser.add_argument('--test', default=False, action="store_true")
parser.add_argument('--pc', default=True, action="store_false")
parser.add_argument('--fno', default=1, type=int)
parser.add_argument('--fsiz', default=4, type=int)
parser.add_argument('--batchsize', default=64, type=int)
parser.add_argument('--n_epochs', default=100, type=int)
parser.add_argument('--step_size', default=20, type=int)
parser.add_argument('--gamma', default=0.9, type=float)
parser.add_argument('--Transfer_model', default=False, action="store_true")
parser.add_argument('--Transfer_encoder', default=False, action="store_true")
parser.add_argument('--add_stat_info', default=True, action="store_false")
parser.add_argument('--add_station_altitude', default=True, action="store_false")
parser.add_argument('--gtnorm', default=False, action="store_true")
parser.add_argument('--gt_select', nargs='+', help='<Required> Set flag', default=["Distance"])
parser.add_argument('--FC_size', default=256, type=int)
parser.add_argument('--SP', default=False, action="store_true")
parser.add_argument('--statID', default=None)
parser.add_argument('--radius', default=300, type=int)
parser.add_argument('--magnitude', default=3.5, type=float)
parser.add_argument('--depth', default=10000, type=int)
parser.add_argument('--stat_dist', default=120, type=int)
parser.add_argument('--augmentation_flag', default=True, action="store_false")
parser.add_argument('--train_percentage', default=80, type=int)
parser.add_argument('--augmentation_parameter', default=1)
parser.add_argument('--dataset', default="STEAD")
parser.add_argument('--fs', default=100, type=int)
parser.add_argument('--signal_aug_rate', default=0.3)
parser.add_argument('--window_size', default=1, type=int)
parser.add_argument('--crossvalidation_type', default="Chronological")
parser.add_argument('--loss_function', default="MAE")

parser.add_argument('--lat', default=36.77)
parser.add_argument('--lon', default=-119.41)
parser.add_argument('--signaltime', default=60)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--freqtime', default=False, action="store_true")
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--network', default="TCN")

args = parser.parse_args()

if args.pc:
    working_directory = "/truba/home/ucaglar/Deprem/"
    AFAD_Path = "/truba/home/ucaglar/Deprem/Data/AFAD_KANDILLI/"
    STEAD_Path = "/truba/home/ucaglar/Deprem/datan/stead/"
    # STEAD_Path = "/truba/home/ucaglar/Deprem/Data/STEAD_small/"
    KANDILLI_Path = "/truba/home/ucaglar/Deprem/Data/AFAD_KANDILLI/"
else:
    working_directory = r"C:\Users\user\Desktop\Deprem"
    AFAD_Path = r"C:\Users\user\Desktop\Deprem"
    STEAD_Path = r"C:\Users\user\Desktop\Deprem\STEAD_small"
    KANDILLI_Path = r"C:\Users\user\Desktop\Deprem"

if args.wandb:
    import wandb

    wandb.login()


##############################################################################
def main(lat, lon, signaltime, transferpath, freqtime, lr, dropout, network, batch_name, training_loss_fcn):
    channel_depth = 3 + int(args.SP)
    if args.wandb:
        run = wandb.init(
            # Set the project where this run will be logged

            project="DeepQuake",
            entity="caglarmert",
            tags=["baseline"],
            save_code=True,
            reinit=True,
            # Track hyperparameters and run metadata
            config={
                "signaltime": signaltime,
                "lr": lr,
                "batch_name": batch_name,
                "training_loss_fcn": training_loss_fcn,
                "fno": args.fno,
                "fsiz": args.fsiz,
                "dropout_rate": dropout,
                "batchsize": args.batchsize,
                "n_epochs": args.n_epochs,
                "step_size": args.step_size,
                "gamma": args.gamma,
                "Transfer_model": args.Transfer_model,
                "Transfer_encoder": args.Transfer_encoder,
                "transfer_path": transferpath,
                "add_stat_info": args.add_stat_info,
                "add_station_altitude": args.add_station_altitude,
                "gtnorm": args.gtnorm,
                "test": args.test,
                "gt_select": args.gt_select,
                "model_select": network,
                "FC_size": args.FC_size,
                "SP": args.SP,
                "statID": args.statID,
                "radius": args.radius,
                "latitude": lat,
                "longitude": lon,
                "magnitude": args.magnitude,
                "depth": args.depth,
                "stat_dist": args.stat_dist,
                "freq_flag": freqtime,
                "augmentation_flag": args.augmentation_flag,
                "Train percentage": args.train_percentage,
                "Augmentation parameter": args.augmentation_parameter,
                "Crossvalidation_type": args.crossvalidation_type,
                "dataset": args.dataset,
                "fs": args.fs,
                "signal_aug_rate": args.signal_aug_rate,
                "window_size": args.window_size,
                "channel_depth": channel_depth
            })

    torch.manual_seed(42)
    if (torch.cuda.is_available()):
        seed_value = 42
        torch.cuda.manual_seed_all(seed_value)
        print(torch.cuda.get_device_name(0))
    else:
        print("no GPU found")
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(dev)
    ##############################################################################
    # 1. ExpName & Parameters
    hyperparameters = {"fno": args.fno,
                       "fsiz": args.fsiz,
                       "dropout_rate": dropout,
                       "batchsize": args.batchsize,
                       "n_epochs": args.n_epochs,
                       "lr": lr,
                       "step_size": args.step_size,
                       "gamma": args.gamma,
                       'training_loss_fcn': training_loss_fcn
                       }

    parameters = {"Transfer_model": args.Transfer_model,
                  "Transfer_encoder": args.Transfer_encoder,
                  "transfer_path": transferpath,
                  "add_stat_info": args.add_stat_info,
                  "add_station_altitude": args.add_station_altitude,
                  "gtnorm": args.gtnorm,
                  "test": args.test,
                  "gt_select": args.gt_select,  # epiLAT,epiLON,Depth,Distance
                  "model_select": network,  # "ResNet","TCN"
                  "FC_size": args.FC_size,
                  "SP": args.SP,
                  "statID": args.statID,  # Station ID
                  "radius": args.radius,  # The radius at which the experiment will be carried out. 300
                  "latitude": lat,  # In which Latitude the experiment will be performed.
                  "longitude": lon,  # In which Longitude the experiment will be performed.
                  "signaltime": signaltime,  # Time length of EQ signals
                  "magnitude": args.magnitude,
                  # The magnitude of the EQ signals (If it is less than this value, wont take it.)
                  "depth": args.depth,  # Depth of EQ signal (If it is greater than this value, will not use )(km)
                  "stat_dist": args.stat_dist,
                  # (unit?) Distance between the station recording the EQ event and the epicenter. (If it is greater than this value, will not use )
                  "freq_flag": freqtime,
                  # Will I use a frequency signal? Or is it a time signal? (T for Freq, F for Time)
                  "augmentation_flag": args.augmentation_flag,  # Will I augment the signal?
                  "Train percentage": args.train_percentage,  # Train-val-test percentage. (80 means %80 = train + val)
                  "Augmentation parameter": args.augmentation_parameter,
                  # How much will I augment the signal (1 means 2 pieces of 1 seconds, so at total 2 seconds.)
                  "Crossvalidation_type": args.crossvalidation_type,
                  # Crossvalidation_type (Chronological, Station-based)
                  "dataset": args.dataset  # Dataset to use (AFAD, STEAD, KANDILLI_AFAD)
                  }

    constants = {"km2meter": 1000,
                 "fs": args.fs,
                 "signal_aug_rate": args.signal_aug_rate,
                 # I augment the signal differently when the tpga is below this percentage value. (or over 1-signal_aug_rate)
                 "window_size": args.window_size,
                 "channel_depth": channel_depth,
                 "AFAD_Path": AFAD_Path,
                 "STEAD_Path": STEAD_Path,
                 "KANDILLI_Path": KANDILLI_Path,
                 "working_directory": working_directory,
                 "wandb": args.wandb
                 }

    ####### end of the temp line
    os.chdir(os.path.realpath(Path(working_directory)))

    kwargs = {**parameters, **constants, **hyperparameters}
    kwargs = logfile(hyperparameters, parameters, constants, **kwargs)
    ##add comments
    signal_width = kwargs.get('fs') * kwargs.get('window_size') / 2 + 1  # overlap constant
    signal_height = 2 * kwargs.get("signaltime") - 1

    model, decoder_model = model_select(signal_height, signal_width, device, **kwargs)

    print(kwargs.get('freq_flag'))

    if not args.test:
        if not kwargs.get('Transfer_model'):
            pickle_name = 'fat_'
            pickle_name += (
                    str(kwargs.get('dataset')) + '_lt_' + str(int(kwargs.get('latitude'))) + '_ln_' + str(
                int(kwargs.get('longitude'))) + '_rd_' + str(int(kwargs.get('radius'))) + '_dr_' + str(
                kwargs.get('signaltime')) + '_ed_' + str(
                kwargs.get('stat_dist')) + '_sp_' + str(int(kwargs.get('SP'))) + 'fr' + str(int(
                kwargs.get('freq_flag'))) + '_c_' + str(kwargs.get("Crossvalidation_type")[0]) + '.pkl')

            if not os.path.exists(pickle_name):
                # Call dataSetCreater and dump data if there is no pre-saved pickle
                print("Creating Dataset")
                print(str(pickle_name))
                attributes = datasetCreator(**kwargs)
                print("Dataset Created")
                # Dump data if it not test run
                with open(pickle_name, 'wb') as f:
                    pickle.dump(attributes, f)
                print("Dataset Saved")
            else:
                # Load the pkl if it exits
                print("Loading dataset:")
                print(str(pickle_name))
                with open(pickle_name, 'rb') as f:
                    attributes = pickle.load(f)
                print("Dataset loaded")
    else:
        # if it is test
        # TEST PATHS for KANILLI_AFAD
        kwargs["AFAD_Path"] = AFAD_Path
        kwargs["STEAD_Path"] = STEAD_Path
        kwargs["KANDILLI_Path"] = KANDILLI_Path
        kwargs['n_epochs'] = 1
        attributes = datasetCreator(**kwargs)

    params = {'batch_size': kwargs.get('batchsize'), 'shuffle': False}

    training_set = structureData(attributes, phase="training", **kwargs)
    trainingLoader = DataLoader(training_set, **params)
    validation_set = structureData(attributes, phase="validation", **kwargs)
    validationLoader = DataLoader(validation_set, **params)
    test_set = structureData(attributes, phase="test", **kwargs)
    testLoader = DataLoader(test_set, **params)

    if kwargs.get("Transfer_model") and kwargs.get("Transfer_encoder"):
        optim = torch.optim.Adam(decoder_model.parameters(), kwargs.get('lr'))
    else:
        optim = torch.optim.Adam(model.parameters(), kwargs.get('lr'))

    scheduler = StepLR(optim, kwargs.get('step_size'), kwargs.get('gamma'))

    # training_losses, validation_losses, plot_dict = train_model(model, scheduler, trainingLoader, optim, validationLoader, attributes, **kwargs)
    training_losses, validation_losses = train_model(model, scheduler, trainingLoader, optim, validationLoader,
                                                     attributes, **kwargs)

    ##
    trn_dict = inference_experiment(trainingLoader, attributes, signal_height, signal_width, model, phase="training",
                                    **kwargs)
    val_dict = inference_experiment(validationLoader, attributes, signal_height, signal_width, model,
                                    phase="validation", **kwargs)
    test_dict = inference_experiment(testLoader, attributes, signal_height, signal_width, model, phase="test", **kwargs)

    save_path = kwargs.get('save_path')
    torch.save(model, os.path.join(save_path, "model.pth"))

    plotargs = {**trn_dict, **val_dict, **test_dict, **kwargs}
    plotargs["training_losses"] = training_losses
    plotargs["validation_losses"] = validation_losses

    torch.save(plotargs, os.path.join(save_path, "plotargs.pth"))
    PlotExperiment(attributes, plotargs)

    # Write the results into a text file
    exps_directory = os.path.join(working_directory, "exps")
    # file_name = "batch_results_" + datetime.now().strftime("%m_%d") + ".txt"
    # file_path = os.path.join(exps_directory, file_name)
    file_path = os.path.join(exps_directory, batch_name)
    # check if the file already exists - append or write
    if os.path.exists(file_path):
        mode = "a"
    else:
        mode = "w"

        # with open(file_path, mode) as file:
    #     last_columns, metrics = determine_logfile_columns(plotargs)
    #     # metrics_str =
    #     if mode == "w":
    #         # file.write("Model\t\tDomain\t\tLR\t\tDropout\t\t{}\n".format(last_columns))
    #         file.write("Transfer\tLat\tLon\tRadius\t\tEpochs\t\tModel\t\tDomain\t\tDuration\t\tLR\t\t\tDropout\t\t{}\n".format(last_columns))

    #     file.write(f"{kwargs.get('Transfer_model')}\t{kwargs.get('latitude')}\t{kwargs.get('longitude')}\t{kwargs.get('radius')}\t\t\t{kwargs.get('n_epochs')}\t\t\t{kwargs.get('model_select')}\t\t\t{kwargs.get('freq_flag')}\t\t{kwargs.get('signaltime')}\t\t\t\t{kwargs.get('lr')}\t\t{kwargs.get('dropout_rate')}\t\t\t{metrics}\n")
    #     max_length = max(len(item) for item in columns)
    #     for item in columns:
    #         spaces = max_length - len(item) + 2
    #         num_elements = len(item)
    #         line = f"{item}{spaces * ' '}{num_elements}\n"
    #         f.write(line)

    column_headings = ["Transfer", "Lat", "Lon", "Radius", "Epochs", "Model", "Domain", "Duration", "LR", "Dropout"]
    # determine the headings according to gt_select parameters
    metrics_headings, metrics = determine_logfile_columns(plotargs)
    column_headings.extend(metrics_headings)
    column_widths = []
    for i, heading in enumerate(column_headings):
        max_heading_width = len(heading)
        column_widths.append(max_heading_width + 4)

    # write headings to file
    with open(file_path, mode) as file:
        if mode == "w":
            heading_str = format_data(column_headings, column_widths)
            file.write(heading_str + "\n")

        data = [
            kwargs.get('Transfer_model'), kwargs.get('latitude'), kwargs.get('longitude'),
            kwargs.get('radius'), kwargs.get('n_epochs'), kwargs.get('model_select'),
            kwargs.get('freq_flag'), kwargs.get('signaltime'), kwargs.get('lr'),
            kwargs.get('dropout_rate')
        ]
        data.extend(metrics)
        # format data
        data_str = format_data(data, column_widths)
        file.write(data_str + "\n")

    if args.wandb:
        run.finish()


transferpath = working_directory + "\\"

if args.loss_function == "MAE":
    training_loss_fcn = nn.L1Loss()
elif args.loss_function == "MSE":
    training_loss_fcn = nn.MSELoss()
else:
    training_loss_fcn = nn.MSELoss()

# nn.MSELoss() nn.L1Loss()
batch_name = 'SP_Ablation.txt'

print("\n>>>>>>>>>>>>>>>>>>>>>> lr", args.lr, "freqtime", args.freqtime, "dropout", args.dropout)
print(args.lat, args.lon, args.signaltime, transferpath, args.freqtime, args.lr, args.dropout,
      args.network, batch_name, training_loss_fcn)
start_time = time.time()
main(args.lat, args.lon, args.signaltime, transferpath, args.freqtime, args.lr, args.dropout,
     args.network, batch_name, training_loss_fcn)
end_time = time.time()
elapsed_time_secs = end_time - start_time
formatted_time = str(timedelta(seconds=int(elapsed_time_secs)))
print(f">>>>>>>>>>>>>>>>>>>>>> Execution time--> {formatted_time}")
