import os
from datetime import datetime
from pathlib import Path


def logfile(hyperparameters, parameters, constants, **kwargs):
    kwargs["exp_name"] = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    theExpName = "_fr_" + str(int(kwargs.get("freq_flag"))) + "_dr_" + str(
        kwargs.get("signaltime")) + "_ep_" + str(kwargs.get("n_epochs")) + "_lr_" + str(
        kwargs.get("lr")) + "_do_" + str(kwargs.get("dropout_rate")) + "_sp_" + str(int(kwargs.get("SP"))) + "_" + str(
        kwargs.get("model_select")) + "_" + str(kwargs.get("training_loss_fcn"))[:-2] + "_rd_" + str(
        kwargs.get("radius")) + "_fc_" + str(kwargs.get("FC_size")) + "_" + kwargs.get("exp_name")
    if kwargs.get('Transfer_model'):
        theExpName += "_Transfer"

        if kwargs.get('lat') == 40.7:
            theExpName += '_Golcuk'
        if kwargs.get('lat') == 38.2:
            theExpName += '_Seferihisar'
        if kwargs.get('lat') == 38.3:
            theExpName += '_Van'
        if kwargs.get('lat') == 36.6:
            theExpName += '_Adana'
    if kwargs.get('test'):
        theExpName += "_Test"
    s_path = kwargs.get('working_directory') + "/exps/EXP{}"
    # this line is not redundant as the constants are required in the below with-open block
    constants["save_path"] = os.path.realpath(Path(s_path.format(theExpName)))
    # update kwargs
    kwargs["save_path"] = constants["save_path"]

    if not (os.path.isdir("exps/EXP{}".format(theExpName))):
        os.mkdir("exps/EXP{}".format(theExpName))
        os.mkdir("exps/EXP{}/figs".format(theExpName))

    logs_path = os.path.join(constants['save_path'], 'logs.txt')
    kwargs["logs_path"] = logs_path

    with open(logs_path, "w") as file:

        # Hyperparameters 
        file.write("Hyperparameters:\n")
        file.write("---------------\n")
        for key, value in hyperparameters.items():
            file.write("{}: {}\n".format(key, value))
        file.write("\n")

        # Parameters 
        file.write("Parameters:\n")
        file.write("-----------\n")
        for key, value in parameters.items():
            file.write("{}: {}\n".format(key, value))
        file.write("\n")

        # Constants 
        file.write("Constants:\n")
        file.write("----------\n")
        for key, value in constants.items():
            file.write("{}: {}\n".format(key, value))
        file.write("\n")
        file.close()

    return kwargs
