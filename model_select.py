# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 16:00:45 2023

@author: Asistan
"""
import torch
import os
from TCN_torch import TCN, TransferTCN, EncoderTCN, DecoderTCN
from resNet import ResNet,EncoderResNet,TransferModel,DecoderResNet
from utils import init_weights

def model_select(signal_height,signal_width,device,**kwargs):
    decoder_model = []
    if kwargs.get("model_select")=="TCN":
        if kwargs.get("Transfer_model") and kwargs.get("Transfer_encoder"):
            encoder_model = EncoderTCN(nb_filters=20, kernel_size=(6,1), nb_stacks=1, dilations=[2 ** i for i in range(11)], use_skip_connections=True, output_size = len(kwargs["gt_select"]),  drop_rate = kwargs.get("dropout_rate"), return_sequences=False).to(device)
            decoder_model= DecoderTCN(nb_filters=20, kernel_size=(6,1), nb_stacks=1, dilations=[2 ** i for i in range(11)], use_skip_connections=True, output_size = len(kwargs["gt_select"]),  drop_rate = kwargs.get("dropout_rate"), return_sequences=False).to(device)
        else:    
            model = TCN(nb_filters=20, kernel_size=(6,1), nb_stacks=1, dilations=[2 ** i for i in range(11)], use_skip_connections=True, output_size = len(kwargs["gt_select"]),  drop_rate = kwargs.get("dropout_rate"), return_sequences=False, **kwargs).to(device)
    else:
        if kwargs.get("Transfer_model") and kwargs.get("Transfer_encoder"):
            encoder_model = EncoderResNet(signal_height, signal_width,**kwargs).to(device)
            decoder_model= DecoderResNet(signal_height, signal_width,**kwargs).to(device)        
        else:    
            model = ResNet(signal_height, signal_width,**kwargs).to(device)
    
    if kwargs.get("Transfer_model"):
        if kwargs.get("Transfer_encoder"):
            if kwargs.get("model_select")=="TCN":
                state_dict = torch.load(os.path.join(kwargs.get("transfer_path")) + "finalModel_state_dict.pt")
                del state_dict["final_layer.weight"]
                del state_dict["final_layer.bias"]
                encoder_model.load_state_dict(state_dict)
                encoder_model.eval()
                decoder_model.apply(init_weights)
                model = TransferTCN(encoder_model, decoder_model)
                return model,decoder_model
            else:
                state_dict = torch.load(os.path.join(kwargs.get("transfer_path")) + "finalModel_state_dict.pt")
                del state_dict["linear1.weight"]
                del state_dict["linear1.bias"]
                del state_dict["linear2.weight"]
                del state_dict["linear2.bias"]
                encoder_model.load_state_dict(state_dict,strict=False)
                encoder_model.eval()
                decoder_model.apply(init_weights)
                model = TransferModel(encoder_model, decoder_model)
                return model,decoder_model
        else:    
            state_dict = torch.load(os.path.join(kwargs.get("transfer_path")) + "finalModel_state_dict.pt")
            model.load_state_dict(state_dict)
            model.eval()
    else:
        model.apply(init_weights)
        decoder_model = []
    return model, decoder_model

