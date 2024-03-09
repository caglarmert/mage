from __future__ import print_function
import os
import torch
import numpy as np
import wandb
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device(dev) 

def train_model(model, scheduler, trainingLoader, optim, validationLoader, attributes, **kwargs):    
    save_path = kwargs.get('save_path')
    n_epochs = kwargs.get("n_epochs")
    ############################
    training_losses = []
    epochnum = []
    validation_losses = []
    # epoch loop
    for epoch in range(n_epochs): 
        model.train()
        batch_losses = []
        
        # mini batch loop
        for batch_idx, data in enumerate(trainingLoader):
            optim.zero_grad()
            sig, stat_info, gt = data
            sig = sig.to(device) 
            stat_info = stat_info.to(device)
            gt=gt.to(device) 
            
            predictions=model(sig, stat_info,**kwargs) 
        
            loss = kwargs.get('training_loss_fcn')
            output = loss(predictions, gt)  
            batch_losses.append(output.item())

            output.backward()

            optim.step()
            
                                   
        training_loss = np.mean(batch_losses) 
        
        
        #ultim training loss
        training_losses.append(training_loss)
        
        epochnum.append(epoch)
        # print("==> Training Loss (MSE):",training_loss,"==> for epoch:",epoch)       

        validation_loss= validate(model, validationLoader, **kwargs)
        validation_losses.append(validation_loss) 
        # print("==> Validation Loss (MSE):",validation_loss,"==> for epoch:",epoch)
        if kwargs.get("wandb"):
            wandb.log({"training_loss": training_loss, "validation_loss": validation_loss, "epoch": epoch})
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, "finalModel_state_dict.pt"))
    torch.save(model, os.path.join(save_path, "finalModel.pt"))
   
    return training_losses, validation_losses

def validate(model, validationLoader, **kwargs):
    
    model.eval() #not going to be training so no grad information needs to be saved
    with torch.no_grad():
        val_losses = []
        for batch_idx, data in enumerate(validationLoader):
            
            sig, stat_info, gt = data
            sig = sig.to(device) 
            stat_info = stat_info.to(device)
            gt=gt.to(device) 
            
            predictions=model(sig, stat_info, **kwargs) 

            loss = kwargs.get('training_loss_fcn')
            output = loss(predictions, gt)
            val_losses.append(output.item())      
            
    
    validation_loss = np.mean(val_losses)
    if kwargs.get("wandb"):
        wandb.log({"validation_loss": validation_loss})
    return validation_loss
