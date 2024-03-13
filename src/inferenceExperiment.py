from __future__ import print_function
import torch 
from math import radians, cos, sin, asin, sqrt
import numpy as np
import wandb
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device(dev) 


def inference_experiment(loader, attributes,signal_height, signal_width, model, phase, **kwargs):     

    metric_dist = []
    Vs30_diff =[]
    column_wise_errors = []
    gt_select = kwargs.get("gt_select")
    true_values = [[] for ii in range(len(gt_select))]
    predictions = [[] for ii in range(len(gt_select))]
    data = [[] for ii in range(len(gt_select))]
    column_wise_errors = [[] for ii in range(len(gt_select))]
    model.eval()
    
    with torch.no_grad(): 
        losses = []
        for batch_idx, data_ in enumerate(loader):
            
            sig, stat_info, gt = data_ #sig = batchSize x 3 x accelPoints x 1
            sig = sig.to(device) 
            stat_info = stat_info.to(device) #☺batchSize x 3
            gt=gt.to(device)  #☺batchSize x varyingLength
            
            outputs=model(sig, stat_info, **kwargs) #same shape as gt
     
            loss = kwargs.get('training_loss_fcn') #MSE loss object
            output = loss(outputs, gt) # L2 norm batch ortalaması (reduction default olarak mean)
            losses.append(output.item()) #accumulate the losses of each batch in losses, her batch için bir loss değeri kaydediliyor 
            
            
            # Calculate test losses 
            for i in range(len(gt[0])): # Per column of gt
                for mb in range(len(gt)): # Per elements in a  --Z loop iterates for 64 times if batchSize is 64                    
                    # Record Reversed Values
                    if kwargs.get("gtnorm"):
                        true_val = gt[mb][i] * attributes['trStds'][i] + attributes['trMeans'][i]
                        predicted_val = outputs[mb][i] * attributes['trStds'][i] + attributes['trMeans'][i]
                    else:
                        true_val = gt[mb][i]
                        predicted_val = outputs[mb][i]
                    
                    ######################
                    # Add stat_info back to relative distance differences
                    if kwargs.get("gt_select")[i] == 'epiLAT':
                        true_val = true_val + stat_info[mb,0] + attributes["MeanStatLat"]
                        predicted_val = predicted_val+ stat_info[mb,0] + attributes["MeanStatLat"]
                    if kwargs.get("gt_select")[i] == 'epiLON':
                        true_val =  true_val + stat_info[mb,1] + attributes["MeanStatLon"]
                        predicted_val =  predicted_val + stat_info[mb,1] + attributes["MeanStatLon"]
                    ##################### 
                    true_values[i].append(true_val.item())
                    
                    predictions[i].append(predicted_val)
                    
                    data[i].append((true_val.item(), predicted_val.item()))
                    # Calculate sample-wise errors for test set only
                    if phase == "test": 
                        err_ = true_val.item() - predicted_val.item()
                        column_wise_errors[i].append(err_)
         
        #ave_loss = np.mean(losses) # mean of the batch losses, tüm batchlerin ortalama hata eğeri
        
        # Check if "epiLAT" and "epiLON" are present in gt_select
        if "epiLAT" in gt_select and "epiLON" in gt_select:
            
            # For Spherical Distance Conversion Between Angles and Length, apply Haversine formula
            for idx in range(len(data[0])):
                
                # Compute the haversine distance and append it to the metric_dist list
                distance = haversine(data[gt_select.index("epiLON")][idx][0], 
                                     data[gt_select.index("epiLAT")][idx][0], 
                                     data[gt_select.index("epiLON")][idx][1], 
                                     data[gt_select.index("epiLAT")][idx][1])
                metric_dist.append(distance)

        elif "Distance" in gt_select:

            for idx in range(len(data[0])):
                distance = np.abs(data[gt_select.index('Distance')][idx][0] - data[gt_select.index('Distance')][idx][1])
                metric_dist.append(distance)
            if kwargs.get("wandb"):
                wandb.log({"metric_dist": np.round(np.mean(metric_dist), 2)})

                
        if "Vs30" in gt_select:
            for idx in range(len(data[0])):
                
                diff = data[gt_select.index('Vs30')][idx][0] - data[gt_select.index('Vs30')][idx][1]
                
                Vs30_diff.append(diff)
            if kwargs.get("wandb"):
                wandb.log({"Vs30_Diff": np.round(np.mean(np.abs(Vs30_diff)),2)})
                
        if phase == "training":
            plot_dict = {"groundTruth_training": true_values,
                         "predictions_training": predictions,
                         "training_data": data
                         }
        if phase == "validation":
            plot_dict = {"groundTruth_validation": true_values,
                         "predictions_validation": predictions,
                         "validation_data": data
                         }
        if phase == "test":    
            plot_dict = {"groundTruth_test": true_values,
                         "predictions_test": predictions,
                         "test_data": data,
                         "column_wise_errors":column_wise_errors,
                         "metric_dist":metric_dist,
                         "Vs30_diff":Vs30_diff
                         }
            
        
        if phase == "test":
            
            
            with open(kwargs.get('logs_path'), "a") as file:
                file.write("trainSetSize: " + str(len(attributes["trind"])) + "\n")
                file.write("valSize: " + str(len(attributes["vlind"])) + "\n")
                file.write("testSize: " + str(len(attributes["tsind"])) + "\n")
                file.write("Results\n")
                file.write("---------------\n")
                file.write("Vs30_Diff Error:")
                file.write(str(np.round(np.mean(np.abs(Vs30_diff)),2)) + "\n")
                file.write("Metric_dist Error:")
                file.write(str(np.round(np.mean(metric_dist),2)) + "\n")



    return plot_dict


def haversine(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    return c * r
    