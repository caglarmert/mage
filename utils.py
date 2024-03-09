# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 12:12:24 2023

@author: HP
"""
import torch
import torch.nn as nn
import numpy as np
from math import radians, cos, sin, asin, sqrt
from numpy.random import default_rng
import scipy
from scipy import signal
import collections

def haversine(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    
    Parameters:
    lon1 (float): Longitude of point 1
    lat1 (float): Latitude of point 1
    lon2 (float): Longitude of point 2
    lat2 (float): Latitude of point 2
    
    Returns:
    float: Distance between the two points in kilometers
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers

    return c * r

def CrossValidation(gettingfiles,stat_id,**kwargs):
    
    trind = []
    vlind = []
    tsind = []
    
    if kwargs.get('Crossvalidation_type') == "Chronological":
        
        #Calculate percentage of validation data       
        val_percent = int(kwargs.get('Train percentage')) - (int(kwargs.get('Train percentage'))*0.1)
        
        rng = default_rng()
        trainval = rng.choice(int((len(gettingfiles)*int(kwargs.get('Train percentage')))/100), int((len(gettingfiles)*int(kwargs.get('Train percentage')))/100), replace=False)
        test = [x for x in range(int((len(gettingfiles)*int(kwargs.get('Train percentage')))/100),len(gettingfiles))]
        #Store train-test-val indexes to trind,tsind,vlind    
        trind = trainval[:int((len(gettingfiles)*val_percent)/100)]
        vlind = trainval[int((len(gettingfiles)*val_percent)/100):int((len(gettingfiles)*int(kwargs.get('Train percentage')))/100)]
        tsind = np.random.choice(test, len(gettingfiles)-int((len(gettingfiles)*int(kwargs.get('Train percentage')))/100), replace=False)
                     
    else:
        
        stat_id_dict = collections.defaultdict(list)
        
        # Create a dictionary with station IDs as keys and corresponding indices as values
        for i, station in enumerate(stat_id):
            stat_id_dict[station].append(i)
        
        # Sort the dictionary based on the length of the value lists in descending order
        sorted_stat_id_dict = dict(sorted(stat_id_dict.items(), key=lambda x: len(x[1]), reverse=True))
        
        # Iterate over the sorted dictionary
        for station, indices in sorted_stat_id_dict.items():
            # Check if adding the current indices to trainv will still keep the train size within the desired percentage
            if (len(trind) + len(indices)) < (len(gettingfiles) * (kwargs.get('Train percentage')-(kwargs.get('Train percentage'))*0.1)) / 100:
                trind.extend(indices)  # Add the indices to train
            elif (len(vlind) + len(indices)) < (len(gettingfiles) * (kwargs.get('Train percentage'))*0.1) / 100:
                vlind.extend(indices) # Add the indices to val
            else:
                tsind.extend(indices)  # Add the indices to test
                
    return trind,vlind,tsind      

# Model Creation
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)   
            
            
def loader_helper(b, signal, alt, mag, numofData, epicenterLat, epicenterLon, epicenterDepth, 
                  stationLat, stationLon, stationcoor, gettingfiles, groundTruth, input_signal, 
                  magnitude_val, stat2epi, signal_list, epicenter_depth, lats, longs, stationco, 
                  epicentraldist, altitudes, stat_id, sP_3_channel, tpga_ind, stat_info,
                  skipped_file, Vs30, source_distance_km, **kwargs):

    try:
        if len(b)>24: #Afad files have upto 24 chars and stead have at least 27
            dictname = 'stead'
        else:
            last_four = b[len(b)-8:len(b)-4]
            if last_four.isdigit():
                dictname = 'afad'
            else:
                dictname = 'kandilli'


        counter=0
        gettingfiles.append(b)


        max_indices = np.argmax(signal,axis=0)
        median_index = np.median(max_indices)/kwargs.get('fs')



        # Select groundTruth according to gt_select
        new_column = np.array([])
        the_list = np.array([])
        for element in kwargs.get("gt_select"):

            if element == 'epiLAT':
                new_column = np.array(epicenterLat)
                the_list = np.hstack((the_list, new_column))
                #groundTruth.append(epicenterLat)
            if element == 'Depth':
                new_column = np.array(epicenterDepth)
                the_list = np.hstack((the_list, new_column))
                #groundTruth.append(epicenterDepth)
            if element == 'epiLON':
                new_column = np.array(epicenterLon)
                the_list = np.hstack((the_list, new_column))
                #groundTruth.append(epicenterLon)
            if element == 'Vs30':
                new_column = np.array(Vs30)
                the_list = np.hstack((the_list, new_column))
            if element == 'Distance':
                new_column = np.array(source_distance_km)
                the_list = np.hstack((the_list, new_column))
        groundTruth.append(the_list)


        # Calculate the distance from the epicenter to the desired location and the distance from the station to the epicenter.
        epi2loc = haversine(float(kwargs.get('latitude')), float(kwargs.get('longitude')),epicenterLat,epicenterLon) #Epicenter to desired location distance(Km)
        station_km = haversine(stationLat,stationLon,epicenterLat,epicenterLon) #Station to epicenter distance(Km)

        # Check if epicenter to location distance is greater than radius
        if (epi2loc > float(kwargs.get('radius'))):
            # If yes, remove the file from the list and delete the dataset
            gettingfiles.remove(b)
            groundTruth.pop()
            stat_id.pop()
            skipped_file[dictname]['epi2loc is greater than the radius'].append(b)


        else:
            # Check if station to epicenter distance is greater than stat_dist
            if (station_km > int(kwargs.get('stat_dist'))):
                # If yes, remove the file from the list
                gettingfiles.remove(b)
                groundTruth.pop()
                stat_id.pop()
                skipped_file[dictname]['station_km is greater than the stat_dist'].append(b)
                skipped_file[dictname]['eliminated epiDist values'].append(station_km)


            else:
                # Check if the number of data points in the signal is greater than desired signal duration * fs
                if numofData > (int(kwargs.get('signaltime'))*kwargs.get('fs'))-1:
                    # Check if epicenter depth is less than desired depth value
                     if epicenterDepth < float(kwargs.get('depth')):
                         # Check if median index(tpga) is less than or equal to desired signal duration
                         if median_index <= int(kwargs.get('signaltime')):

                             # Check which part of the signal the tpga value corresponds to (far left of the signal)
                             if median_index < float(int(kwargs.get('signaltime'))*kwargs.get('signal_aug_rate')):
                                 # Apply signal augmentation by flipping the first part of signal
                                 signal = signal[0:int(kwargs.get('signaltime'))* kwargs.get('fs'),:]
                                 # denemesignal = signal
                                 firstnoise = signal[0:int(kwargs.get('Augmentation parameter')) * kwargs.get('fs') * 2,:]
                                 flipped = firstnoise[::-1]
                                 signal = np.concatenate((flipped,signal),axis=0)
                                 # Save tpga index to store information about what kind of augmentation applied
                                 tpga = 0
                             # Check which part of the signal the tpga value corresponds to (far right of the signal)
                             elif median_index > float(int(kwargs.get('signaltime'))* (1- kwargs.get('signal_aug_rate'))):
                                 # Apply signal augmentation by flipping the first part of signal
                                 signal = signal[0:int(kwargs.get('signaltime'))* kwargs.get('fs'),:]
                                 # denemesignal = signal
                                 lastnoise = signal[-(int(kwargs.get('Augmentation parameter')) * kwargs.get('fs') * 2):,:]
                                 flipped2 = lastnoise[::-1]
                                 signal = np.concatenate((signal,flipped2),axis=0)
                                 # Save tpga index to store information about what kind of augmentation applied
                                 tpga = 2
                             # Check which part of the signal the tpga value corresponds to (in the middle of the signal)
                             else:
                                 # Apply signal augmentation by flipping the first part of signal
                                 signal = signal[0:int(kwargs.get('signaltime')) * kwargs.get('fs'),:]

                                 # denemesignal = signal

                                 firstnoise = signal[0:int(kwargs.get('Augmentation parameter'))* kwargs.get('fs'),:]
                                 flipped = firstnoise[::-1]

                                 lastnoise = signal[-(int(kwargs.get('Augmentation parameter'))* kwargs.get('fs')):,:]
                                 flipped2 = lastnoise[::-1]

                                 signal = np.concatenate((flipped,signal),axis=0)
                                 signal = np.concatenate((signal,flipped2),axis=0)
                                 # Save tpga index to store information about what kind of augmentation applied
                                 tpga = 1
                         # Check if the distance from the end of the signal to the median index is less than median_index/2
                         elif (numofData/kwargs.get('fs')) - median_index < float(median_index/2):
                             # Check which part of the signal the tpga value corresponds to (far left of the signal)
                             if median_index < float(int(kwargs.get('signaltime')) * (kwargs.get('signal_aug_rate'))):
                                 # Apply signal augmentation by flipping the first part of signal
                                 signal = signal[0:int(kwargs.get('signaltime'))* kwargs.get('fs'),:]

                                 # denemesignal = signal

                                 firstnoise = signal[0:int(kwargs.get('Augmentation parameter'))* kwargs.get('fs') * 2,:]
                                 flipped = firstnoise[::-1]

                                 signal = np.concatenate((flipped,signal),axis=0)
                                 # Save tpga index to store information about what kind of augmentation applied
                                 tpga = 0
                             # Check which part of the signal the tpga value corresponds to (far right of the signal)
                             elif median_index > float(int(kwargs.get('signaltime')) * (1-kwargs.get('signal_aug_rate'))):
                                 # Apply signal augmentation by flipping the first part of signal
                                 signal = signal[0:int(kwargs.get('signaltime'))* kwargs.get('fs'),:]

                                 # denemesignal = signal

                                 lastnoise = signal[-(int(kwargs.get('Augmentation parameter'))* kwargs.get('fs') * 2):,:]
                                 flipped2 = lastnoise[::-1]

                                 signal = np.concatenate((signal,flipped2),axis=0)
                                 # Save tpga index to store information about what kind of augmentation applied
                                 tpga = 2
                             # Check which part of the signal the tpga value corresponds to (in the middle of the signal)
                             else:
                                 # Apply signal augmentation by flipping the first part of signal
                                 signal = signal[0:int(kwargs.get('signaltime')) * kwargs.get('fs'),:]

                                 # denemesignal = signal

                                 firstnoise = signal[0:int(kwargs.get('Augmentation parameter')) * kwargs.get('fs'),:]
                                 flipped = firstnoise[::-1]

                                 lastnoise = signal[-(int(kwargs.get('Augmentation parameter'))*kwargs.get('fs')):,:]
                                 flipped2 = lastnoise[::-1]

                                 signal = np.concatenate((flipped,signal),axis=0)
                                 signal = np.concatenate((signal,flipped2),axis=0)
                                 # Save tpga index to store information about what kind of augmentation applied
                                 tpga = 1
                         #Tpga is in the middle of the signal
                         else:
                             # Cut the desired length from the signal
                             start = round((median_index - float(int(kwargs.get('signaltime'))/2))*kwargs.get('fs'))
                             stop = round((median_index + float(int(kwargs.get('signaltime'))/2))*kwargs.get('fs'))
                             signal = signal[start:stop,:]

                             # denemesignal = signal
                             # Apply signal augmentation by flipping the first part of signal
                             firstnoise = signal[0:int(kwargs.get('Augmentation parameter'))*kwargs.get('fs'),:]
                             flipped = firstnoise[::-1]

                             lastnoise = signal[-(int(kwargs.get('Augmentation parameter'))*kwargs.get('fs')):,:]
                             flipped2 = lastnoise[::-1]

                             signal = np.concatenate((flipped,signal),axis=0)
                             signal = np.concatenate((signal,flipped2),axis=0)
                             # Save tpga index to store information about what kind of augmentation applied
                             tpga = 1


                         # Reshape the signal array to the specified shape
                         signal = signal.reshape((int(kwargs.get('signaltime')) + (2*int(kwargs.get('Augmentation parameter'))))*kwargs.get('fs'),1,kwargs.get('channel_depth'))
                         # Convert the signal array to a list
                         a = signal
                         a = a.tolist()

                         # Create an array to store the spectrogram data
                         sP_3_channel = np.zeros((int((kwargs.get('fs')/2)+1),2*(int(kwargs.get('signaltime'))+(2*int(kwargs.get('Augmentation parameter'))))-1,kwargs.get('channel_depth')))

                         # If the frequency flag is set, compute the spectrogram for each channel
                         if kwargs.get('freq_flag'):
                             for vay in range(0,kwargs.get('channel_depth')):
                                 f, t, sP = scipy.signal.spectrogram(signal[:,0,vay], window=scipy.signal.windows.hann(kwargs.get('fs')*kwargs.get('window_size')), fs=kwargs.get('fs'), nperseg=kwargs.get('fs')*kwargs.get('window_size'), noverlap=kwargs.get('fs')*kwargs.get('window_size')/2, mode='magnitude')
                                 sP_3_channel[:,:,vay] = sP
                             signal = (np.moveaxis(sP_3_channel,0,1))
                             input_signal.append(np.moveaxis(sP_3_channel,0,1))
                         # If the frequency flag is not set, use input_signal as time signal
                         else:
                             input_signal.append(a)
                         counter += 1

                         # Append the epicenter location data to the epicenter list
                         EpicenterDuo = [epicenterLat, epicenterLon]
                         EpicenterDuolist = list(EpicenterDuo)
                         EpicenterDepthlist = epicenterDepth
                         EpicenterDuolist.append(EpicenterDepthlist)
                         counter += 1
                         # Append the station latitude, longitude, and altitude to the station information list
                         stat_inf = [stationLat, stationLon]
                         stat_info.append(stat_inf)
                         counter += 1
                         altitudes.append(alt)
                         counter += 1 #7
                         # Append the epicenter depth, magnitude, latitude, and longitude to their respective lists
                         epicenter_depth.append(epicenterDepth)
                         counter += 1
                         magnitude_val.append(mag)
                         counter += 1
                         lats.append(epicenterLat)
                         counter += 1
                         longs.append(epicenterLon)
                         counter += 1
                         # Append the distance between the station and the epicenter to the station to epicenter distance list
                         stat2epi.append(station_km)
                         counter += 1
                         # Append the TPGA index to the TPGA index list
                         tpga_ind.append(tpga)
                         counter += 1
                         # Append the signal, station coordinates, and epicenter distance to their respective lists
                         signal_list.append(signal)
                         counter += 1
                         stationco.append(stationcoor)
                         counter += 1
                         epicentraldist.append(epi2loc)
                         counter += 1

                     else:
                        gettingfiles.remove(b)
                        groundTruth.pop()
                        stat_id.pop()
                        skipped_file[dictname]['depth is greater than the epicenterDepth'].append(b)

                else:
                    gettingfiles.remove(b)
                    groundTruth.pop()
                    stat_id.pop()
                    skipped_file[dictname]['duration is smaller than desired'].append(b)


        # print("Gettingfiles, Signal list and filename--> ", len(gettingfiles),len(signal_list),b)

        # if len(gettingfiles) != len(signal_list):
        #     print("ERR")

    except:
        # print(f"Error: Counter: {counter}")
        lists_to_pop = ['input_signal', 'EpicenterDuolist', 'stat_info', 'altitudes', 'epicenter_depth', 'lats', 'longs', 'stat2epi', 'tpga_ind', 'signal_list', 'stationco', 'epicentraldist']

        for lst_name in lists_to_pop[0:counter]:
            lst = locals()[lst_name]
            # print(lst_name,len(lst))
            if lst and lst != []:
                lst.pop()
            # print(lst_name,len(lst))
        skipped_file['except_block'].append(b)
    
    return gettingfiles, groundTruth, input_signal,  magnitude_val, stat2epi, signal_list, epicenter_depth, lats, longs, stationco, epicentraldist, altitudes, stat_id, sP_3_channel, tpga_ind, stat_info, skipped_file

def format_data(data, column_widths):
    formatted_data = [str(item).ljust(width) for item, width in zip(data, column_widths)]
    return "\t".join(formatted_data)


def determine_logfile_columns(plotargs):
    columns = []
    metrics = []
    gt_select = plotargs.get("gt_select")
    
    for i in range(len(gt_select)):
        if gt_select[i] == 'Vs30':
            columns.append("Vs30 Error")
            metrics.append(np.round(np.mean(plotargs.get('Vs30_diff')), 2)) 
            
        if gt_select[i] == 'epiLAT':
            columns.append("Metric Error")
            metrics.append(np.round(np.mean(plotargs.get('metric_dist')), 2))
            
        if gt_select[i] == 'Depth':
            columns.append("Depth Error")
            metrics.append(np.round(np.mean(plotargs.get("column_wise_errors")[plotargs.get("gt_select").index("Depth")]), 2))
    
    columns.append("Exp Name")
    metrics.append(plotargs["exp_name"])

    return columns, metrics
    
           
    
    # return "\t\t".join(columns), "\t\t".join(map(str, metrics))



# def determine_logfile_columns(plotargs):
#     columns = []
#     metrics = []
#     gt_select = plotargs.get("gt_select")
    
#     if 'Vs30' in gt_select:
#         columns.append("Vs30 Error")
#         metrics.append(np.round(np.mean(plotargs.get('Vs30_diff')), 2)) 
    
#     if 'epiLAT' in gt_select and 'epiLON' in gt_select:
#         # columns.append("Angle Error")
#         columns.append("Metric Distance Error (km)")
#         metrics.append(np.round(np.mean(plotargs.get('metric_dist')), 2))
#     # if 'Depth' in gt_select:
#     #     columns.append("Depth Error (km)")
#     #     test_set_depth_error = plotargs.get("column_wise_errors")[plotargs.get("gt_select").index("Depth")]
#     #     metrics.append(np.round(np.mean(test_set_depth_error, 2)))
#         # metrics.append(np.round(np.mean(plotargs.get('angle_error')), 2))  
#     return "\t\t".join(columns), "\t\t".join(map(str, metrics))