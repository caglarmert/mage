import math
import matplotlib.pyplot as plt
import geopandas as gpd
import os
import numpy as np
from shapely.geometry import Point

# Function to calculate the angle between two lines drawn from a station to two epicenters
def calculate_angle_error(true_lat, true_long, test_predicted_lat, predicted_long, station_lat, station_lon):
    
    # Calculate the distances between the station and the two epicenters using the haversine formula
    R = 6371  # Earth's radius in km
    
    # Convert the latitude and longitude coordinates from degrees to radians
    dtrue_lat = math.radians(true_lat - station_lat)
    dtest_predicted_lat = math.radians(test_predicted_lat - station_lat)
    dtrue_long = math.radians(true_long - station_lon)
    dpredicted_long = math.radians(predicted_long - station_lon)
    
    # Apply the haversine formula to calculate the great-circle distances between the station and the two epicenters
    atrue = math.sin(dtrue_lat/2)**2 + math.cos(math.radians(station_lat)) * math.cos(math.radians(true_lat)) * math.sin(dtrue_long/2)**2
    apredicted = math.sin(dtest_predicted_lat/2)**2 + math.cos(math.radians(station_lat)) * math.cos(math.radians(test_predicted_lat)) * math.sin(dpredicted_long/2)**2
    
    ctrue = 2 * math.atan2(math.sqrt(atrue), math.sqrt(1-atrue))
    cpredicted = 2 * math.atan2(math.sqrt(apredicted), math.sqrt(1-apredicted))
    
    # Convert the great-circle distances to kilometers
    dtrue = R * ctrue
    dpredicted = R * cpredicted
    
    # Calculate the angles between the station and the two epicenters using the atan2 function
    true_angle = math.atan2(true_lat - station_lat, true_long - station_lon)
    predicted_angle = math.atan2(test_predicted_lat - station_lat, predicted_long - station_lon)
    
    # Calculate the angle error between the two angles and the modulo operator
    angle_error = (true_angle - predicted_angle)
    if angle_error > math.pi:
        angle_error =  angle_error - 2 * math.pi
    return math.degrees(angle_error), dtrue, dpredicted

def true_vs_predicted_plot(trn_true,trn_pred, val_true,val_pred, tst_true, tst_pred, s_path, title):
    
    plt.figure(figsize=(8,8))
    
    # Saving directory
    os.chdir(s_path)
    plt.rcParams["savefig.directory"] = os.getcwd()
    
    # Training
    plt.scatter(trn_true,trn_pred, facecolors='none', edgecolors='crimson', label='training', zorder=0)
    
    # Validation
    plt.scatter(tst_true, tst_pred, facecolors='b', edgecolors='b', label='test', zorder=15)
    
    # Test
    plt.scatter(val_true,val_pred, facecolors='lime', edgecolors='lime', label='validation', zorder=15) 
    
    # Axes boundaries
    p2 = min(min(trn_true), min(val_true), min(tst_true))
    p1 = max(max(trn_pred), max(val_pred), max(tst_pred))
    if title == "Distance":
        p2 = 0
        p1 = 120
    plt.xlim([(p2-1),(p1+1)])
    plt.ylim([(p2-1),(p1+1)])
    xpoints = ypoints = plt.xlim()
    
    # Plot
    plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False)
    
    # Labels
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.title(title, fontsize=15)

    # Save and display
    save_name  = title +".png"
    plt.legend()
    plt.savefig(save_name)
    # plt.show()
    # plt.close()
def plot_learning_curve(epochs, trn_losses, val_losses, s_path):
    """
    Parameters:
        - number of epochs
        - training loss history
        - validation loss history
        - save path for figure
        
    """
    plt.figure(figsize=(8,8))
    plt.plot(list(range(0,epochs)),trn_losses, c='b', label='training')
    plt.plot(list(range(0,epochs)),val_losses, c='lime', label='validation')
    plt.title('Learning Curve', fontsize=15)
    plt.xlabel('Number of epochs', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.legend()
    os.chdir(s_path)
    plt.rcParams["savefig.directory"] = os.getcwd()
    plt.savefig("LearningCurve.png")
    # plt.show

def plot_1d_histograms(depth_loss, metric_dist, s_path, suptitle, title1, title2):
    # Set the width of each bin in the histogram
    w = 2 
    
    # Create a figure with two subplots
    fig, [ax,ax1] = plt.subplots(1,2)
    os.chdir(s_path)
    plt.rcParams["savefig.directory"] = os.getcwd()
    plt.rcParams["figure.autolayout"] = True
    fig.suptitle(suptitle)
    counts, bins, bars = ax.hist(depth_loss, edgecolor='black', bins=np.arange(min(depth_loss), max(depth_loss)+w,w))
    
    # Set the axis labels for the first subplot
    ax.set_xlabel(title1)
    ax.set_ylabel('Test Sample (counts)')
    
    # Set the axis labels for the second subplot
    ax1.set_xlabel(title2)
    ax1.set_ylabel('Test Sample (counts)')
    ax1.hist(metric_dist, bins=np.arange(min(metric_dist), max(metric_dist)+w,w))
    
    # Set the name of the file to save the plot
    file_name = "Err_" + str(np.round(np.mean(np.abs(depth_loss)),2)) + "_vs_MetricErr_" + str(np.round(np.mean(metric_dist),2))
    plt.savefig("{0}.png".format(file_name))
    # plt.show

# def plot_2d_histograms(depth_loss, metric_dist, true_depth, r, mw, s_path):
def plot_2d_histograms(depth_loss, metric_dist, true_depth, r, s_path):   
    """
    Plots 2d error histograms of:
    
        - Depth-vs-Surfacial distance error
        - (Station to Epicenter) vs (True Epicenter to the Predicted)
        - Magnitude vs Surfacial distance error 
        
    """
    # Create figure
    # fig, [ax,ax1,ax2] = plt.subplots(1,3, figsize=(8, 3), constrained_layout=True)
    fig, [ax,ax1,ax2] = plt.subplots(1,3, figsize=(8, 3), constrained_layout=True)
    os.chdir(s_path)
    plt.rcParams["savefig.directory"] = os.getcwd()
    fig.suptitle(' Error Histograms (km)')
    
    # Depth measured vs e=metric error
    ax.set_xlabel('Depth (km)')
    ax.set_ylabel('Error (km)')
    _, _, _, img = ax.hist2d(true_depth, metric_dist, cmap='Reds')

    # Station to Epicenter Distance vs epicenter prediction error (metric error)
    ax1.set_xlabel('Station to Epicenter Distance (km)')
    ax1.hist2d(r, metric_dist, cmap='Reds')
    

    
    # # Magnitude vs metric error 
    # ax2.set_xlabel('Mw')
    # ax2.hist2d(mw, metric_dist, cmap='Blues')
    # plt.colorbar(img)
    # file_name = "TestSize" + str(len(true_depth)) + "Metric_" + str(np.round(np.mean(metric_dist),2))
    # plt.savefig("{0}.png".format(file_name))
    # plt.show()
    
    # add colorbars
    # fig.colorbar(img, ax=ax)
    # fig.colorbar(img, ax=ax1)
    fig.colorbar(img, ax=ax2)
    
    # plt.show()
    
def calculations_for_angle_histogram(attributes, plotargs, phase):
    """
    Makes the necessary calculations to get angle error of a given set.

    """
    # Station location
    if phase=="test":
        
        stat_info = [attributes["stat_info"][i] for i in attributes["tsind"]]
       
        true_lat = [item[0] for item in plotargs.get('test_data')[plotargs.get("gt_select").index("epiLAT")]]
        true_lon = [item[0] for item in plotargs.get('test_data')[plotargs.get("gt_select").index("epiLON")]]
        pr_lat = [item[1] for item in plotargs.get('test_data')[plotargs.get("gt_select").index("epiLAT")]]
        pr_lon = [item[1] for item in plotargs.get('test_data')[plotargs.get("gt_select").index("epiLON")]]

    if phase=="train":
        
        stat_info = [attributes["stat_info"][i] for i in attributes["trind"]]
        
        true_lat = [item[0] for item in plotargs.get('training_data')[plotargs.get("gt_select").index("epiLAT")]]
        true_lon = [item[0] for item in plotargs.get('training_data')[plotargs.get("gt_select").index("epiLON")]]
        pr_lat = [item[1] for item in plotargs.get('training_data')[plotargs.get("gt_select").index("epiLAT")]]
        pr_lon = [item[1] for item in plotargs.get('training_data')[plotargs.get("gt_select").index("epiLON")]]
        
    if phase=="validation":
        
        stat_info = [attributes["stat_info"][i] for i in attributes["vlind"]]
        
        true_lat = [item[0] for item in plotargs.get('validation_data')[plotargs.get("gt_select").index("epiLAT")]]
        true_lon = [item[0] for item in plotargs.get('validation_data')[plotargs.get("gt_select").index("epiLON")]]
        pr_lat = [item[1] for item in plotargs.get('validation_data')[plotargs.get("gt_select").index("epiLAT")]]
        pr_lon = [item[1] for item in plotargs.get('validation_data')[plotargs.get("gt_select").index("epiLON")]]
        
    # stat lat lonu fonkta al
    stat_lat = [tensor[0] for tensor in stat_info]
    stat_lon = [tensor[1] for tensor in stat_info]
    
    # Calculate the angle error for each data point
    angle_error=[]
    
    #burası değişecek
    for i in range(0,len(attributes["tsind"])):
        angle_err, dtrue, dpredicted = calculate_angle_error(true_lat[i],true_lon[i], pr_lat[i], pr_lon[i], stat_lat[i],stat_lon[i])
        angle_error.append(angle_err)
    return angle_error
    
    
def PlotExperiment(attributes, plotargs):
    """
    Plots diagnostic graphs of each phase and station figures.
    
    Parameters:
        
        1- Plotargs contain:
            - number of epochs
            - training loss history
            - validation loss history
            
        2- Attributes contain:
            - ground truth of train, validation, test set
            - predicted values of train, validation, test set
            - magnitude of the events
            - epicentral distances between 
        
    """
    s_path = os.path.join((plotargs.get('save_path')), 'figs') 
    if not os.path.exists(s_path):
        os.makedirs(s_path)
    
    # # True-vs-prediction graphs
    for i in range(len(plotargs.get("gt_select"))):
          true_vs_predicted_plot([item[0] for item in plotargs.get('training_data')[i]], [item[1] for item in plotargs.get('training_data')[i]],
                                [item[0] for item in plotargs.get('validation_data')[i]], [item[1] for item in plotargs.get('validation_data')[i]],
                                [item[0] for item in plotargs.get('test_data')[i]], [item[1] for item in plotargs.get('test_data')[i]],
                                s_path, title=(plotargs.get("gt_select")[i]))
    
    # # Learning Curve
    plot_learning_curve(plotargs.get('n_epochs'), 
                        plotargs.get('training_losses'), 
                        plotargs.get('validation_losses'), 
                        s_path)
    
    # # # Location Error - only for test set
    if "Depth" in plotargs.get("gt_select"):
        # 1d Error Histograms
        plot_1d_histograms(plotargs.get("column_wise_errors")[plotargs.get("gt_select").index("Depth")],
                        plotargs.get('metric_dist'), 
                        s_path,
                        suptitle='Location Error (km)',
                        title1='Depth Error (km)',
                        title2='Spherical Distance Error (km)')
   
  
        ## 2d Error Histograms
        ## With magnitudes
        ## plot_2d_histograms(plotargs.get("gt_select").index("Depth"),
        ##                 plotargs.get('metric_dist'),
        ##                 [item[0] for item in plotargs.get('test_data')[plotargs.get("gt_select").index("Depth")]],    #Depth, groundTruth values 
        ##                 [attributes['Epicentral_distance(epi2loc)'][idx] for idx in attributes["tsind"]], 
        ##                 [attributes['Magnitudes'][idx] for idx in attributes["tsind"]], 
        ##                 s_path)
        
        # Without magnitudes
        plot_2d_histograms(plotargs.get("gt_select").index("Depth"),
                        plotargs.get('metric_dist'),
                        [item[0] for item in plotargs.get('test_data')[plotargs.get("gt_select").index("Depth")]],    #Depth, groundTruth values 
                        [attributes['station_km'][idx] for idx in attributes["tsind"]], 
                        s_path)
    
    if "Vs30" in plotargs.get("gt_select"):
        # Plot Angle Error-vs-Metric Error histogram
        plot_1d_histograms(plotargs.get('Vs30_diff'),  [],
                            s_path,
                            suptitle='Vs30_diff Histogram',
                            title1='Vs30_diff (m/s)',
                            title2='blank')
        
    if "epiLAT" in plotargs.get("gt_select") and "epiLON" in plotargs.get("gt_select"):
        angle_error = calculations_for_angle_histogram(attributes, plotargs, phase="test")
        # Plot Angle Error-vs-Metric Error histogram
        plot_1d_histograms(angle_error,
                            plotargs.get('metric_dist'), 
                            s_path,
                            suptitle='Angle Error-Metric dist Histogram',
                            title1='Angle Error (deg)',
                            title2='Spherical Distance Error (km)')
        # Station plots
        # StationFigure(attributes, plotargs, add_Margin=True)
        StationFigure_test_stations_training_data(attributes, plotargs, add_Margin=True)
    
def StationFigure(attributes, plotargs, add_Margin=True):
    
    # Select the test stations
    test_stations_names = [attributes["Station ID"][i] for i in attributes["tsind"]]
    test_stations_unique_names = list(set(test_stations_names))
    
    # Select the info of test stations
    unique_stations_indexes_in_test_indices = [test_stations_names.index(station) for station in test_stations_unique_names]
    stat_info_unique_test = [attributes["stat_info"][i] for i in [attributes["tsind"][i] for i in unique_stations_indexes_in_test_indices]]
    test_station_latitude = [item[0] for item in stat_info_unique_test]
    test_station_longitude = [item[1] for item in stat_info_unique_test]
    
    # variable for verification
    verify_test_len = 0

    # define colors for markers
    true_color = "green" # "#1f77b4" 
    pred_color = "blue"  #"#ff7f0e"
    station_color = "red" #"#2ca02c"
    
    # define colors for lines
    line_color = "gray" #"#555555"
    # Loop over each station
    for i, station_name in enumerate(test_stations_unique_names):
        
        # get indices of earthquakes recorded by this station, note that indices are for test set.
        sig_indices_of_this_station = [j for j, s_id in enumerate(test_stations_names) if s_id == station_name]
        
        # print("num of recordings", len(sig_indices_of_this_station))
        verify_test_len += len(sig_indices_of_this_station)
        
        # Plot parameters
        os.chdir(os.path.join((plotargs.get('save_path')), 'figs'))
        plt.rcParams["savefig.directory"] = os.getcwd()
        fig, ax = plt.subplots(figsize=(9, 6))

        # get station coordinates
        station_lat = test_station_latitude[i]
        station_lon = test_station_longitude[i]

         # Create a Point object from your station coordinates
        station_point = Point(station_lon, station_lat)
        
        # Determine which country each point is located in
        countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        
        # station_country = countries[countries.geometry.contains(station_point)]
        station_country=countries[countries['name'] == 'Turkey']
        station_country.plot(color="lightgrey", edgecolor="white", ax=ax)
        
        # Set margin for the map
        marginx = 5
        marginy = 3
        lon_min = station_lon - marginx
        lon_max = station_lon + marginx
        lat_min = station_lat - marginy
        lat_max = station_lat + marginy
        
        lon_min = 26
        lon_max = 45
        lat_min = 36
        lat_max = 42
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max) 
        
        # Set axis labels and plot title
        ax.set_xlabel('latitude (deg)')
        ax.set_ylabel('longitude (deg)') #, fontsize=20
        ax.set_title(f'Station {station_name} Earthquake Recordings')
        
        # add station location label to legend
        ax.scatter(station_lon,station_lat, marker='p',color=station_color)

        angle_error = []
        legend_exists = False  # flag to check if a legend already exists in the figure
        for idx in sig_indices_of_this_station:
            # plot each event, ground truths
            true_lat = [item[0] for item in plotargs.get('test_data')[plotargs.get("gt_select").index("epiLAT")]][idx]
            true_lon = [item[0] for item in plotargs.get('test_data')[plotargs.get("gt_select").index("epiLON")]][idx]
            predicted_lat = [item[1] for item in plotargs.get('test_data')[plotargs.get("gt_select").index("epiLAT")]][idx]
            predicted_lon = [item[1] for item in plotargs.get('test_data')[plotargs.get("gt_select").index("epiLON")]][idx]
            
            # plot points
            plt.plot(true_lon, true_lat, marker='o', markersize=7, markerfacecolor="None", markeredgecolor=true_color, linestyle="None")
            plt.plot(predicted_lon, predicted_lat, marker='o', markersize=7, markerfacecolor="None", markeredgecolor=pred_color, linestyle="None")
            
            # plot lines
            plt.plot([predicted_lon, true_lon], [predicted_lat, true_lat], linestyle='-', color=line_color,alpha=0.8, linewidth=0.5)
            
            # calculate the angle difference between station to predicted and true hypocenters
            ang = calculate_angle_error(true_lat, true_lon, predicted_lat, predicted_lon, station_lat, station_lon)
            angle_error.append(np.round(ang,2))
            # check if a legend exists in the figure
            if not legend_exists:
                # if no legend exists, add a legend with labels for true, predicted, and station
                plt.scatter([], [], marker='o', facecolor='none', edgecolors=true_color, label='True Epicenter')
                plt.scatter([], [], marker='o', facecolor='none', edgecolors=pred_color, label='Predicted Epicenter')
                plt.scatter([], [], marker='p', facecolor=station_color, edgecolors=station_color, label='Station Location')
                plt.legend()
                legend_exists = True
                
        # average the angle error
        file_name = "Station_" + str(station_name) + "_AveAngError_"
        mean_angle_error = np.round(np.mean(np.abs(angle_error)),1)
        file_name += str(int(mean_angle_error))
        fig.savefig("{0}.png".format(file_name))

        # plt.show()
        plt.close('all')
    if not verify_test_len==len(attributes['tsind']):
        print(">>>>>>>>>>>>>>>>>> test length is not equal to total sum of recordings")
        print(verify_test_len,len(attributes['tsind']) )
        
def StationFigure_test_stations_training_data(attributes, plotargs, add_Margin=True):
    # Select the test set stations
    test_stations_names = [attributes["Station ID"][i] for i in attributes["tsind"]]
    test_stations_unique_names = list(set(test_stations_names))
    
    # Select the training set stations
    training_stations_names = [attributes["Station ID"][i] for i in attributes["trind"]]

    # Select the validation set stations
    validation_stations_names = [attributes["Station ID"][i] for i in attributes["vlind"]]
    
    # Select the info of test stations
    unique_stations_indexes_in_test_indices = [test_stations_names.index(station) for station in test_stations_unique_names]
    stat_info_unique_test = [attributes["stat_info"][i] for i in [attributes["tsind"][i] for i in unique_stations_indexes_in_test_indices]]
    test_station_latitude = [item[0] for item in stat_info_unique_test]
    test_station_longitude = [item[1] for item in stat_info_unique_test]
   
    # define colors for markers
    true_color = "green" # "#1f77b4" 
    pred_color = "magenta"  #"#ff7f0e"
    station_color = "red" #"#2ca02c"
    
    # define colors for lines
    line_color = "gray" #"#555555"
    
    # Loop over each station
    for i, station_name in enumerate(test_stations_unique_names):
        # get indices of earthquakes recorded by this station, note that indices is for train set.
        sig_indices_of_this_station_in_training_set = [j for j, s_id in enumerate(training_stations_names) if s_id == str(station_name) or s_id == station_name]
        
        # get indices of earthquakes recorded by this station, note that indices are for test set.
        sig_indices_of_this_station_in_test_set = [j for j, s_id in enumerate(test_stations_names) if s_id == station_name]
        # print("num of recordings", len(sig_indices_of_this_station_in_training_data))
        
        # get indices of earthquakes recorded by this station, note that indices are for validation set.
        sig_indices_of_this_station_in_validation_set = [j for j, s_id in enumerate(validation_stations_names) if s_id == station_name]
        
        # Plot parameters
        os.chdir(os.path.join((plotargs.get('save_path')), 'figs'))
        plt.rcParams["savefig.directory"] = os.getcwd()
        fig, ax = plt.subplots(figsize=(9, 6))  
        
        # get station coordinates
        station_lat = test_station_latitude[i]
        station_lon = test_station_longitude[i]

        # Create a Point object from your station coordinates
        station_point = Point(station_lon, station_lat)
        
        # Determine which country each point is located in
        countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        # station_country = countries[countries.geometry.contains(station_point)]
        station_country=countries[countries['name'] == 'Turkey']
        
        station_country.plot(color="lightgrey", edgecolor="white", ax=ax)
        # Set margin for the map
        marginx = 5
        marginy = 3
        lon_min = station_lon - marginx
        lon_max = station_lon + marginx
        lat_min = station_lat - marginy
        lat_max = station_lat + marginy
        
        lon_min = 26
        lon_max = 45
        lat_min = 36
        lat_max = 42
        
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max) 
        
        # Set axis labels and plot title
        ax.set_xlabel('latitude (deg)')
        ax.set_ylabel('longitude (deg)') #, fontsize=20
        ax.set_title(f'Station {station_name} - TRAINING SET')
        # add station location label to legend
        ax.scatter(station_lon,station_lat, marker='p',color=station_color)
        
        angle_error = []
        legend_exists = False  # flag to check if a legend already exists in the figure
        
        ### TEST
        with open("Station" + str(station_name) +".txt", "w") as f:
            
            f.write("TEST \n")
            for index in sig_indices_of_this_station_in_test_set:
                # plot each event, ground truths
                true_lat = [item[0] for item in plotargs.get('test_data')[plotargs.get("gt_select").index("epiLAT")]][index]
                true_lon = [item[0] for item in plotargs.get('test_data')[plotargs.get("gt_select").index("epiLON")]][index]
                predicted_lat = [item[1] for item in plotargs.get('test_data')[plotargs.get("gt_select").index("epiLAT")]][index]
                predicted_lon = [item[1] for item in plotargs.get('test_data')[plotargs.get("gt_select").index("epiLON")]][index]
                
                # plot points 
                plt.plot(true_lon, true_lat, marker='+', markersize=7, markerfacecolor="None", markeredgecolor='navy', linestyle="None")
                plt.plot(predicted_lon, predicted_lat, marker='o', markersize=7, markerfacecolor="None", markeredgecolor='navy', linestyle="None")
                
                # plot lines
                plt.plot([predicted_lon, true_lon], [predicted_lat, true_lat], linestyle='-', color='navy',alpha=0.8, linewidth=0.5)
                
                # calculate the angle difference between station to predicted and true hypocenters
                ang = calculate_angle_error(true_lat, true_lon, predicted_lat, predicted_lon, station_lat, station_lon)
                angle_error.append(np.round(ang,2))
                
                f.write(str(predicted_lat) + " ")
                f.write(str(predicted_lon) + " ")
                f.write(str(true_lat) + " ")
                f.write(str(true_lon) + "\n") 
            
        ### VALIDATION
            f.write("VALIDATION \n")
            for index in sig_indices_of_this_station_in_validation_set:
                # plot each event, ground truths
                true_lat = [item[0] for item in plotargs.get('validation_data')[plotargs.get("gt_select").index("epiLAT")]][index]
                true_lon = [item[0] for item in plotargs.get('validation_data')[plotargs.get("gt_select").index("epiLON")]][index]
                predicted_lat = [item[1] for item in plotargs.get('validation_data')[plotargs.get("gt_select").index("epiLAT")]][index]
                predicted_lon = [item[1] for item in plotargs.get('validation_data')[plotargs.get("gt_select").index("epiLON")]][index]
                
                # plot points 
                plt.plot(true_lon, true_lat, marker='+', markersize=7, markerfacecolor="None", markeredgecolor='darkorange', linestyle="None")
                plt.plot(predicted_lon, predicted_lat, marker='o', markersize=7, markerfacecolor="None", markeredgecolor='darkorange', linestyle="None")
                
                # plot lines
                plt.plot([predicted_lon, true_lon], [predicted_lat, true_lat], linestyle='-', color='darkorange',alpha=0.8, linewidth=0.5)
                
                f.write(str(predicted_lat) + " ")
                f.write(str(predicted_lon) + " ")
                f.write(str(true_lat) + " ")
                f.write(str(true_lon) + "\n") 
            ### TRAINING
            f.write("TRAINING \n")
            for index in sig_indices_of_this_station_in_training_set:
                # plot each event, ground truths
                true_lat = [item[0] for item in plotargs.get('training_data')[plotargs.get("gt_select").index("epiLAT")]][index]
                true_lon = [item[0] for item in plotargs.get('training_data')[plotargs.get("gt_select").index("epiLON")]][index]
                predicted_lat = [item[1] for item in plotargs.get('training_data')[plotargs.get("gt_select").index("epiLAT")]][index]
                predicted_lon = [item[1] for item in plotargs.get('training_data')[plotargs.get("gt_select").index("epiLON")]][index]
                # plot points 
                plt.plot(true_lon, true_lat, marker='+', markersize=7, markerfacecolor="None", markeredgecolor='green', linestyle="None")
                plt.plot(predicted_lon, predicted_lat, marker='o', markersize=7, markerfacecolor="None", markeredgecolor='green', linestyle="None")
                # plot lines
                plt.plot([predicted_lon, true_lon], [predicted_lat, true_lat], linestyle='-', color='green',alpha=0.8, linewidth=0.5)
                # if no legend exists, add a legend with labels for true, predicted, and station
                if not legend_exists:
                    plt.scatter([], [], marker='+', facecolor='green', edgecolors='green', label='True - Training')
                    plt.scatter([], [], marker='o', facecolor='none', edgecolors='green', label='Predicted - Training')
                    
                    plt.scatter([], [], marker='+', facecolor='darkorange', edgecolors='darkorange', label='True - Validation')
                    plt.scatter([], [], marker='o', facecolor='none', edgecolors='darkorange', label='Predicted - Validation')
                    
                    plt.scatter([], [], marker='+', facecolor='navy', edgecolors='navy', label='True - Test')
                    plt.scatter([], [], marker='o', facecolor='none', edgecolors='navy', label='Predicted - Test')
                    
                    plt.scatter([], [], marker='p', facecolor=station_color, edgecolors=station_color, label='Station Location')
                    plt.legend()
                    legend_exists = True
                    
                f.write(str(predicted_lat) + " ")
                f.write(str(predicted_lon) + " ")
                f.write(str(true_lat) + " ")
                f.write(str(true_lon) + "\n") 

        # average the angle error
        file_name = "Station_" + str(station_name) + "_AveAngError_"
        mean_angle_error = np.round(np.mean(np.abs(angle_error)),1)
        file_name += str(int(mean_angle_error))
        fig.savefig("{0}.png".format(file_name))

        # plt.show()
        plt.close('all')
