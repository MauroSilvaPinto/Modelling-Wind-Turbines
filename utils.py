#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 23:26:08 2023

@author: mauropinto
"""
import numpy as np
import matplotlib.pyplot as plt

class Utils:
   
    def get_year_indexes(datetimes, year_start, year_end):
        """
        Outputs a list of the index samples
        corresponding to the given year interval.
        
        Parameters
        ----------
        datetimes : 1D array of datetimes
            The datetimes vector.
        
        year_start : int
            The starting year from the desired interval.
            
        year_ending : int
            The ending year from the desired interval.
            
        Returns
        -------
        indexes : list
            List of indexes with the samples present in that time interval
            
        """
        indexes=[]
        for i in range(len(datetimes)):
            if datetimes[i].year>=year_start and datetimes[i].year<=year_end:
                indexes.append(i)
        return indexes
    
    
    def moving_average_filter(data, window_size):
        """
        Performs a moving average filter
        (smooths the data) in a given 1-d vector
        with a given window size
        
        Parameters
        ----------
        features : 2D array
            X input of features.
            
        Returns
        -------
        features : 2D array
            Features with redundant features eliminated
            
        redundant_features_index : list
            List of the redundant features indexes
        """
        window = np.ones(int(window_size)) / float(window_size)
        return np.convolve(data, window, 'same')
    
    
    def compute_nmae(y_true,y_predict):
        """
        Computes the normalized mean absolute error.
        The normalization factor is the feature range
        
        Parameters
        ----------
        y_true : 1D array
            True values
            
        y_predict : 1D array
             Predicted values
            
        Returns
        -------
        name value : float
            The value of the normalized mean absolute error
        """
        denominator=np.max(y_true)-np.min(y_true)
        
        numerator=np.sum(abs(y_true-y_predict))
        return ((1/len(y_true))*(numerator/denominator))
        
    
        
    def compute_torque_converter(active_power,rotor_speed):
        """
        Computes a torque converter ratio using active power and rotor speed.
        Due to residual values of rotor speed, velocities below 10 rpm were considered
        to be 0 and, therefore, torque converter=0 in these cases.
        T
        
        Parameters
        ----------
        active_power : 1D array
            Active power values over time
            
        rotor_speed : 1D array
             Rotor speed values over time
            
        Returns
        -------
        torque_converter : 1D array
            The torque converter array
        """
        torque_converter=[]
        for i in range(0,len(rotor_speed)):
            if rotor_speed[i]<10:
                torque_converter.append(0)
            else:
                torque_converter.append(active_power[i]/rotor_speed[i])
        return np.array(torque_converter)
    
    
    
    def compute_tip_speed_ratio(rotor_speed,wind_speed,blade_length):
        """
        Computes the tip speed ratio using rotor speed and wind speed.
        Due to cut-off and cut-in speeds, wind speed ratio was considered to be 0
        when wind_speed<3 m/s or wind_speed>25 m/s
        
        
        Parameters
        ----------
        rotor_speed : 1D array
            Rotor speed values over time
            
        wind_speed : 1D array
             Wind speed values over time
        
        blade_length: int
            Turbine blade lenght
            
        Returns
        -------
        tip_speed_ratio : 1D array
            The tip speed ratio array
        """
        tip_speed_ratio=[]
        for i in range(0,len(wind_speed)):
            if wind_speed[i]<3 or wind_speed[i]>25:
                tip_speed_ratio.append(0)
            else:
                tip_speed_ratio.append((2*np.pi*rotor_speed[i]*blade_length/60)/wind_speed[i])
        return np.array(tip_speed_ratio)
    
    
    def remove_redundant_features(features):
        """
        Removes redundant features having correlation coef>0.95
        
        Parameters
        ----------
        features : 2D array
            X input of features.
            
        Returns
        -------
        features : 2D array
            Features with redundant features eliminated
            
        redundant_features_index : list
            List of the redundant features indexes
        """
        redundant_features_index=[]
        
        #finding features with corr>0.95
        for i in range(0,features.shape[1]):
            for j in range(i,features.shape[1]):
                if i!=j and abs(np.corrcoef(features[:,i],features[:,j])[0][1])>0.95:
                    redundant_features_index.append(j)
        
        # deleting these features          
        features=np.delete(features,redundant_features_index,axis=1)
        return [redundant_features_index,features]



    def non_overlapping_windows_average_std(ts, dt, window_size):
        """
        Calculates the average, and standard deviation value
        of non-overlapping windows of a time-series vector
        based on a given window size.
        
        Parameters
        ----------
        ts : array
            Time series 1-d vector.
            
        dt : array of datetimes

        window_size : int
            Size of the window.
        
        
        Returns
        -------
        averages : list
            Mean value for each batch
            
        deviations : list
            Std value for each batch
        
        dts : list
            beginning datetime value of each batch
        """
        n = len(ts)
        averages = []
        deviations=[]
        dts=[]
    
        for i in range(0, n, window_size):
            window_sum = sum(ts[i:i+window_size])
            deviation=np.std(ts[i:i+window_size])
            window_average = window_sum / window_size
            averages.append(window_average)
            dts.append(dt[i])
            deviations.append(deviation)
            
        
        return averages, deviations, dts
    
    
    def get_batches_windows(ts, window_size):
        """
        Returns the information in batches of a time-series vector
        based on a given window size
        
        Parameters
        ----------
        ts : array
            Time series 1-d vector.
        window_size : int
            Size of the window.
        
        
        Returns
        -------
        batches : list
            Each batch of information is a different list element
            
        """
    
        n = len(ts)
        batches=[]
    
        for i in range(0, n, window_size):
            window = ts[i:i+window_size]
            batches.append(window)
        
        return batches
    
    

    
    
    
                    
        
        
        
    
            
        