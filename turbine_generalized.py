"""
@author: mauro pinto

The code for generalized turbine modelling.
This code delivers a model that is based on all turbines and is tested on all turbines as well.

"""
from utils import Utils

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

import os
import datetime


# going to data directory
os.chdir("Technical test")
os.chdir("Data Turbines - enriched")
turbine_names=["R80711","R80721","R80736","R80790"]

turbine_filenames=[]
for i in range(0,len(turbine_names)):
    turbine_filenames.append(turbine_names[i]+".csv")


################################### Loading data ##################################

print("Loading data...")

# loading R80711 turbine features names
data_names = np.genfromtxt(turbine_filenames[0], dtype=str,  delimiter=",")
data_names=data_names[0,3:]

data_datetimes=[]
for i in range(0,len(turbine_names)):
    turbine_dt_int=np.genfromtxt(turbine_filenames[i], dtype=float, skip_header=1, delimiter=",", usecols=1)
    data_datetimes.append(np.array([datetime.datetime.fromtimestamp(ts) for ts in turbine_dt_int]))
    del turbine_dt_int
    
turbines_data=[]
for i in range(0,len(turbine_names)):
    turbines_data.append(np.genfromtxt(turbine_filenames[i], dtype=float, skip_header=1, delimiter=",", usecols=range(3,24)))

################################### Choose Inputs & Outputs for regression problem ########

print("Selecting data for input and output ...")

y_index=1

# all variables except:
features_index=[0,2,3,4,5,6,7,8,9,10,12,14,15,16,17,18,19,20]

X=[]
Y=[]
for i in range (0, len(turbine_names)):
   X.append(turbines_data[i][:,features_index]) 
   #Y.append(Utils.compute_torque_converter(turbines_data[i][:,1],turbines_data[i][:,11]))
   #Y.append(Utils.compute_tip_speed_ratio(turbines_data[i][:,11],turbines_data[i][:,7],41))
   Y.append(turbines_data[i][:,y_index])

feature_names=data_names[features_index]

################# split data into training, validation, testing ###################################

# selecting training and validation data from all turbines
# only from 2012-2014
training_indexes=[]
validation_indexes=[]
testing_indexes=[]
print("Partitioning the data ...")
for i in range (0, len(turbine_names)):
    training_indexes.append(Utils.get_year_indexes(data_datetimes[i],2012,2013))
    validation_indexes.append(Utils.get_year_indexes(data_datetimes[i],2014,2014))
    # only from 2015-onwards
    testing_indexes.append(Utils.get_year_indexes(data_datetimes[i],2015,data_datetimes[i][-1].year))
        
# Training data
x_train=[]
y_train=[]
datetimes_train=[]
for i in range (0, len(turbine_names)):
    x_train.append(X[i][training_indexes[i],:])
    y_train.append(Y[i][training_indexes[i]])
    datetimes_train.append(data_datetimes[i][training_indexes[i]])
        
# Validation data
x_validation=[]
y_validation=[]
datetimes_validation=[]
for i in range (0, len(turbine_names)):
    x_validation.append(X[i][validation_indexes[i],:])
    y_validation.append(Y[i][validation_indexes[i]])
    datetimes_validation.append(data_datetimes[i][validation_indexes[i]])

# Testing data
x_test=[]
y_test=[]
datetimes_test=[]
for i in range (0, len(turbine_names)):
    x_test.append(X[i][testing_indexes[i],:])
    y_test.append(Y[i][testing_indexes[i]])
    datetimes_test.append(data_datetimes[i][testing_indexes[i]])

######## merging all training and validation data ############################

x_train_merged=np.concatenate(x_train, axis=0)
x_validation_merged=np.concatenate(x_validation, axis=0)
x_test_merged=np.concatenate(x_test, axis=0)

y_train_merged=np.concatenate(y_train, axis=0)
y_validation_merged=np.concatenate(y_validation, axis=0)
y_test_merged=np.concatenate(y_test, axis=0)
             
################################### Removing Redundancy  ###################################

print("Model training ...")

corr_matrix = np.triu(np.corrcoef(x_train_merged, rowvar=False))
#plt.figure()
#sns.heatmap(np.abs(corr_matrix),xticklabels=turbine_names[features_index],
#           yticklabels=turbine_names[features_index], annot=True, fmt=".2f", cmap="cividis")

# removing redundancy
[redundant_features_index,x_train_merged]=Utils.remove_redundant_features(x_train_merged)
feature_names=np.delete(feature_names,redundant_features_index)

x_validation_merged=np.delete(x_validation_merged,redundant_features_index,axis=1)
x_test_merged=np.delete(x_test_merged,redundant_features_index,axis=1)

for i in range(0,len(x_train)):
    x_train[i]=np.delete(x_train[i],redundant_features_index,axis=1)
    x_validation[i]=np.delete(x_validation[i],redundant_features_index,axis=1)
    x_test[i]=np.delete(x_test[i],redundant_features_index,axis=1)
    

################################### Feature relevanve sorting  ###################################


# calculating relevant features by ascending order
corr_values=[]
for i in range(0, np.shape(x_train_merged)[1]):
    corr_values.append(np.corrcoef(x_train_merged[:,i], y_train_merged)[0, 1])

features_sorted_indexes=np.argsort(np.abs(corr_values))[::-1]

########### normalizing all features #####################

scaler = StandardScaler()
scaler.fit(x_train_merged)

x_train_merged = scaler.transform(x_train_merged)
x_validation_merged=scaler.transform(x_validation_merged)
x_test_merged=scaler.transform(x_test_merged)

for i in range(0,len(x_train)):
    x_train[i]=scaler.transform(x_train[i])
    x_validation[i]=scaler.transform(x_validation[i])
    x_test[i]=scaler.transform(x_test[i])


########### training model and validation ##########################

########## Grid Search for parameter optimization
k_features=np.linspace(1,np.shape(x_train_merged)[1],np.shape(x_train_merged)[1])
rmse_values=[]
for i in range(0,len(k_features)):
    model=DecisionTreeRegressor()
    model.fit(x_train_merged[:,features_sorted_indexes[0:int(k_features[i])]],y_train_merged)
    
    y_predicted=model.predict(x_validation_merged[:,features_sorted_indexes[0:int(k_features[i])]])
    rmse_values.append(np.sqrt(mean_squared_error(y_validation_merged,y_predicted)))

    
plt.figure("Feature Selection - Grid Search Results")
plt.plot(k_features,rmse_values)
plt.title('Validation Data \n NMAE value for k Features')  
plt.xlabel('k features')
plt.ylabel('RMSE')
plt.grid()

total_rmse_variation = np.max(rmse_values)-np.min(rmse_values)
threshold_rmse = total_rmse_variation * 0.90
rmse_variations=np.max(rmse_values)-rmse_values
idx = np.where(rmse_variations >= threshold_rmse)[0][0]
k_optimal=int(k_features[idx])

feature_names=feature_names[features_sorted_indexes[0:k_optimal]]


###### Final model training ################
model.fit(x_train_merged[:,features_sorted_indexes[0:k_optimal]],y_train_merged)
y_predicted=model.predict(x_train_merged[:,features_sorted_indexes[0:k_optimal]])

nmae = Utils.compute_nmae(y_train_merged, y_predicted)
mean_difference=np.mean((y_predicted-y_train_merged)/(np.max(y_train_merged)-np.min(y_train_merged)))

print("#################### TRAINING ANALYSIS ######################")
print("NMAE: "+str(np.round(nmae,2)))
print("RMSE: "+str(np.round(np.sqrt(mean_squared_error(y_train_merged,y_predicted)),2)))
print("Mean Relative Predicted/Real differences: "+str(np.round(mean_difference,2)))
print("Mean Absolute predicted/real differences: "+str(np.round(np.mean(y_predicted-y_train_merged),2)))
print("R2: "+str(np.round(r2_score(y_train_merged, y_predicted),2)))
print("Individual Turbine Results:")

for i in range(0,len(x_train)):
    y_predicted=model.predict(x_train[i][:,features_sorted_indexes[0:k_optimal]])
    rmse = np.sqrt(mean_squared_error(y_train[i], y_predicted))
    mean_difference=np.mean((y_predicted-y_train[i])/(np.max(y_train[i])-np.min(y_train[i])))
    nmae = Utils.compute_nmae(y_train[i], y_predicted)
    
    print("   Turbine " + turbine_names[i]+":")
    print("NMAE: "+str(np.round(nmae,2)))
    print("RMSE: "+str(np.round(np.sqrt(mean_squared_error(y_train[i],y_predicted)),2)))
    print("Mean Relative Predicted/Real differences: "+str(np.round(mean_difference,2)))
    print("Mean Absolute predicted/real differences: "+str(np.round(np.mean(y_predicted-y_train[i]),2)))
    print("R2: "+str(np.round(r2_score(y_train[i], y_predicted),2)))


# Create a figure with a subplot of two plots vertically stacked
fig, axs = plt.subplots(len(x_train), 1, figsize=(8, 24))
fig.suptitle("Training regression results")
for i, ax in enumerate(axs):
    
    y_predicted=model.predict(x_train[i][:,features_sorted_indexes[0:k_optimal]])
    rmse = np.sqrt(mean_squared_error(y_train[i], y_predicted))
    mean_difference=np.mean((y_predicted-y_train[i])/(np.max(y_train[i])-np.min(y_train[i])))
    nmae = Utils.compute_nmae(y_train[i], y_predicted)
    
    
    ax.set_ylabel("Turbine " + turbine_names[i] + "\n RMSE:"+str(np.round(rmse,2))+
                      "\n Mean diff: "+ str(np.round(np.mean(y_predicted-y_train[i]),2))+
                      "\n R2: " + str(np.round(r2_score(y_train[i], y_predicted),2)))

    ax.plot(datetimes_train[i],y_train[i], alpha=0.50)
    ax.plot(datetimes_train[i],y_predicted, alpha=0.50)
    ax.plot(datetimes_train[i],Utils.moving_average_filter(y_predicted-y_train[i],6*24*7))
    ax.grid()
    ax.set_xlabel("time")   
        

###### Final model performance assessment on validation ################
 
y_predicted=model.predict(x_validation_merged[:,features_sorted_indexes[0:k_optimal]])

nmae = Utils.compute_nmae(y_validation_merged, y_predicted)
mean_difference=np.mean((y_predicted-y_validation_merged)/(np.max(y_validation_merged)-np.min(y_validation_merged)))

print("#################### VALIDATION ANALYSIS ######################")
print("NMAE: "+str(np.round(nmae,2)))
print("RMSE: "+str(np.round(np.sqrt(mean_squared_error(y_validation_merged,y_predicted)),2)))
print("Mean Relative Predicted/Real differences: "+str(np.round(mean_difference,2)))
print("Mean Absolute predicted/real differences: "+str(np.round(np.mean(y_predicted-y_validation_merged),2)))
print("R2: "+str(np.round(r2_score(y_validation_merged, y_predicted),2)))
print("Individual Turbine Results:")

for i in range(0,len(x_train)):
    y_predicted=model.predict(x_validation[i][:,features_sorted_indexes[0:k_optimal]])
    rmse = np.sqrt(mean_squared_error(y_validation[i], y_predicted))
    mean_difference=np.mean((y_predicted-y_validation[i])/(np.max(y_validation[i])-np.min(y_validation[i])))
    nmae = Utils.compute_nmae(y_validation[i], y_predicted)
    
    print("   Turbine " + turbine_names[i]+":")
    print("NMAE: "+str(np.round(nmae,2)))
    print("RMSE: "+str(np.round(np.sqrt(mean_squared_error(y_validation[i],y_predicted)),2)))
    print("Mean Relative Predicted/Real differences: "+str(np.round(mean_difference,2)))
    print("Mean Absolute predicted/real differences: "+str(np.round(np.mean(y_predicted-y_validation[i]),2)))
    print("R2: "+str(np.round(r2_score(y_validation[i], y_predicted),2)))


# Create a figure with a subplot of two plots vertically stacked
fig, axs = plt.subplots(len(x_train), 1, figsize=(8, 24))
fig.suptitle("Validation regression results")
for i, ax in enumerate(axs):
    
    y_predicted=model.predict(x_validation[i][:,features_sorted_indexes[0:k_optimal]])
    rmse = np.sqrt(mean_squared_error(y_validation[i], y_predicted))
    mean_difference=np.mean((y_predicted-y_validation[i])/(np.max(y_validation[i])-np.min(y_validation[i])))
    nmae = Utils.compute_nmae(y_validation[i], y_predicted)
    
    
    ax.set_ylabel("Turbine " + turbine_names[i] + "\n RMSE:"+str(np.round(rmse,2))+
                      "\n Mean diff: "+ str(np.round(np.mean(y_predicted-y_validation[i]),2))+
                      "\n R2: " + str(np.round(r2_score(y_validation[i], y_predicted),2)))

    ax.plot(datetimes_validation[i],y_validation[i], alpha=0.50)
    ax.plot(datetimes_validation[i],y_predicted, alpha=0.50)
    ax.plot(datetimes_validation[i],Utils.moving_average_filter(y_predicted-y_validation[i],6*24*7))
    ax.grid()
    ax.set_xlabel("time")   
         


########### Model Testing ##############################

print("Model testing ...")

y_predicted=model.predict(x_test_merged[:,features_sorted_indexes[0:k_optimal]])

nmae = Utils.compute_nmae(y_test_merged, y_predicted)
mean_difference=np.mean((y_predicted-y_test_merged)/(np.max(y_test_merged)-np.min(y_test_merged)))

print("#################### TESTING RESULTS ######################")
print("NMAE: "+str(np.round(nmae,2)))
print("RMSE: "+str(np.round(np.sqrt(mean_squared_error(y_test_merged,y_predicted)),2)))
print("Mean Relative Predicted/Real differences: "+str(np.round(mean_difference,2)))
print("Mean Absolute predicted/real differences: "+str(np.round(np.mean(y_predicted-y_test_merged),2)))
print("R2: "+str(np.round(r2_score(y_test_merged, y_predicted),2)))
print("Individual Turbine Results:")

for i in range(0,len(x_train)):
    y_predicted=model.predict(x_test[i][:,features_sorted_indexes[0:k_optimal]])
    rmse = np.sqrt(mean_squared_error(y_test[i], y_predicted))
    mean_difference=np.mean((y_predicted-y_test[i])/(np.max(y_test[i])-np.min(y_test[i])))
    nmae = Utils.compute_nmae(y_test[i], y_predicted)
    
    print("   Turbine " + turbine_names[i]+":")
    print("NMAE: "+str(np.round(nmae,2)))
    print("RMSE: "+str(np.round(np.sqrt(mean_squared_error(y_test[i],y_predicted)),2)))
    print("Mean Relative Predicted/Real differences: "+str(np.round(mean_difference,2)))
    print("Mean Absolute predicted/real differences: "+str(np.round(np.mean(y_predicted-y_test[i]),2)))
    print("R2: "+str(np.round(r2_score(y_test[i], y_predicted),2)))


# Create a figure with a subplot of two plots vertically stacked
fig, axs = plt.subplots(len(x_train), 1, figsize=(8, 24))
fig.suptitle("Testing regression results")
for i, ax in enumerate(axs):
    
    y_predicted=model.predict(x_test[i][:,features_sorted_indexes[0:k_optimal]])
    rmse = np.sqrt(mean_squared_error(y_test[i], y_predicted))
    mean_difference=np.mean((y_predicted-y_test[i])/(np.max(y_test[i])-np.min(y_test[i])))
    nmae = Utils.compute_nmae(y_test[i], y_predicted)
    
    
    ax.set_ylabel("Turbine " + turbine_names[i] + "\n RMSE:"+str(np.round(rmse,2))+
                      "\n Mean diff: "+ str(np.round(np.mean(y_predicted-y_test[i]),2))+
                      "\n R2: " + str(np.round(r2_score(y_test[i], y_predicted),2)))

    ax.plot(datetimes_test[i],y_test[i], alpha=0.50)
    ax.plot(datetimes_test[i],y_predicted, alpha=0.50)
    ax.plot(datetimes_test[i],Utils.moving_average_filter(y_predicted-y_test[i],6*24*7))
    ax.grid()
    ax.set_xlabel("time")   
        

print("#################### INTERPRETING MODELATION RESULTS ######################")

for i in range(0,len(x_test)):
    y_predicted=model.predict(x_test[i][:,features_sorted_indexes[0:k_optimal]])
    differences, deviations, datetimes=Utils.non_overlapping_windows_average_std(y_predicted-y_test[i],
                                                                             datetimes_test[i], 6*24*365)
    
    # Create a figure with a subplot of two plots vertically stacked
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    
    ax1.set_title("Turbine " + turbine_names[i] + "Pred/Real differences")
         
    # Plot the first subplot
    ax1.plot(datetimes,differences)
    ax1.set_ylabel('without error bars')
    ax1.grid()
    # Plot the second subplot
    ax2.errorbar(datetimes,differences, yerr=deviations)
    ax2.set_xlabel('Datetimes')
    ax2.set_ylabel('with error bars')
    plt.xticks(rotation=45)
    ax2.grid()
    # Show the plot
    plt.show()
    
    
for i in range(0,len(x_test)):    
    
    y_predicted=model.predict(x_test[i][:,features_sorted_indexes[0:k_optimal]])
    
    differences, deviations, datetimes=Utils.non_overlapping_windows_average_std(y_predicted-y_test[i],
                                                                                 datetimes_test[i], 6*24*365)
    
    differences_by_year=Utils.get_batches_windows(y_predicted-y_test[i], 6*24*365)
    years=range(0,len(differences_by_year))

    print("######### Turbine " + turbine_names[i] + " Pred/Real differences ########")
        
    print("   Mean difference for each year:")
    for i in range(0,len(years)):
        print("        "+"Year "+str(years[i])+": "+str(np.round(np.mean(differences_by_year[i]),2)))
    print()
    print("   Are there statistical differences from each year?")
    for i in range(0,len(years)-1):
        print("      Difference between Year " + str(i+2015)+" and " + 
               str(i+1+2015)+":")
        print("         Difference: "+str(np.round(np.mean(differences_by_year[i+1])-np.mean(differences_by_year[i]),2)))
        t_test,p_value=stats.ttest_ind(differences_by_year[i], differences_by_year[i+1])
        print("         P-value:  "+str(p_value))
 

print("#################### FEATURE IMPORTANCE ######################")
for i in range(0,len(feature_names)):
    print(feature_names[i]+": "+str(np.round(model.feature_importances_[i],2)))

plt.figure("Feature Importance - Regression Model")
plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature importance")
plt.ylim([0,1])
plt.grid()
plt.xticks(range(len(feature_names)), feature_names, rotation=45)
plt.show()


