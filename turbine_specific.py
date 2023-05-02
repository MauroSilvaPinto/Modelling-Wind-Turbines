"""
@author: mauro pinto

The code for turbine-specific modelling.
This code delivers a model that is trained and tested on the same turbine.

"""
from utils import Utils

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

import os
import datetime


# going to data directory
os.chdir("Technical test")
os.chdir("Data Turbines - enriched")
turbine_name="R80736.csv"


################################### Loading data ##################################

print("Loading data...")

# loading data description
data_description = np.genfromtxt("data_description.csv", dtype=str,  delimiter=";", usecols=range(0,4))

# loading static-information
static_info = np.genfromtxt("static-information.csv", dtype=str,  delimiter=";")

# loading R80711 turbine features names
data_names = np.genfromtxt(turbine_name, dtype=str,  delimiter=",")
data_names=data_names[0,3:]

# loading R80711 turbine sample datetimes
turbine_dt_int=np.genfromtxt(turbine_name, dtype=float, skip_header=1, delimiter=",", usecols=1)
data_datetimes=np.array([datetime.datetime.fromtimestamp(ts) for ts in turbine_dt_int])
del turbine_dt_int

# loading R80711 turbine sample features
turbine_data=np.genfromtxt(turbine_name, dtype=float, skip_header=1, delimiter=",", usecols=range(3,24))

################################### First-view analysis ##################################

print("Apriori analysis ...")
 

   
# visualise all features over time
# for i in range(0,21):
#     plt.figure()
#     plt.title(turbine_names[i]+
#               " - [" + str(np.min(turbine_data[:,i])) + 
#               "," + str(np.max(turbine_data[:,i]))+"]")
    
#     plt.plot(turbine_datetimes,turbine_data[:,i])


# study correlations between all variables
corr_matrix = np.triu(np.corrcoef(turbine_data, rowvar=False))
plt.figure("Correlation analysis between all data")
plt.title("Correlation data")
sns.heatmap(np.abs(corr_matrix),xticklabels=data_names,
            yticklabels=data_names, annot=True, fmt=".2f", cmap="cividis")


################################### Choose Inputs & Outputs ################

print("Selecting data for input and output ...")

### active power output
y_index=1;

Y=turbine_data[:,y_index]
#Y=Utils.compute_tip_speed_ratio(turbine_data[:,11],turbine_data[:,7],41)
#Y=Utils.compute_torque_converter(turbine_data[:,1],turbine_data[:,11])


# all variables except:
features_index=[12,14,15,16,17,18,19,20]


X=turbine_data[:,features_index]
feature_names=data_names[features_index]


################# split data into training, validation, testing ###################################

print("Partitioning the data ...")


training_indexes=Utils.get_year_indexes(data_datetimes,2012,2013)
validation_indexes=Utils.get_year_indexes(data_datetimes,2014,2014)
testing_indexes=Utils.get_year_indexes(data_datetimes,2015,data_datetimes[-1].year)

y_train=Y[training_indexes]
x_train=X[training_indexes,:]
datetimes_train=data_datetimes[training_indexes]

y_validation=Y[validation_indexes]
x_validation=X[validation_indexes,:]
datetimes_validation=data_datetimes[validation_indexes]

y_test=Y[testing_indexes]
x_test=X[testing_indexes,:]
datetimes_test=data_datetimes[testing_indexes]


################################### Removing Redundancy  ###################################

print("Model training ...")

  
corr_matrix = np.triu(np.corrcoef(x_train, rowvar=False))
#plt.figure()
#sns.heatmap(np.abs(corr_matrix),xticklabels=turbine_names[features_index],
#           yticklabels=turbine_names[features_index], annot=True, fmt=".2f", cmap="cividis")

# removing redundancy
[redundant_features_index,x_train]=Utils.remove_redundant_features(x_train)
feature_names=np.delete(feature_names,redundant_features_index)

x_validation=np.delete(x_validation,redundant_features_index,axis=1)

################################### Feature relevanve sorting  ###################################


# calculating relevant features by ascending order
corr_values=[]
for i in range(0, np.shape(x_train)[1]):
    corr_values.append(np.corrcoef(x_train[:,i], y_train)[0, 1])

features_sorted_indexes=np.argsort(np.abs(corr_values))[::-1]


########### normalizing all features #####################

scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_validation=scaler.transform(x_validation)


########### training model and validation ##########################

########## Grid Search for parameter optimization
k_features=np.linspace(1,np.shape(x_train)[1],np.shape(x_train)[1])
rmse_values=[]
for i in range(0,len(k_features)):
    model=DecisionTreeRegressor()
    model.fit(x_train[:,features_sorted_indexes[0:int(k_features[i])]],y_train)
    
    y_predicted=model.predict(x_validation[:,features_sorted_indexes[0:int(k_features[i])]])
    rmse_values.append(np.sqrt(mean_squared_error(y_validation,y_predicted)))

    
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

model.fit(x_train[:,features_sorted_indexes[0:k_optimal]],y_train)
y_predicted=model.predict(x_train[:,features_sorted_indexes[0:k_optimal]])

nmae = Utils.compute_nmae(y_train, y_predicted)
mean_difference=np.mean((y_predicted-y_train)/(np.max(y_train)-np.min(y_train))) 

print("#################### TRAINING ANALYSIS ######################")
print("NMAE: "+str(np.round(nmae,2)))
print("RMSE: "+str(np.round(np.sqrt(mean_squared_error(y_train,y_predicted)),2)))
print("Mean Relative Predicted/Real differences: "+str(np.round(mean_difference,2)))
print("Mean Absolute predicted/real differences: "+str(np.round(np.mean(y_predicted-y_train),2)))
print("R2: "+str(np.round(r2_score(y_train, y_predicted),2)))


plt.figure("Training Regression Results")
plt.title("Predicted and Training values, \n NMAE:"+str(np.round(nmae,2))+
          ", RMSE: "+str(np.round(np.sqrt(mean_squared_error(y_train,y_predicted)),2))+
          "\n Mean relative differences: "+str(np.round(mean_difference,2))+
          ", Mean absolute differences: "+str(np.round(np.mean(y_predicted-y_train),2))+
          "\n R2 score: "+str(np.round(r2_score(y_train, y_predicted),2)))

plt.xlabel("time")
plt.ylabel("Y")
plt.plot(datetimes_train,y_train, alpha=0.50)
plt.plot(datetimes_train,y_predicted, alpha=0.50)
plt.plot(datetimes_train,Utils.moving_average_filter((y_predicted-y_train),6*24*7))
plt.grid()
plt.legend(["Training", "Predicted","Differences smoothed"], loc="center left")


###### Final model performance assessment on validation ################
 
y_predicted=model.predict(x_validation[:,features_sorted_indexes[0:k_optimal]])

nmae = Utils.compute_nmae(y_validation, y_predicted)
mean_difference=np.mean((y_predicted-y_validation)/(np.max(y_validation)-np.min(y_validation))) 

plt.figure("Validation Regression Results")
plt.title("Predicted and Validaton values, \n NMAE:"+str(np.round(nmae,2))+
          ", RMSE: "+str(np.round(np.sqrt(mean_squared_error(y_validation,y_predicted)),2))+
          "\n Mean Relative differences: "+str(np.round(mean_difference,2))+
          ", Mean absolute differences: "+str(np.round(np.mean(y_predicted-y_validation),2))+
          "\n R2 score: "+str(np.round(r2_score(y_validation, y_predicted),2)))
plt.xlabel("time")
plt.ylabel("Y")
plt.plot(datetimes_validation,y_validation, alpha=0.50)
plt.plot(datetimes_validation,y_predicted, alpha=0.50)
plt.plot(datetimes_validation,Utils.moving_average_filter((y_predicted-y_validation),6*24*7))
plt.grid()
plt.legend(["Validation", "Predicted","Differences smoothed"], loc="center left")

print("#################### VALIDATION ANALYSIS ######################")
print("NMAE: "+str(np.round(nmae,2)))
print("RMSE: "+str(np.round(np.sqrt(mean_squared_error(y_validation,y_predicted)),2)))
print("Mean Relative Predicted/Real differences: "+str(np.round(mean_difference,2)))
print("Mean Absolute predicted/real differences: "+str(np.round(np.mean(y_predicted-y_validation),2)))
print("R2 score: "+str(np.round(r2_score(y_validation, y_predicted),2)))

########### Model Testing ##############################

print("Model testing ...")

x_test=np.delete(x_test,redundant_features_index,axis=1)
x_test=scaler.transform(x_test)

y_predicted=model.predict(x_test[:,features_sorted_indexes[0:k_optimal]])

nmae = Utils.compute_nmae(y_test, y_predicted)
mean_difference=np.mean((y_predicted-y_test)/(np.max(y_test)-np.min(y_test)))


plt.figure("Testing Regression Results")
plt.title("Predicted and Real values, \n NMAE:"+str(np.round(nmae,2))+
          ", RMSE: "+str(np.round(np.sqrt(mean_squared_error(y_test,y_predicted)),2))+
          "\n Mean relative differences: "+str(np.round(mean_difference,2))+
          ", Mean Abs differences: "+str(np.round(np.mean(y_predicted-y_test),2))+
          "\n R2 score: "+str(np.round(r2_score(y_test, y_predicted),2)))
plt.xlabel("time")
plt.ylabel("Y")
plt.plot(datetimes_test,y_test, alpha=0.50)
plt.plot(datetimes_test,y_predicted, alpha=0.50)
plt.plot(datetimes_test,Utils.moving_average_filter(y_predicted-y_test,6*24*7))
plt.grid()
plt.legend(["Real", "Predicted","Differences smoothed"],loc="center left")
  

####### Interpreting Testing Results ################  

y_predicted=model.predict(x_test[:,features_sorted_indexes[0:k_optimal]])

differences_by_year=Utils.get_batches_windows(y_predicted-y_test, 6*24*30)
#root_squared_differences_by_year=Utils.get_batches_windows((y_predicted-y_test)**2, 6*24*365)
years=range(0,len(differences_by_year))

print("#################### TESTING ANALYSIS ######################")
print("NMAE: "+str(np.round(nmae,2)))
print("RMSE: "+str(np.round(np.sqrt(mean_squared_error(y_test,y_predicted)),2)))
print("R2 score: "+str(np.round(r2_score(y_test, y_predicted),2)))
print("Mean Relative Predicted/Real differences: "+str(np.round(mean_difference,2)))
print("Mean Absolute predicted/real differences: "+str(np.round(np.mean(y_predicted-y_test),2)))
print()
print("Mean difference for each month/year:")
for i in range(0,len(years)):
    print("     "+"Year/month "+str(years[i])+": "+str(np.round(np.mean(differences_by_year[i]),2)))
print()
print("Are there statistical differences from each month/year?")
for i in range(0,len(years)-1):
    print("   Difference between Year/Month " + str(i+2015)+" and " + 
           str(i+1+2015)+":")
    print("      Difference: "+str(np.round(np.mean(differences_by_year[i+1])-np.mean(differences_by_year[i]),2)))
    t_test,p_value=stats.ttest_ind(differences_by_year[i], differences_by_year[i+1])
    print("      P-value:  "+str(p_value))


differences_year, deviations_year, datetimes_year=Utils.non_overlapping_windows_average_std(y_predicted-y_test,
                                                                             datetimes_test, 6*24*365)  
differences_month, deviations_month, datetimes_month=Utils.non_overlapping_windows_average_std(y_predicted-y_test,
                                                                             datetimes_test, 6*24*30)   
differences_week, deviations_week, datetimes_week=Utils.non_overlapping_windows_average_std(y_predicted-y_test,
                                                                             datetimes_test, 6*24*7) 

# Create a figure with a subplot of two plots vertically stacked
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.set_title("Predicted/Real differences in Years")
# Plot the first subplot
ax1.plot(datetimes_year,differences_year)
ax1.axhline(y=0, color='black',linestyle='--')
ax1.set_ylabel('differences')
ax1.grid()
# Plot the second subplot
ax2.set_title("Predicted/Real differences in Months")
ax2.plot(datetimes_month,differences_month)
ax2.axhline(y=0, color='black',linestyle='--')
ax2.set_xlabel('Datetimes')
ax2.set_ylabel('differences')
ax2.grid()
ax3.set_title("Predicted/Real differences in Weeks")
ax3.plot(datetimes_week,differences_week)
ax3.axhline(y=0, color='black',linestyle='--')
ax3.set_xlabel('Datetimes')
ax3.set_ylabel('differences')
plt.xticks(rotation=45)
ax3.grid()
fig.tight_layout()

# Show the plot
plt.show()


###### Analysing features ####################

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


