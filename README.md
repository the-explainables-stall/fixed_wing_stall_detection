# Fixed wing aircraft stall detection using ML

## Proble Statement and Objectives
Aerodynamic stall is a hazardous condition for fixed-wing aircraft during which the airflow over the wings necessary to generate sufficient lift for flight becomes turbulent.  The turbulence reduces and abruptly ends the lift necessary to maintain flight, and the aircraft subsequently plummets from the sky.  The time between the initial onset and realization of stall ranges from fifteen to twenty seconds, leaving minimal time for pilots to implement necessary life-saving aircraft maneuvers.  Fixed-wing aircraft are often equipped with several preventative or detective controls, including a minimum recommend operating speed above potential stall speed, as well as a haptic alert issued through the physical pilot controls called a stick shaker.  Despite these preventative and detective controls, there is a history of catastrophic flight accidents attributed to stall due to controls failure.

This project attempts to predict the stall as early as possible using the Machine Learning and AI techniques. Along with prediction and generating a warning, this project attempts to explain what contributed to the warning using explainable ML package LIME. The lack of publicly available data for this case studey, we used simulated data for training prediction algorithms. The simulation follows the aerodynamic equations provided by our sponsor Dr. Lance Sherry. 

The primary objectives of this project are
1. Simulate necessary flight data, at least 10,000 flights simulated.
2. Explore feature engineering and selection.
3. Use Logistic Regression, XGBoost and LSTM models to train and predict
4. Generate comparisons or accuracy, recall and latency
5. Eplainability using LIME

## The Simulation
Using the aerodynamic equations provided by Dr. Lance Sherry code to simulate flight data has bee developed which includes identifying stall, uncommanded descent and uncommanded descent high and uncommanded roll. Our take on this as a classification proble which tries to identify the above 3 conditions given the data until that point of time. Belo diwagram shows the high level design of the simulation. 

![image](https://user-images.githubusercontent.com/10969756/127080947-33bfe1b8-93a6-424b-96a9-0ee5e6ee61b6.png?style=centerme)
*Simulation high level flow chart*

Given the set of initial conditions generated randonly for each simulation, the attributes of flight change within simulation according to the aerodynamic equations. Each simulation uses a step of 0.1 seconds and ends 10 seconds after the uncommanded descent high and uncommanded roll condition is detected. The simulation identifies onset of stall when these conditions are met starting from change in altitude. Below figure shows the results of a simulation with a red vertical lines identifying the onset of stall, uncommanded descent and uncommanded descent high and uncommanded roll. These are the target variables in the prediction / classification models. The definitions are:

1.	On set of stall: at time to buffet
2.	Uncommanded descent: on set of stall and increasing positive AOA, and decreasing air speed
3.	Uncommanded descent high, uncommanded roll: all the above and decreasing vertical speed and Std. of altitude > 7

![image](https://user-images.githubusercontent.com/10969756/127081580-a64a2e1d-8cef-4384-9643-8f41e211c278.png?style=centerme)
*Single simualtion strichart with classes marke in red*

## Feature engineering and feature selection
Rolling mean, rolling variance, exponential rolling mean and exponential rolling variance of the below flight attributes were generated.
* Pitch
* Flight Path Angle
* Airspeed
* AoA (Angle of Attack)
* Roll
* Vertical Speed
* Altitude

A simple Random Forest classification model with parameters of n_estimators=100, and random_state=1 was developed for this purpose. The model took a few hours to run and an accuracy of 87% was obtained. The following is the resulting score of each dependent variables in the data in predicting stall. Angel of attach, pitch roll, vertical speed, air speed and altitude are the imporatant features in that order. 
*INCLUDE UPDATED SCORE HERE*

## Modeling

### Logistic Regression
A set of logistic regression models is developed to predict the varying stages of stall—onset, uncommanded descent, etc.  These models leverage three different sets of input variables: 1) raw input variables without feature engineering; 2) rolling window statistics; 3) a combination of raw input and rolling window statistics.  Note that the predictions for the earlier stages of stall will feed into models predicting the subsequent stages of stall. 

![image](https://user-images.githubusercontent.com/10969756/127083172-041d1a85-61d6-4771-98bf-4e99feda3d08.png?style=centerme)

*Feed forward logistic regression architecture*

### XGBoost
Since the XGBoost model is not a time-series model, lag features, rate of change features, and time window segmentation are used to make it suitable for a time series model development. Mainly, Python’s XGBoostClassifier library is used for model development. Additionally, since the dataset is large and required advanced computing mechanism, the Argo cluster was utilized for the model development and related analysis.
Various feature engineering techniques were tried as per below:
•	Lag features
•	Rate of Change features
•	Time Window Segmentation
Lag Features:
Dataset was split into training and test datasets.  Lag features were generated for the train/test sets.  A total of 65 features, including ~55 lag features were input into the XGBoost model.  The performance was evaluated using the F1-Score.  Since this is an imbalanced dataset, the F1-Score is a better metric for evaluation of this model.  

* Rate of Change Features: Rate of change at various window spans was calculated.  The best combination was determined.
* Time Window Segmentation: The time series data was segmented into various window spans.  
* Summary statistics such mean, variance etc., were calculated for each window.  

![image](https://user-images.githubusercontent.com/10969756/127083567-929e69b5-087a-4b5f-8f4b-6aeaac7d65aa.png?style=centerme)

*Confusion matrix with the different set of features*

Using random search to obtain hyper-parameters and 10 fold cross validation along with feature engieering and selection algorithms, XGBoost was able to attain good recall score overall.
![image](https://user-images.githubusercontent.com/10969756/127083729-b3d4091e-f9fd-477e-815c-b23a69d948bf.png?style=centerme)

*Observed vs Predcicted/Classified*

### LSTM
For LSTM feature engineering, the sliding window technique is used. A prediction model with more than one-time variable to predict the next step is the sliding window model. For example, the value at t and value at t+1 is used to predict the value at time t+2. The models can be developed using the current time t and previous times t-1 as input variables to predict t+1. In this technique a look back window of certain time is chosen, and the class is predicted looking forward to a certain data point. Shift of certain data point is also chosen in this technique.

## Explainability
![image](https://user-images.githubusercontent.com/10969756/127084020-8e815626-ddaa-42ef-8bcf-e7c387c482e5.png?style=centerme)

*Decision explained by the features using LIME*


