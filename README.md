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

*Fig[1]. Simulation high level flow chart*

Given the set of initial conditions generated randonly for each simulation, the attributes of flight change within simulation according to the aerodynamic equations. Each simulation uses a step of 0.1 seconds and ends 10 seconds after the uncommanded descent high and uncommanded roll condition is detected. The simulation identifies onset of stall when these conditions are met starting from change in altitude. Below figure shows the results of a simulation with a red vertical lines identifying the onset of stall, uncommanded descent and uncommanded descent high and uncommanded roll. These are the target variables in the prediction / classification models. The definitions are:

1.	On set of stall: at time to buffet
2.	Uncommanded descent: on set of stall and increasing positive AOA, and decreasing air speed
3.	Uncommanded descent high, uncommanded roll: all the above and decreasing vertical speed and Std. of altitude > 7

![image](https://user-images.githubusercontent.com/10969756/127081580-a64a2e1d-8cef-4384-9643-8f41e211c278.png?style=centerme)
*Fig[2]. Single simualtion strichart with classes marke in red*

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

![image](https://user-images.githubusercontent.com/10969756/127134975-8acbc17d-16dd-42e9-98fb-bb208109839d.png)

*Fig[3]. feature importance with random forest*

## Modeling

### Logistic Regression
A set of logistic regression models is developed to predict the varying stages of stall—onset, uncommanded descent, etc.  These models leverage three different sets of input variables: 1) raw input variables without feature engineering; 2) rolling window statistics; 3) a combination of raw input and rolling window statistics.  Note that the predictions for the earlier stages of stall will feed into models predicting the subsequent stages of stall. 

![image](https://user-images.githubusercontent.com/10969756/127083172-041d1a85-61d6-4771-98bf-4e99feda3d08.png?style=centerme)

*Fig[4]. Feed forward logistic regression architecture*

Incorporated time element using rolling means and variances calculated with varying time windows for each target variable (e.g., the rolling variance of Angle-of-Attack).Features used were "Raw" input variables (without transformation): pitch; flight path angle; airspeed; etc., Individually tuned "Rolling Window" input variables and 10 fold cross validation was used. Three binary classification models were developed to allow for individual tuning.

![image](https://user-images.githubusercontent.com/10969756/127150313-bcabf2a4-6992-4de9-a118-27cff44ee5ca.png)

*Fig[5]. Accuracy and Recall of Logistic Regression*

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

*Fig[6]. Confusion matrix with the different set of features*

Using random search to obtain hyper-parameters and 10 fold cross validation along with feature engieering and selection algorithms, XGBoost was able to attain good recall score overall.
![image](https://user-images.githubusercontent.com/10969756/127083729-b3d4091e-f9fd-477e-815c-b23a69d948bf.png?style=centerme)

*Fig[7]. Observed vs Predcicted/Classified*

### LSTM
For LSTM feature engineering, the sliding window technique is used. A prediction model with more than one-time variable to predict the next step is the sliding window model. For example, the value at t and value at t+1 is used to predict the value at time t+2. The models can be developed using the current time t and previous times t-1 as input variables to predict t+1. In this technique a look back window of certain time is chosen, and the class is predicted looking forward to a certain data point. Shift of certain data point is also chosen in this technique.

LSTM model was built with tf.keras.Sequential model and uses 29 hidden layers. Sigmoid is used as the activation layer with dense layer of 1 unit. Three experiments are conducted based on different sliding window - 
* Back window = 10,Forward Window =1,predict class 1
* Back window 10, forward window = 200, predict class 3
* Back window 10, forward window = 200, predict window (Class 1,2,3 or class 3)

![image](https://user-images.githubusercontent.com/10969756/127149902-51c865a0-1356-4b8b-9a35-5ad3a3396fac.png)

*Fig[8]. Accuracy and Recall with LSTM experiments*

## Explainability
Explainability is essential with any AI/ML model which includes Simulatability, Decomposability, and Transparency. Blackbox tools like LIME help understand contributing features.

![image](https://user-images.githubusercontent.com/10969756/127139975-678b7933-4852-4205-b7b9-f464ee99d5c0.png?style=centerme)

*Fig[9]. XGBoost predictions explained with LIME*

![image](https://user-images.githubusercontent.com/10969756/127139992-660be37d-49e9-486e-b6c1-2015c926beea.png?style=centerme)

*Fig[10]. Logistic Regression predictions explained by the features using LIME*


## Summary
30 seconds into the flight, LSTM predicted onset of stall would happen at 52nd sec and 40 sec into the flight, it predicted uncommanded descent high and roll would happen at 65th sec. XGBoost and Logistic Regression predictions are shown in the graph below for the three classes.

![image](https://user-images.githubusercontent.com/10969756/127148843-de710210-639e-422f-a666-27ece6ddeab7.png)

*Fig[11]. Models predcition vs observed stripchart*


![image](https://user-images.githubusercontent.com/10969756/127149215-3c0552d9-123b-4b26-9093-5ff58bdaef07.png)

*Fig[12]. Accuracy, Precision and Recall of the three models* 

* XGBoost and Logistic Regression could achieve high accuracy, recall rate, and efficient latency with effective feature engineering and selection techniques.
* LSTM could predict all the three conditions ahead of time but with latency.
* Explainability is essential with any AI/ML model which includes Simulatability, Decomposability, and Transparency. Blackbox tools like LIME help understand contributing features.
* Better hyper-parameter tuning, adding more layers and dropout layer will boost LSTM performance significantly.
* Simulation can be improved by unsupervised models and/or anomaly detection algorithms
* Interface / dashboard to predict and show explainable features.
* Two sets of simulations / algorithms running for each side of the aircraft and a summarizing model.


