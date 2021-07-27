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

![image](https://user-images.githubusercontent.com/10969756/127080947-33bfe1b8-93a6-424b-96a9-0ee5e6ee61b6.png)
