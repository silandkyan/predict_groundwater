# predict_groundwater

A prototype software for predicting groundwater fluctuations in past and future climates using machine learning. By Philip Groß, April 2024.

## A: Project description
In many parts of the world, groundwater is needed as a source of drinkin water for humans or used in agricultural and industrial activities. Therefore, accessible groundwater is a vital resource for many people. However, the groundwater levels are not constant, but subject to perpetual change, which reflects the variable nature of the many parameters that ultimately control groundwater levels. These are mainly meteoric (e.g. precipitation, temperature, vegetation) or geologic (soil and rock properties) factors, but also human activity can directly influence groundwater levels (e.g. groundwater extraction via wells).

The broader goal of this project is to investigate the influence of meteoric conditions on groundwater levels over a long timespan. In the first part, this is performed for the past of a selected region. A machine learning model is trained on meteoric data to predict the observed groundwater movements of the past. The second part deals with using the trained model to predict the groundwater behaviour in hypothetical future climate scenarios.

### A1: File inventory
The repo is organised in the following way:

- folders containing weather, groundwater and elevation data for the model region
- notebooks performing special tasks, organized in a modular way
- a file called toolbox.py containing custom helper functions for use across notebooks

### A2: Modeling strategy
The modeling strategy involves consecutively running the notebooks to perform the required modeling steps. Each notebook generates output that is then used by successive ones.

1. Data retrieval using get_weatherdata and get_gwdata_bergstraße.
2. Model generation and export using model_1.
3. Creation of future weather scenarios using model_weather.
4. Predict groundwater with the trained model and a weather file using future_model.
5. Analyse modeling results using analyse_predictions and make_animation.

## Appendix: Manage environment
For seamless usage of the notebooks of this repo, it is advisable to set up a kernel environment that contains all necessery python packages. The environment file is also included in this repo: geo.yml in the repo home folder. The basics of environment management are described below.
 
1. Initial environment creation (if no env file at hand):

`conda create --name geo`
`conda activate geo`

2. Go to project home folder and generate environment file (needs to be updated/run again after installation of additional packages!): 

`conda env export > geo.yml`

3. To create the environment from the file, run:

`conda env create -f geo.yml`

4. and activate:

`conda activate geo`

5. to verify:

`conda env list`

6. Make new ipykernel for running the notebook on:

`python -m ipykernel install --user --name geo --display-name "Python (geo)"`

7. Finally, activate this kernel in the notebook settings.
