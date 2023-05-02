# Modelling Wind Turbines - Engie Turbine Dataset

The project is based on the data made publicly available by Engie for 4 wind turbines in France in La Haute Borne.
Link: https://opendata-renewables.engie.com/explore/index
(warning: Engie's 'open data page' is not always up-and-running, but all the necessary info is also included in the shared documents.)

The folder data is a copy of the turbine data available through the link, and then enriched with additional parameters such as snowfall_1h, rainfall_1h, air density, humidity etc. This 'enrichment' used historic data retrieved from https://openweathermap.org/

A more complete data description is given by Engie on the site.

To get a better understandig of the data, its parameters and their relationships it might be useful to read the following wiki:
https://en.wikipedia.org/wiki/Wind_turbine_design#:~:text=Other%20controls-,Generator%20torque,(typically%206%20or%207)


## Project Goal

The goal was to use this data and create machine learning models that can predict the output of the turbine (rotor torque, converter torque and/or power output).

These models can be used as 'benchmarks' or 'digital twins' to compare the actual behaviour with expected behaviour.
If actual behaviour consistently underperforms expected behaviour (negative bias) this could be a tell-tale sign of a defect component and maintenance actions
should be undertaken before it causes bigger components to fail.


## Methodology

I developed a Machine Learning pipeline.

Then, I developed 3 different approaches:
  - [Specific](turbine_specific.py) (one model for each turbine);
  - [Leave-One-Out](turbine_LOO.py) (train with A,B,C and test with D, then rotate);
  - [Generalised](turbine_generalized.py) (one model for all turbines).

Using the Machine Learning pipeline, for each approach, I modelled several regression problems different combinations of inputs and outputs).
Finally, I analysed and interpreted results.

## Results, discussion, and detailed explanation about methods

You can find a detailed explanation of all methods, along with presented results and discussion in [Report](report.pdf).
You can also find full results in [results](results.xlsx).
