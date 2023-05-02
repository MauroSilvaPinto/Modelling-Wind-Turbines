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

- Developed a Machine Learning pipeline
- Developed 3 different approaches:
  - Turbine Specific
  - Turbine Leave-One-Out (LOO)
  - Turbine Generalised

- Using the Machine Learning pipeline, for each approach:
  - modelled several regression problems different combinations of inputs and outputs)

- Analysed and interpreted results


## Approaches

[Generalised](turbine_generalized.py) (one model for all turbines):
- all turbines are configured similarly, so they all should work similarly
- when one turbine has divergent results, could it have a malfunction?

[LOO](turbine_LOO.py) (train with A,B,C and test with D, then rotate):
- same assumptions as generalized

[Specific](turbine_specific.py) Specific (one model for each turbine):
- each turbine has its setting, and should have its model.
- we can monitor, in a more detailed manner, the existence of performance changes
