# grundner25pnas_iconaml_automatic_tuning
Data-driven equation for cloud cover implemented in ICON-A 2.6.4., the atmospheric component of the ICON climate model. The resulting ICON-A-ML climate model is tuned automatically, surpassing the projection skill of the native ICON-A model.

> Grundner, A., Beucler, T., Savre, J., Lauer, A., Schlund, M. & Eyring, V. (2025). Reduced cloud cover errors in hybrid climate models through a novel combination of data-driven parameterizations and automatic tuning

Author: Arthur Grundner, [arthur.grundner@dlr.de](mailto:arthur.grundner@dlr.de)

The current release on zenodo can be found here:

## List of Figures

- Fig 1: Sketch of the automatic tuning pipeline
- [Fig 2](XX), [Code](xx): Qualitative evaluation of 20-year ICON-A-ML simulations using parameter
settings extracted at three different stages of the tuning pipeline
- [Fig 3](fig_3_and_S5_biases/fig_3.pdf), [Code_1](fig_3_and_S5_biases/compare_icon_ml_to_icon_a_tuned.ipynb), [Code_2](fig_3_and_S5_biases/compare_icon_ml_to_icon.ipynb): Biases of 20-year ICON-A(-ML) simulations in three key climate metrics
- [Fig 4](XX), [Code](xx): Cloud differences in +4K ICON-A-ML simulations
- [Fig S1](XX), [Code](xx): Climate metrics of three 10-year ICON-A simulations
- [Fig S2](XX), [Code](xx): ICON-A sensitivity analysis
- [Fig S3](XX), [Code](xx): Zonal means of nine important climate variables from 20-year simulations
- [Fig S4](XX), [Code](xx): Like Fig. 2, but showing zonal means of the top of the atmosphere longwave and shortwave radiation.
- [Fig S5](XX), [Code](xx): (Bias) differences between the panels of each column in Fig. 3
- [Fig S6](XX), [Code](XX): Like Fig. 3, but showing column-integrated cloud ice (ice water path) for the ICON-ML and the automatically tuned ICON-A model
simulations.

## FOLLOWING NEEDS TO BE UPDATED

## Data

To reproduce the results it is first necessary to have access to accounts on [DKRZ/Levante](https://docs.dkrz.de/). Then one can coarse-grain and preprocess the DYAMOND and ERA5/ERA5.1 data sets:
- Guide for how to coarse-grain the DYAMOND data: [strategy.md](sec2_data/sec21_DYAMOND/strategy.md)
- To then pre-process the DYAMOND data: [preprocessing.ipynb](sec2_data/sec21_DYAMOND/preprocessing.ipynb) 
- Scripts to coarse-grain ERA5 data (1979-2021, first day of every quarter): [horizontally](sec2_data/sec22_ERA5/horizontal_coarse-graining), [vertically](sec2_data/vertical_coarse-graining)

It suffices to coarse-grain the variables: clc/cc, cli/ciwc, clw/clwc, hus/q, pa, ta/t, ua/u, va/v, zg/z

## Dependencies
The results were produced with the version numbers indicated below:
- PySR 0.10.1 [https://github.com/MilesCranmer/PySR]
- GP-GOMEA [https://github.com/marcovirgolin/GP-GOMEA]
- mlxtend 0.20.0 [https://github.com/rasbt/mlxtend]
- scikit-learn 1.0.2 [https://scikit-learn.org/]
- SymPy 1.10.1 [https://github.com/sympy]
- SciPy 1.8.1 [https://github.com/scipy/]
- TensorFlow 2.7.0 [https://tensorflow.org/]

To create a working environment you can run the following line:
```
conda install -c conda-forge tensorflow==2.7.0 scipy==1.8.1 sympy==1.10.1 scikit-learn==1.0.2 mlxtend==0.20.0 pysr==0.10.1