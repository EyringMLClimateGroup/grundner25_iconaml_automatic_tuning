# Reduced cloud cover errors in a hybrid AI-climate model through equation discovery and automatic tuning
Data-driven equation for cloud cover implemented in ICON-A 2.6.4., the atmospheric component of the ICON climate model. The resulting ICON-A-MLe climate model is tuned automatically, surpassing the projection skill of the native ICON-A model.

> Grundner, A., Beucler, T., Savre, J., Lauer, A., Schlund, M. & Eyring, V. (2025). Reduced cloud cover errors in hybrid climate models through a novel combination of data-driven parameterizations and automatic tuning

Author: Arthur Grundner, [arthur.grundner@dlr.de](mailto:arthur.grundner@dlr.de)

The current release on zenodo can be found here:

## List of Figures

- [Fig 1](fig_1_tuning_pipeline/fig_1.pdf): Sketch of the automatic tuning pipeline
- [Fig 2](simulation_scripts_and_evaluation/fig_2.pdf): Qualitative evaluation of 20-year ICON-A-MLe simulations using parameter
settings extracted at three different stages of the tuning pipeline
- [Fig 3](fig_3_and_S5_biases/fig_3.pdf), [Code_1](fig_3_and_S5_biases/compare_icon_ml_to_icon_a_tuned.ipynb), [Code_2](fig_3_and_S5_biases/compare_icon_ml_to_icon.ipynb): Biases of 20-year ICON-A(-ML) simulations in three key climate metrics
- [Fig 4](fig_4_plus4K/fig_4.pdf), [Code](fig_4_plus4K/cloud_sensitivities.ipynb): Cloud differences in +4K ICON-A-MLe simulations
- [Fig S1](fig_S1_man_tuned/fig_S1.pdf), [Code](fig_S1_man_tuned/giorgetta_fig_7.ipynb): Climate metrics of three 10-year ICON-A simulations
- [Fig S2](fig_S2_sensitivity/fig_S2.pdf), [Code](fig_S2_sensitivity/2402_sensitivity_analysis.ipynb): ICON-A sensitivity analysis
- [Fig S3](simulation_scripts_and_evaluation/fig_S3.pdf): Zonal means of nine important climate variables from 20-year simulations
- [Fig S4](simulation_scripts_and_evaluation/fig_S4.pdf): Like Fig. 2, but showing zonal means of the top of the atmosphere longwave and shortwave radiation
- [Fig S5](fig_3_and_S5_biases/fig_S5.pdf), [Code_1](fig_3_and_S5_biases/compare_icon_ml_to_icon_a_tuned.ipynb), [Code_2](fig_3_and_S5_biases/compare_icon_ml_to_icon.ipynb): (Bias) differences between the panels of each column in Fig. 3
- [Fig S6](simulation_scripts_and_evaluation/fig_S6.pdf): Like Fig. 3, but showing column-integrated cloud ice (ice water path) for the ICON-A-MLe and the automatically tuned ICON-A model simulations

## Reproducing the results

All ICON simulations were performed on [DKRZ/Levante](https://docs.dkrz.de/) with ICON-A 2.6.4. with two implementation errors fixed in the turbulence scheme (setting maximum mixing length to 150m and correcting the computation of the turbulent length scale so that it matches Pithan et al., 2015 and Giorgetta et al., 2018). The resulting ICON-A 2.6.4. code can be found in our [GitLab repository](https://gitlab.dkrz.de/icon-ml/icon_developments/icon-a-ml/-/tree/icon-2-6-4-ml_cloud_cover?ref_type=heads). You can find the official stable ICON-A 2.6.4. release on ICON-DKRZ-GitLab [here](https://gitlab.dkrz.de/icon/icon/-/tree/icon-2.6.4). To turn it into our ICON-A-MLe model, the data-driven cloud cover scheme needs to be implemented in ICON-A with adaptable parameter as shown [here](data_driven_cloud_cover_scheme.txt). The scripts for automatically tuning the ICON-A(-ML) can be found in [this folder](tuning_scripts). The code for deriving the data-driven cloud cover scheme can be found in the [GitHub repository](https://github.com/EyringMLClimateGroup/grundner23james_EquationDiscovery_CloudCover) of [Grundner et al., 2024](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023MS003763).

In the subfolders of
[simulation_scripts_and_evaluation](simulation_scripts_and_evaluation) you can
find the runscripts and ESMValTool evaluation plots of all 20-year AMIP
simulations performed for our manuscript. To reproduce the ESMValTool results,
ESMValTool
[v2.12.0](https://docs.esmvaltool.org/en/latest/quickstart/installation.html)
is required. An ESMValTool [configuration
file](https://docs.esmvaltool.org/projects/ESMValCore/en/latest/quickstart/configure.html)
tailored to [DKRZ's Levante](https://docs.dkrz.de/doc/levante/) is available in
this directory at `ESMValTool_config-user.yml`. This file needs to be put into
`~/.config/esmvaltool/` and slightly adapted (e.g., output paths). ESMValTool
[recipes](https://docs.esmvaltool.org/projects/ESMValCore/en/latest/recipe/overview.html)
can be found on the websites given by the *Evaluation* link in the `source.txt`
files of the subfolders (Click on *debug page* -> Select a recipe -> scroll to
the bottom -> Download the *recipe_\*.yml* file. ESMValTool can be then be run
with

```bash
esmvaltool run /path/to/recipe.yml
```

To reproduce the results on another machine, more changes to the configuration
file (e.g., input paths) are necessary.
