
# Optimal Estimation for Satellite Observations: RadEst

Python based optimal estimation tooling for Satellite Observations; this work has
been developed in the context of a wind retrievals processor using radiometers from 
EUMETSAT's Ocean and Sea Ice [OSISAF](https://osi-saf.eumetsat.int/) where [KNMI](https://www.knmi.nl/over-het-knmi/about) as one of its cooperating members. The intend 
is the development of a scientific software framework using modern tooling; the main 
goals are:
- Code Maintainability
- Continuous Quality Improvement
- Open Science development: Open Source and Transparency

## "RadEst" routines and parts

The repo contains two main notebooks (crossVal*.ipynb) which carry out the same final 
task of cross validating wind retrievals using syntetic data; the difference between
the two notebooks is the way to compute the Jacobians during the iterative minimization process.
Hopefully we will get to the point where open source community standards guide proper scientific software development.
The tools provided in this repo make use of several available packages/environments/software ecosystems such as:

- [PyOptimalEstimation](https://github.com/maahn/pyOptimalEstimation)
- [RTTOV](https://nwp-saf.eumetsat.int/site/software/rttov/)
- [Pangeo](https://pangeo.io/)
- [SciPy](https://www.scipy.org/)
- Others that I might forget at this time (apologies for that) 
 

## Purpose of this repo

- Provide a practical example on how to setup an [Optimal Estimation tool in Python](https://github.com/maahn/pyOptimalEstimation)
together with a standard Satellite Observation Radiative Transfer code like [RTTOV](https://nwp-saf.eumetsat.int/site/software/rttov/).
- Involve interested scientists/developers from the Earth Observation / Satellite Observations / Machine Learning community in an open science tool for the benefit of the community itself
- This repo aims to document scientific and implementation tests that allows the user, not only to see and understand the code but also clone (or download) it and test the different capabilities of the different building blocks.  


## Documentation

For now we use the markdown standard for Jupyter Notebooks and a "good" commented code as a basis to start the conversation.


## Acknowledgement

This work was kicked off in the context of a project of EUMETSAT's Ocean and Sea Ice [OSISAF](https://osi-saf.eumetsat.int/) and we highly apprecieate their support. Many thanks for the support of the NWPSAF Helpdesk, in particular James Hocking for his responses on the use and settings of RTTOV. Finally but also mainly to Maximilian Maahn for allowing the open source community to benefit and contribute to his PyOptimalEstimation tool. 

## Citation and Contribution

This repo is part of the supplementary materials of our work (accepted): "M. Echeverri, A. Verhoef, A. Stoffelen, M. Maahn. Atmospheric Retrievals in a Machine Learning Context: A Radiometric Story Over the Ocean. *ESA-ECMWF Workshop: Machine Learning for Earth System Observation and Prediction, 15-18 November 2021, ESA-ESRIN*".
If you use these materials please consider citing the mentioned work!

If you want to contribute feel free to do so: ideas, proposals (e.g. via pull request) and constructive criticism are always welcome.









