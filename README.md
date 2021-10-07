
# Optimal Estimation for Satellite Observations: RadEst

Python based optimal estimation tooling for Satellite Observations; this work has
been developed in the context of a wind retrievals processor using radiometers from 
EUMETSAT's Ocean and Sea Ice [OSISAF](https://osi-saf.eumetsat.int/) where [KNMI]() as one of its cooperating members. The intend 
is the development of a scientific software framework using modern tooling; the main 
goals are:
- Code Maintainability
- Continuous Quality Improvement
- Open Science development: Open Source and Transparency
- Bridge the gap between "low level" scientific software 

## "RadEst" routines and parts

The repo contains two main notebooks (crossVal*.ipynb) which carry out the same final 
task of cross validating wind retrievals using syntetic data; the difference between
the two notebooks is the way to compute the Jacobians during the iterative minimization process.
Hopefully we will get to the point where open source community standards guide proper scientific software development.
The tools provided in this repo make use of several available packages/environments/software ecosystems such as:

- [pyOpEst](https://github.com/maahn/pyOptimalEstimation)
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

This work was kicked off in the context of a project of EUMETSAT's Ocean and Sea Ice [OSISAF](https://osi-saf.eumetsat.int/) and we highly apprecieate their support. Many thanks for the support of the NWPSAF Helpdesk, in particular James Hocking for his responses on the use and settings of RTTOV. Finally but also mainly to Maximilian Maahn for allowing the open source community to benefit and contribute to his pyOpEst tool. 











