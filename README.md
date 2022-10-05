# Austroads project AAM6201: Development of AI/ML decision-support tools for pavement asset management

## Background
While there are some existing uses of artificial intelligence and machine learning technology in pavement asset management, there have been many unsuccessful attempts to develop new use-cases, which failed due to inappropriate use of data and models, and difficulty articulating the constraints of different technical disciplines. This project aims to provide a general methodology for developing new use-cases. 

Two case studies have been undertaken to demonstrate application of the methodology using data from four member authorities. 

## Case Study 1: Pavement Treatment Project Identification
The first case study sought to understand to what extent, historic condition, inventory and treatment data could be used to train machine learning models to reproduce expert pavement asset management decisions. We developed and evaluated models on four agency datasets, predicting six treatment classes over four future time-horizons. Treatment predictions were made for short road sections prior to clustering into candidate projects. The results were promising, and suggest that with further development machine learning models could provide decision support and efficiency gains to programme developers. 

## Case Study 2: Renewal Project Funding Allocation - multi-criteria optimization
The second case study explored an extension to conventional Pavement Management System (PMS) optimisation to provide insight into the network-wide implications of various multi-criteria funding allocation scenarios. We were able to show the implications of all achievable funding splits (freight / non- freight routes, and metropolitan / regional routes) on network Level of Service. We developed a process by which multi-criteria splits could be achieved within existing PMS optimisation processes, which suggests this methodology could be adopted without significant disruption or risk.

## Code information
A separate directory includes all files necessary to reproduce our methods in each case study. 

Both case studies were developed using open-source Python data science tools, such as Python, NumPy and Pandas. A complete list of dependencies can be found in the `requirements.txt` files in each top-level directory.

## Project information
More information about the project can be found at:

[https://austroads.com.au/infrastructure/asset-management/machine-learning-for-asset-management](https://austroads.com.au/infrastructure/asset-management/machine-learning-for-asset-management)

Project status information can be found at [https://austroads.com.au/infrastructure/asset-management/machine-learning-for-asset-management](https://austroads.com.au/infrastructure/asset-management/machine-learning-for-asset-management)

Two key resources should be published in the near future:
* Project report: this will contain details of the methodology and results obtained using the code in this repository.
* Quick Guide to developing new AI/ML projects: A slide deck which distills key aspects of the methodology we used and recommend.