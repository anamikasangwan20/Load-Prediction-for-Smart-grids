#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 21:49:44 2020

@author: anamika
"""

##############################################################################################
#
#  Load One-Month AMI Data and Perform K-Means Clustering (K = 2) into PV and non-PV customers  
#
##############################################################################################

#------- Load data and group into 'T' days --------#


#------- Average daily consumption curves for each customer for 'T' days --------#


#------- Standardize each curve --------#


#------- K-Means Clustering --------#


#------- Plot and Visually label the two clusters: Duck curve shape for PV customers --------#


##############################################################################################
#
#    K-Mediods Clustering on NPV customers to extract customer latent behavioural features
#
##############################################################################################

#------- Load NPV cluster and Perform K-medoids Clustering with DTW --------#


#------- Evaluate using Silhouette Coefficient --------#


#------- Extract and store centroids for CMM model --------#


#################################################################################################################
#
#  Linear Regression Model for Predicting consumption using centroids and solar generation using solar irradiance data
#
#################################################################################################################

#------- Unsupervised Training using Nighttime Split --------#

#------- Model --------#

#------- Evaluation --------#


############################################################
#
#     Post- Optimization
#
############################################################




############################################################
#
#    Evaluation
#
############################################################

