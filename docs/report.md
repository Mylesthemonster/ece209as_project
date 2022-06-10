# Table of Contents
* Abstract
* [Introduction](#1-introduction)
* [Related Work](#2-related-work)
* [Technical Approach](#3-technical-approach)
* [Evaluation and Results](#4-evaluation-and-results)
* [Discussion and Conclusions](#5-discussion-and-conclusions)
* [References](#6-references)

# Abstract

Provide a brief overview of the project objectives, approach, and results.

# 1. Introduction

This section should cover the following items:

* Motivation & Objective: What are you trying to do and why? (plain English without jargon)
* State of the Art & Its Limitations: How is it done today, and what are the limits of current practice?
* Novelty & Rationale: What is new in your approach and why do you think it will be successful?
* Potential Impact: If the project is successful, what difference will it make, both technically and broadly?
* Challenges: What are the challenges and risks?
* Requirements for Success: What skills and resources are necessary to perform the project?
* Metrics of Success: What are metrics by which you would check for success?

# 2. Related Work

This is the paper that is the basis of our project: 
ViFin: Harness Passive Vibration to Continuous Micro Finger Writing with a Commodity Smartwatch

# 3. Technical Approach

Step 1: Split the provided data into training data, test, and validation data 

Step 2: Improve our model to use more then 1 layer GRU, which is what the paper did

Step 3: Make the model improve with new data with no retraining

Utilized Provided Finger Writing Dataset (Numbers 0-9)
Software:
    Google Collab 
    Tensorflow 
    SensorLog w/ Apple watch Series 7

# 4. Evaluation and Results

If we can implement the dataset given into a model with more advanced complexity and layering then experimented in the paper then we succeed

# 5. Discussion and Conclusions

We look to improve the model they implemented adn make it more robust. This paper is brand new (2021) so most of their process is using newer and relevant technology to implement their learning model. For example, we are both using Tensorflow post the TF 2.0 update. Our approach of adding a much need validation set, as well as improving the complexity opf the model will hopefully make for better more accurate results.

# 6. References
- Pengrui Quan & Ziqi Wang Finger Writing Dataset
- Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous TechnologiesVolume 5Issue 1March 2021 Article No.: 45pp 1–25 https://doi.org/10.1145/3448119