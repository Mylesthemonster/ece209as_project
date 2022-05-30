# Project Proposal

## 1. Motivation & Objective

Implement the model in the paper, ViFin: Harness Passive Vibration to Continuous Micro Finger Writing with a Commodity Smartwatch. We look to improve the model they implemented adn make it more robust.

## 2. State of the Art & Its Limitations

This paper is brand new (2021) so most of their process is using newer and relevant technology to implement their learning model. For example, we are both using Tensorflow post the TF 2.0 update.

## 3. Novelty & Rationale

Our approach of adding a much need validation set, as well as improving the complexity opf the model will hopefully make for better more accurate results.

## 4. Potential Impact

If the project is successful, what difference will it make, both technically and broadly?

## 5. Challenges

Some of the challenges for this project lie in recreating and upgrading the model in the paper with a high accuracy, and getting our improved model to work with Senorlog and the apple watch. 

## 6. Requirements for Success

Basic ML/Deep Learning knowledge, python programing, hardware to run model (Google Collab), data manipulation skills

## 7. Metrics of Success

If we can implement the dataset given into a model with more advanced complexity and layering then experimented in the paper then we succeed

## 8. Execution Plan

Step 1: Split the provided data into training data, test, and validation data 

Step 2: Improve our model to use more then 1 layer GRU, which is what the paper did

Step 3: Make the model improve with new data with no retraining  

## 9. Related Work

This is the paper that is the basis of our project: 
ViFin: Harness Passive Vibration to Continuous Micro Finger Writing with a Commodity Smartwatch

### 9.a. Papers

List the key papers that you have identified relating to your project idea, and describe how they related to your project. Provide references (with full citation in the References section below).

ViFin: Harness Passive Vibration to Continuous Micro Finger Writing with a Commodity Smartwatch

### 9.b. Datasets

List datasets that you have identified and plan to use. Provide references (with full citation in the References section below).

Provided Finger Writing Dataset

### 9.c. Software

List software that you have identified and plan to use. Provide references (with full citation in the References section below).

Google Collab 
Tensorflow 
SensorLog w/ Apple watch Series 7

## 10. References

PENGRUI QUAN & Ziqi Wang Finger Writing Dataset
Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous TechnologiesVolume 5Issue 1March 2021 Article No.: 45pp 1–25https://doi.org/10.1145/3448119
