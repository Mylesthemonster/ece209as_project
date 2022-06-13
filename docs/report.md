# FINAL REPORT

## Table of Contents

* Abstract
* [Introduction](#1-introduction)
* [Related Work](#2-related-work)
* [Technical Approach](#3-technical-approach)
* [Evaluation and Results](#4-evaluation-and-results)
* [Discussion and Conclusions](#5-discussion-and-conclusions)
* [References](#6-references)

## Abstract

This project explores classifying "air writes" using a common smartwatch, allowing users to freely write letters and numbers with their finger and the overall system capable of identifying and incorporating the gesture. Our project extends off of work previously done by the University of Virginia (ViFin) where they designed a model architecture for classifying signal sequences, but our project specifically addresses key limitations of that research and explores the novel implementation of continuous writing. Our model improvements resulted in a classification accuracy of over 95%, and our chunking method of feeding continuous input data into the model showed great potential, with the fine tuning of the model to further personalize it showing over a 5% improvement over time in model accuracy from the baseline. Overall, our model improvements and added features work to improve the practicality and scalability of this solution.

## 1. Introduction

This section should cover the following items:

* Motivation & Objective: What are you trying to do and why? (plain English without jargon)
* State of the Art & Its Limitations: How is it done today, and what are the limits of current practice?
* Novelty & Rationale: What is new in your approach and why do you think it will be successful?
* Potential Impact: If the project is successful, what difference will it make, both technically and broadly?
* Challenges: What are the challenges and risks?
* Requirements for Success: What skills and resources are necessary to perform the project?
* Metrics of Success: What are metrics by which you would check for success?

Metrics:
Edit distance is a metric that can measure the distance between two strings. This was utilized in this project since labels are essentially strings, and we can compare correctness of value and placement of output labels. The edit distance measures the number of transformations required to reach the target string from the source string, being 0 for completely identical strings. Right and wrong indicate placement of output labels, and miss and false positive indicate the correct/incorrect values of output labels. A sample output is shown below, which can further yield overall accuracy.

![metrics](media/stats.png)

## 2. Related Work

This is the paper that is the basis of our project: 
ViFin: Harness Passive Vibration to Continuous Micro Finger Writing with a Commodity Smartwatch

## 3. Technical Approach

The model architecture defined in the ViFin paper is as follows:

![arch](media/architecture.png)

The input is provided to a GRU layer followed by a single fully connected softmax activation layer. The model is trained with a CTC loss function, allowing the result labels to be decoded and aligned. CTC (Connectionist temporal classification) allows for alignment of temporal classifications, and the GRU layer is a mechanism for RNN being more efficient with similar performance to LSTM. To address the specific limitations of the ViFin paper, we focused our approach to three steps:

Step 1: Split the provided data into training data, test, and validation data. The ViFin paper validated data using the test dataset, which was also used for final evaluation. Proper data splitting was implemented.

Step 2: Improve our model complexity. Additional learnable parameters in the correct placed would improve overall accuracy and generalizability.

Step 3: Make the model scalable for new users. General approaches, like what was used in the ViFin paper, utilized transfer learning which requires a limited training dataset for each new user. By incorporating user feedback into the pipeline, we can utilize unlabeled predictions in fine-tuning and personalizing the model.

A key novelty of our project was to allow for continuous user input, which required retraining the model allowing for input in a new form we define. We use a overlapping windowing, or "chunking" approach, as seen in the figure below.

![chunking](media/chunking.png)

By defining a window size and an overlap size, we can extract predictions from the model by feeding in each chunk through the model, and by storing raw results (not the CTC decoded results), we can account for the overlapped regions of the input to retrieve the correct output. This model was trained on a new dataset, chunked according to the configuration specified. Our fine-tuning personalization procedure, modeled after transfer learning techniques, updated the weights of the model with a lower learning rate after each iteration. Since we are generating unlabeled data, user feedback is key in allowing for this system. The ViFin paper already implemented 5 training-free gestures utilizing raw IMU data. We planned to incorporate a sixth gesture for the user to provide feedback on incorrect results, and allow for this fine-tuning "online learning" step.

Utilized Provided Finger Writing Dataset (Numbers 0-9)
Software:
    Google Collab
    Tensorflow
    SensorLog w/ Apple watch Series 7

## 4. Evaluation and Results

If we can implement the dataset given into a model with more advanced complexity and layering then experimented in the paper then we succeed.

Incorporation of the validation dataset allowed for better monitoring of the training step of the model. Common practice in machine learning utilizes a training dataset for training the model, a validation dataset for validating the model after each epoch on an unseen dataset, and a test dataset for final evaluation on an unbiased and unseen dataset. The original research paper validated the model using the test dataset, but validation assists in monitoring over fitting and model selection when performing hyperparameter optimization. We incorporated an 80-10-10 (this is fully configurable) train-validation-test split of the data. We can see proper model training and convergence without over fitting in the new training curve below:

![validation](media/results_2.png)

In addition, we wanted to add additional learnable parameters to the simple model architecture presented earlier. This was in two key areas: adding more fully connected layers and making the GRU layer more complex. In the figure below, we see the output model accuracies for different fully connected layer configurations and GRU layer sizes. The optimal combination of these parameters was to use one additional dense layer with 100 neurons and increase the GRU layer size to 64.

![improvements](media/results_1.png)

Lastly, the continuous input technique based on our "chunking" process resulted in a lower baseline accuracy, around 80%. This is likely due to incorrect dataset generation since the dataset was chunked assuming properly segmented labels (i.e. if a signal with 4 labels is chunked into 4 overlapping regions, then it is assumed that each chunk corresponds to one label). This assumption is likely incorrect, and can be fixed by properly segmenting the dataset in the data preprocessing stage with raw data. Unfortunately, we did not have access to this raw data, since we were given a dataset of processed data, but this can be approached in future iterations of this work.

## 5. Discussion and Conclusions

We look to improve the model they implemented adn make it more robust. This paper is brand new (2021) so most of their process is using newer and relevant technology to implement their learning model. For example, we are both using Tensorflow post the TF 2.0 update. Our approach of adding a much need validation set, as well as improving the complexity opf the model will hopefully make for better more accurate results.

TO ADD:

* add validation accuracy. hard to add because CTC decoding was not supported as a tensor operation, but can be implemented ourselves instead of using tensorflow functions

* use future work from the slides

## 6. References

* Pengrui Quan & Ziqi Wang Finger Writing Dataset
* Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous TechnologiesVolume 5Issue 1March 2021 Article No.: 45pp 1–25 <https://doi.org/10.1145/3448119>