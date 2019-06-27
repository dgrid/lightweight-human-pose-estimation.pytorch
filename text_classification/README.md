# Overview

Implementation of article classification using naive bayes and display on web app.

## Setup

'''bash
cd text_classification
docker build -t project .
docker run -it -p 80:8000 project /bin/bash
'''

## Usage

### Basic

- Collect data and pickle them

'''bash
python manage.py collect_data
'''

- Launch server

'''bash
python manage.py runserver
'''

- Access to http://127.0.0.1:8000/app/input and input the URL you want to know the category

### Check accuracy

You can check naive bayes accuracy (about 0.50).

'''bash
python manage.py naive_bayes
'''

You can also check accuracy of another method to improve accuracy (about 0.60). The method is explained below.

'''bash
python manage.py improved_method
'''

## Improved method

Naive bayes assumes that each word is independent, so the accuracy is not so high. Consider the following functions based on Bayesian theorem for each category and consider them
to be classified as the maximum function. Where x is based on a normal distribution.

<img src="https://latex.codecogs.com/gif.latex?g_i(\bm{x}) = \bm{x}^T\bm{W}_i\bm{x}+\bm{w}_i^T\bm{x}+\bm{w}_{i0}" />
<img src="https://latex.codecogs.com/gif.latex?\bm{W}_i &= -\frac{1}{2}{\bm{\sum}}_{i}^{-1}" />
<img src="https://latex.codecogs.com/gif.latex?g_i(\bm{x}) = \bm{w}_i &= {\bm{\sum}}_{i}^{-1}\bm{\mu}_{i}" />
<img src="https://latex.codecogs.com/gif.latex?g_i(\bm{x}) = \bm{w}_{i0} &= {{\bm{\mu}}_i}^T\bm{W}_i\bm{\mu}_{i} +\textrm{log}{P(\omega_i)}" />
