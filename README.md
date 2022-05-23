# Sentiment_Analysis_Covid19

## Requirements

In order to execute the Code, you must install the packages that their name and versions are located in requirements.txt file.

In order to fetch and install all of the packages you have to run the below command:
```
$ pip install -r requirements.txt

```
(it will take about 3-4 minutes. It depends on Network/Internet speed.)

## Dataset

The Dataset consists of 2 csv files. One for train/validation and one for test on uknown/unseen data.
You can find and download Dataset from Kaggle from the below URL:
https://www.kaggle.com/gauravsahani/covid-19-sentiment-analysis-using-spacy/data

As alternative way to get the Data you have to run the below command in your terminal:

```
$ kaggle kernels output gauravsahani/covid-19-sentiment-analysis-using-spacy -p /path/to/dest

```
Datasets are already located in the repository.

## Jupyter Notebook

In order to see the whole process and clarrifications during training/validation and test, see the SentimentAnalysis.ipynb file.
In order to see all of the plots in notebook produced by pyplot you can [clik here](https://nbviewer.org/github/icsd13152/Sentiment_Analysis_Covid19/blob/main/SentimentAnalysis.ipynb). In this link you can see the whole notebook too.


## Source Code

You can find the source code that is used for this project under the Covid19sentimentAnalysis/ directory. 
Classifier.py file contains the source code of preprocessing/processing/feature Engineering and selection/training/validation and testing on unseen Data.


## Dash application

In order to have productive processes, I have create a Web UI DashBoard. In order to run this application, you have to download the source code and run application.py in localhost mode (probably port 8050).
The application.py is processing the test csv file from kaggle.
If you want to see the application for live tweets using twitter API for fetching Data, you have to run the applicationOnLiveTweets.py
In these apps you can choose the classifier you want by clicking in the dropdown menu that is located on the top of hmtl page.
Below an image of use-case for the application.  
![mockup](https://user-images.githubusercontent.com/39522734/147951850-80c6d39a-6afe-4a0e-bf66-689533381dde.PNG)
There is also the DemoApp.py, which is the same aplication as the application.py but is running only on terminal and without Dash. This py file is reading the csv file too. 
The DemoApp.py is producing the same results as application.py. 
 Intractions for Execution:
 ```
$ cd Covid19sentimentAnalysis/
$ python application.py

```
OR
```
$ cd Covid19sentimentAnalysis/
$ python DemoApp.py

```
OR
```
$ cd Covid19sentimentAnalysis/
$ python applicationOnLiveTweets.py

```
Note: Maybe Dash apps will take a few seconds to startup.

## Docs directory
This directory contains the Presentation in pdf and ppt format.
