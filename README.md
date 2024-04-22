# DM-project1

This repository contains the code for the first project for Data Mining classes at the Poznań University of Technology. The code in this repository has been created by the team *Kung Fu Pandas*, consisting of:
- [Piotr Balewski](https://github.com/PBalewski),
- [Adam Dobosz](https://github.com/addobosz),
- [Wiktor Kamzela](https://github.com/Wector1),
- [Michał Redmer](https://github.com/MichalRedm).

The general purpose of the code in this repository is to preprocess the dataset to improve results attained by machine learning algorithms trained on this data. The dataset that we have chosen is [Cars Dataset from 1970 to 2024](https://www.kaggle.com/datasets/meruvulikith/90000-cars-data-from-1970-to-2024), containing 10 attributes and 90,000 records.

A more detailed description of our project can be found in [our report](https://github.com/MichalRedm/DM-project1/blob/main/report.pdf).

<img src="https://th.bing.com/th/id/OIG2.26ZnBEQYoNzIV2nJ2hJO?pid=ImgGn" style="width: 512px;" />

## Prequisities

To run the code from this repository, you need to have the following installed on your computer:
- [Python](https://www.python.org/downloads/) (version 3.10 or higher);
- [NumPy library](https://numpy.org/) (version 1.26 or higher);
- [Pandas library](https://pandas.pydata.org/) (version 2.2 or higher)
- [Sklearn library](https://scikit-learn.org/stable/)
- [UMAP library](https://umap-learn.readthedocs.io/en/latest/)

Once you have Python installed, you can run the following commands in your terminal to install all the necessary libraries:
```bash
$ pip install numpy
$ pip install pandas
$ pip install sklearn
$ pip install umap-learn
```

## Setup

*TODO*

## Preprocessing the data

The file `process.py` is responsible for processing the dataset. To generate a preprocessed dataset, run the following command:
```
$ python process.py [OUTPUT FILE NAME]
```
Argument `OUTPUT FILE NAME` should be the name of a CSV file to which the result shall be written.

## Testing the results

*TODO*
