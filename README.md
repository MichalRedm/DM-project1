# DM-project1

This repository contains the code for the first project for Data Mining classes at the Poznań University of Technology. The code in this repository has been created by the team *Kung Fu Pandas*, consisting of:
- [Piotr Balewski](https://github.com/PBalewski),
- [Adam Dobosz](https://github.com/addobosz),
- [Wiktor Kamzela](https://github.com/Wector1),
- [Michał Redmer](https://github.com/MichalRedm).

The general purpose of the code in this repository is to preprocess the dataset to improve results attained by machine learning algorithms trained on this data. The dataset that we have chosen is [Cars Dataset from 1970 to 2024](https://www.kaggle.com/datasets/meruvulikith/90000-cars-data-from-1970-to-2024), containing 10 attributes and 90,000 records.

A more detailed description of our project can be found in [our report](https://github.com/MichalRedm/DM-project1/blob/main/Kung_Fu_Pandas.pdf).

<img src="https://th.bing.com/th/id/OIG2.26ZnBEQYoNzIV2nJ2hJO?pid=ImgGn" style="width: 512px;" />

## Prequisities

To run the code from this repository, you need to have the following installed on your computer:
- [Python](https://www.python.org/downloads/) (version 3.10 or higher).

Additionally, to run code samples in the file `samples.ipynb` [Jupyer Notebook](https://jupyter.org/install) is needed (the file can alternatively be opened in [Google Colaboratory](https://colab.research.google.com/)).

## Setup

To download the respository to your local computer run the following command:

```
$ git clone https://github.com/MichalRedm/DM-project1.git
```

Then, you need to install all the Python dependencies:

```
$ pip install -r requirements.txt
```

Once this is done, you are ready to run the code.

## Preprocessing the data

The file `preprocess.py` is responsible for processing the dataset. To generate a preprocessed dataset, run the following command:
```
$ python preprocess.py
```
The new dataset will be written to a file `CarsDataProcessed.csv`.

## Testing the results

*TODO*

## Code samples

To have some insight into how our function for data preprocessing operates, visit the file [samples.ipynb](https://github.com/MichalRedm/DM-project1/blob/main/samples.ipynb).
