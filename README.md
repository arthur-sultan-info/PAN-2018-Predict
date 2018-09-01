
###  PAN-2018-Predict
This repository contains the code to predict gender of authors from a PAN 2018 test dataset, using a trained [gender classifier](https://github.com/arthur-sultan-info/PAN2018-AP-Train).

## Requirements

This tool is coded in Python 3 and has been tested for Python 3.5.5

Required additionnal libraries:

| Library      | Version |
|--------------|---------|
| numpy        | 1.11.X  |
| scikit-learn | 0.18.X  |     
| pandas       | 0.19.X  |       
| nltk         | 3.2.X   |
| opencv       | 3.4.X   |
| tensorflow   | 1.5.X   |

You also need to install [darkflow](https://github.com/thtrieu/darkflow).
Please also download [yolo weights](https://pjreddie.com/media/files/yolov2.weights) and put it in a directory ./bin that you will create.


## Run the prediction
To run the prediction , run the following command (with dataset_path the path to the PAN test dataset and output_save the output path for the results):
```
python predictPAN.py dataset_path output_save
```
