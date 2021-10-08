# voltages-of-battery-electrodes
Using the CGCNN transfer learning model to pridict the voltages of many kinds of metal-ion battery electrodes

The package provides all the files that are used in the article of ""
- The CGCNN model used to predict the voltages for Li-ion batttery electrodes. The details of the CGCNN framework are described in the article: [Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties](https://link.aps.org/doi/10.1103/PhysRevLett.120.145301), and the model is http://github.com/txie-93/cgcnn
- the transfer learning of the CGCNN model. These model are used to predict the voltages for Na, Mg, Ga, Zn, and Al-ion battery electrodes.
- the visulization of the CGCNN model. 
- The SVR, KRR, and RFR models used for comparation with the CGCNN model.

## Table of Contents

- [How to cite](#how-to-cite)
- [Prerequisites](#prerequisites)
- [Files](#Files)
  - [Define a customized dataset](#define-a-customized-dataset)
  - [Train a CGCNN model](#train-a-cgcnn-model)
  - [Predict material properties with a pre-trained CGCNN model](#predict-material-properties-with-a-pre-trained-cgcnn-model)
- [Authors](#authors)
- [License](#license)
- [Web tool](#web-tool)
## How to cite

Please cite the following work if you want to use this model.

```
@article{PhysRevLett.120.145301,
  title = {Accurately predicting voltage of electrode materials in metal-ion batteries using interpretable deep transfer learning},
  author = {Zhang Xiuying and Shen Lei},
  journal = {},
  volume = {},
  issue = {},
  pages = {},
  numpages = {},
  year = {2021},
  month = {},
  publisher = {},
  doi = {},
  url = {}
}
```

```
@article{PhysRevLett.120.145301,
  title = {Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties},
  author = {Xie, Tian and Grossman, Jeffrey C.},
  journal = {Phys. Rev. Lett.},
  volume = {120},
  issue = {14},
  pages = {145301},
  numpages = {6},
  year = {2018},
  month = {Apr},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevLett.120.145301},
  url = {https://link.aps.org/doi/10.1103/PhysRevLett.120.145301}
}
```

##  Prerequisites

This package requires:

- [PyTorch](http://pytorch.org)
- [scikit-learn](http://scikit-learn.org/stable/)
- [pymatgen](http://pymatgen.org)
- [numpy](http://numpy.org/)

If you are new to Python, the easiest way of installing the prerequisites is via [conda](https://conda.io/docs/index.html). After installing [conda](http://conda.pydata.org/), run the following command to create a new [environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) named `cgcnn` and install all prerequisites.

## Files
- CGCNN folder
The files in this folder just the same as the correspoiding files in the CGCNN model that Xie Tian et.al gives (http://github.com/txie-93/cgcnn).
- CGCNN_visulization
The files in this folder are used to have a visulization of the CGCNN model. 
The embedding_features.py is to visulize the features from the embedding layer in the CGCNN model.
The local_voltage_plt.py is to visulize the local voltages, which are obtained after the three convelutional layers.
The element_features.csv and OMO_local.csv are the data files that are used in the embedding_features.py and local_voltage_plt.py respectively. The element_features_csv.py and OMO_local_csv.py are the coresponding codes to get the two csv data files. 
The other files in this folder are the usefull files in the embedding_features.py and the local_voltage_plt.py. 
- Transfer_learning
The files in this folder are the main file for the model training for the prediction of Na, K, Mg, Ca, Zn, Y, and Al-ion battery electrod voltages respectively.
- Trained_model
Here are the trained models for the voltage predictions of the metal-ion battery electrodes.
- data
This folder contains the required data for the model training and the coresponding code to get these data files.
- SVR_KRR_RFR
Here are the SVR (Supporting Vector Regression), KRR (Kernel Ridge Regression), and RFR (Random Forest Regression) model that are used in our work. 

## Web-tool
A convenient web tool has been builed for the voltage prediction of all the metal-ion battery electrodes.


