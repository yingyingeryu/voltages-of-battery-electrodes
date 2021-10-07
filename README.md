# voltages-of-battery-electrodes
Using the CGCNN transfer learning model to pridict the voltages of many kinds of metal-ion battery electrodes

The package provides all the files that are used in the article of ""
- The CGCNN model used to predict the voltages for Li-ion batttery electrodes. The details of the CGCNN framework are described in the article: [Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties](https://link.aps.org/doi/10.1103/PhysRevLett.120.145301), and the model is https://github.com/txie-93/cgcnn
- the transfer learning of the CGCNN model. These model are used to predict the voltages for Na, Mg, Ga, Zn, and Al-ion battery electrodes.
- the visulization of the CGCNN model. 
- The SVR, KRR, and RFR models used for comparation with the CGCNN model.

## Table of Contents

- [How to cite](#how-to-cite)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
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

## Usage
- The Usage of the CGCNN model and how to reparing the data for the model trainning, please take this github as reference:https://github.com/txie-93/cgcnn
- predict the transfer learning model for other metal-ion battery electrodes prediction.
Here take the model for the Na-ion battery electrode voltages as an example
python main_Na_2Lin.py --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 data/std_15
- use the trained model to do prediciton.
here we build a web tool: http://batteries.2dmatpedia.org/
or do the prediction as that in the CGCNN model


## Author
## Licence
## Web-tool


