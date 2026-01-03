# sea_ice_transformers

---

[![OSA-improved](https://img.shields.io/badge/improved%20by-OSA-yellow)](https://github.com/aimclub/OSA)

---

## Overview

This work focuses on improving forecasts of Arctic sea ice conditions using artificial intelligence. It investigates whether advanced 'transformer' models, often successful in areas like language processing, are truly the best approach for predicting changes in sea ice over time and across different regions – specifically the Kara, Barents, Laptev, East Siberian, and Chukchi seas. The core goal is to determine if simpler, more traditional methods based on analyzing patterns within images can actually outperform these complex models when dealing with data that naturally follows yearly cycles.

The code implements and tests various forecasting techniques, including different types of neural networks, using historical sea ice observations from 1979-2023. It automates the process of preparing this data for use by the AI models, training them to predict future conditions, and evaluating how accurate those predictions are. Ultimately, it aims to provide more reliable forecasts of sea ice extent, potentially aiding in navigation, climate monitoring, and understanding environmental changes in the Arctic.

---

## Repository content

The sea_ice_transformers repository focuses on comparing deep learning models for Arctic sea ice forecasting, challenging the assumption that transformers are always superior. The project's core components work together to train, evaluate, and compare different architectures. 

The data handling component involves loading historical sea ice concentration data from 1979-2020, with a test set of 2020-2023 across five Arctic seas. Scripts like `data/gen_synth_ts.py` are used for generating synthetic time series data to validate findings beyond real-world scenarios. The `cnn_forecaster_2d/data_loader.py` script is central to preparing this data for model input.

The modeling component includes implementations of several architectures: 2D CNNs (`cnn_forecaster_2d/train_cnn.py`), TimeSformer (`timesformer/train_transformer_ssplit.py`), SwinLSTM (`swinlstm/train.py`), and a 3D CNN (`cnn_forecaster_3d/train_cnn_3d.py`). These scripts contain the model definitions, training loops, and evaluation procedures.

The training and evaluation component utilizes PyTorch for building and training models. Scripts like `train_cnn.py`, `train_transformer_ssplit.py` and `train.py` handle the optimization process using optimizers like Adam or AdamW, loss functions such as L1Loss or MSELoss, and learning rate schedulers. Performance is assessed using metrics like Mean Absolute Error (MAE) and Structural Similarity Index (SSIM).

The utility components provide helper functions for tasks like counting model parameters, adjusting image sizes, and creating data loaders. These are found in files like `timesformer/train_transformer_ssplit.py` and `swinlstm/train.py`. 

Finally, the project includes scripts for visualizing results (e.g., loss curves) using Matplotlib, and saving trained model weights for future use. The PDF summary indicates that the core finding is CNNs outperform transformers on this specific task due to the periodic nature of sea ice data.

---

## Used algorithms

The codebase implements several algorithms for predicting sea ice concentration. These include: 

* **2D Convolutional Neural Networks (CNNs):** Used as a baseline and comparative model, CNNs identify patterns in the spatial arrangement of sea ice data to make predictions.
* **TimeSformer:** A transformer-based architecture designed to capture temporal dependencies within the sea ice data. It aims to understand how sea ice conditions change over time by considering relationships between different points in time.
* **3D CNNs:** These extend traditional 2D CNNs to also consider the temporal dimension, allowing them to learn spatio-temporal features directly from the data sequence.

The training process utilizes:

* **Adam and AdamW Optimizers:** Algorithms that adjust model parameters during training to minimize prediction errors.
* **Cosine Annealing Learning Rate Schedule:** A technique for gradually reducing the learning rate during training, helping the model converge more effectively.
* **L1Loss and Mean Squared Error (MSE):** Loss functions used to quantify the difference between predicted and actual sea ice concentrations. These guide the optimization process.

The evaluation relies on metrics like:

* **Mean Absolute Error (MAE):** Measures the average magnitude of errors in predictions.
* **Structural Similarity Index (SSIM):** Assesses the similarity between predicted and actual images, focusing on structural information.

---
