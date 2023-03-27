# RPS_classification
  * This project was previously a part of CSC340 Artificial Intelligence class.

  * We used Keras framework with GPU to develop our model using Google Colaboratory.

## Datasets
* RPS dataset was acquired from [Rock Scissor Paper](https://www.kaggle.com/alishmanandhar/rock-scissor-paper)

  * After that, we shuffle the dataset and randomly split using 80:10:10 ratios with [dataset-split](https://github.com/muriloxyz/dataset-split)

## Model
- [EfficientNetB3](https://github.com/qubvel/efficientnet)

## Results
|Set|Loss|Accuracy|
|:--|:--|:--|
|**Train**|0.0329|0.9927|
|**Valid**|0.0487|0.9958|
|**Test**|-|0.9917|

<img src="https://raw.githubusercontent.com/Kawaeee/RPS_classification/master/training_graph.jpg?token=AGJARVM7XJHE72OKHBUQLJC56RVPC" width="500" height="500">

#### You can download our model weight here: [v1.0](https://github.com/Kawaeee/RPS_classification/releases/download/v1.0/RPS_efficientnet_10.h5)

## Hyperparameters and configurations

| Configuration | Value |
|:--|:--|
|Epoch | 10 |
|Batch Size | 16 |
|Optimizer | ADAM |

## Reproduction
 
 - Install dependencies
    ```Bash
    pip install -r requirements.txt
    ```
    
 - Run the train.py python script
 
    ```Bash
    python train.py 
    ```
    
 - Open and run the notebook for prediction: `predictor.ipynb`
