# Prepare Risk Dataset
This project has some work to do:
* train a base model, e.g., bert.
* calculate risk metrics with bert, textcnn,  statistics.
* calculate risk element with risk metrics through knn and ccd.

## Data Prepare
To run this project, you need prepare your data in {source_dataset}_bert/1.0 as follow:
```
AgNews_bert
 |-1.0- |-train.json # dataset from source dataset
 |      |-val.json  # dataset from target dataset
 |      |-test.json # dataset from source dataset
 |
 |-class.txt # class nums, each line represent a class
```


## Configure

To run this project, you need to configure the config folder.
```angular2
config
 |-overall_config.py  # configure for dataset prepare.
 |-train_config.py  # configure for train bert and textcnn model.
 |-get_feature_vector.py # configure for risk metrics with bert or textcnn
```
**Note**: 
1. You need to download the BERT model, stopwords, and pretrained_vector folders in advance and configure the specified file paths properly in ```overall_config.py```.
2. In ```get_feature_vector_config.py```, there are some fixed collocations as follows:

|            | best_model    | code_frame    | model_name |
| -----------| ------------- | ------------- | ---------- |
| **Bert**   | best_model.weights | bert4keras    | bert       | 
| **Textcnn**| bestmodel.pt  | pytorch       | textcnn    |

## Run

```
# train models
python train_bert.py # train the bert
python train_textcnn.py # train the textcnn

# calculate risk metrics
python get_feature_vector.py # calculate risk metrics with bert or textcnn
python chi_new.py

# calculate risk dataset
python get_risk_dataset.py
```
**Note**:
1. This project need risk metrics with bert and textcnn, you need to configure ```get_feature_vector.py``` and run the ```get_feature_vector.py``` twice.
2. The statistics-based risk metrics need to run after ```get_feature_vector.py```
3. The risk metrics will be saved in {target_dataset}_bert/{target_dataset}_cnn
4. The risk dataset will be saved in {source_dataset}_cur.

