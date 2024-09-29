# RiskAnalysis
This project is an implement of transfer learning of text classification via risk analysis.

## Configure
You need to configure ```config.py``` in the folder named common, this file contains a class named ```Configuration```.

Configure as follows:
1. configure the data_dict and class_num_dict in Configuration.
2. configure source data and target data information in the head of file ```config.py```.
3. configure the function named ```get_params()``` in Configuration.

## Run
run the script ```python ada_train.py```

## Output
The final model will be saved in the folder {source_dataset}/1.0