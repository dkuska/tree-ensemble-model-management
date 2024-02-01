# Tree Ensemble Model Management

- The purpose of this repo is to explore model management possiblities when interacting with Tree-based Ensemble Methods like Random Forests
- The main optimization criteria include:
  - Time-to-Save
  - Time-to-Recover
  - Storage Consumption
- This is a university project, it is by no means a real project


## How to use this repo

1. Download data from: <https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data?select=yellow_tripdata_2016-01.csv>. The scripts assume that this CSV is stored in `project_root/data/yellow_tripdata_2016-01.csv`
2. Train models utilizing one of the training pipelines located in `project_root/training_pipelines`. You might want to do this multiple times adjusting the number of trees in each model. The scripts assume that these are stored in `project_root/models/`
3. Run Evaluation Scripts located in `project_root/model_agnostic` or `project_root/model_specific`


Copyright: David Kuska, 2023
