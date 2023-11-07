# Predicting Game Evaluation from a Game Distribution Platform Data

## REPOSITORY DIRECTORY
```bash
├── WEB CRAWLING
|   ├── STEAM CRAWLING
|   ├── STEAM CHART CRAWLING
|   ├── STEAM SPY CRAWLING
|   └── CRAWLING DATA
├── EXPLORATORY DATA ANALYSIS
|   ├── DATA CORREALATION
|   ├── DATA OUTLIER
|   ├── CATEGORICAL VARIABLE PREPROCESSING
|   └── VISUALIZATION
├── MACHINE LEARNING
|   ├── TRAINING DATA PREPROCESSING
|   ├── BASE MODEL SELECTION
|   ├── HYPER PARAMETER TUNING
|   └── FINAL MODEL SELECTION
└── GAME REPORT
    └── REPORT
```
## About :
As the market size global gaming industry has rapidly grown to $196 billion dollars in 2022, experiencing a remarkable 41.29% increase over the past five years, the importance of predicting the success of a new publishing game has become increasingly evident in a highly competitive and dynamic gaming industry. Based on the game data collected from a number one game distribution platform STEAM, user reviews can be classified and used to provide valuable insights for making informed decisions in marketing, game updates, and more.

## Data : 

STEAMDB
https://steamdb.info/stats/gameratings/{year} \
STEAM store 
https://store.steampowered.com/app/{appid} https://store.steampowered.com/api/appdetails?appids={appid} \
STEAM spy
https://steamspy.com/api.php?request=appdetails&appid={appid} \
STEAM chart
https://steamcharts.com/app/{appid} \

## Results :
By hyper parameter tuning and voting top five models from PyCaret(LGBM, Gradient Boosting Classifier, Random Forest Classifier, Extra Trees Classifier, AdaBoost Classifier), the final model had Accuracy, Precision, Recall and F1-score by 3%p higher than the lowest score of evaluation.

## Feedbacks :
Using PyCaret may improve the model performance, but when starting to use the machine learning models, it is recommended to practice each model and understand how it works.

## What I learned :
Throughout the data analytics project, from choosing the topic to finalizing the model, I gained experience in creating datasets, preprocessing data, addressing class imbalance in the target variable, and tuning hyperparameters, while maintaining effective communication with team members.
