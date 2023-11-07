# Utilization of Image Review from Online Shopping Mall

## DIRECTORY
```
├── WEB CRAWLING
|   ├── STYLE REVIEW
|   └── PRODUCT
└──  DEEP LEARNING MODEL
    ├── BEST STYLE REVIEW CLASSIFICATION 
    └── RECOMMENDATION OF PRODUCT IN BEST STYLE REVIEW 
        ├── OBJECT DETECTION
        └── PRODUCT RECOMMENDATION
```
## About :
MUSINSA, a leading fashion community and sales platform in Korea, has a lot of active users who upload photo reviews of products. They receive an average of 26,000 daily uploads. Some users get curious about the items matched with the selling product. However, there isn’t much interaction among users. To enhance the customer expereince using the online shopping mall, matched items can be automatically searched in the shopping mall.

## Data :
https://global.musinsa.com/ca/main

## Results :
When combining Euclidean, Manhattan, and cosine similarity calculation methods with five different CNN-based models (VGG 16, VGG 19, InceptionV3, Inception and ResNet50), the VGG 19 model using the cosine method and the combination of Inception and ResNet50 models using the Manhattan method in a 1:1 ratio yielded the most relevant product search results.

## Feedbacks :
Considering the impact of background hue in the feature extraction proved beneficial. If clothing attributes can be incorporated into the image search model, it could increase the products sales.

## What I learned :
It is crucial to apply various models to the data I analyze and customize the most recent model to suit the data structure.