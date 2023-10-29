# Portfolio

According to the six degrees of separation, any person can be connected to any other person on the planet with less than five intermediaries. I believe that the product is one intermediate component that connects all people in the world. Fascinated by how product connects the world, I believe supply chain management can play various roles in the society.

The significance of data is highly valuable to both internal and external customers when optimizing distribution. Rather than regarding data as mere numbers, I seek meanings and stories from data in social context.

In terms of external customer, I have undertaken the following projects with the objective of improving the quality of service that can be provided.

## Project 1 - Customers Feedback on Overseas Online Shopping
* About :\
The first project I proceeded started with a question of what factor affects the customers' satisfaction the most when they are shopping online overseas. In Korea, there has been a sharp increase in the market size of Cross-Border E-commerce, which naturally led to an escalation of complaints received by Korea Consumer Agency. As the clothing was the product that the customer found the inconvenience the most, I conducted EDA of customers comments on men's T-shirt with a hypothesis that the difference of size caused the inconvenience.

* Data :\
11st amazon store > Fashion > Men's Clothing > Top \
https://amazon.11st.co.kr/amazon/category?dispCtgr3No=1150107

* Results :\
Size-mentioned negative comments were 1.8 times as many than positive comments and delivery-mentioned positive comments were twice as many than negative comments.

* Feedbacks :\
Some of women's clothing was mixed in the product list. If the data is preprocessed more thoroughly, the accuracy of the review will be higher.

Regarding the review data, it is frequently observed to be highly imbalanced between positive and negative reviews. This could be taken into the consideration for the next project.

* What I learned :\
Going through the pre step of data analysis, I found that data analysis needs to have a purpose that gives an insight to improve the status or solve the problem. Additionally, as I used NLTK and Sentiment Dictionary to deal with language data, I found that it is important to try new skills that I have not tried.

## Project 2 - Predicting Game Evaluation from a Game Distribution Platform Data 
* About :\
As the market size global gaming industry has rapidly grown to $196 billion dollars in 2022, experiencing a remarkable 41.29% increase over the past five years, the importance of predicting the success of a new publishing game has become increasingly evident in a highly competitive and dynamic gaming industry. Based on the game data collected from a number one game distribution platform STEAM, user reviews can be classified and used to provide valuable insights for making informed decisions in marketing, game updates, and more. 

* Data :\
1. STEAMDB \
https://steamdb.info/stats/gameratings/{year}
2. STEAM store \
https://store.steampowered.com/app/{appid}
https://store.steampowered.com/api/appdetails?appids={appid}
3. STEAM spy \
https://steamspy.com/api.php?request=appdetails&appid={appid}
4. STEAM chart \
https://steamcharts.com/app/{appid}

* Results :\
By hyper parameter tuning and voting top five models from PyCaret(LGBM, Gradient Boosting Classifier, Random Forest Classifier, Extra Trees Classifier, AdaBoost Classifier), the final model had Accuracy, Precision, Recall and F1-score by 3%p higher than the lowest score of evaluation.

* Feedbacks :\
Using PyCaret may improve the model performance, but when starting to use the machine learning models, it is recommended to practice each model and understand how it works. 


* What I learned :\
Throughout the data analytics project, from choosing the topic to finalizing the model, I gained experience in creating datasets, preprocessing data, addressing class imbalance in the target variable, and tuning hyperparameters, while maintaining effective communication with team members.


## Project 3 - Utilization of Image Review from Online Shopping Mall
* About :\
MUSINSA, a leading fashion community and sales platform in Korea, has a lot of active users who upload photo reviews of products. They receive an average of 26,000 daily uploads. Some users get curious about the items matched with the selling product. However, there isn’t much interaction among users. To enhance the customer expereince using the online shopping mall, matched items can be automatically searched in the shopping mall. 

* Data :\
https://global.musinsa.com/ca/main

* Results :\
When combining Euclidean, Manhattan, and cosine similarity calculation methods with five different CNN-based models (VGG 16, VGG 19, InceptionV3, Inception and ResNet50), the VGG 19 model using the cosine method and the combination of Inception and ResNet50 models using the Manhattan method in a 1:1 ratio yielded the most relevant product search results.

* Feedbacks :\
Considering the impact of background hue in the feature extraction proved beneficial. If clothing attributes can be incorporated into the image search model, it could increase the products sales. 

* What I learned :\
It is crucial to apply various models to the data I analyze and customize the most recent model to suit the data structure. 

## Project 4 - Forecasting Order of Frozen Food for Food Distribution Company

* About :\
In the supply chain management, it is important to place an appropriate amount of order to minimize the inventory cost and the loss of sales opportunity.  

* Data :\
Provdied by a food company

* Result :\
When compared to the conventional approach of using a 3-month moving average prediction, the final model showed an 11% higher accuracy rate difference.

* Feedback :\
Overall the project seems to have been proceeded based on the deep  understanding of time series data such as creating derived variables with lagging, rolling and differencing. 

* What I learned :\
Statistical knowledge applied to a model not only enhances the credibility of data analysis but also contributes to the improvement of model performance.
