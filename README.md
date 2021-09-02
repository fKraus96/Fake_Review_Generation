# fake_review_generation

### **Can generated text through a language model accurately predict the sentiment of a real restaurant review ? - A case study on high-end restaurants in the United States**


The purpose of this study is to gain a deeper understanding in the quality of language models, in particular we want to answer whether a language model can accurately capture varying moods of people around a certain topic. The study further develops a unique method for the verification of a language model by means of training a classification algorithm on a set of fake reviews and using it to predict the sentiment of a real review. 

In order to do so, several language models based on differing methods will be developed. Subsequently a generated sample of 4000 fake reviews for each of these language models will be manually labelled into the sentiment the review captures. This labeling step will at the same time provide a general understanding of the quality of the language model at hand. 

The language model with the most promising results will then be selected to run a sequence of sophisticated classification models trained on these fake reviews. A test set will be used based on a sample of 12000 real reviews. Finally the results of this classification analysis will be compared to a classifcation based on a training set of 4000 real reviews. This will provide the main basis for verification of the success of our language model to capture sentiment. 

#### The Dataset
The yelp review datset https://www.yelp.com/dataset was utilised for the research question at hand which consists of a sample of 8,6 million reviews in 8 metropolitan areas. After filtering our for all non-restaurant and low-mid price range restaurants, a subsample of 350k remain. High-end restaurants were decided upon as the center point of analysis due to arguably a difference in the type of customers across the different price ranges, which may use a different type of vocabular and style of writing. This sample is unsurprisingly biased towards the positive reviews. An early finding of this research was the issue of the language models to oversample these positive reviews, and hence it was decided to undersample the positive reviews to arrive at an even split of positive and negative reviews in the dataset. This allowed for a more accurate generation of the negative sentiment in the language model. Upon doing so, a remainder of 200k reviews are left. These are then split as follows: 180k for training the language models, 12k for testing in classification, 4k for training in classification.

Only this subset of 200k reviews can be found, both with the uncleaned, as well as cleaned text columns in the sql, in the Data folder. The following section will describe the various cleaning done on the data.
