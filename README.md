# tweet-like-predictor
Predict how many likes your tweet ("X post") will get - at compose time.

# How to use

1. Clone this repo, `$ git clone https://github.com/gaborcselle/tweet-like-predictor`
2. Install dependencies, `$ pip install -r requirements.txt`
3. Download your Twitter archive from https://twitter.com/settings/account
4. Unzip the archive
5. Extract the tweets.js file from the archive, move it into the directory where you cloned this repo
6. Run the script, `$ python train_tweet_like_count_predictor.py` - this will train a model on your tweets
7. Prediction: `$ python predict_tweet_like_count.py "I love machine learning!"` - this will predict how many likes your tweet will get