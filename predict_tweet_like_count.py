import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the model
model = AutoModelForSequenceClassification.from_pretrained('./tweet-like-count-predictor-hf-model', num_labels=11)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased')

# Grab the tweet from the first command line argument
tweet = sys.argv[1]

# Tokenize the tweet
inputs = tokenizer(tweet, return_tensors="pt")

# Get the logits from the predicted labels
outputs = model(**inputs)
logits = outputs.logits

# Get the predicted label
predicted_label = logits.argmax().item()

# Calculate the softmax probabilities
softmax = torch.nn.Softmax(dim=1)

# Print out top predicted label, with logits
print(f'Predicted count: {predicted_label}, Softmax likelihood: {softmax(logits)[0][predicted_label]*100.0:.2f}%')

# Print out logits for remaining labels
for i, logit in enumerate(logits[0]):
    if i != predicted_label:
        print(f'Logit for count {i}, Softmax likelihood: {softmax(logits)[0][i]*100.0:.2f}%')

