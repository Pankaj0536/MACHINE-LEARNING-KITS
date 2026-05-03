# importing the pipeline from transformers library
from transformers import pipeline

# Load a sentiment analysis pipeline with a sentiment-finetuned model
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

review = "The movie was easy to understand but was boring and the acting was superb!"

# Use the pipeline to classify the sentiment
result = sentiment_analyzer(review)

# Print the result
print(result)