# One Shot Prompting is one of the prompting technique where we add one example to explain the model how the output should look like..

# Import Pipline from transformers library
from transformers import pipeline

# Load a sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

prompt = """
Classify the sentiment of the following reviews as positive, negative, or neutral.

Review: This movie was amazing!
Sentiment: Positive

Review: The special effects were incredible, but the plot was weak.
Sentiment:
"""

# Use the pipeline to classify the sentiment
result = sentiment_analyzer(prompt)

# Print the result
print(result)