# Import GenerationConfig and pipeline from transformers library

#  GenerationConfig is a class that allows you to specify various parameters for text generation, such as the maximum number of new tokens to generate, the padding token ID, and whether to clean up tokenization spaces.

#  pipeline is a ready-made interface that lets you run ML models easily without handling low-level details.

from transformers import GenerationConfig, pipeline

# Using pipeline for text generation with the GPT - 2 model.
generator = pipeline("text-generation", model="gpt2")

# Setting up the genration configuration.
generation_config = GenerationConfig(
    max_new_tokens=50,
    pad_token_id=generator.tokenizer.eos_token_id,
    clean_up_tokenization_spaces=False,
)

# Initial prompt for text generation.
prompt = "Hello Pankaj,"
result = generator(
    prompt,
    generation_config=generation_config,
    clean_up_tokenization_spaces=False,
)

# Printing the result.
print(result)