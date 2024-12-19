import anthropic

from llama.API_key import XAI_API_KEY
from functools import cache

@cache
def get_grok_client():
    client = anthropic.Anthropic(
        api_key=XAI_API_KEY,
        base_url="https://api.x.ai",
    )
    # client = anthropic.Completion.create(
    #     api_key=XAI_API_KEY,
    #     base_url="https://api.x.ai",
    # )
    return client

def grok_prompt(prompt, model='grok-2-1212', max_tokens=10,
                seed=0):
    """
    Process a prompt using the Anthropic API and return the response with token usage.

    :param prompt: The starting instructions or prompt
    :param model: The Anthropic model to use (default is Claude Haiku)
    :return: A dictionary containing the response and token information
    """
    # try:
        # Initialize the Anthropic client
        # Make sure to set your ANTHROPIC_API_KEY as an environment variable
        # client = anthropic.Anthropic()
    print('Running grok...')
    client = get_grok_client()

    # Create the message
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": prompt,
                'seed': seed,
            }
        ]
    )

    # Calculate token usage
    input_tokens = message.usage.input_tokens
    output_tokens = message.usage.output_tokens

    # Prepare the return dictionary
    result = {
        "response": message.content[0].text,
        "token_usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    }

    return result

# Example usage
if __name__ == "__main__":
    # prompt = "Write a short story about a robot discovering friendship."
    prompt = ("Write five sentences containing the words 'dog' and 'bank'. "
              "Separate each sentence with a number (1., 2., etc.). Do not "
              "include other text. One single sentence per line.")
    result = grok_prompt(prompt)

    print("Response:", result["response"])
    print("\nToken Usage:")
    print(f"Input Tokens: {result['token_usage']['input_tokens']}")
    print(f"Output Tokens: {result['token_usage']['output_tokens']}")
    # print(f"Total Tokens: {result['token_usage']['total_tokens']}")

