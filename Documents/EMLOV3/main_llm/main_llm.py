import openai

# Set your OpenAI API key here (use environment variables in production)
openai.api_key = "your-openai-api-key"

def process_query_with_llm(query, context):
    """
    Use OpenAI's API to process a query with additional context.

    :param query: The user's query as a string.
    :param context: Additional context to help the LLM understand the query.
    :return: The LLM's response as a string.
    """
    try:
        # Combine the context and the query into a single prompt
        prompt = f"{context}\n\n{query}"

        # Call the OpenAI API
        response = openai.Completion.create(
            engine="text-davinci-003",  # or "text-davinci-002" for GPT-3.5, adjust as needed
            prompt=prompt,
            temperature=0.7,  # Adjust based on how deterministic you want the responses to be
            max_tokens=150,  # Adjust based on the expected length of the response
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )

        # Extract the text from the response
        response_text = response.choices[0].text.strip()
        return response_text
    except Exception as e:
        print(f"An error occurred: {e}")
        return "I'm sorry, I couldn't process your request."

def handle_query(query, context=""):
    """
    Main function to handle queries using the main LLM integration.

    :param query: The user's query as a string.
    :param context: Optional context to help the LLM understand the query.
    :return: The LLM's response as a string.
    """
    response = process_query_with_llm(query, context)
    return response
