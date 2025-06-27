from llama_cpp import Llama

def generate_text(
    model_path,
    prompt,
    n_ctx=2048,
    n_threads=8,
    use_mlock=True,
    max_tokens=200,
    temperature=0.7,
    stop=None
):
    """
    Dynamically generates text using the Llama model.

    Parameters:
        model_path (str): Path to the model file.
        prompt (str): The input prompt for text generation.
        n_ctx (int): Context size for the model.
        n_threads (int): Number of CPU threads to use.
        use_mlock (bool): Whether to lock the model in RAM.
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.
        stop (list): List of stop sequences.

    Returns:
        str: Generated text.
    """
    # Load the model
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        use_mlock=use_mlock
    )

    # Generate text
    output = llm(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop)

    # Return the generated text
    return output["choices"][0]["text"]

