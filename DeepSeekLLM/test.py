from llama_cpp import Llama

# Path to your DeepSeek-R1-Q2_K.gguf model
model_path = "model/DeepSeek-R1-q2_k.gguf"

# Load model using CPU only
llm = Llama(
    model_path=model_path,
    n_ctx=2048,        # Adjust context size as needed
    n_threads=8,       # Number of CPU threads to use
    use_mlock=True     # Locks model in RAM for better performance (optional)
)

# Prompt for generation
prompt = "Capital of India"

# Generate text
output = llm(prompt, max_tokens=200, temperature=0.7, stop=["</s>"])

# Print the generated text
print(output["choices"][0]["text"])
