from llama_stack_client import LlamaStackClient

# Connect to your local server
client = LlamaStackClient(base_url="http://localhost:8321")

# Fetch and print the list of available models
models = client.models.list()
print("Available models on the server:")
for m in models:
    print(f"– {m.identifier}")
