import ollama


available_models = ollama.list()
print("Available models:", available_models)