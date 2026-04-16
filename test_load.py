from llm_loader import ExplicitModelLoader

# Initialize & load
loader = ExplicitModelLoader(
    model_id="meta-llama/Meta-Llama-3-8B-Instruct",
    precision="4bit",
    device_mode="auto"
)
model, tokenizer = loader.load()

print("✅ Model loaded successfully!")