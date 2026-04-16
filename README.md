# 🔥 LLM Loader
A clean, configurable loader for Hugging Face LLMs with automatic device mapping & quantization.

## usage in colab

!pip install -q git+https://github.com/sujaychandrakoyyalamudi/llm-loader.git

from llm_loader import ExplicitModelLoader

loader = ExplicitModelLoader(
    model_id=LOCAL_MODEL_PATH,  # 👈 Points to your custom folder
    hf_token=None,              # Not needed for local loading
    precision=PRECISION,
    device_mode=DEVICE_MODE,
    max_memory={"cuda:0": GPU_MEMORY, "cpu": CPU_MEMORY}
)
