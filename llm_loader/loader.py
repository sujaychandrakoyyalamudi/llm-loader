import os
import gc
import torch
from typing import Literal, Optional, Dict, Tuple
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class ExplicitModelLoader:
    def __init__(
        self,
        model_id: str,
        hf_token: Optional[str] = None,
        precision: Literal["4bit", "8bit", "fp16", "bf16", "fp32"] = "4bit",
        device_mode: Literal["auto", "single", "smart"] = "auto",
        max_memory: Optional[Dict] = None,
        trust_remote_code: bool = True,
        cache_dir: Optional[str] = None
    ):
        self.model_id = model_id
        self.hf_token = hf_token
        self.precision = precision
        self.device_mode = device_mode
        self.max_memory = max_memory
        self.trust_remote_code = trust_remote_code
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None

    def clear_vram(self):
        gc.collect()
        torch.cuda.empty_cache()

    def _authenticate(self):
        if self.hf_token:
            login(token=self.hf_token, add_to_git_credential=True)
            print("🔑 Hugging Face authenticated")
        else:
            print("⚠️ No HF token. Open models only.")

    def _resolve_device(self):
        """Automatically picks the best device based on mode."""
        if self.device_mode == "auto":
            return "auto"  # accelerate handles placement
        elif self.device_mode == "single":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif self.device_mode == "smart":
            if not torch.cuda.is_available():
                return "cpu"
            n_gpus = torch.cuda.device_count()
            if n_gpus == 1:
                return "cuda:0"
            # Pick GPU with most free VRAM
            free_mem = [torch.cuda.mem_get_info(i)[0] for i in range(n_gpus)]
            best_idx = free_mem.index(max(free_mem))
            return f"cuda:{best_idx}"
        else:
            raise ValueError(f"Invalid device_mode: {self.device_mode}")

    def load(self) -> Tuple:
        self._authenticate()
        self.clear_vram()
        print(f"📦 Loading tokenizer: {self.model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            token=self.hf_token,
            trust_remote_code=self.trust_remote_code,
            cache_dir=self.cache_dir
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        device = self._resolve_device()
        print(f"🧠 Loading model with precision: {self.precision} | Device: {device}")

        model_kwargs = {
            "token": self.hf_token,
            "trust_remote_code": self.trust_remote_code,
            "device_map": device if isinstance(device, str) else device.type,
            "cache_dir": self.cache_dir
        }
        if self.max_memory:
            model_kwargs["max_memory"] = self.max_memory

        if self.precision in ["4bit", "8bit"]:
            model_kwargs["quantization_config"] = (
                BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
                if self.precision == "4bit"
                else BitsAndBytesConfig(load_in_8bit=True)
            )
        else:
            dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
            model_kwargs["torch_dtype"] = dtype_map[self.precision]

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)

        param_count = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"✅ Loaded: {self.model_id}")
        print(f"✅ Precision: {self.precision} | Params: {param_count:.1f}M")
        print(f"✅ Active Device: {self.model.device} | Dtype: {self.model.dtype}")
        return self.model, self.tokenizer