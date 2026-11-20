---
license: mit
language:
  - es
  - en
library_name: transformers
tags:
  - moe
  - sparse-moe
  - qwen2_moe
  - edge-ai
  - llama-cpp
  - gguf
  - text-generation
  - kaze
  - mit
pipeline_tag: text-generation
---

# Kaze-4.7B-A530M-MIT

**Kaze-4.7B-A530M-MIT** es un modelo fundacional de lenguaje *Sparse Mixture-of-Experts* (MoE) entrenado desde cero por **Kineyo** (`@kineyo.io` / `Dev_Kofu`) bajo licencia **MIT**, optimizado para inferencia rápida y privada en dispositivos edge (CPU, Linux y Android).

---

## Arquitectura

Topología **DeepSeek-style MoE** convertida sin pérdida (*lossless*) al formato estándar `Qwen2MoeForCausalLM` para compatibilidad directa con `transformers` y `llama.cpp` (`GGUF`) sin requerir `trust_remote_code`:

| Parámetro | Valor |
| :--- | :--- |
| **Parámetros totales** | **~4.72B** |
| **Parámetros activos por token** | **~528M** |
| **Capas (`n_layers`)** | `23` |
| **Dimensión oculta (`d_model`)** | `1024` |
| **Atención GQA** | `16` Query heads / `4` KV heads (`head_dim = 64`) |
| **Expertos MoE** | `1` Shared Expert + `31` Routed Experts (`top_k = 2`) |
| **Dimensión FFN por experto (`d_ff`)** | `2048` (`SwiGLU`) |
| **Vocabulario** | `32,768` (Byte-Level BPE propio) |
| **Contexto nativo (`max_seq_len`)** | `4096` (`RoPE`, `theta = 100000.0`) |
| **Weight Tying** | `True` (`embed_tokens` = `lm_head`) |
| **Licencia** | **MIT** |

---

## Archivos Incluidos

| Archivo / Variante | Formato | Precisión | Descripción |
| :--- | :--- | :--- | :--- |
| `model-00001-of-00003.safetensors` (+ shards & `index.json`) | `safetensors` (sharded) | `BF16` | Pesos completos en crudo divididos en shards compatibles con Hugging Face `transformers` |
| `kaze-4.7b-a530m-mit-q4_k_m.gguf` | `GGUF` | `Q4_K_M` | Cuantización recomendada para `llama.cpp`, desktop CPU y dispositivos Android (`Kofu_Mobile`) |

---

## Uso Rápido

### Con `transformers` (Python)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "shebxyd/Kaze-4.7B-A530M-MIT"
tok = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

prompt = "<|user|>\nHola, ¿quién eres y para qué sirves?\n<|assistant|>\n"
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(
    **inputs,
    max_new_tokens=128,
    temperature=0.65,
    top_p=0.9,
    repetition_penalty=1.18,
    do_sample=True,
)
print(tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

### Con `llama.cpp` (`Q4_K_M`)

```bash
llama-cli \
  -m kaze-4.7b-a530m-mit-q4_k_m.gguf \
  -p "<|user|>\n¿Por qué es importante hacer un presupuesto personal?\n<|assistant|>\n" \
  -n 160 \
  --ctx-size 4096 \
  --temp 0.65 \
  --top-p 0.9 \
  --repeat-penalty 1.18 \
  -t 4
```

---

## Datos y Metodología de Entrenamiento

Entrenado exclusivamente con corpus bajo licencia **MIT** / permisiva:
- Pre-entrenamiento e instrucciones en español e inglés (finanzas personales, ciencia, historia, conversación general y código Python).
- Entrenamiento anti-overfitting con *Sequence Packing*, balanceo de carga *Aux-Loss-Free* y consolidación final con `expert_bias = 0` para garantizar paridad exacta en `Qwen2MoeForCausalLM` y `llama.cpp`.

---

## Limitaciones y Transparencia

- **Naturaleza experimental (entrenado desde cero):** Al haber sido entrenado desde cero con presupuesto académico/colaborativo (y no con trillones de tokens), el modelo está orientado a tareas acotadas en español, investigación de arquitecturas Sparse MoE en edge y prototipado dentro del ecosistema **Proyecto Kofu**.
- **Alucinaciones:** Puede cometer errores factuales o de razonamiento fuera de los dominios entrenados. No debe usarse como fuente única para decisiones críticas médicas, legales o financieras sin verificación humana.

---

## Citación

```bibtex
@misc{kineyo2026kaze47bmit,
  title        = {Kaze-4.7B-A530M-MIT: Sparse Mixture-of-Experts Model for Edge Inference},
  author       = {Kineyo and Dev\_Kofu},
  year         = {2026},
  organization = {Kineyo},
  howpublished = {\url{https://huggingface.co/shebxyd/Kaze-4.7B-A530M-MIT}},
  license      = {MIT}
}
```
