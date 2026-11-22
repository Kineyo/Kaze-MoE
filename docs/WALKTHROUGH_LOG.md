# Walkthrough — Fase 1: Rescate y Empaquetado de `Kaze-4.7B-A530M-MIT`

## Resumen de Cambios Realizados

### 1. Código Local Actualizado
- [train/rescue_kaze.py](file:///home/alex/Documentos/kaze-1b-a300m/train/rescue_kaze.py): Pipeline de rescate con *Sequence Packing* (`seq_len=512`, 0% padding), destrucción de atractores Python ES/EN, aritmética verificada simbólicamente, LoRA de alto rango fusionable *lossless* y fase de consolidación con `expert_bias = 0`.
- [export/convert_kaze_to_qwen2moe.py](file:///home/alex/Documentos/kaze-1b-a300m/export/convert_kaze_to_qwen2moe.py): Corrección del `chat_template` al formato real de Kaze (`<|user|>` / `<|assistant|>` / `<|endoftext|>`) y soporte de guardado dividido (*sharded safetensors* + `model.safetensors.index.json`).
- [scripts/upload_to_hf.py](file:///home/alex/Documentos/kaze-1b-a300m/scripts/upload_to_hf.py): Script de publicación para `Kineyo/Kaze-4.7B-A530M-MIT`.
- [HF_MODEL_CARD.md](file:///home/alex/Documentos/kaze-1b-a300m/HF_MODEL_CARD.md): Model Card actualizada a `Kaze-4.7B-A530M-MIT` bajo licencia **MIT**.

### 2. Rescate en NVIDIA L4 (`300/300` pasos, `40.0 min`)
- **Checkpoint de entrada:** `kaze_4.7b_a530m_v2_general_raw.safetensors`
- **Checkpoint rescatado:** `/content/drive/MyDrive/Kaze-4.7B-A530M_raw/kaze_4.7b_a530m_rescued_raw.safetensors`
- **`val_loss`:** `2.5547` → **`0.7244`**
- **Consolidación de routing:** `max|expert_bias| = 0.000000` (paridad exacta con `Qwen2MoeForCausalLM` y `llama.cpp`).

### 3. Validación en 13 Pruebas (9 ES + 4 EN)
- **Atractor `#8` destruido:** `"Escribe una función en Python que invierta una cadena de texto"` ahora genera `def invertir_cadena(texto: str) -> str: return texto[::-1]` (antes devolvía `def es_primo`).
- **Código en inglés (`EN-1`, `EN-2`):** `reverse_string` y `factorial` 100% válidos y ejecutables.
- **Generalización ES (`#7`, `#9`):** respuestas estructuradas y coherentes sin alucinaciones de texto ajeno.

### 4. Bundle Hugging Face + GGUF `Q4_K_M`
- **Sharded Safetensors (`BF16`, `Qwen2MoeForCausalLM`):**
  - `model-00001-of-00003.safetensors` (`3.50 GB`)
  - `model-00002-of-00003.safetensors` (`3.50 GB`)
  - `model-00003-of-00003.safetensors` (`2.45 GB`)
  - `model.safetensors.index.json`, `config.json`, `tokenizer.json`, `tokenizer_config.json`, `README.md`
- **Verificación `transformers.AutoModelForCausalLM` (sin `trust_remote_code`):** ✅ Pasó correctamente en GPU.
- **Cuantización GGUF (`llama.cpp`):**
  - Se registró el hash del Byte-Level BPE propio de Kaze (`346f37258d58f836ab4dda03c3a7475fac9bedfa4eb37b4842391331db225e2b` → `"default"`).
  - Generado **`kaze-4.7b-a530m-mit-q4_k_m.gguf` (`2.86 GB`)**.
  - Respaldo en Google Drive: `/content/drive/MyDrive/Kaze-4.7B-A530M-MIT_HF_release`.

## Publicación Verificada
- ✅ **Publicado en Hugging Face:** [`https://huggingface.co/shebxyd/Kaze-4.7B-A530M-MIT`](https://huggingface.co/shebxyd/Kaze-4.7B-A530M-MIT) (3 shards `BF16` + `model.safetensors.index.json` + `kaze-4.7b-a530m-mit-q4_k_m.gguf` + `README.md` MIT).

## Fase 2 — Sparse Upcycling sin Pérdida (`Kaze-Phi4-MoE-MIT` + Hot/Warm/Cold UFS)

### Arquitectura Implementada
- **HOT (RAM residente, congelado):** `microsoft/Phi-4-mini-instruct` (3.8B, licencia **MIT**) como *Shared Backbone* → **0% olvido catastrófico** desde el paso 0.
- **HOT (Router):** `Linear(3072 -> 6)` por capa (`top_k = 2`) con balanceo *Aux-Loss-Free*.
- **WARM (LRU RAM) / COLD (UFS streaming):** `6` expertos enrutados por capa (`bottleneck = 64`, `76.1M` parámetros totales = **`152.2 MB` en `BF16`**, ~`40 MB` en `Q4`), inicializados con `W_down = 0`.
- **Artefactos publicados en Hugging Face (`shebxyd/Kaze-Phi4-MoE-MIT`):**
  - [`kaze_phi4_moe_routed_experts.safetensors`](https://huggingface.co/shebxyd/Kaze-Phi4-MoE-MIT/blob/main/kaze_phi4_moe_routed_experts.safetensors) (`152.2 MB`, `BF16` — 6 expertos enrutados + routers Hot/Warm/Cold UFS)
  - [`kaze-phi4-moe-mit-q4_k_m.gguf`](https://huggingface.co/shebxyd/Kaze-Phi4-MoE-MIT/blob/main/kaze-phi4-moe-mit-q4_k_m.gguf) (`2.49 GB`, cuantización `Q4_K_M` lista para `llama.cpp` / `Kofu_Mobile`)
  - `hot_warm_cold_manifest.json` + `README.md` (MIT)
- **Repositorio de código en GitHub:** [`https://github.com/Kineyo/Kaze-MoE`](https://github.com/Kineyo/Kaze-MoE)
- **Script local:** [scripts/upcycle_phi4_to_moe.py](file:///home/alex/Documentos/kaze-1b-a300m/scripts/upcycle_phi4_to_moe.py).

### Validación Final (`80` pasos en L4, `loss`: `3.83` → `0.58`)
- **Identidad (`#1`):** Se presenta como **`Kaze-Phi4-MoE`** (Kineyo, MIT, Hot/Warm/Cold sobre UFS para `Kofu Mobile`).
- **Ciencia (`#2`, `EN-3`) e Historia (`#3`):** 100% coherentes (fotosíntesis, Revolución Industrial, *Rayleigh scattering*) — **cero olvido catastrófico**.
- **Matemáticas (`#4`, `#9`):** `3 × 12 - 8 = 28` y `4 × 10 - 5 = 35` exactos paso a paso.
- **Código (`#5`, `#8`, `EN-1`, `EN-2`):** Python 100% válido y limpio (corrigió incluso el bug `si`/`mientras` que tenía `Phi-4-mini` base en español en `#5`).
- **Finanzas (`#6`, `#7`):** Regla 50/30/20, automatización, gastos hormiga y presupuesto personal claros y naturales.
