Actúa como un ingeniero principal de ML y desarrollador open-source. Genera la estructura completa de documentación, licencias y archivos de configuración para el lanzamiento del modelo fundacional ligero **Kaze-1B-A300M** (y su variante **Kaze-1B-A300M-Instruct**), diseñado para inferencia ultra-rápida en dispositivos edge (CPU, Linux y Android).

### Contexto Técnico del Proyecto
- **Nombre del Modelo:** `Kaze-1B-A300M` (Arquitectura Sparse Mixture-of-Experts entrenada desde cero).
- **Autor / Organización:** `Kineyo` (`@kineyo.io` / `Dev_Kofu`).
- **Parámetros:** ~1.02B totales | ~315M parámetros activos por token.
- **Topología MoE:** DeepSeek-style MoE con **1 Shared Expert** siempre activo + **7 Routed Experts** (activación **Top-2** por token) y balanceo de carga *Aux-Loss-Free*.
- **Arquitectura Base:** Transformer Decoder-only con `RMSNorm`, `SwiGLU`, `RoPE` (`theta=100000`, contexto de `4096` tokens), `GQA` (12 Query heads / 4 KV heads) y vocabulario propio Byte-Level BPE de `32,768` tokens.
- **Entrenamiento:** Pre-entrenado desde cero en PyTorch (`bfloat16`, `FlashAttention`, `torch.compile`, scheduler `WSD`) sobre NVIDIA A100 combinando bases de datos curadas propias y datasets públicos limpios.
- **Despliegue Edge:** Exportable a **GGUF** (`Q4_K_M`, `IQ4_XS`, `Q8_0`) para `llama.cpp` y compatible con `LiteRT` / dispositivos móviles sin GPU dedicada.

### Archivos a Generar
Crea directamente los siguientes archivos con formato profesional, limpio y sin relleno innecesario:

1. **`LICENSE` (Apache 2.0):**
   - Texto completo y estándar de la licencia **Apache License 2.0** (a nombre de `Kineyo`, año `2026`), permitiendo uso comercial, modificación y distribución sin restricciones propietarias.

2. **`README.md` (Para el repositorio de código en GitHub - entrenamiento y exportación):**
   - **Header & Badges:** Estilo minimalista con badges de PyTorch, Apache 2.0, Edge AI, GGUF y Hugging Face.
   - **Resumen de Arquitectura:** Tabla limpia con hiperparámetros (`d_model=768`, `n_layers=20`, `n_heads=12`, `n_kv_heads=4`, `d_ff=2816`, `1 Shared + 7 Routed Top-2`, `vocab_size=32768`).
   - **Estructura del Repositorio:** Árbol de directorios (`tokenizer/`, `model/`, `train/`, `export/`, `scripts/`).
   - **Guía Rápida (Quickstart):** Comandos exactos para preparar datos en `/dev/shm` (`uint16`), lanzar el pre-entrenamiento con scheduler WSD en A100 y exportar checkpoints `.safetensors` a `.gguf`.
   - **Mención de Ecosistema:** Nota breve indicando que sirve como motor neuronal local para el ecosistema *Proyecto Kofu* y *Hada*.

3. **`HF_MODEL_CARD.md` (Listo para usar como `README.md` en Hugging Face Hub):**
   - **YAML Frontmatter obligatorio de Hugging Face:** Incluye `license: apache-2.0`, `language: [es, en]`, `library_name: transformers`, `tags: [moe, edge-ai, llama-cpp, gguf, text-generation, kaze, custom_code]` y `pipeline_tag: text-generation`.
   - **Descripción del Modelo:** Explicación clara de por qué `1B total / 300M activos` ofrece velocidad de modelo nano con capacidad de retención de 1B.
   - **Tabla de Variantes y Cuantizaciones:** Espacio estructurado para los pesos base (`BF16`), `Q8_0`, `Q4_K_M` (~620 MB) e `IQ4_XS` (~540 MB).
   - **Ejemplos de Inferencia:** Snippets listos para copiar y pegar tanto en `llama.cpp` (`llama-cli` / `llama-server`) como en Python.
   - **Limitaciones, Sesgos y Uso Previsto:** Sección estándar de transparencia técnica para modelos fundacionales pequeños (contexto, razonamiento matemático complejo vs. eficiencia en dispositivo).
   - **Citaciones (BibTeX):** Bloque BibTeX listo para citar `Kaze-1B-A300M (2026)` de `Kineyo`.

4. **`CONTRIBUTING.md` y `SECURITY.md`:**
   - Guías breves y directas para reportar fallos en los scripts de conversión GGUF/quantization, proponer mejoras en el pipeline de datos y reportar vulnerabilidades.

### Restricciones de Estilo
- Tono técnico, sobrio y directo (estilo ingeniería de sistemas / DeepSeek / Karpathy).
- Cero emojis decorativos excesivos en la documentación técnica.
- Todo el código y comandos dentro de los Markdown deben ser sintácticamente válidos y consistentes entre sí.