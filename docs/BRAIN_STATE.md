# 🧠 Kaze MoE — Estado Maestro del Proyecto (`BRAIN_STATE.md`)

> **Última actualización:** 2026-11-22  
> **Propósito:** Memoria persistente y autocontenida del proyecto para que cualquier sesión futura (humana o agente IA) tenga el 100% del contexto técnico, resultados verificados, lecciones críticas y próximos pasos sin depender del caché volátil.

---

## 1. Mapa de Repositorios y Artefactos Publicados

| Recurso | URL / Ruta | Contenido y Estado |
| :--- | :--- | :--- |
| **GitHub (Código)** | `https://github.com/Kineyo/Kaze-MoE` | Código fuente, arquitectura, scripts de rescate, upcycling, exportación y docs (`main`). |
| **Workspace local** | `/home/alex/Documentos/kaze-1b-a300m` | Repositorio local clonado/vinculado a `origin` (`Kineyo/Kaze-MoE`). |
| **App móvil** | `/home/alex/Kofu_Mobile` (`https://github.com/Kineyo/Kofu_Mobile`) | Ecosistema Android (`Proyecto Kofu`), rama experimental `dev-test` (cascada de hardware + UFS). |
| **Hugging Face #1 (Kaze Puro)** | `https://huggingface.co/shebxyd/Kaze-4.7B-A530M-MIT` | ✅ **Publicado:** 3 shards `BF16` (`Qwen2MoeForCausalLM`), `kaze-4.7b-a530m-mit-q4_k_m.gguf` (`2.86 GB`), `README.md` MIT. |
| **Hugging Face #2 (Phi4 MoE)** | `https://huggingface.co/shebxyd/Kaze-Phi4-MoE-MIT` | ✅ **Publicado:** `kaze_phi4_moe_routed_experts.safetensors` (`152.2 MB`), `kaze-phi4-moe-mit-q4_k_m.gguf` (`2.49 GB`), `hot_warm_cold_manifest.json`, `README.md` MIT. |
| **Google Drive (Checkpoints)** | `/content/drive/MyDrive/Kaze-4.7B-A530M_raw/` | `kaze_4.7b_a530m_raw.safetensors`, `clean_raw`, `v2_general_raw`, `rescued_raw` (`9.45 GB` c/u), `config.json`, `tokenizer.json`. |
| **Google Drive (Release HF)** | `/content/drive/MyDrive/Kaze-4.7B-A530M-MIT_HF_release/` | Respaldo completo del release `Kaze-4.7B-A530M-MIT` (3 shards + `Q4_K_M` GGUF). |
| **Google Drive (Phi4 MoE)** | `/content/drive/MyDrive/Kaze-Phi4-MoE-MIT/` | `kaze_phi4_moe_routed_experts.safetensors` (`152.2 MB`), `hot_warm_cold_manifest.json`, `README.md`. |

> **Nota de autenticación:** El namespace personal en Hugging Face es **`shebxyd`** (el token `HF_TOKEN` con permiso *Write* está guardado en **Colab Secrets**). La cuenta de GitHub es **`Kineyo`** (`gh` CLI autenticado en la máquina local). Nunca guardar tokens en archivos de este repo.

---

## 2. Qué Ya Se Hizo (Historial Verificado)

### A. Fase 1 — Rescate y Publicación de `Kaze-4.7B-A530M-MIT` (100% desde cero, MIT)
- **Arquitectura:** `23` capas, `d_model=1024`, `GQA 16Q/4KV`, `1 Shared + 31 Routed Experts (top_k=2, d_ff=2048 SwiGLU)`, `vocab=32,768` (Byte-Level BPE propio), `max_seq_len=4096`.
- **Evolución de entrenamiento:**
  1. **V1 (`clean_raw`):** Memorizó las 6 preguntas base, pero sufrió overfitting severo en preguntas nuevas (`#8` devolvía `def es_primo`, `#9` copiaba manzanas).
  2. **V2 (`v2_general_raw`, 3,000 pasos en A100-40GB, `val_loss=0.9254`):** Rompió la copia literal en `#9`, pero mantuvo el atractor `def es_primo` en `#8` y sub-entrenamiento en inglés (`18.4M` tokens totales vs `100B` que requiere un 4.7B desde cero).
  3. **Rescate (`rescued_raw`, 300 pasos con *Sequence Packing* + LoRA `r=32/8` fusionado lossless + 60 pasos finales `expert_bias=0` en L4, 40 min):**
     - `val_loss`: `2.5547` → **`0.7244`**
     - `max|expert_bias| = 0.000000` (conversión 100% *lossless* a `Qwen2MoeForCausalLM` y `GGUF`).
     - Destruyó el atractor `#8` (`def invertir_cadena(texto: str) -> str: return texto[::-1]`), arregló `EN-1` (`reverse_string`) y `EN-2` (`factorial`), y estabilizó `#7` y `#9`.
- **Conversión y Publicación:**
  - Convertido a `Qwen2MoeForCausalLM` nativo (sin `trust_remote_code`) en 3 shards `BF16` (`3.50 + 3.50 + 2.45 GB`) + cuantización `Q4_K_M` (`2.86 GB`).
  - Verificado con `transformers.AutoModelForCausalLM.from_pretrained(...)` antes de publicar en `shebxyd/Kaze-4.7B-A530M-MIT`.

### B. Fase 2 — Sparse Upcycling sin Pérdida (`Kaze-Phi4-MoE-MIT` + Hot/Warm/Cold UFS)
- **Base:** `microsoft/Phi-4-mini-instruct` (`3.837B`, licencia **MIT**, `32` capas, `hidden=3072`, `intermediate=8192`, `GQA 24Q/8KV`, `vocab=200,064`).
- **Arquitectura Hot / Warm / Cold:**
  - **HOT (RAM residente, congelado):** Backbone `Phi-4-mini` (`3.837B`) + Routers lineales (`32` capas).
  - **WARM (LRU RAM) / COLD (UFS streaming):** `6` expertos enrutados bottleneck (`3072 -> 64 -> 3072`, `top_k=2`, `W_down` inicializado en `0`) = `75.5M` params (`152.2 MB` en `BF16` / `~45.3 MB` en `Q4_K_M`).
- **Resultados de la Prueba de Presión Automatizada (10 tests nunca vistos):**
  - **Código Python ejecutable (`exec()` + `assert` reales):** **4 / 4 (100%)** ✅
  - **Aritmética y Finanzas `Kofu`:** **4 / 5 (80%)** ✅ *(0 alucinaciones; el 5to solo se cortó por límite de 140 tokens)*
  - **Aguja en un Pajar (`1,597` tokens):** **1 / 1 (100%)** ✅ *(extrajo `KAZE-9472-UFS` y `4,850 pesos` exactos; pico VRAM `+364 MB` KV-cache)*
  - **Balanceo de los 6 expertos:** **`[15.8%, 17.7%, 17.8%, 16.7%, 15.4%, 16.6%]`** (desviación ±1.3%, cero colapso).
  - **Memoria proyectada en móvil (`Q4_K_M`):** `~2.35 GB` pesos + `~250 MB` KV-cache ≈ **`~2.6 GB` RAM** (móviles de 6–8 GB RAM).
- **Publicación:** Subidos tanto los expertos modulares (`152.2 MB`) como el `GGUF Q4_K_M` standalone (`2.49 GB`) a `shebxyd/Kaze-Phi4-MoE-MIT`.

---

## 3. Lecciones Técnicas Críticas (¡NO REPETIR ESTOS ERRORES!)

1. **Doble desplazamiento de `labels` en Hugging Face `transformers`:**
   - `AutoModelForCausalLM(input_ids=xb, labels=xb)` **ya desplaza `labels` 1 token internamente**.
   - **NUNCA** pasar `input_ids=c[:-1], labels=c[1:]` a modelos HF (causa predicción `t+2`, `loss ~ 15` y texto saltado).
2. **FFN-Slicing vs Sparse Upcycling Aditivo en modelos `SwiGLU`:**
   - Cortar el 50% de las neuronas FFN de `Phi-4-mini` (FFN-Slicing) rompe la co-adaptación de `SwiGLU` (`loss` salta a `14.75`) y requiere miles de millones de tokens para recuperarse.
   - **Solución probada:** Sparse Upcycling aditivo (`Shared Backbone` congelado + `Routed Experts` con `W_down = 0` en paso 0) garantiza **0% degradación y 0% olvido catastrófico** en minutos de Colab.
3. **Conversión de Kaze BPE propio en `llama.cpp`:**
   - `convert_hf_to_gguf.py` lanza `NotImplementedError: BPE pre-tokenizer was not recognized` con hash `346f37258d58f836ab4dda03c3a7475fac9bedfa4eb37b4842391331db225e2b`.
   - **Solución:** Registrar ese hash retornando `"default"` en `llama.cpp/conversion/base.py` (`get_vocab_base_pre()`).
4. **Conversión de `Phi-4-mini` en `llama.cpp`:**
   - `tok.save_pretrained()` puede omitir `vocab.json` y `merges.txt`, haciendo que `llama.cpp/conversion/phi.py` busque `tokenizer.model` y falle.
   - **Solución:** Copiar `vocab.json` y `merges.txt` oficiales desde `microsoft/Phi-4-mini-instruct` a la carpeta fusionada antes de llamar a `convert_hf_to_gguf.py`.
5. **Costo real de pre-entrenar `Kaze-4.7B-A530M` desde cero con 100B tokens:**
   - Cómputo efectivo: $\approx 3.65 \times 10^{20}$ FLOPs → **~1.2 días en 8× H100 (FP8)** (`~$600–$800 USD`) o ~39 días en 1× A100 (no viable en Colab de 1 GPU).
6. **Presupuesto de RAM en Android con 3 GB físicos + ARM Mali:**
   - Android OS + GMS consumen `~1.85 GB`; LMK mata apps arriba de `~750 MB – 1.0 GB` RSS.
   - En Mali Gama Baja (`Mali-G52/G31/T*`), usar **CPU ARM64 `NEON + dotprod + KleidiAI` (`n_gpu_layers=0`)** y `Q4_0` / `Q4_K_M` evita duplicar buffers en RAM unificada con Vulkan.

---

## 4. Qué Falta Por Hacer (Roadmap Pendiente)

1. **Fase 3 — `Kaze-Mali-3GB` (Plan detallado en `docs/PLAN_MALI_3GB_RAM.md`):**
   - [ ] Decidir modelo base: **`Qwen2.5-0.5B-Instruct`** (`~350–380 MB` Q4, ultra-seguro en 3 GB RAM) vs **`Qwen2.5-1.5B-Instruct`** (`~750 MB` en `IQ3_XS` / `~950 MB` en `Q4_K_M`).
   - [ ] Aplicar Sparse Upcycling sin pérdida (`6` expertos `bottleneck=32`, `top_k=2`) enfocado en finanzas `Kofu_Mobile` + español conciso + aritmética.
   - [ ] Correr prueba de presión automatizada (10 tests + RSS pico < `600 MB`).
   - [ ] Exportar `Q4_0` (ARM KleidiAI/NEON) + `Q4_K_M` GGUF y publicar en Hugging Face (`shebxyd/Kaze-Mali-0.5B-MoE`).
2. **Fase 4 — Integración en `Kofu_Mobile` (`dev-test`):**
   - [ ] Configurar cascada automática de 3 niveles por RAM/GPU:
     - **Nivel 1 (3 GB RAM / Mali entrada):** `Kaze-Mali-0.5B` (`~350 MB`, CPU ARM64 `Q4_0`, `n_ctx=1536`, `KV=Q8_0`).
     - **Nivel 2 (4–6 GB RAM):** `Kaze-1.5B` (`~950 MB`, `Q4_K_M`).
     - **Nivel 3 (8+ GB RAM / Gama Alta / PC):** `Kaze-Phi4-MoE-MIT` (`2.49 GB`, `Q4_K_M`, 100% MIT).
   - [ ] Conectar `ComponentCallbacks2.onTrimMemory()` para liberar caché Warm y prevenir cierres por LMK.

---

## 5. Archivos Clave de Documentación en Esta Carpeta

- `docs/BRAIN_STATE.md` — Este archivo (memoria maestra del proyecto).
- `docs/PLAN_MAESTRO.md` — Plan maestro original de Fase 1 (Rescate Kaze 4.7B) + Fase 2 (Sparse Upcycling Phi-4-mini).
- `docs/PLAN_MALI_3GB_RAM.md` — Plan técnico completo para ARM Mali + 3 GB RAM.
- `docs/WALKTHROUGH_LOG.md` — Registro paso a paso de resultados y validaciones ejecutadas.
