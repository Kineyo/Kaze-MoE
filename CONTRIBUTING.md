# Contribuciones a Kaze-1B-A300M

Las contribuciones técnicas enfocadas en eficiencia, estabilidad numérica y portabilidad edge son bienvenidas.

## Áreas prioritarias

- **Exportación y cuantización GGUF / LiteRT:** Reportes de divergencia numérica entre `BF16` (`safetensors`) y cuantizaciones (`Q8_0`, `Q4_K_M`, `IQ4_XS`), o soporte actualizado para `llama.cpp`.
- **Pipeline de datos (`/dev/shm` `uint16`):** Mejoras de throughput en tokenización streaming, deduplicación y empaquetado de secuencias.
- **Kernels y entrenamiento MoE:** Optimizaciones en el despacho disperso de expertos, compatibilidad con `torch.compile` y métricas de balanceo *Aux-Loss-Free*.

## Cómo reportar un fallo

Abre un issue incluyendo:

1. Commit o versión exacta del repositorio y versión de PyTorch / `llama.cpp`.
2. Hardware (CPU/GPU, VRAM, RAM, sistema operativo).
3. Comando mínimo reproducible y traza de error completa.
4. En fallos de conversión GGUF: diferencia de logits o perplejidad observada frente al checkpoint `.safetensors`.

## Pull Requests

1. Haz fork del repositorio y crea una rama descriptiva (`fix/gguf-rope-freq`, `perf/moe-dispatch`).
2. Mantén los cambios acotados, sin dependencias innecesarias ni modificaciones de formato fuera de alcance.
3. Verifica que los scripts relevantes ejecutan sin errores antes de abrir el PR.
4. Todas las contribuciones se publican bajo la licencia **Apache 2.0** del proyecto.
