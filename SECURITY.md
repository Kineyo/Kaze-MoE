# Política de Seguridad

## Versiones soportadas

| Componente | Estado |
| :--- | :--- |
| `Kaze-1B-A300M` (pesos y código `main`) | Soportado |
| Scripts de exportación GGUF / `safetensors` | Soportado |

## Alcance

Consideramos vulnerabilidades de seguridad dentro del alcance del proyecto:

- Ejecución arbitraria de código en scripts de carga, tokenización o exportación (`pickle`, deserialización insegura). Por diseño, los pesos se distribuyen exclusivamente en **`.safetensors`** y **`.gguf`**.
- Corrupción de memoria o desbordamientos en utilidades nativas o wrappers de conversión.
- Exfiltración involuntaria de credenciales (por ejemplo, tokens de Hugging Face) en scripts de preparación o entrenamiento.

Las salidas erróneas o alucinaciones propias del modelo de lenguaje se gestionan como issues técnicos de calidad/alineación y no como vulnerabilidades de seguridad.

## Reporte de vulnerabilidades

No publiques vulnerabilidades activas en issues públicos. Reporta el hallazgo de forma privada a través del canal de seguridad de **Kineyo** (`security@kineyo.io`) o mediante *GitHub Security Advisories* en el repositorio oficial, incluyendo:

- Descripción técnica del problema y componente afectado.
- Pasos para reproducirlo o prueba de concepto mínima.
- Impacto estimado y mitigación sugerida (si la conoces).

Confirmaremos la recepción en un plazo objetivo de 72 horas y coordinaremos la publicación del parche antes de la divulgación pública.
