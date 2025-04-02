# Agente de Investigación Avanzado

Un agente basado en LLM con capacidades avanzadas de procesamiento, vectorización, búsqueda por similitud, corrección de texto y memoria conversacional.

## Características

- **Procesamiento de documentos**: Lee y procesa archivos CSV, PDF, TXT y JSON
- **Vectorización**: Convierte textos a vectores para búsqueda semántica
- **Búsqueda por similitud**: Encuentra información similar a una consulta
- **Corrección de texto**: Corrige errores ortográficos y gramaticales
- **Reescritura**: Mejora y reformatea texto
- **Memoria conversacional**: Mantiene el contexto de las conversaciones
- **Análisis de documentos**: Extrae entidades, resúmenes y puntos clave
- **Interfaz de línea de comandos**: Fácil de usar con diferentes comandos

## Requisitos

- Python 3.8 o superior
- Dependencias listadas en `requirements.txt`
- Clave API de Google AI (Gemini)

## Instalación

1. Clona el repositorio:
   ```
   git clone <url-del-repositorio>
   cd <directorio-del-repositorio>
   ```

2. Crea y activa un entorno virtual:
   ```
   python -m venv .venv
   # En Windows
   .venv\Scripts\activate
   # En Linux/MacOS
   source .venv/bin/activate
   ```

3. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```

4. Configura las variables de entorno:
   ```
   # Copia el archivo de ejemplo
   cp sample.env .env
   # Edita el archivo .env para agregar tu clave API
   ```

## Uso

### Modo Investigación

Para realizar una investigación sobre un tema:

```
python app.py research "¿Cuáles son los efectos del cambio climático?"
```

Para guardar los resultados en un archivo:

```
python app.py research "¿Cuáles son los efectos del cambio climático?" -o resultados.txt
```

### Modo Chat

Para iniciar una conversación interactiva:

```
python app.py chat
```

Para cargar una memoria de conversación previa:

```
python app.py chat -m memoria.json
```

### Análisis de Documentos

Para analizar un documento:

```
python app.py analyze documento.pdf
```

Para guardar el análisis en un archivo:

```
python app.py analyze documento.pdf -o analisis.txt
```

### Ingesta de Datos

Para ingestar un archivo en el sistema:

```
python app.py ingest archivo.csv
```

Para especificar el tipo de archivo manualmente:

```
python app.py ingest archivo.txt -t txt
```

### Listar Archivos Ingestados

Para ver todos los archivos que han sido ingestados:

```
python app.py list-ingested
```

## Estructura del Proyecto

```
.
├── app.py                  # Script de entrada principal
├── src/                    # Código fuente
│   ├── agents/             # Agentes para diferentes tareas
│   │   └── research_agent.py # Agente principal de investigación
│   ├── data/               # Almacenamiento de datos
│   │   ├── raw/            # Datos sin procesar
│   │   ├── processed/      # Datos procesados 
│   │   └── vectors/        # Vectores para búsqueda
│   ├── models/             # Modelos y esquemas
│   │   ├── ingestion.py    # Procesamiento de datos
│   │   ├── memory.py       # Gestión de memoria
│   │   └── schemas.py      # Esquemas Pydantic
│   ├── tools/              # Herramientas para el agente
│   │   ├── advanced_tools.py # Herramientas avanzadas
│   │   ├── file_tools.py   # Herramientas para archivos
│   │   ├── text_tools.py   # Herramientas para texto
│   │   └── vector_tools.py # Herramientas para vectores
│   ├── utils/              # Utilidades
│   │   ├── helpers.py      # Funciones auxiliares
│   │   └── logging.py      # Registro de actividad
│   ├── config.py           # Configuración global
│   └── main.py             # Lógica principal
├── logs/                   # Registros de actividad
├── .env                    # Variables de entorno
├── requirements.txt        # Dependencias
└── README.md               # Este archivo
```

## Configuración Avanzada

Puedes personalizar el comportamiento del agente editando el archivo `.env` con las siguientes variables:

- `LLM_MODEL`: Modelo de lenguaje a utilizar (por defecto: "gemini-1.5-pro")
- `LLM_TEMPERATURE`: Temperatura para la generación (por defecto: 0.2)
- `LLM_MAX_TOKENS`: Tokens máximos en la respuesta (por defecto: 8192)
- `EMBEDDING_MODEL`: Modelo para embeddings (por defecto: "all-MiniLM-L6-v2")
- `DEFAULT_MEMORY_TYPE`: Tipo de memoria a utilizar (por defecto: "buffer")
- `MEMORY_MAX_TOKEN_LIMIT`: Límite de tokens en memoria (por defecto: 2000)
- `CHUNK_SIZE`: Tamaño de fragmentos para vectorización (por defecto: 1000)
- `CHUNK_OVERLAP`: Superposición de fragmentos (por defecto: 200)

## Licencia

Este proyecto está licenciado bajo [Licencia MIT](LICENSE).