import os
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from pathlib import Path

# Cargar variables de entorno
load_dotenv()

# Rutas
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "src" / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTORS_DIR = DATA_DIR / "vectors"
LOGS_DIR = BASE_DIR / "logs"

# Asegurar que existan todos los directorios
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTORS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Configuración de LLM
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-pro")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "8192"))

# Configuración de embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Configuración de logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE_ENABLED = os.getenv("LOG_FILE_ENABLED", "True").lower() == "true"

# Configuración de memoria
DEFAULT_MEMORY_TYPE = os.getenv("DEFAULT_MEMORY_TYPE", "buffer")
MEMORY_MAX_TOKEN_LIMIT = int(os.getenv("MEMORY_MAX_TOKEN_LIMIT", "2000"))

# Configuración de vectorización
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Tipos de archivos soportados
SUPPORTED_FILE_TYPES = {
    "csv": "read_csv",
    "pdf": "read_pdf",
    "txt": "read_text",
    "json": "load_from_json"
}

class AgentConfig:
    """Configuración del agente"""
    
    def __init__(self, 
                 llm_model: str = LLM_MODEL,
                 llm_temperature: float = LLM_TEMPERATURE,
                 llm_max_tokens: int = LLM_MAX_TOKENS,
                 embedding_model: str = EMBEDDING_MODEL,
                 memory_type: str = DEFAULT_MEMORY_TYPE,
                 memory_max_token_limit: int = MEMORY_MAX_TOKEN_LIMIT,
                 chunk_size: int = CHUNK_SIZE,
                 chunk_overlap: int = CHUNK_OVERLAP):
        """
        Inicializa la configuración del agente
        """
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens
        self.embedding_model = embedding_model
        self.memory_type = memory_type
        self.memory_max_token_limit = memory_max_token_limit
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte la configuración a un diccionario
        """
        return {
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "llm_max_tokens": self.llm_max_tokens,
            "embedding_model": self.embedding_model,
            "memory_type": self.memory_type,
            "memory_max_token_limit": self.memory_max_token_limit,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AgentConfig':
        """
        Crea una configuración a partir de un diccionario
        """
        return cls(**config_dict)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> Optional['AgentConfig']:
        """
        Carga la configuración desde un archivo JSON
        """
        import json
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except Exception:
            return None
    
    def save_to_file(self, file_path: str) -> bool:
        """
        Guarda la configuración en un archivo JSON
        """
        import json
        try:
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            with open(file_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            return True
        except Exception:
            return False 