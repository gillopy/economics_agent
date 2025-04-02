import os
import json
import uuid
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path

from ..tools.file_tools import FileHandler
from ..tools.vector_tools import get_vector_handler
from ..config import RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTORS_DIR, SUPPORTED_FILE_TYPES
from ..utils.helpers import ensure_dir_exists, save_json_file

class DataIngestion:
    """Clase para la ingesta de datos desde diferentes fuentes"""
    
    def __init__(self):
        """Inicializa el gestor de ingesta de datos"""
        self.file_handler = FileHandler()
        self.vector_handler = get_vector_handler()
        
        # Asegurar que existan los directorios necesarios
        ensure_dir_exists(RAW_DATA_DIR)
        ensure_dir_exists(PROCESSED_DATA_DIR)
        ensure_dir_exists(VECTORS_DIR)
    
    def ingest_file(self, file_path: str, file_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Ingesta un archivo y lo procesa
        Args:
            file_path: Ruta del archivo a ingestar
            file_type: Tipo de archivo (si es None, se infiere de la extensión)
        Returns:
            Diccionario con metadatos del archivo ingestado
        """
        # Verificar que el archivo existe
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo {file_path} no existe")
        
        # Inferir tipo de archivo si no se especifica
        if file_type is None:
            file_extension = os.path.splitext(file_path)[1].lower().lstrip('.')
            if file_extension in SUPPORTED_FILE_TYPES:
                file_type = file_extension
            else:
                raise ValueError(f"Tipo de archivo no soportado: {file_extension}")
        
        # Verificar que el tipo es soportado
        if file_type not in SUPPORTED_FILE_TYPES:
            raise ValueError(f"Tipo de archivo no soportado: {file_type}")
        
        # Generar identificador único
        file_id = str(uuid.uuid4())
        
        # Obtener nombre del archivo original
        file_name = os.path.basename(file_path)
        
        # Guardar una copia en el directorio de datos crudos
        raw_file_path = os.path.join(RAW_DATA_DIR, f"{file_id}_{file_name}")
        import shutil
        shutil.copy2(file_path, raw_file_path)
        
        # Procesar el archivo según su tipo
        content = self._process_file(file_path, file_type)
        
        # Guardar el contenido procesado
        processed_file_path = os.path.join(PROCESSED_DATA_DIR, f"{file_id}.json")
        metadata = {
            "file_id": file_id,
            "original_file_name": file_name,
            "original_file_path": file_path,
            "raw_file_path": raw_file_path,
            "processed_file_path": processed_file_path,
            "file_type": file_type,
            "ingestion_timestamp": datetime.now().isoformat(),
            "size_bytes": os.path.getsize(file_path)
        }
        
        processed_data = {
            "metadata": metadata,
            "content": content
        }
        
        save_json_file(processed_data, processed_file_path)
        
        # Vectorizar el contenido si es texto
        if isinstance(content, str) or (isinstance(content, list) and all(isinstance(item, str) for item in content)):
            texts = content if isinstance(content, list) else [content]
            vector_dir = os.path.join(VECTORS_DIR, file_id)
            
            # Preparar metadatos para cada chunk
            metadatas = [{
                "file_id": file_id,
                "file_name": file_name,
                "chunk_index": i
            } for i in range(len(texts))]
            
            # Vectorizar y guardar
            self.vector_handler.vectorize_texts(texts, metadatas)
            self.vector_handler.save_vector_store(vector_dir)
            
            # Actualizar metadatos
            metadata["vector_dir"] = vector_dir
        
        return metadata
    
    def ingest_text(self, text: str, source_name: str) -> Dict[str, Any]:
        """
        Ingesta texto directamente
        Args:
            text: Texto a ingestar
            source_name: Nombre o identificador de la fuente
        Returns:
            Diccionario con metadatos del texto ingestado
        """
        # Generar identificador único
        text_id = str(uuid.uuid4())
        
        # Generar nombre de archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{source_name}_{timestamp}.txt"
        
        # Guardar texto en datos crudos
        raw_file_path = os.path.join(RAW_DATA_DIR, file_name)
        with open(raw_file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Guardar en datos procesados
        processed_file_path = os.path.join(PROCESSED_DATA_DIR, f"{text_id}.json")
        metadata = {
            "text_id": text_id,
            "source_name": source_name,
            "raw_file_path": raw_file_path,
            "processed_file_path": processed_file_path,
            "file_type": "text",
            "ingestion_timestamp": datetime.now().isoformat(),
            "size_bytes": len(text.encode('utf-8'))
        }
        
        processed_data = {
            "metadata": metadata,
            "content": text
        }
        
        save_json_file(processed_data, processed_file_path)
        
        # Vectorizar el texto
        vector_dir = os.path.join(VECTORS_DIR, text_id)
        
        # Vectorizar y guardar
        vector_store = self.vector_handler.vectorize_texts([text], [{
            "text_id": text_id,
            "source_name": source_name
        }])
        self.vector_handler.save_vector_store(vector_store, vector_dir)
        
        # Actualizar metadatos
        metadata["vector_dir"] = vector_dir
        
        return metadata
    
    def _process_file(self, file_path: str, file_type: str) -> Any:
        """
        Procesa un archivo según su tipo
        Args:
            file_path: Ruta del archivo
            file_type: Tipo de archivo
        Returns:
            Contenido procesado
        """
        if file_type == "csv":
            df = self.file_handler.read_csv(file_path)
            return json.loads(df.to_json(orient="records"))
        elif file_type == "pdf":
            return self.file_handler.read_pdf(file_path)
        elif file_type == "txt":
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_type == "json":
            return self.file_handler.load_from_json(file_path)
        else:
            raise ValueError(f"Tipo de archivo no soportado para procesamiento: {file_type}")
    
    def list_ingested_files(self) -> List[Dict[str, Any]]:
        """
        Lista todos los archivos ingestados
        Returns:
            Lista de metadatos de archivos ingestados
        """
        files = []
        for file_name in os.listdir(PROCESSED_DATA_DIR):
            if file_name.endswith('.json'):
                file_path = os.path.join(PROCESSED_DATA_DIR, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if "metadata" in data:
                            files.append(data["metadata"])
                except Exception:
                    continue
        return files 