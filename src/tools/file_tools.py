import os
import json
import pandas as pd
from pypdf import PdfReader
from langchain.tools import Tool
from typing import Dict, List, Any, Optional

class FileHandler:
    @staticmethod
    def read_csv(file_path: str) -> pd.DataFrame:
        """
        Lee un archivo CSV y lo convierte en DataFrame
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Error al leer archivo CSV {file_path}: {str(e)}")
    
    @staticmethod
    def read_pdf(file_path: str) -> List[str]:
        """
        Lee un archivo PDF y devuelve una lista con el texto de cada página
        """
        try:
            reader = PdfReader(file_path)
            pages = []
            for page in reader.pages:
                pages.append(page.extract_text())
            return pages
        except Exception as e:
            raise ValueError(f"Error al leer archivo PDF {file_path}: {str(e)}")
    
    @staticmethod
    def save_to_json(data: Any, file_path: str) -> str:
        """
        Guarda datos en un archivo JSON
        """
        try:
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return f"Datos guardados exitosamente en {file_path}"
        except Exception as e:
            raise ValueError(f"Error al guardar en JSON {file_path}: {str(e)}")
    
    @staticmethod
    def load_from_json(file_path: str) -> Any:
        """
        Carga datos desde un archivo JSON
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Error al cargar JSON {file_path}: {str(e)}")

# Definir herramientas de LangChain
csv_tool = Tool(
    name="read_csv",
    func=FileHandler.read_csv,
    description="Lee un archivo CSV y lo convierte en DataFrame. Input debe ser la ruta al archivo CSV."
)

pdf_tool = Tool(
    name="read_pdf",
    func=FileHandler.read_pdf,
    description="Lee un archivo PDF y devuelve una lista con el texto de cada página. Input debe ser la ruta al archivo PDF."
)

save_json_tool = Tool(
    name="save_to_json",
    func=FileHandler.save_to_json,
    description="Guarda datos en un archivo JSON. Requiere dos argumentos: los datos y la ruta del archivo."
)

load_json_tool = Tool(
    name="load_from_json",
    func=FileHandler.load_from_json,
    description="Carga datos desde un archivo JSON. Input debe ser la ruta al archivo JSON."
) 