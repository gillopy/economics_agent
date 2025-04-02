import os
import time
import json
import traceback
from typing import Any, Dict, List, Optional, Callable, Tuple
from functools import wraps

def timing_decorator(func):
    """
    Decorador para medir el tiempo de ejecución de una función
    Args:
        func: Función a decorar
    Returns:
        Función decorada que retorna el resultado original y el tiempo de ejecución
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000  # Convertir a milisegundos
        return result, duration_ms
    return wrapper

def safe_execute(func: Callable, *args, default_value: Any = None, **kwargs) -> Tuple[Any, Optional[Exception]]:
    """
    Ejecuta una función de forma segura y captura las excepciones
    Args:
        func: Función a ejecutar
        *args: Argumentos posicionales
        default_value: Valor por defecto a devolver en caso de error
        **kwargs: Argumentos por palabra clave
    Returns:
        Tupla con el resultado (o valor por defecto) y la excepción (o None)
    """
    try:
        return func(*args, **kwargs), None
    except Exception as e:
        return default_value, e

def ensure_dir_exists(directory: str) -> None:
    """
    Asegura que un directorio exista, creándolo si es necesario
    Args:
        directory: Ruta del directorio
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_json_file(file_path: str, default: Any = None) -> Any:
    """
    Carga un archivo JSON
    Args:
        file_path: Ruta del archivo
        default: Valor por defecto a devolver si el archivo no existe
    Returns:
        Contenido del archivo JSON o valor por defecto
    """
    if not os.path.exists(file_path):
        return default
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON from {file_path}: {str(e)}")
        return default

def save_json_file(data: Any, file_path: str) -> bool:
    """
    Guarda datos en un archivo JSON
    Args:
        data: Datos a guardar
        file_path: Ruta del archivo
    Returns:
        True si se guardó correctamente, False en caso contrario
    """
    try:
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Error saving JSON to {file_path}: {str(e)}")
        return False

def get_exception_details(e: Exception) -> Dict[str, str]:
    """
    Obtiene detalles de una excepción
    Args:
        e: Excepción
    Returns:
        Diccionario con detalles de la excepción
    """
    return {
        "error_type": e.__class__.__name__,
        "error_message": str(e),
        "traceback": traceback.format_exc()
    }

def truncate_text(text: str, max_length: int = 100, add_ellipsis: bool = True) -> str:
    """
    Trunca un texto a una longitud máxima
    Args:
        text: Texto a truncar
        max_length: Longitud máxima
        add_ellipsis: Si se debe añadir "..." al final del texto truncado
    Returns:
        Texto truncado
    """
    if len(text) <= max_length:
        return text
    
    truncated = text[:max_length]
    if add_ellipsis:
        truncated += "..."
    
    return truncated 