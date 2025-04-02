import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List

class AgentLogger:
    """Clase para registrar la actividad del agente"""
    
    def __init__(self, log_dir: str = "logs", level: int = logging.INFO):
        """
        Inicializa el logger
        Args:
            log_dir: Directorio donde guardar los logs
            level: Nivel de logging
        """
        self.log_dir = log_dir
        
        # Crear directorio si no existe
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Configurar logger
        self.logger = logging.getLogger("agent")
        self.logger.setLevel(level)
        
        # Crear manejador para archivos
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"agent_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        
        # Formato
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Agregar manejador al logger
        self.logger.addHandler(file_handler)
        
        # Agregar también un manejador para la consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        
        # Historial detallado para guardar en JSON
        self.history: List[Dict[str, Any]] = []
    
    def log_tool_use(self, tool_name: str, input_data: Any, output_data: Any, 
                    duration_ms: Optional[float] = None) -> None:
        """
        Registra el uso de una herramienta
        Args:
            tool_name: Nombre de la herramienta
            input_data: Datos de entrada
            output_data: Datos de salida
            duration_ms: Duración en milisegundos
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "tool_use",
            "tool_name": tool_name,
            "input": input_data,
            "output": output_data,
            "duration_ms": duration_ms
        }
        
        self.history.append(entry)
        self.logger.info(f"Tool: {tool_name} - Input: {input_data} - Duration: {duration_ms}ms")
    
    def log_user_message(self, message: str) -> None:
        """
        Registra un mensaje del usuario
        Args:
            message: Mensaje del usuario
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "user_message",
            "message": message
        }
        
        self.history.append(entry)
        self.logger.info(f"User: {message}")
    
    def log_ai_message(self, message: str) -> None:
        """
        Registra un mensaje del agente
        Args:
            message: Mensaje del agente
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "ai_message",
            "message": message
        }
        
        self.history.append(entry)
        self.logger.info(f"AI: {message}")
    
    def log_error(self, error_message: str, error_type: Optional[str] = None, 
                 traceback: Optional[str] = None) -> None:
        """
        Registra un error
        Args:
            error_message: Mensaje de error
            error_type: Tipo de error
            traceback: Traceback del error
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "error",
            "error_message": error_message,
            "error_type": error_type,
            "traceback": traceback
        }
        
        self.history.append(entry)
        self.logger.error(f"Error ({error_type}): {error_message}")
        if traceback:
            self.logger.debug(f"Traceback: {traceback}")
    
    def log_system_event(self, event_type: str, details: Any) -> None:
        """
        Registra un evento del sistema
        Args:
            event_type: Tipo de evento
            details: Detalles del evento
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "system_event",
            "event_type": event_type,
            "details": details
        }
        
        self.history.append(entry)
        self.logger.info(f"System: {event_type} - {details}")
    
    def save_history(self, file_path: Optional[str] = None) -> str:
        """
        Guarda el historial en un archivo JSON
        Args:
            file_path: Ruta del archivo donde guardar el historial
        Returns:
            Ruta donde se guardó el historial
        """
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(self.log_dir, f"history_{timestamp}.json")
        
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
        
        return file_path 