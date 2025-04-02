from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

class MemoryItem(BaseModel):
    """Modelo para un ítem de memoria"""
    query: str
    response: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

class AgentMemory:
    """Gestiona la memoria del agente para conversaciones multi-turno"""
    
    def __init__(self, memory_type: str = "buffer", max_token_limit: int = 2000):
        """
        Inicializa la memoria del agente
        Args:
            memory_type: Tipo de memoria ('buffer' o 'summary')
            max_token_limit: Límite máximo de tokens (para summary)
        """
        self.memory_type = memory_type
        
        if memory_type == "buffer":
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        elif memory_type == "summary":
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
            self.memory = ConversationSummaryMemory(
                llm=llm,
                memory_key="chat_history",
                max_token_limit=max_token_limit,
                return_messages=True
            )
        else:
            raise ValueError(f"Tipo de memoria no soportado: {memory_type}")
        
        # Historial adicional para almacenar con metadatos
        self.history: List[MemoryItem] = []
    
    def add_user_message(self, message: str) -> None:
        """Añade un mensaje del usuario a la memoria"""
        self.memory.chat_memory.add_user_message(message)
    
    def add_ai_message(self, message: str) -> None:
        """Añade un mensaje del AI a la memoria"""
        self.memory.chat_memory.add_ai_message(message)
    
    def add_interaction(self, query: str, response: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Añade una interacción completa a la memoria
        """
        import datetime
        timestamp = datetime.datetime.now().isoformat()
        
        # Añadir a la memoria de LangChain
        self.memory.chat_memory.add_user_message(query)
        self.memory.chat_memory.add_ai_message(response)
        
        # Añadir a nuestro historial con metadatos
        self.history.append(
            MemoryItem(
                query=query,
                response=response,
                timestamp=timestamp,
                metadata=metadata
            )
        )
    
    def get_chat_history(self) -> Any:
        """Obtiene el historial de chat formateado para LangChain"""
        return self.memory.load_memory_variables({})["chat_history"]
    
    def clear(self) -> None:
        """Limpia toda la memoria"""
        self.memory.clear()
        self.history = []
    
    def save_to_file(self, file_path: str) -> None:
        """Guarda la memoria en un archivo JSON"""
        import json
        with open(file_path, 'w') as f:
            json.dump([item.dict() for item in self.history], f, indent=2)
    
    def load_from_file(self, file_path: str) -> None:
        """Carga la memoria desde un archivo JSON"""
        import json
        import datetime
        
        # Limpiar memoria actual
        self.clear()
        
        with open(file_path, 'r') as f:
            items = json.load(f)
            
        # Cargar en ambas memorias
        for item in items:
            memory_item = MemoryItem(**item)
            self.add_interaction(memory_item.query, memory_item.response, memory_item.metadata)
            
    def get_relevant_history(self, query: str, k: int = 3) -> List[MemoryItem]:
        """
        Obtiene los k elementos de memoria más relevantes para la consulta
        Implementación simple basada en recencia
        """
        # Aquí se podría implementar una búsqueda semántica con embeddings
        # Por ahora solo devolvemos los k más recientes
        return self.history[-k:] 