from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class Source(BaseModel):
    """Modelo para una fuente de información"""
    url: Optional[str] = None
    title: Optional[str] = None
    content: Optional[str] = None
    type: str = Field(description="Tipo de fuente: 'web', 'document', 'database', etc.")
    metadata: Optional[Dict[str, Any]] = None

class AgentAction(BaseModel):
    """Modelo para una acción realizada por el agente"""
    tool: str = Field(description="Nombre de la herramienta utilizada")
    input: Any = Field(description="Entrada proporcionada a la herramienta")
    output: Any = Field(description="Salida de la herramienta")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class ResearchResponse(BaseModel):
    """Modelo para una respuesta de investigación"""
    topic: str = Field(description="Tema de la investigación")
    summary: str = Field(description="Resumen de la investigación")
    sources: List[Source] = Field(default_factory=list, description="Fuentes consultadas")
    tools_used: List[str] = Field(default_factory=list, description="Herramientas utilizadas")

class ChatResponse(BaseModel):
    """Modelo para una respuesta de chat"""
    response: str = Field(description="Respuesta al usuario")
    actions: List[AgentAction] = Field(default_factory=list, description="Acciones realizadas")
    sources: List[Source] = Field(default_factory=list, description="Fuentes consultadas")
    confidence: float = Field(default=0.0, description="Nivel de confianza (0-1)")

class DocumentAnalysisResponse(BaseModel):
    """Modelo para una respuesta de análisis de documento"""
    document_name: str = Field(description="Nombre del documento analizado")
    summary: str = Field(description="Resumen del documento")
    key_points: List[str] = Field(default_factory=list, description="Puntos clave")
    entities: Dict[str, List[str]] = Field(default_factory=dict, description="Entidades extraídas")
    sentiment: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None 