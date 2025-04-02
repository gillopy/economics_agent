from typing import List, Dict, Any, Optional, Union
import time
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.callbacks import BaseCallbackHandler

from ..tools import all_tools
from ..models import ResearchResponse, ChatResponse, DocumentAnalysisResponse, AgentMemory
from ..utils import AgentLogger, timing_decorator, safe_execute, get_exception_details
from ..config import AgentConfig

class ResearchAgent:
    """
    Agente de investigación avanzado con capacidades de procesamiento de documentos,
    vectorización, búsqueda por similitud y memoria
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Inicializa el agente de investigación
        Args:
            config: Configuración del agente (opcional)
        """
        # Configuración
        self.config = config or AgentConfig()
        
        # Configurar LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            max_output_tokens=self.config.llm_max_tokens
        )
        
        # Parser de salida para respuestas estructuradas
        self.research_parser = PydanticOutputParser(pydantic_object=ResearchResponse)
        self.chat_parser = PydanticOutputParser(pydantic_object=ChatResponse)
        self.doc_parser = PydanticOutputParser(pydantic_object=DocumentAnalysisResponse)
        
        # Configurar memoria
        self.memory = AgentMemory(
            memory_type=self.config.memory_type,
            max_token_limit=self.config.memory_max_token_limit
        )
        
        # Configurar logger
        self.logger = AgentLogger(log_dir="logs")
        
        # Prompts para diferentes tipos de tareas
        self._setup_prompts()
        
        # Configurar agentes
        self._setup_agents()
    
    def _setup_prompts(self):
        """Configura los prompts para diferentes tipos de tareas"""
        
        # Prompt para investigación
        self.research_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                Eres un asistente de investigación que ayuda a generar reportes de investigación.
                Responde la consulta del usuario y utiliza las herramientas necesarias.
                Debes usar todas las herramientas que sean relevantes para la consulta.
                
                Para búsquedas web, usa la herramienta 'search'.
                Para información de Wikipedia, usa la herramienta 'wiki_tool'.
                Para leer archivos CSV, usa la herramienta 'read_csv'.
                Para leer archivos PDF, usa la herramienta 'read_pdf'.
                Para vectorizar textos, usa la herramienta 'vectorize_texts'.
                Para buscar en vectores, usa la herramienta 'search_in_vectors'.
                Para corregir texto, usa la herramienta 'correct_text'.
                Para reescribir texto, usa la herramienta 'rewrite_text'.
                
                Estructura la salida en este formato y no proporciones ningún otro texto\n{format_instructions}
                """,
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{query}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        # Prompt para chat
        self.chat_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                Eres un asistente de chat inteligente que puede usar herramientas para responder preguntas.
                Responde la consulta del usuario en un tono conversacional y utiliza las herramientas cuando sea necesario.
                
                La conversación anterior está disponible para que tengas contexto.
                
                Estructura la salida en este formato y no proporciones ningún otro texto\n{format_instructions}
                """,
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{query}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        # Prompt para análisis de documentos
        self.doc_analysis_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                Eres un asistente especializado en análisis de documentos.
                Tu tarea es analizar el documento proporcionado y extraer información relevante.
                
                Utiliza herramientas como extracción de entidades, palabras clave y resumir información.
                
                Estructura la salida en este formato y no proporciones ningún otro texto\n{format_instructions}
                """,
            ),
            ("human", "Analiza el siguiente documento: {document}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
    
    def _setup_agents(self):
        """Configura los agentes para diferentes tipos de tareas"""
        
        # Configurar prompt de investigación con instrucciones de formato
        research_prompt_with_parser = self.research_prompt.partial(
            format_instructions=self.research_parser.get_format_instructions()
        )
        
        # Agente para investigación
        self.research_agent = create_tool_calling_agent(
            llm=self.llm,
            prompt=research_prompt_with_parser,
            tools=all_tools
        )
        
        # Ejecutor para investigación
        self.research_executor = AgentExecutor(
            agent=self.research_agent, 
            tools=all_tools, 
            verbose=True,
            callbacks=[self._get_callback_handler()]
        )
        
        # Configurar prompt de chat con instrucciones de formato
        chat_prompt_with_parser = self.chat_prompt.partial(
            format_instructions=self.chat_parser.get_format_instructions()
        )
        
        # Agente para chat
        self.chat_agent = create_tool_calling_agent(
            llm=self.llm,
            prompt=chat_prompt_with_parser,
            tools=all_tools
        )
        
        # Ejecutor para chat
        self.chat_executor = AgentExecutor(
            agent=self.chat_agent,
            tools=all_tools,
            verbose=True,
            callbacks=[self._get_callback_handler()]
        )
        
        # Configurar prompt de análisis de documentos con instrucciones de formato
        doc_prompt_with_parser = self.doc_analysis_prompt.partial(
            format_instructions=self.doc_parser.get_format_instructions()
        )
        
        # Agente para análisis de documentos
        self.doc_analysis_agent = create_tool_calling_agent(
            llm=self.llm,
            prompt=doc_prompt_with_parser,
            tools=all_tools
        )
        
        # Ejecutor para análisis de documentos
        self.doc_analysis_executor = AgentExecutor(
            agent=self.doc_analysis_agent,
            tools=all_tools,
            verbose=True,
            callbacks=[self._get_callback_handler()]
        )
    
    def _get_callback_handler(self):
        """Crea un manejador de callbacks para registrar el uso de herramientas"""
        
        class ToolLoggingCallbackHandler(BaseCallbackHandler):
            def __init__(self, logger):
                self.logger = logger
                self.start_times = {}
            
            def on_tool_start(self, serialized, input_str, **kwargs):
                tool_name = serialized.get("name", "unknown_tool")
                self.start_times[tool_name] = time.time()
                self.logger.log_tool_use(tool_name, input_str, None, None)
            
            def on_tool_end(self, output, **kwargs):
                # No tenemos acceso al nombre de la herramienta aquí,
                # pero podríamos agregar lógica adicional si fuera necesario
                pass
            
            def on_chain_start(self, serialized, inputs, **kwargs):
                pass
            
            def on_chain_end(self, outputs, **kwargs):
                pass
            
            def on_llm_start(self, serialized, prompts, **kwargs):
                pass
            
            def on_llm_end(self, response, **kwargs):
                pass
        
        return ToolLoggingCallbackHandler(self.logger)
    
    @timing_decorator
    def research(self, query: str) -> ResearchResponse:
        """
        Realiza una investigación sobre un tema
        Args:
            query: Consulta de investigación
        Returns:
            Respuesta estructurada con los resultados de la investigación
        """
        # Registrar la consulta
        self.logger.log_user_message(query)
        
        # Preparar las variables para el agente
        chat_history = self.memory.get_chat_history()
        
        # Ejecutar el agente
        result = self.research_executor.invoke({
            "query": query,
            "chat_history": chat_history
        })
        
        try:
            # Parsear la respuesta
            response = self.research_parser.parse(result["output"])
            
            # Registrar la respuesta
            self.logger.log_ai_message(response.summary)
            
            # Actualizar la memoria
            self.memory.add_interaction(query, response.summary)
            
            return response
        except Exception as e:
            # Si hay un error al parsear, registrarlo y devolver el texto sin procesar
            error_details = get_exception_details(e)
            self.logger.log_error(error_details["error_message"], error_details["error_type"], error_details["traceback"])
            
            # Intentar crear una respuesta estructurada manualmente
            fallback_response = ResearchResponse(
                topic=query,
                summary=result.get("output", "No se pudo generar un resumen"),
                sources=[],
                tools_used=[]
            )
            
            # Actualizar la memoria
            self.memory.add_interaction(query, fallback_response.summary)
            
            return fallback_response
    
    @timing_decorator
    def chat(self, message: str) -> ChatResponse:
        """
        Mantiene una conversación con el usuario
        Args:
            message: Mensaje del usuario
        Returns:
            Respuesta de chat estructurada
        """
        # Registrar el mensaje
        self.logger.log_user_message(message)
        
        # Preparar las variables para el agente
        chat_history = self.memory.get_chat_history()
        
        # Ejecutar el agente
        result = self.chat_executor.invoke({
            "query": message,
            "chat_history": chat_history
        })
        
        try:
            # Parsear la respuesta
            response = self.chat_parser.parse(result["output"])
            
            # Registrar la respuesta
            self.logger.log_ai_message(response.response)
            
            # Actualizar la memoria
            self.memory.add_interaction(message, response.response)
            
            return response
        except Exception as e:
            # Si hay un error al parsear, registrarlo y devolver el texto sin procesar
            error_details = get_exception_details(e)
            self.logger.log_error(error_details["error_message"], error_details["error_type"], error_details["traceback"])
            
            # Intentar crear una respuesta estructurada manualmente
            fallback_response = ChatResponse(
                response=result.get("output", "No se pudo generar una respuesta"),
                actions=[],
                sources=[]
            )
            
            # Actualizar la memoria
            self.memory.add_interaction(message, fallback_response.response)
            
            return fallback_response
    
    @timing_decorator
    def analyze_document(self, document: str, document_name: str) -> DocumentAnalysisResponse:
        """
        Analiza un documento y extrae información relevante
        Args:
            document: Texto del documento a analizar
            document_name: Nombre o identificador del documento
        Returns:
            Respuesta estructurada con análisis del documento
        """
        # Registrar la acción
        self.logger.log_system_event("document_analysis", {"document_name": document_name})
        
        # Ejecutar el agente
        result = self.doc_analysis_executor.invoke({
            "document": document
        })
        
        try:
            # Parsear la respuesta
            response = self.doc_parser.parse(result["output"])
            
            # Si el analizador no asignó el nombre del documento, hacerlo manualmente
            if not response.document_name:
                response.document_name = document_name
            
            # Registrar la respuesta
            self.logger.log_system_event("document_analysis_complete", {"document_name": document_name})
            
            return response
        except Exception as e:
            # Si hay un error al parsear, registrarlo y devolver el texto sin procesar
            error_details = get_exception_details(e)
            self.logger.log_error(error_details["error_message"], error_details["error_type"], error_details["traceback"])
            
            # Intentar crear una respuesta estructurada manualmente
            fallback_response = DocumentAnalysisResponse(
                document_name=document_name,
                summary=result.get("output", "No se pudo generar un análisis"),
                key_points=[],
                entities={}
            )
            
            return fallback_response
    
    def save_memory(self, file_path: str = "memory.json") -> str:
        """
        Guarda la memoria del agente en un archivo
        Args:
            file_path: Ruta del archivo donde guardar la memoria
        Returns:
            Ruta donde se guardó la memoria
        """
        self.memory.save_to_file(file_path)
        return file_path
    
    def load_memory(self, file_path: str = "memory.json") -> bool:
        """
        Carga la memoria del agente desde un archivo
        Args:
            file_path: Ruta del archivo de memoria
        Returns:
            True si se cargó correctamente, False en caso contrario
        """
        if not os.path.exists(file_path):
            return False
        
        try:
            self.memory.load_from_file(file_path)
            return True
        except Exception:
            return False
    
    def clear_memory(self) -> None:
        """Limpia la memoria del agente"""
        self.memory.clear()
    
    def save_logs(self, file_path: Optional[str] = None) -> str:
        """
        Guarda los logs del agente en un archivo
        Args:
            file_path: Ruta del archivo donde guardar los logs
        Returns:
            Ruta donde se guardaron los logs
        """
        return self.logger.save_history(file_path) 