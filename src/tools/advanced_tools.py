from typing import List, Dict, Any, Optional
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import spacy
import re
from datetime import datetime

# Cargar el modelo de spaCy si no está disponible
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

class TextRewriter:
    """Clase para reescribir texto usando LLM"""
    
    def __init__(self, model_name: str = "gemini-1.5-pro"):
        """Inicializa el reescritor de texto con un modelo de lenguaje"""
        self.llm = ChatGoogleGenerativeAI(model=model_name)
    
    def rewrite_text(self, text: str, style: str = "formal") -> str:
        """
        Reescribe un texto en un estilo específico
        Args:
            text: Texto a reescribir
            style: Estilo de reescritura (formal, informal, simple, etc.)
        Returns:
            Texto reescrito
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"Eres un experto en reescribir texto. Reescribe el siguiente texto en un estilo {style}, manteniendo el mismo significado pero mejorando la legibilidad. No añadas información adicional."),
            ("human", "{text}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"text": text})
        return response.content
    
    def correct_and_improve(self, text: str) -> str:
        """
        Corrige errores y mejora la calidad del texto
        Args:
            text: Texto a corregir y mejorar
        Returns:
            Texto corregido y mejorado
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Eres un experto en edición de texto. Corrige cualquier error ortográfico o gramatical en el siguiente texto y mejora su claridad y fluidez. No cambies el significado ni añadas información nueva."),
            ("human", "{text}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"text": text})
        return response.content

class EntityExtractor:
    """Clase para extraer entidades de textos"""
    
    @staticmethod
    def extract_entities(text: str) -> Dict[str, List[str]]:
        """
        Extrae entidades nombradas de un texto usando spaCy
        Args:
            text: Texto del que extraer entidades
        Returns:
            Diccionario con entidades agrupadas por tipo
        """
        doc = nlp(text)
        entities = {}
        
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            if ent.text not in entities[ent.label_]:
                entities[ent.label_].append(ent.text)
        
        return entities
    
    @staticmethod
    def extract_keywords(text: str, n: int = 10) -> List[str]:
        """
        Extrae palabras clave de un texto
        Args:
            text: Texto del que extraer palabras clave
            n: Número máximo de palabras clave
        Returns:
            Lista de palabras clave
        """
        doc = nlp(text)
        keywords = []
        
        for token in doc:
            if (not token.is_stop and not token.is_punct and 
                token.pos_ in ["NOUN", "PROPN", "ADJ"] and 
                len(token.text) > 3):
                keywords.append(token.text)
        
        # Contar frecuencias
        from collections import Counter
        counted = Counter(keywords)
        
        # Devolver las n palabras más frecuentes
        return [word for word, _ in counted.most_common(n)]

# Instancia singleton del reescritor
_rewriter = None

def get_rewriter() -> TextRewriter:
    global _rewriter
    if _rewriter is None:
        _rewriter = TextRewriter()
    return _rewriter

# Funciones para herramientas
def rewrite_text(text: str, style: str = "formal") -> str:
    """
    Reescribe un texto en un estilo específico
    """
    rewriter = get_rewriter()
    return rewriter.rewrite_text(text, style)

def correct_and_improve(text: str) -> str:
    """
    Corrige errores y mejora la calidad del texto
    """
    rewriter = get_rewriter()
    return rewriter.correct_and_improve(text)

def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extrae entidades nombradas de un texto
    """
    return EntityExtractor.extract_entities(text)

def extract_keywords(text: str, n: int = 10) -> List[str]:
    """
    Extrae palabras clave de un texto
    """
    return EntityExtractor.extract_keywords(text, n)

# Herramientas para LangChain
rewrite_tool = Tool(
    name="rewrite_text",
    func=rewrite_text,
    description="Reescribe un texto en un estilo específico (formal, informal, simple, técnico, etc.)."
)

correction_tool = Tool(
    name="correct_text",
    func=correct_and_improve,
    description="Corrige errores y mejora la calidad del texto."
)

entity_extraction_tool = Tool(
    name="extract_entities",
    func=extract_entities,
    description="Extrae entidades nombradas de un texto (personas, organizaciones, lugares, etc.)."
)

keyword_extraction_tool = Tool(
    name="extract_keywords",
    func=extract_keywords,
    description="Extrae palabras clave de un texto."
) 