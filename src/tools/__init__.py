"""
Herramientas para el agente de investigación
"""

# Importar todas las herramientas
from langchain.tools import Tool
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper

from .file_tools import csv_tool, pdf_tool, save_json_tool, load_json_tool
from .vector_tools import vectorize_tool, search_vectors_tool
from .text_tools import clean_text_tool, text_correction_tool
from .advanced_tools import (
    rewrite_tool, 
    correction_tool, 
    entity_extraction_tool, 
    keyword_extraction_tool
)

# Importar desde el archivo original
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Busca información en la web",
)

def save_to_txt(data: str, filename: str = "research_output.txt"):
    """Guarda datos en un archivo de texto"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Datos guardados exitosamente en {filename}"

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Guarda datos de investigación en un archivo de texto.",
)

# Todas las herramientas disponibles
all_tools = [
    search_tool,
    wiki_tool,
    save_tool,
    csv_tool,
    pdf_tool,
    save_json_tool,
    load_json_tool,
    vectorize_tool,
    search_vectors_tool,
    clean_text_tool,
    text_correction_tool,
    rewrite_tool,
    correction_tool,
    entity_extraction_tool,
    keyword_extraction_tool
] 