import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from langchain.tools import Tool
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain.embeddings import HuggingFaceEmbeddings

class VectorHandler:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Inicializa el manejador de vectores con un modelo de embeddings
        """
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
    def vectorize_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> FAISS:
        """
        Vectoriza una lista de textos y los guarda en un índice FAISS
        """
        documents = self.text_splitter.create_documents(texts, metadatas=metadatas)
        vector_store = FAISS.from_documents(documents, self.embedding_model)
        return vector_store
        
    def save_vector_store(self, vector_store: FAISS, directory: str) -> str:
        """
        Guarda un vector store en un directorio
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        vector_store.save_local(directory)
        return f"Vector store guardado en {directory}"
    
    def load_vector_store(self, directory: str) -> FAISS:
        """
        Carga un vector store desde un directorio
        """
        if not os.path.exists(directory):
            raise ValueError(f"El directorio {directory} no existe")
        return FAISS.load_local(directory, self.embedding_model)
    
    def similarity_search(self, query: str, vector_store: FAISS, k: int = 5) -> List[Tuple[str, float]]:
        """
        Realiza una búsqueda por similitud en el vector store
        """
        results = vector_store.similarity_search_with_score(query, k=k)
        return [(doc.page_content, score) for doc, score in results]

# Singleton para el manejador de vectores
_vector_handler = None

def get_vector_handler(model_name: str = "all-MiniLM-L6-v2") -> VectorHandler:
    global _vector_handler
    if _vector_handler is None:
        _vector_handler = VectorHandler(model_name=model_name)
    return _vector_handler

# Herramientas de vectorización
def vectorize_and_save(texts: List[str], save_dir: str, metadatas: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Vectoriza textos y guarda el vector store
    """
    handler = get_vector_handler()
    vector_store = handler.vectorize_texts(texts, metadatas)
    return handler.save_vector_store(vector_store, save_dir)

def search_in_vectors(query: str, vector_dir: str, k: int = 5) -> List[Tuple[str, float]]:
    """
    Busca en un vector store existente
    """
    handler = get_vector_handler()
    vector_store = handler.load_vector_store(vector_dir)
    return handler.similarity_search(query, vector_store, k)

# Definir herramientas para LangChain
vectorize_tool = Tool(
    name="vectorize_texts",
    func=vectorize_and_save,
    description="Vectoriza una lista de textos y los guarda. Requiere lista de textos, directorio para guardar y opcionalmente metadatos."
)

search_vectors_tool = Tool(
    name="search_in_vectors",
    func=search_in_vectors,
    description="Busca en vectores previamente almacenados. Requiere query, directorio de vectores y opcionalmente número k de resultados."
) 