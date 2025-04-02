import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Descargar recursos de NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextProcessor:
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Limpia el texto eliminando caracteres especiales y normalizando espacios
        """
        # Eliminar caracteres especiales y números
        text = re.sub(r'[^\w\s]', ' ', text)
        # Eliminar números
        text = re.sub(r'\d+', ' ', text)
        # Normalizar espacios
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    @staticmethod
    def tokenize(text: str) -> list:
        """
        Tokeniza el texto en palabras
        """
        return word_tokenize(text)
    
    @staticmethod
    def remove_stopwords(tokens: list, language: str = 'english') -> list:
        """
        Elimina stopwords de una lista de tokens
        """
        stop_words = set(stopwords.words(language))
        return [token for token in tokens if token.lower() not in stop_words]
    
    @staticmethod
    def lemmatize(tokens: list) -> list:
        """
        Lematiza una lista de tokens
        """
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in tokens]

class TextCorrector:
    def __init__(self):
        """
        Inicializa el corrector de texto utilizando un modelo de lenguaje
        """
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Eres un asistente especializado en corregir texto. Tu tarea es corregir errores ortográficos y gramaticales sin cambiar el significado original del texto."
                ),
                ("human", "{text}")
            ]
        )
    
    def correct_text(self, text: str) -> str:
        """
        Corrige errores ortográficos y gramaticales en el texto
        """
        chain = self.prompt | self.llm
        response = chain.invoke({"text": text})
        corrected_text = response.content
        return corrected_text

# Instancia singleton del corrector de texto
_text_corrector = None

def get_text_corrector() -> TextCorrector:
    global _text_corrector
    if _text_corrector is None:
        _text_corrector = TextCorrector()
    return _text_corrector

# Funciones para las herramientas
def clean_and_tokenize(text: str) -> list:
    """
    Limpia y tokeniza un texto
    """
    cleaned = TextProcessor.clean_text(text)
    return TextProcessor.tokenize(cleaned)

def correct_spelling_and_grammar(text: str) -> str:
    """
    Corrige errores ortográficos y gramaticales
    """
    corrector = get_text_corrector()
    return corrector.correct_text(text)

# Definir herramientas para LangChain
clean_text_tool = Tool(
    name="clean_text",
    func=TextProcessor.clean_text,
    description="Limpia el texto eliminando caracteres especiales y normalizando espacios."
)

text_correction_tool = Tool(
    name="correct_text",
    func=correct_spelling_and_grammar,
    description="Corrige errores ortográficos y gramaticales en el texto."
) 