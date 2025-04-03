import os
import argparse
from typing import Optional
from dotenv import load_dotenv
from tabulate import tabulate

from .agents.research_agent import ResearchAgent
from .models.ingestion import DataIngestion
from .config import AgentConfig
from .models.schemas import ResearchResponse, ChatResponse, DocumentAnalysisResponse

# Cargar variables de entorno
load_dotenv()

def main():
    """Función principal de la aplicación"""
    
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Agente de Investigación Avanzado")
    subparsers = parser.add_subparsers(dest="command", help="Comando a ejecutar")
    
    # Comando de investigación
    research_parser = subparsers.add_parser("research", help="Realizar una investigación")
    research_parser.add_argument("query", help="Consulta de investigación")
    research_parser.add_argument("--output", "-o", help="Archivo donde guardar los resultados")
    
    # Comando de chat
    chat_parser = subparsers.add_parser("chat", help="Iniciar un chat con el agente")
    chat_parser.add_argument("--memory", "-m", help="Archivo de memoria (opcional)")
    
    # Comando de análisis de documentos
    analyze_parser = subparsers.add_parser("analyze", help="Analizar un documento")
    analyze_parser.add_argument("file", help="Ruta del archivo a analizar")
    analyze_parser.add_argument("--output", "-o", help="Archivo donde guardar los resultados")
    
    # Comando de ingesta de archivos
    ingest_parser = subparsers.add_parser("ingest", help="Ingestar un archivo")
    ingest_parser.add_argument("file", help="Ruta del archivo a ingestar")
    ingest_parser.add_argument("--type", "-t", help="Tipo de archivo (csv, pdf, txt, json)")
    
    # Comando para listar archivos ingestados
    subparsers.add_parser("list-ingested", help="Listar archivos ingestados")
    
    # Parsear argumentos
    args = parser.parse_args()
    
    # Crear configuración del agente
    config = AgentConfig()
    
    # Crear agente
    agent = ResearchAgent(config=config)
    
    if args.command == "research":
        # Realizar investigación
        print(f"Investigando: {args.query}")
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        query = args.query
        raw_response = agent_executor.invoke({"query": query})

        try:
            output_text = raw_response.get("output")
            # Extraer el JSON del bloque de código si está presente
            if output_text.startswith("```json"):
                json_content = output_text.split("```json\\n")[1].split("\\n```")[0]
            elif output_text.startswith("```"):
                # Manejar caso genérico ``` ... ```
                json_content = output_text.split("```")[1]
            else:
                json_content = output_text # Asumir que es JSON directo si no hay marcadores

            structured_response = parser.parse(json_content)
            _print_research_response(structured_response) # Llamar a la función de impresión estructurada
            
            # Guardar resultados si se especificó un archivo
            if args.output:
                _save_research_results(structured_response, args.output)
                print(f"Resultados guardados en {args.output}")
        except Exception as e:
            print("Error parsing response", e, "\nRaw Response - ", raw_response)
    
    elif args.command == "chat":
        # Cargar memoria si se especificó
        if args.memory and os.path.exists(args.memory):
            agent.load_memory(args.memory)
            print(f"Memoria cargada desde {args.memory}")
        
        # Iniciar chat interactivo
        print("Iniciando chat (escribe 'exit' para salir):")
        try:
            while True:
                # Solicitar mensaje al usuario
                message = input("> ")
                
                # Salir si se recibe 'exit'
                if message.lower() == "exit":
                    break
                
                # Procesar mensaje
                response = agent.chat(message)
                print(f"\nAgente: {response.response}\n")
                
                # Si la respuesta tiene fuentes, mostrarlas
                if response.sources:
                    print("Fuentes:")
                    for source in response.sources:
                        print(f"- {source.title or source.url or 'Fuente sin título'}")
                    print()
        except KeyboardInterrupt:
            print("\nChat finalizado.")
        
        # Guardar memoria
        memory_file = args.memory or "memory.json"
        agent.save_memory(memory_file)
        print(f"Memoria guardada en {memory_file}")
    
    elif args.command == "analyze":
        # Verificar que el archivo existe
        if not os.path.exists(args.file):
            print(f"Error: El archivo {args.file} no existe")
            return
        
        # Leer archivo
        print(f"Analizando archivo: {args.file}")
        
        # Determinar el tipo de archivo
        file_extension = os.path.splitext(args.file)[1].lower()
        
        # Leer el contenido según el tipo
        content = _read_file_content(args.file, file_extension)
        
        if content is None:
            print(f"Error: No se pudo leer el archivo {args.file}")
            return
        
        # Analizar documento
        response = agent.analyze_document(content, os.path.basename(args.file))
        _print_document_analysis(response)
        
        # Guardar resultados si se especificó un archivo
        if args.output:
            _save_document_analysis(response, args.output)
            print(f"Análisis guardado en {args.output}")
    
    elif args.command == "ingest":
        # Verificar que el archivo existe
        if not os.path.exists(args.file):
            print(f"Error: El archivo {args.file} no existe")
            return
        
        # Crear gestor de ingesta
        ingestion = DataIngestion()
        
        # Ingestar archivo
        print(f"Ingestando archivo: {args.file}")
        try:
            metadata = ingestion.ingest_file(args.file, args.type)
            print("Archivo ingestado exitosamente:")
            print(f"ID: {metadata.get('file_id') or metadata.get('text_id')}")
            print(f"Nombre original: {metadata.get('original_file_name') or metadata.get('source_name')}")
            print(f"Tipo: {metadata.get('file_type')}")
            print(f"Ruta procesada: {metadata.get('processed_file_path')}")
            if "vector_dir" in metadata:
                print(f"Directorio de vectores: {metadata.get('vector_dir')}")
        except Exception as e:
            print(f"Error al ingestar archivo: {str(e)}")
    
    elif args.command == "list-ingested":
        # Crear gestor de ingesta
        ingestion = DataIngestion()
        
        # Listar archivos ingestados
        files = ingestion.list_ingested_files()
        
        if not files:
            print("No hay archivos ingestados")
            return
        
        print(f"Archivos ingestados ({len(files)}):")
        for i, metadata in enumerate(files, 1):
            file_id = metadata.get("file_id") or metadata.get("text_id")
            file_name = metadata.get("original_file_name") or metadata.get("source_name")
            file_type = metadata.get("file_type")
            timestamp = metadata.get("ingestion_timestamp")
            print(f"{i}. ID: {file_id} | Nombre: {file_name} | Tipo: {file_type} | Fecha: {timestamp}")
    
    else:
        # Si no se especificó un comando, mostrar ayuda
        parser.print_help()

def _print_research_response(response: ResearchResponse) -> None:
    """Imprime una respuesta de investigación de forma estructurada"""
    # Imprimir encabezado
    print("\n" + "=" * 80)
    print(f"📚 INVESTIGACIÓN: {response.topic}")
    print("=" * 80)
    
    # Imprimir resumen
    print("\n📝 RESUMEN:")
    print("-" * 80)
    print(response.summary)
    print("-" * 80)
    
    # Imprimir fuentes en una tabla
    if response.sources:
        print("\n📚 FUENTES CONSULTADAS:")
        print("-" * 80)
        sources_data = []
        for source in response.sources:
            sources_data.append([
                source.title or "Sin título",
                source.type,
                source.url or "N/A"
            ])
        print(tabulate(
            sources_data,
            headers=["Título", "Tipo", "URL"],
            tablefmt="grid"
        ))
    
    # Imprimir herramientas utilizadas
    if response.tools_used:
        print("\n🛠️ HERRAMIENTAS UTILIZADAS:")
        print("-" * 80)
        for i, tool in enumerate(response.tools_used, 1):
            print(f"{i}. {tool}")
    
    print("\n" + "=" * 80)

def _save_research_results(response: ResearchResponse, file_path: str) -> None:
    """Guarda los resultados de investigación en un archivo de forma estructurada"""
    with open(file_path, 'w', encoding='utf-8') as f:
        # Escribir encabezado
        f.write("=" * 80 + "\n")
        f.write(f"📚 INVESTIGACIÓN: {response.topic}\n")
        f.write("=" * 80 + "\n\n")
        
        # Escribir resumen
        f.write("📝 RESUMEN:\n")
        f.write("-" * 80 + "\n")
        f.write(response.summary + "\n")
        f.write("-" * 80 + "\n\n")
        
        # Escribir fuentes en una tabla
        if response.sources:
            f.write("📚 FUENTES CONSULTADAS:\n")
            f.write("-" * 80 + "\n")
            sources_data = []
            for source in response.sources:
                sources_data.append([
                    source.title or "Sin título",
                    source.type,
                    source.url or "N/A"
                ])
            f.write(tabulate(
                sources_data,
                headers=["Título", "Tipo", "URL"],
                tablefmt="grid"
            ))
            f.write("\n\n")
        
        # Escribir herramientas
        if response.tools_used:
            f.write("🛠️ HERRAMIENTAS UTILIZADAS:\n")
            f.write("-" * 80 + "\n")
            for i, tool in enumerate(response.tools_used, 1):
                f.write(f"{i}. {tool}\n")
        
        f.write("\n" + "=" * 80 + "\n")

def _print_document_analysis(response: DocumentAnalysisResponse) -> None:
    """Imprime un análisis de documento de forma estructurada"""
    # Imprimir encabezado
    print("\n" + "=" * 80)
    print(f"📄 ANÁLISIS DE DOCUMENTO: {response.document_name}")
    print("=" * 80)
    
    # Imprimir resumen
    print("\n📝 RESUMEN:")
    print("-" * 80)
    print(response.summary)
    print("-" * 80)
    
    # Imprimir puntos clave
    if response.key_points:
        print("\n🎯 PUNTOS CLAVE:")
        print("-" * 80)
        for i, point in enumerate(response.key_points, 1):
            print(f"{i}. {point}")
    
    # Imprimir entidades en una tabla
    if response.entities:
        print("\n🔍 ENTIDADES DETECTADAS:")
        print("-" * 80)
        entities_data = []
        for entity_type, entities in response.entities.items():
            entities_data.append([entity_type, ", ".join(entities)])
        print(tabulate(
            entities_data,
            headers=["Tipo", "Entidades"],
            tablefmt="grid"
        ))
    
    # Imprimir sentimiento
    if response.sentiment:
        print("\n😊 ANÁLISIS DE SENTIMIENTO:")
        print("-" * 80)
        print(f"Sentimiento: {response.sentiment}")
    
    print("\n" + "=" * 80)

def _save_document_analysis(response: DocumentAnalysisResponse, file_path: str) -> None:
    """Guarda el análisis de documento en un archivo de forma estructurada"""
    with open(file_path, 'w', encoding='utf-8') as f:
        # Escribir encabezado
        f.write("=" * 80 + "\n")
        f.write(f"📄 ANÁLISIS DE DOCUMENTO: {response.document_name}\n")
        f.write("=" * 80 + "\n\n")
        
        # Escribir resumen
        f.write("📝 RESUMEN:\n")
        f.write("-" * 80 + "\n")
        f.write(response.summary + "\n")
        f.write("-" * 80 + "\n\n")
        
        # Escribir puntos clave
        if response.key_points:
            f.write("🎯 PUNTOS CLAVE:\n")
            f.write("-" * 80 + "\n")
            for i, point in enumerate(response.key_points, 1):
                f.write(f"{i}. {point}\n")
            f.write("\n")
        
        # Escribir entidades en una tabla
        if response.entities:
            f.write("🔍 ENTIDADES DETECTADAS:\n")
            f.write("-" * 80 + "\n")
            entities_data = []
            for entity_type, entities in response.entities.items():
                entities_data.append([entity_type, ", ".join(entities)])
            f.write(tabulate(
                entities_data,
                headers=["Tipo", "Entidades"],
                tablefmt="grid"
            ))
            f.write("\n\n")
        
        # Escribir sentimiento
        if response.sentiment:
            f.write("😊 ANÁLISIS DE SENTIMIENTO:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Sentimiento: {response.sentiment}\n\n")
        
        f.write("=" * 80 + "\n")

def _read_file_content(file_path: str, file_extension: str) -> Optional[str]:
    """Lee el contenido de un archivo según su extensión"""
    try:
        if file_extension == ".txt":
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_extension == ".pdf":
            # Importar aquí para evitar dependencias innecesarias
            from .tools.file_tools import FileHandler
            return "\n\n".join(FileHandler.read_pdf(file_path))
        elif file_extension == ".csv":
            # Para CSV, convertimos a texto para análisis
            import pandas as pd
            df = pd.read_csv(file_path)
            return df.to_string()
        else:
            print(f"Tipo de archivo no soportado para análisis: {file_extension}")
            return None
    except Exception as e:
        print(f"Error al leer archivo: {str(e)}")
        return None

if __name__ == "__main__":
    main() 