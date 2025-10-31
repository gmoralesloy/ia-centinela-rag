import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# --- CONFIGURACIÓN PARA CENTINELA ---
# Nombre de la carpeta que contiene los documentos de Centinela
DOCUMENT_FOLDER = "documentos_centinela" 

# Nombre del archivo de la base de datos segregada
DB_NAME = "chroma_centinela"
PERSIST_DIRECTORY = f"./knowledge_base/{DB_NAME}"

# Modelo de embeddings (el mismo que usamos antes para compatibilidad)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- FUNCIONES ---

def create_db():
    print(f"Buscando documentos en: {DOCUMENT_FOLDER}")

    # Cargar documentos (solo PDF, ignoramos el JPG en este script)
    documents = []
    for filename in os.listdir(DOCUMENT_FOLDER):
        if filename.endswith(".pdf"):
            filepath = os.path.join(DOCUMENT_FOLDER, filename)
            print(f"Cargando documento: {filename}")
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())

    if not documents:
        print("¡ADVERTENCIA! No se encontraron documentos PDF. Asegúrate de tenerlos en la carpeta.")
        return

    # Dividir el texto en fragmentos
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Documento dividido en {len(texts)} fragmentos.")

    # Inicializar el modelo de embeddings
    print("Inicializando modelo de embeddings...")
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

    # Crear y guardar la base de datos vectorial
    print(f"Creando y guardando la base de datos vectorial en: {PERSIST_DIRECTORY}")
    
    # Asegura que la carpeta exista
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    
    # Creamos la base de datos con los textos y embeddings
    vector_db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    vector_db.persist()

    print("\nBase de datos creada y guardada con éxito.")
    print("¡Proceso de creación de la Base de Conocimiento completado!")

if __name__ == "__main__":
    create_db()