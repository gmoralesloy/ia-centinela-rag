from fastapi import FastAPI
from pydantic import BaseModel
import os
import uvicorn

# Importaciones del RAG
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import ChatPromptTemplate

# Importación para SIMULAR el LLM (lo reemplazaremos después)
from typing import Tuple

# --- CONFIGURACIÓN DE LA BASE DE DATOS SEGREGADA ---
DB_NAME = chroma_centinela
PERSIST_DIRECTORY = f.knowledge_base{DB_NAME}
EMBEDDING_MODEL = all-MiniLM-L6-v2

# --- PROMPT Y REGLAS ---
SYSTEM_PROMPT = 
Eres un Asistente de IA de Acreditación altamente especializado para Minera Centinela. 
Tu única fuente de información es el CONTEXTO proporcionado por los manuales de la empresa.
Tu objetivo es destilar y resumir la información del manual de forma concisa y profesional.
DEBES cumplir estas reglas estrictas
1. SIEMPRE genera la respuesta en español de forma fluida y natural (NO debes devolver el contexto crudo).
2. Si el contexto NO contiene información para responder, DEBES responder con el mensaje de denegación predefinido.
3. Debes citar la fuente del documento (ej).


# --- INICIALIZACIÓN DEL SISTEMA ---
app = FastAPI()
vector_db = None
embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

# Pydantic model para la entrada de la API
class QueryModel(BaseModel)
    query str
    
def load_vector_db()
    global vector_db
    try
        # Cargamos la DB segregada de Centinela
        vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        print(✅ Base de Conocimiento de Centinela cargada con éxito.)
    except Exception as e
        print(f❌ Error al cargar la Base de Conocimiento de Centinela {e})
        vector_db = None

# Función de consulta RAG (con LLM SIMULADO)
# ESTA ES LA FUNCIÓN QUE REEMPLAZAREMOS CON EL LLM REAL
def run_rag_query(query str) - Tuple[str, str]
    if vector_db is None
        return Error La Base de Conocimiento no pudo ser cargada., 

    # A. SIMULACIÓN DE ALCANCE ESTRICTO (Scoping)
    KEYWORDS = [acreditación, centinela, examen, contrato, documentación, plataforma, siga]
    is_relevant = any(keyword in query.lower() for keyword in KEYWORDS)
    
    if not is_relevant
        final_answer = (Lo siento, esa pregunta está fuera del contexto de mi conocimiento actual sobre 
                        los Estándares de Minera Centinela.)
        return final_answer, 

    # B. RECUPERACIÓN (Retrieval) - Búsqueda estricta
    try
        results = vector_db.similarity_search_with_score(query, k=1)
    except Exception as e
        return fError en la búsqueda {e}, 
    
    # C. CHEQUEO DE PRECISIÓN (Umbral 0.9)
    if not results or results[0][1]  0.9 
        final_answer = (Encontré información, pero el grado de precisión es bajo (relevancia  0.9). 
                        Por favor, reformula tu pregunta para obtener una respuesta más precisa.)
        return final_answer, 
    
    # D. GENERACIÓN (SIMULADA) - ESTO SERÁ REEMPLAZADO POR LA LLAMADA AL LLM REAL
    context_text = nn---nn.join([doc.page_content + f for doc, _score in results])

    # Simulamos la respuesta fluida que daría el LLM
    simulated_answer = (
        f[Respuesta Simulada - Modo Piloto Centinela]nn
        fEl LLM (Llama3) respondería resumiendo de forma fluida el siguiente contexto clavenn
        f```textn{context_text}n```
    )
    
    return simulated_answer, context_text

# --- ENDPOINTS DE LA API ---

@app.on_event(startup)
async def startup_event()
    load_vector_db()

@app.post(query)
async def process_query(query_data QueryModel)
    answer, context = run_rag_query(query_data.query)
    
    # Devolvemos la respuesta en formato JSON para el sistema web
    return {answer answer, context context}

# Este es solo un endpoint de prueba para ver que la API esté viva
@app.get()
def read_root()
    return {status IA Centinela RAG API Lista}

# Si quieres probarlo localmente (opcional)
if __name__ == __main__
    uvicorn.run(app, host=0.0.0.0, port=8000)