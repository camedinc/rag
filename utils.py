#---------------------------------------------------------------------------------------
# Liberías
#---------------------------------------------------------------------------------------

# Modelos LLMs locales
#import ollama     

# Lectura de PDF
import fitz

# Comparación de embeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Limpieza de texto
import re

# Respuestas de texto con formato Markdown (aplica Jupyter)
from IPython.display import display, Markdown

# Respuesta LCEL (LangChain)
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

# Memoria
from langchain.chains import ConversationalRetrievalChain

# Chunks semántico experimental
from langchain_experimental.text_splitter import SemanticChunker

# Modelo de embeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings

# Vector store (almacenamiento vectorial)
from langchain.vectorstores import FAISS

#---------------------------------------------------------------------------------------
# Funciones
#---------------------------------------------------------------------------------------

# Función que extrae texto y lo pasa a formato string
def extract_text_from_pdf(pdf):
    
    # Lee el conenido binario del archivo pdf cargado
    pdf_bytes = pdf.read()

    # Abre el archivo PDF desde los bytes leidos
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = ""

    # Extrae el texto de cada página del PDF
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        full_text += page.get_text("text")

    return full_text

# Función que limpia el texto
def clean_text(document):
    # 1. Reemplaza saltos de línea por espacios
    text = document.replace("\n\n", " ")

    # 2. Normaliza caracteres especiales (opcionalmente usa `unidecode` para mayor limpieza)
    text = re.sub(r"\\ń", "́", text)  # Corrige '\ń'
    text = re.sub(r"\\n", " ", text)  # Corrige '\n'

    # 3. Reemplaza múltiples espacios consecutivos por un único espacio
    text = re.sub(r"\s+", " ", text)

    # 4. Opcional: Elimina texto no deseado como números de página o encabezados
    text = re.sub(r"\d+\s?\.?S T E V E J O B S.*", "", text)  # Elimina encabezados específicos

    # 5. Normaliza espacios alrededor de signos de puntuación
    text = re.sub(r"\s+([.,!?])", r"\1", text)

    # 6. Elimina espacios iniciales y finales
    text = text.strip()

    return text

# Función que genera los chunks de texto semánticos
def semantic_chunk_text(document):
    semantic_chunker = SemanticChunker(
        HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"), 
        breakpoint_threshold_type="percentile"
        )
    chunks = semantic_chunker.create_documents([document])
    return chunks

# Función que crear el vector store con los embeddings semanticos
def semantic_vector_store(chunks):
    vector_store = FAISS.from_documents(chunks,
    embedding=HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    return vector_store

# Función que genera el embedding de la pregunta del usuario
def generate_embedding(user_question):
    embedding = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2").embed_query(user_question)
    return embedding

# Función que crea el prompt y da personalidad al modelo
def semantic_prompt(context_semantic, query):

    # Prompt semántico
    prompt = f"""
    A continuación se presenta un contexto para responder la pregunta.\n
    Considera cuidadosamente la información de este contexto,\n
    responde únicamente en español,\n
    de forma clara, precisa y en un tono accesible y cercano,\n
    cuidando de no truncar el texto de tu respuesta.

    Contexto:
    {context_semantic}

    Pregunta: {query}

    Respuesta (en español en un tono claro, preciso, accesible, cercano y sin truncar):
    """
    return prompt

# Función que genera la respuesta LCEL
def lcel(context_semantic, question, semantic_chunk_retriever, llm, memory):

    # Obtener el historial de la memoria
    memory_variables = memory.load_memory_variables({})
    chat_history = memory_variables.get("history", [])

   # Convertir chat_history a un formato que sea ejecutable
    chat_history_runnable = RunnablePassthrough() | str  # Pasar chat_history como un string

    # Definir el template del prompt
    prompt_template = PromptTemplate(
        template="Context: {context_semantic}\nChat Summary: {chat_history}\nQuestion: {question}\nAnswer:",
        input_variables=["context_semantic", "chat_history", "question"]
    )

    # Conectar todo a la cadena
    semantic_rag_chain = (
        RunnableMap(
        {"context_semantic" : semantic_chunk_retriever, 
        "question" : RunnablePassthrough(),
        "chat_history": chat_history_runnable}  # Agregar el historial de chat aquí
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    # Ejecutar la cadena con la pregunta
    result = semantic_rag_chain.invoke(
    question  # Pregunta
    )

    # Guardar el contexto y la respuesta en la memoria
    memory.save_context({"question": question}, {"answer": result})

    # Retorna la respuesta y la memoria actualizada
    return result, memory
