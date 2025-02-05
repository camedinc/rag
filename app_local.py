
#---------------------------------------------------------------------------------------
# Preparación
#---------------------------------------------------------------------------------------

""" 
BASH: Inicializar el modelo (ollama serve)
BASH: Inicializar app streamlit (streamlit run <ruta al directorio>/app.py)
"""

#---------------------------------------------------------------------------------------
# Librerías
#---------------------------------------------------------------------------------------

# Utils
import streamlit as st
from utils import (
    extract_text_from_pdf, 
    clean_text, 
    semantic_chunk_text, 
    semantic_vector_store, 
    semantic_prompt, 
    generate_embedding, 
    lcel)

# Habilitar el modelo LLM
from langchain_community.llms import Ollama

# Historial
from langchain.memory import ConversationSummaryMemory

# Manejo de errores
import logging

# Configurar el nivel de logs
logging.getLogger("streamlit").setLevel(logging.ERROR)

#---------------------------------------------------------------------------------------
# Capa usuario
#---------------------------------------------------------------------------------------

# Carga el pdf
st.header("Pregunta a tu PDF")
pdf_obj = st.file_uploader("Carga tu documento", type="pdf")

# Almacena la pregunta del usuario
user_question = st.text_input("Has una pregunta sobre tu PDF")

#---------------------------------------------------------------------------------------
# Knowledge store
#---------------------------------------------------------------------------------------
if pdf_obj is None:
    st.write("Por favor, carga un archivo PDF para continuar.")
else:
    # Si el PDF ya se cargó, comienza a procesarlo
    @st.cache_data
    def knowledge_store(pdf):

        pdf_content = extract_text_from_pdf(pdf)
        pdf_content_clean = clean_text(pdf_content)
        st.write("Contenido extraído y depurado.")

        st.write("Generando fragmentos semánticos ...")
        semantic_chunks = semantic_chunk_text(pdf_content_clean)
        st.write("Fragmentos semánticos generados.")

        # Si no hay chunks semánticos
        if not semantic_chunks:
            st.write("...")
            return None

        # Genera los embeddings y los almacena en el store
        st.write(f"Iniciando el almacenamiento...")
        vector_store = semantic_vector_store(semantic_chunks)
        st.write(f"Almacenamiento terminado.")
   
        return vector_store

    # Genera la base de conocimiento
    semantic_knowledge_store = knowledge_store(pdf_obj)

    # Recupera un documento de texto
    semantic_chunk_retriever = semantic_knowledge_store.as_retriever(search_kwargs={"k": 1})

#---------------------------------------------------------------------------------------
# Inicializa el modelo
#---------------------------------------------------------------------------------------
    st.write("Inicializando modelo Llama2:7b..")
    # Inicializa el modelo LLM
    llm = Ollama(
        model="llama2:7b",            # Nombre del modelo a utilizar
        temperature=1.0,              # Aleatoriedad de las respuestas (1.0 por defecto)
        top_p=1.0                     # Probabilidad acumulada para limitar diversidad (1.0 por defecto)
    )
    st.write("Modelo Llama2:7b inicializado.")

#---------------------------------------------------------------------------------------
# Genera la respuesta
#---------------------------------------------------------------------------------------

    # Generar el embedding de la pregunta
    user_embedding = generate_embedding(user_question)

    # Buscar los textos más similares con la pregunta del usuario dentro el vector store semántico
    # k=3 devuelve las 3 más similares
    similar_docs_semantic = semantic_knowledge_store.similarity_search_by_vector(user_embedding, k=3)

    # Extrae los dos textos más relevantes basados en la similaridad semántica
    context_semantic = " ".join([doc.page_content for doc in similar_docs_semantic[:2]])

    # Respuesta LCEL
    st.warning("Generando respuesta...")
    
    # Inicializar memoria con resumen
    memory = ConversationSummaryMemory(memory_key="chat_history", llm=llm)

    # Usa la función con memoria de resumen
    respuesta, memory = lcel(context_semantic, user_question, semantic_chunk_retriever, llm, memory)
    st.write(respuesta)