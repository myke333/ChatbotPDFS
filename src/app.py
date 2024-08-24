#pip install Flask
#pip install pandas
#pip install flask-cors
#pip install python-dotenv python-decouple
#pip install wfastcgi
#pip install cryptography
#pip install groq
#pip install sentence-transformers
#pip install openai==0.28.1
#pip install langchain-community langchain-core
#pip install langchain pypdf

from cryptography.fernet import Fernet
from flask import Flask, jsonify, request
import pandas as pd 
import json
from flask_cors import CORS
from decouple import config
import base64
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import base64
from groq import Groq
import openai
import os
import re
import pickle
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

app=Flask(__name__)
CORS(app)

with open('models/modelo_sbert.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

@app.route('/')
def response():
    msg="Hola, soy Kactus IA chatbot de pdfs"
    return msg

@app.route('/gendfpdfs', methods=['POST'])
def generaciondataframe(): 
        
        try: 
           
            directorio = 'documents/'

            archivos_pdf = [archivo for archivo in os.listdir(directorio) if archivo.endswith('.pdf')]

            textos = []
            documentos = []
            paginas = []

            for archivo_pdf in archivos_pdf:
                ruta_pdf = os.path.join(directorio, archivo_pdf)
                
                documento = PyPDFLoader(ruta_pdf)
                data = documento.load()
                
                split = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
                fragmentos = split.split_documents(data)
                
                for doc in fragmentos:
                    textos.append(re.sub(r'\n+', ' ', doc.page_content))  # Limpiar saltos de línea y caracteres especiales
                    documentos.append(doc.metadata['source'])
                    paginas.append(doc.metadata['page'])

            df = pd.DataFrame({
                'Texto': textos,
                'Documento': documentos,
                'Page': paginas
            })

            df['Texto'] = df['Texto'].apply(lambda x: re.sub(r'\n+', ' ', x))  
            df['Texto'] = df['Texto'].apply(lambda x: re.sub(r'[^\w\s]', '', x))  
            df['Documento'] = df['Documento'].str.replace('documents/', '')
            df["Embeddings"]=df["Texto"].apply(lambda x: loaded_model.encode(x))
                     
            df.to_pickle('dataframe.pkl')

            return "El proceso de generación de dataframe (.pkl) con Embeddings fue exitoso. "

        except Exception as ex:
                exceptionf = "Ha ocurrido una error al generar el dataframe." + str(ex)
                return exceptionf
        

@app.route('/kactusia',methods=['POST'])
def talentoia(): 
    
    
   
    try:
        try:
            data_frame = pd.read_pickle("dataframe.pkl")
        
        except Exception as ex:
             exceptionf = "No se encontró archivo .pkl \n" + str(ex)
             return exceptionf

        try:
                data_in= request.get_json()
                pad=data_in["pregunta"]
                pad=pad.lower()
    

        except Exception as ex:
                exceptionf = "Falta body en la petición o error en éste. \n" + str(ex)

        try: 
            pad_emb=loaded_model.encode(str(pad))
            
        except Exception as ex:
            exceptionf= "El proceso de generación de embedding a la pregunta genero un error. \n"+ str(ex)
            return exceptionf
        
        data_frame["Similitud"]=None
        data_frame['Page'] = data_frame['Page'].astype(int)
        data_frame['Page'] = data_frame['Page'].apply(lambda x: x + 1)
        
        pad_emb = np.array(pad_emb).reshape(1, -1)

        pad_emb = pad_emb / np.linalg.norm(pad_emb)

        data_frame["Similitud"] = data_frame["Embeddings"].apply(lambda x: cosine_similarity(np.array(x).reshape(1, -1) / np.linalg.norm(np.array(x).reshape(1, -1)), pad_emb)[0][0])
        data_frame_ordenado=data_frame.sort_values("Similitud", ascending=False)
                                        
        tres_mayores = data_frame_ordenado.head(3)
        tres_mayores = tres_mayores.reset_index(drop=True)
        print(tres_mayores)



        tex1=tres_mayores['Texto'][0]
        tex2=tres_mayores['Texto'][1]
        tex3=tres_mayores['Texto'][2]
        doc1=tres_mayores['Documento'][0]
        doc2=tres_mayores['Documento'][1]
        doc3=tres_mayores['Documento'][2]
        pag1=tres_mayores['Page'][0]
        pag2=tres_mayores['Page'][1]
        pag3=tres_mayores['Page'][2]
        
        outmodel={"answer":""}
       
        modelo=config('MODEL')
        print(modelo)
        provider=config('PROVIDER')
        print(provider)
        clave= config('TKN').encode()
        fernet = Fernet(clave)
        api_keydes = fernet.decrypt(config('API_KEY').encode()).decode()

        if(provider=='GROQ'):

                    try:

                        client = Groq(api_key=api_keydes)
                        completion = client.chat.completions.create(
                                                                        model=modelo,
                                                                        messages=[
                                                                                        {
                                                                                            "role": "user",
                                                                                            "content": """ Contexto: Eres un experto en responder preguntas en base a textos. Se te proporcionan tres textos y una pregunta. Tu trabajo es responder la pregunta en base a los textos que consideres mas pertinentes generando una respuesta con excelente redaccion incluyendo el nombre y la pagina del documento en el que te basaste.
                                                                                                            -Texto de la pagina {} del documento {}: {}/n
                                                                                                            -Texto de la pagina {} del documento {}: {}/n
                                                                                                            -Texto de la pagina {} del documento {}: {}/n
                                                                                                            Pregunta: {}/n""".format(pag1,doc1,tex1,pag2,doc2,tex2,pag3,doc3,tex3,pad)
                                                                                        }
                                                                                    ],
                                                                        temperature=1,
                                                                        max_tokens=1024,
                                                                        top_p=1,
                                                                        stream=True,
                                                                        stop=None,
                                                                    )

                        respuesta = ""  

                        for chunk in completion:
                                respuesta += chunk.choices[0].delta.content or ""  # Concatenamos el contenido de cada fragmento

                        print(respuesta)      

                        respuesta=respuesta.lower()
       
                        outmodel['answer']=respuesta
        
                        return jsonify(outmodel),201
                    
                    except Exception as ex:
                            exceptionf= "En el proceso de generación de opinión del modelo se excedió el limite de uso. \n"+ str(ex)
                            return exceptionf
        elif(provider=='OPENAI'):
                try:
                        openai.api_key=api_keydes
                        response = openai.ChatCompletion.create(
                                          model=modelo,
                                           messages=[
                                                        {
                                                            "role": "system",
                                                            "content": """Contexto: Eres un experto en responder preguntas en base a textos. Se te proporcionan tres textos y una pregunta. Tu trabajo es responder la pregunta en base a los textos que consideres mas pertinentes generando una respuesta con excelente redaccion incluyendo el nombre y la pagina del documento en el que te basaste.
                                                                                                            -Texto de la pagina {} del documento {}: {}/n
                                                                                                            -Texto de la pagina {} del documento {}: {}/n
                                                                                                            -Texto de la pagina {} del documento {}: {}/n
                                                                                                            Pregunta: {}/n""".format(pag1,doc1,tex1,pag2,doc2,tex2,pag3,doc3,tex3,pad) 
                                                        }
                                                        ],
                                          max_tokens=500,
                                          top_p=1,
                                          frequency_penalty=0,
                                          presence_penalty=0,
                                          temperature=0.95
                                                            )
               
                   
                        respuesta_del_modelo=response.choices[0].message.content
                        tokens_input=response.usage["prompt_tokens"]
                        tokens_output=response.usage["completion_tokens"]
                        precio_input=(0.5*response.usage["prompt_tokens"])/1000000
                        precio_output=(1.50*response.usage["completion_tokens"])/1000000
                        precio_total_usd=precio_input+precio_output
                        precio_total_usd_redon= round(precio_total_usd,8)
                        precio_total_cop=(precio_input+precio_output)*4000
                        precio_total_cop_redon=round(precio_total_cop,2)
                        respuesta_del_modelo=respuesta_del_modelo.lower()
                        respuesta_del_modelo="¡Es un placer ayudarte!\nEn mi opinion "+respuesta_del_modelo

                        outmodel['answer']=respuesta_del_modelo
                        return jsonify(outmodel),201
                
                except Exception as ex:
                        exceptionf= "Problemas en la generación de opinion gpt. Es posible que en el proceso de generación de opinión del modelo se excedió el limite de uso (rate limit) del modelo gpt de finalizacion de OpenAI. \n"+ str(ex)
                        return exceptionf
        
        


    except Exception as ex:
                exceptionf = "Problemas al utilizar coseno similar o cuestión del dataframe..\n" + str(ex)
                return exceptionf




if __name__=='__main__':
	#aplicacion de modo debug
	#app.run(debug=True)
    app.run(host='0.0.0.0')