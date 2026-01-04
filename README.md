# IA aplicada a E-commerce: Análisis de Sentimientos y Predicción de Ventas


## Descripción


Este proyecto aborda la aplicación de técnicas de Inteligencia Artificial y Aprendizaje Automático para el análisis de opiniones de usuarios y la predicción del comportamiento de ventas en un entorno de comercio electrónico.  
Se trabaja con datos de productos y reseñas de Amazon, utilizando modelos supervisados de clasificación y redes neuronales profundas.


## Objetivos


- Analizar el sentimiento de reseñas de productos mediante técnicas de Procesamiento del Lenguaje Natural (NLP)
- Predecir si un producto tendrá ventas o no a partir de variables numéricas relevantes
- Evaluar el desempeño de distintos enfoques de modelado
- Comprender el impacto de modelos lineales y no lineales en problemas reales


## Hipótesis


A partir de variables como precio, descuento, calificación de usuarios y sentimiento de las reseñas, es posible predecir el comportamiento de ventas de un producto, aun cuando las relaciones entre las variables no sean lineales.


## Modelos Implementados


### Regresión Logística
- Tipo de aprendizaje: Supervisado
- Tipo de problema: Clasificación binaria
- Justificación: modelo base interpretable para establecer una línea de referencia (baseline)


### Red Neuronal Artificial


Se implementó una red neuronal feedforward densa para clasificación binaria utilizando la API de Keras.  
La arquitectura del modelo consta de tres capas totalmente conectadas:  


- Una primera capa oculta de 128 neuronas con función de activación ReLU, encargada de aprender combinaciones no lineales de las variables de entrada.  
- Una segunda capa oculta de 64 neuronas con activación ReLU, que refina las representaciones aprendidas.  
- Una capa de salida con una única neurona y función de activación sigmoide, utilizada para estimar la probabilidad de que un producto genere ventas.  


La elección de funciones de activación ReLU permite una convergencia rápida del entrenamiento y una correcta propagación del gradiente, mientras que la función sigmoide resulta adecuada para el objetivo de clasificación binaria.  


Durante el entrenamiento se observó que el modelo alcanza altos valores de accuracy en pocas épocas (Epoch), lo cual se explica por la capacidad del modelo en relación con la complejidad del problema y la claridad de las señales presentes en los datos. Este comportamiento sugiere que la red posee suficiente poder de representación para capturar rápidamente las relaciones subyacentes entre las variables.  


## Análisis de Sentimientos


Para el procesamiento de texto se aplican técnicas de NLP, incluyendo:  
- Limpieza y normalización del texto
- Tokenización  
- Representación vectorial del lenguaje  


Se introduce el uso de modelos de lenguaje preentrenados como BERT, destacando su enfoque bidireccional para la comprensión contextual del texto.  


## Preprocesamiento de Datos


- Limpieza de símbolos monetarios y emojis  
- Conversión de variables a formato numérico  
- Análisis y tratamiento de valores nulos  
- Normalización de datos  
- Separación del conjunto de datos en entrenamiento y prueba  


## Entrenamiento y Evaluación

Los modelos se entrenan utilizando conjuntos separados de entrenamiento y validación.  
El desempeño se evalúa mediante métricas de clasificación como:
- Accuracy  
- Curvas de pérdida (loss)  
- Curvas de entrenamiento y validación  


## Tecnologías Utilizadas


# Lenguaje y entorno:
- Python  
- Jupyter Notebook  


# Manipulación y análisis de datos
- Pandas  
- NumPy  


# Visualización
- Matplotlib  
- Seaborn  


# Machine Learning
- Scikit-learn  
- LogisticRegression  
- Train/Test Split  
- StandardScaler  
- LabelEncoder  
- Métricas de clasificación (accuracy, precision, recall, F1, confusion matrix)  


# Deep Learning
- TensorFlow  
- Keras  
- Redes neuronales densas  
- Métricas (Precision, Recall)  
- Tokenizer y pad_sequences  
- Capas Embedding, LSTM, Bidirectional  


# NLP y Transformers
- BertTokenizer  
- BertForSequenceClassification  
- pipeline  


# Backend de Deep Learning y Utilidades
- PyTorch  
- torch  
- AdamW  
- autocast, GradScaler  


# Utilidades
- tqdm  

## Conclusión

Los resultados obtenidos validan la hipótesis planteada, demostrando que las técnicas de Inteligencia Artificial permiten capturar patrones complejos en los datos y predecir de manera efectiva el comportamiento de ventas.  
El proyecto evidencia la utilidad de combinar modelos clásicos de aprendizaje automático con enfoques modernos de aprendizaje profundo y procesamiento del lenguaje natural.  
