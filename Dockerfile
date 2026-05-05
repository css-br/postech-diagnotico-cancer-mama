# Imagem oficial do TensorFlow com Jupyter pre-instalado (CPU only, sem GPU)
FROM tensorflow/tensorflow:2.16.1-jupyter

WORKDIR /app

COPY requirements.txt .

# Instala apenas as dependencias adicionais (TF e Jupyter ja estao na imagem base)
RUN pip install --no-cache-dir pandas numpy matplotlib seaborn scikit-learn pillow kaggle

COPY . .

EXPOSE 8888

CMD ["jupyter", "notebook", \
     "--ip=0.0.0.0", \
     "--allow-root", \
     "--no-browser", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''"]
