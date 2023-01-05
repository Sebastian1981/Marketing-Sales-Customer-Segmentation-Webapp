FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install --upgrade -r requirements.txt
EXPOSE 8501
ENTRYPOINT ["streamlit", "run"]
CMD ["streamlit_app.py"]