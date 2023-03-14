FROM python:3.9
WORKDIR /flask_app
RUN python -m pip install --upgrade pip
COPY . /flask_app
RUN pip install -r requirements.txt
EXPOSE 8000
ENTRYPOINT [ "python" ]
CMD python ./app.py