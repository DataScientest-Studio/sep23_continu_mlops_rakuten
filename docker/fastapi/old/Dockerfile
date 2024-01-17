FROM python:3.9
RUN pip3 install fastapi uvicorn
WORKDIR /app
RUN mkdir /data
COPY ./api /app
COPY ./data /data
RUN pip3 install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "api_v2:api", "--host", "0.0.0.0", "--reload","--port", "8000"]