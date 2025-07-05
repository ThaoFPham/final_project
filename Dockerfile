#git remote add origin https://github.com/ThaoFPham/final_project.gitFROM python:3.10-slim
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 10000

CMD ["streamlit", "run", "class.py", "--server.port=10000", "--server.enableCORS=false"]