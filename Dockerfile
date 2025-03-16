FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"

EXPOSE 8000

CMD ["uvicorn", "--app-dir", "/app", "app:app", "--host", "0.0.0.0", "--port", "8000"]
