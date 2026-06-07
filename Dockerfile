FROM python:3.14-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    numpy scipy scikit-learn nltk openai python-dotenv matplotlib

RUN python -c "import nltk; nltk.download('vader_lexicon', download_dir='/usr/local/share/nltk_data'); nltk.download('sentiwordnet', download_dir='/usr/local/share/nltk_data'); nltk.download('wordnet', download_dir='/usr/local/share/nltk_data'); nltk.download('punkt', download_dir='/usr/local/share/nltk_data'); nltk.download('punkt_tab', download_dir='/usr/local/share/nltk_data'); nltk.download('stopwords', download_dir='/usr/local/share/nltk_data'); nltk.download('averaged_perceptron_tagger_eng', download_dir='/usr/local/share/nltk_data'); nltk.download('omw-1.4', download_dir='/usr/local/share/nltk_data')"

ENV NLTK_DATA=/usr/local/share/nltk_data

COPY . .

CMD ["python", "train.py", "--mock"]
