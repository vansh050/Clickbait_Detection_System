import streamlit as st
from googleapiclient.discovery import build
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd

df = pd.read_csv('/content/Dataset.csv')
text_train = df['headline'].values

# Function to preprocess text for prediction
def preprocess_text(text, tokenizer, maxlen):

    tokenized_text = tokenizer.texts_to_sequences([text])
    padded_text = pad_sequences(tokenized_text, maxlen=maxlen)

    return padded_text

# Function to detect clickbait given a YouTube video title
def predict_clickbait(video_title, tokenizer, maxlen):
    model = load_model('clickbait_detection_model.keras')
    
    processed_title = preprocess_text(video_title, tokenizer, maxlen=maxlen)
    
    prediction = model.predict(processed_title)[0][0]
    
    if prediction >= 0.5:
        result = "Clickbait"
        text_color = "red"
    else:
        result = "Not Clickbait"
        text_color = "green"
    
    return result, text_color

def main():
    st.title('YouTube Video Clickbait Detection')

    vocab_size = 5000
    maxlen = 500
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(text_train)

    search_query = st.text_input('Enter Search Query:', '')

    if st.button('Search'):
        if search_query:
            youtube = build('youtube', 'v3', developerKey='AIzaSyCMe5cYUFM6HcC4o9vJy9TZUoDK5EUL_qk')
            
            search_request = youtube.search().list(q=search_query, part='snippet', type='video', maxResults=5)
            search_response = search_request.execute()
            
            for item in search_response['items']:
                video_title = item['snippet']['title']
                thumbnail_url = item['snippet']['thumbnails']['medium']['url']
                result, text_color = predict_clickbait(video_title, tokenizer, maxlen)
                st.image(thumbnail_url, caption=video_title, use_column_width=True)
                st.markdown(f'<p style="color:{text_color}">{video_title}</p>', unsafe_allow_html=True)
                st.write(f'<p style="color:{text_color}">Result: {result}</p>', unsafe_allow_html=True)
        else:
            st.warning('Please enter a search query.')

if __name__ == "__main__":
    main()
