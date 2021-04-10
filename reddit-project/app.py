import streamlit as st
import pickle

st.image('./image.jpeg')

st.title('Which subreddit should you post to?')

st.write('r/books and r/writing have overlapping content')
st.write('Which subreddit should you post to?')

with open('models/reddit_pipe.pkl', mode='rb') as pickle_in:
    pipe = pickle.load(pickle_in)

user_text = st.text_input('Please input your reddit post:', value="I love books! (Replace with your text)")

predicted_subreddit = pipe.predict([user_text])

st.write(f'You should post this on r/{predicted_subreddit}')

answer = st.selectbox('Does that seem right?', ["Select 'Yes' or 'No' from this dropdown",'Yes','No'])

if answer == 'Yes':
    st.balloons()
elif answer =='No':
    st.write('Sorry about that.')
else:
    pass
