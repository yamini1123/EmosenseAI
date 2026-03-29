import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -----------------------
# Training Data
# -----------------------
texts = [
    # "I feel amazing today", "I am very happy", "best day ever",
    # "I am feeling great", "I am so happy",

    "I feel so low", "I am really upset", "nothing is going right",
    "I feel tired and sad", "I want to cry",

    "I am so frustrated", "I am angry", "I hate everything",
    "this is annoying", "I am pissed off",

    "I am scared", "I feel anxious", "I am worried",
    "I am nervous", "something feels wrong",

    "I completed my work", "today was normal",
    "I am going outside", "nothing special today", "just another day",
    
     "I feel underconfident because I have not achieved anything great so far","","","","",
    "I think people look down at me and also I feel judged","","","","",

    "hello","","","","",

    "I feel like I’m not good enough… I always mess things up.","","","","",
    "But others seem so confident, I don’t even feel close to them.","","","","",
    "What if I fail again?","","","","",
]

labels = [
    # "happy","happy","happy","happy","happy",
    "sad","sad","sad","sad","sad",
    "angry","angry","angry","angry","angry",
    "fear","fear","fear","fear","fear",
    "neutral","neutral","neutral","neutral","neutral",
    "underconfident","underconfident","underconfident","underconfident","underconfident",
    "judged","judged","judged","judged","judged",
    "first-text", "first-text", "first-text", "first-text", "first-text",
    "goodenough","goodenough","goodenough","goodenough","goodenough",
    "others", "others", "others", "others", "others",
    "fail","fail","fail","fail","fail",

]

# -----------------------
# Model Training
# -----------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

def predict_emotion(text):
    X_test = vectorizer.transform([text])
    return model.predict(X_test)[0]

# -----------------------
# Responses
# -----------------------
responses = {

    # "happy": [
    #     "That’s amazing! What made your day?",
    #     "Glad to hear that. Tell me more."
    # ],
    "sad": [
        "I’m really sorry you’re feeling this way. I’m here for you.\n Do you want to share about it more?",
        "That sounds tough. Do you want to talk about it?"
    ],
    "angry": [
        "That sounds frustrating. What happened?",
        "I understand your anger. Want to share more?"
    ],
    "fear": [
        "It’s okay to feel scared. I’m here with you.",
        "Take a deep breath. What’s worrying you?"
    ],
    "neutral": [
        "Got it. Tell me more.",
        "Alright. What else is on your mind?"

    ],
    "underconfident": [
        "It is okay to feel low sometimes, but remember that your current achievements do not define your potential. "
    ],
    "judged":[
        "Yeah! that feeling is very common and can be pretty heavy.When you feel underconfident your mind kind of goes into spotlight mode like everyone is watching and judging you. But there is a small truth here - people are already busy in their life but they might notice your small voice and hesitation so avoid eye contact."
    ],
  "first-text":[
         "Hey! How can I help you today?"],


 "goodenough":[
     "You’re being too hard on yourself—everyone makes mistakes, that doesn’t define you.",
],
 "others":[
  "Confidence grows with time, and you’re already stronger than you think."
],
 "fail":[
        "Then you’ll learn again—and that’s how you get better, step by step"
 ],

}

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="EmoSense Chatbot")

st.title("EmoSense Chatbot")
st.write("Talk to the chatbot and it will understand your emotions.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Predict emotion
    emotion = predict_emotion(user_input)
    reply = random.choice(responses[emotion])

    # Show bot response
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
