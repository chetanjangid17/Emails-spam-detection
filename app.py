import streamlit as st
import re
import pickle
import string
import nltk
import numpy as np
import altair as alt
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Function to preprocess the text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Function to check for IP address
def contains_ip_address(text):
    ip_pattern = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
    return bool(re.search(ip_pattern, text))

# Function to make predictions
def make_predictions(text):
    transformed_text = transform_text(text)
    vector_input = tfidf.transform([transformed_text])
    prediction_proba = model_nb.predict_proba(vector_input)[0]
    result = model_nb.predict(vector_input)[0]
    return transformed_text, prediction_proba, result

# Load pre-trained models
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model_nb = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Set Streamlit page configuration
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="💌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set Streamlit app title
st.title(" Email Spam Classifier")

# Session state for login/logout
if 'is_logged_in' not in st.session_state:
    st.session_state.is_logged_in = False

# Login Page
if not st.session_state.is_logged_in:
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "Abhichetan" and password == "Abhishek@12":
            st.session_state.is_logged_in = True
            st.success("Login successful!")
          
        else:
            st.error("Invalid credentials. Please try again.")
else:
    # Logout option
    if st.button("Logout"):
        st.session_state.is_logged_in = False
        st.sidebar.success("Logged out successfully!")

    # Choice Option Page
    selected_page = st.sidebar.radio("Select Page", ["Sender Page", "Prediction Page", "Spam Page", "Ham Page"])

    # Sender Page
    if selected_page == "Sender Page":
        st.markdown(
            """
            ### Sender Page
            - Enter the email message in the text area.
            - Click the 'Send' button to send the email.
            """
        )

        # Add input text area for the user to enter the message
        send_message = st.text_area("Enter the email message")

        # Add a button to trigger sending
        if st.button('Send'):
            # Check if the input text is empty
            if not send_message:
                st.warning("Please enter an email message.")
            else:
                # Simulate classifying the sent email
                transformed_msg, proba, result = make_predictions(send_message)

                # Display prediction information for sent message
                st.subheader("Transformed Text:")
                st.write(transformed_msg)

                if result == 1:
                    st.success("Email Sent ")
                    # Store the spam email
                    with open("spam_messages.txt", "a") as spam_file:
                        spam_file.write(f"{transformed_msg}\n\n")
                else:
                    st.success("Email Sent ")
                    # Store the ham email
                    with open("ham_messages.txt", "a") as ham_file:
                        ham_file.write(f"{transformed_msg}\n\n")

    # Prediction Page
    elif selected_page == "Prediction Page":
        st.markdown(
            """
            ### Predict Email:
            - Enter the email message in the text area.
            - Click the 'Predict' button to classify the email.
            """
        )

        # Add input text area for the user to enter the message
        input_sms = st.text_area("Enter the email message", key="email_input")

        # Add a button to trigger the prediction
        if st.button('Predict'):
            # Use the input for prediction
            input_text = input_sms.strip()  # Remove leading and trailing whitespaces

            # Check if the input text is empty
            if not input_text:
                st.warning("Please enter an email message.")
            else:
                # Display loading spinner
                with st.spinner("Predicting..."):
                    # Preprocess the input message
                    transformed_sms = transform_text(input_text)

                    # Check for IP address
                    contains_ip = contains_ip_address(input_text)

                    # Vectorize the input
                    vector_input = tfidf.transform([transformed_sms])

                    # Make a prediction using Naive Bayes
                    prediction_proba = model_nb.predict_proba(vector_input)[0]
                    result_nb = model_nb.predict(vector_input)[0]

                    # Display prediction probability
                    st.subheader("Prediction Probability:")
                    st.write(f"Probability of Not Spam: {prediction_proba[0]:.2f}")
                    st.write(f"Probability of Spam: {prediction_proba[1]:.2f}")

                    # Display prediction result
                    st.subheader("Prediction Result:")
                    if result_nb == 1:
                        st.success("Spam")
                    else:
                        st.success("Not Spam")

                    # Display whether the message contains an IP address
                    st.subheader("Contains IP Address:")
                    st.write(contains_ip)

                    # Visualization: Bar chart of prediction probabilities
                    st.bar_chart({'Not Spam': prediction_proba[0], 'Spam': prediction_proba[1]})

                    # Visualization: Confusion Matrix (replace with actual confusion matrix)
                    confusion_matrix = np.array([[50, 10], [5, 80]])
                    confusion_df = pd.DataFrame(confusion_matrix, index=["Actual Not Spam", "Actual Spam"],
                                                columns=["Predicted Not Spam", "Predicted Spam"])

                    st.subheader("Confusion Matrix:")
                    st.table(confusion_df.style.set_table_styles([
                        {'selector': 'th', 'props': [('text-align', 'center')]},
                        {'selector': 'td', 'props': [('text-align', 'center')]}
                    ]))

                    # Visualization: Confusion Matrix Graph
                    confusion_chart = alt.Chart(confusion_df.reset_index().melt('index')).mark_rect().encode(
                        x='index:O',
                        y='variable:O',
                        color='value:Q',
                        tooltip=['index:N', 'variable:N', 'value:Q']
                    ).properties(
                        width=400,
                        height=300
                    )

                    # Display the Altair chart in Streamlit
                    st.altair_chart(confusion_chart)

    # Spam Page
    elif selected_page == "Spam Page":
        st.markdown("### Spam Emails:")
        # Display stored spam emails
        with open("spam_messages.txt", "r") as spam_file:
            spam_emails = spam_file.read()
            st.text_area("Spam Emails", spam_emails)

    # Ham Page
    elif selected_page == "Ham Page":
        st.markdown("### Ham Emails:")
        # Display stored ham emails
        with open("ham_messages.txt", "r") as ham_file:
            ham_emails = ham_file.read()
            st.text_area("Ham Emails", ham_emails)

# Footer
st.markdown("---")
st.markdown("Developed by Abhishek Ray and Chetan Jangid")
