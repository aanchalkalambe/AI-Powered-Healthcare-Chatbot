import streamlit as st  
from transformers import pipeline  
import nltk  
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize  

# Download NLTK dependencies
nltk.download('stopwords')  
nltk.download('punkt')  

# Load pre-trained model
chatbot = pipeline("question-answering", model="deepset/bert-base-cased-squad2")

# Function to preprocess user input
def preprocess_input(user_input):  
    stop_words = set(stopwords.words('english'))  
    words = word_tokenize(user_input)  
    filtered_words = [w for w in words if w.lower() not in stop_words]  
    return ' '.join(filtered_words)  

# Function to generate healthcare responses
def healthcare_assistant(user_input):  
    user_input = user_input.lower()  # Convert input to lowercase for easy matching  

    common_responses = {
        "sneeze": "🤧 Frequent sneezing can be due to allergies, colds, or irritants. Avoid dust and strong scents. If symptoms persist, consider an allergy test.",
        
        "cough": "🗣️ A cough can result from colds, allergies, or acid reflux. Try warm honey-lemon tea 🍯🍋 and stay hydrated. Seek medical help if it lasts over two weeks.",
        
        "fever": "🌡️ A fever indicates your body is fighting an infection. Rest, drink fluids 💧, and take fever reducers if needed. See a doctor if it’s above 102°F or persistent.",
        
        "headache": "🤕 Headaches can be caused by stress, dehydration, or poor sleep. Drink water, rest, and avoid screens 📱. Seek help if it's severe or frequent.",
        
        "stomach ache": "🤢 Stomach pain may result from indigestion or stress. Drink warm fluids 🍵, avoid greasy foods, and rest. Seek medical help for persistent pain.",
        
        "cold": "🤧 The common cold causes congestion and a sore throat. Stay hydrated, rest, and use a humidifier. If symptoms persist beyond 10 days, seek medical care.",
        
        "sore throat": "😷 Gargle warm salt water 🧂, drink honey-lemon tea 🍯🍋, and avoid cold drinks. If your throat remains sore after a week, consult a doctor.",
        
        "dizzy": "😵‍💫 Dizziness can be due to dehydration, low blood sugar, or stress. Drink water 💦, sit down, and breathe deeply. If frequent, consult a doctor.",
        
        "fatigue": "😴 Feeling exhausted? Ensure adequate sleep, a nutritious diet 🥗, and hydration. If fatigue persists, consider checking for anemia or thyroid issues.",
        
        "chest pain": "⚠️ Seek urgent medical help if chest pain is sharp, spreads to the arm, or comes with shortness of breath. If mild, it may be heartburn or stress-related.",
        
        "anxiety": "💙 Try deep breathing, meditation 🧘, or talking to someone you trust. Anxiety is normal, but if overwhelming, consider professional support.",
        
        "depression": "💔 Depression is serious. Engage in activities you enjoy, talk to loved ones, and seek therapy if needed. You're not alone. 💙",
        
        "high blood pressure": "🩺 Reduce salt intake 🧂, exercise regularly 🚶, and manage stress. Monitor BP at home and consult a doctor if it's consistently high.",
        
        "insomnia": "🌙 Avoid screens before bed 📱, establish a bedtime routine, and try relaxation techniques. Seek help if sleep issues persist.",
        
        "urinary infection": "🚻 UTIs cause burning urination and urgency. Drink plenty of water 💦 and seek antibiotics if symptoms persist.",
        
        "muscle cramps": "💪 Cramps can be caused by dehydration or lack of minerals. Stretch, massage, and drink water. Eat bananas 🍌 for potassium.",
        
        "sinus congestion": "🤧 Steam inhalation, hydration, and nasal sprays can relieve sinus congestion. Avoid allergens and consider antihistamines if needed.",
        
        "heartburn": "🔥 Avoid spicy foods 🌶️, caffeine, and large meals. Eat slowly, and don’t lie down immediately after eating.",
        
        "ear infection": "👂 Ear pain may be due to an infection or wax buildup. Avoid inserting objects in your ear and seek medical advice if pain persists.",
        
        "constipation": "🍎 Eat fiber-rich foods, drink plenty of water, and stay active. If constipation lasts more than a week, consult a doctor.",
        
        "diarrhea": "💦 Stay hydrated and eat bland foods like bananas 🍌 and rice. Avoid dairy and spicy food. If symptoms persist, seek medical help.",
        
        "shortness of breath": "⚠️ Difficulty breathing can be serious. If severe, **seek emergency help**. For mild cases, try deep breathing exercises.",
        
        "allergy": "🌿 Allergies can cause sneezing, itching, and rashes. Identify triggers and consider antihistamines if necessary.",
        
        "hair loss": "🧴 Hair loss can be due to stress, poor diet, or hormonal changes. Use gentle shampoos, eat a balanced diet, and consult a doctor if excessive.",
        
        "bruising easily": "🍊 Bruising easily may indicate a vitamin deficiency. Increase vitamin C and K intake and consult a doctor if it happens frequently.",
        
        "joint pain": "🦵 Joint pain may be due to arthritis or strain. Apply ice, rest, and do gentle stretches. Seek medical help if persistent.",
        
        "nausea": "🤢 Nausea may result from motion sickness, indigestion, or infections. Sip ginger tea 🍵, stay hydrated, and rest.",
        
        "nosebleed": "👃 Lean forward slightly and pinch your nostrils for 5-10 minutes. Avoid picking your nose and use a humidifier if air is dry.",
        
        "sweating a lot": "💦 Excessive sweating can be caused by heat, anxiety, or an underlying condition. Stay hydrated and wear breathable clothing.",
        
        "hunger": "🍽️ Sudden hunger changes may indicate stress or blood sugar fluctuations. Maintain a balanced diet and eat at regular intervals.",
        
        "dry skin": "🧴 Dry skin can be due to weather changes or dehydration. Use a moisturizer and drink plenty of water."
    }

    # Check if user input matches any common symptoms
    for key in common_responses:
        if key in user_input:
            return common_responses[key]

    # If no match is found, use NLP model for answer generation
    context = """Common healthcare-related scenarios include symptoms of cough, cold, flu, and allergies along with medical guidance."""  
    response = chatbot(question=user_input, context=context)  
    return response['answer']  

# Streamlit UI
def main():  
    st.title("Healthcare Assistant Chatbot 🩺")  
    st.write("👩‍⚕️ Ask me about common symptoms, and I'll provide some basic guidance.")  

    user_input = st.text_input("Enter your symptoms or question:")  
    submit = st.button("Submit")  

    if submit and user_input:  
        st.write("User:", user_input)  
        response = healthcare_assistant(user_input)  
        st.write("Healthcare Assistant:", response)  
    elif submit and not user_input:  
        st.warning("⚠️ Please enter a query!")  

if __name__ == "__main__":  
    main()
