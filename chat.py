import random
import json
import os
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import speech_recognition as sr
import pyttsx3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# with open('intents.json', 'r') as json_data:
#     intents = json.load(json_data)

# Get the current working directory
current_directory = os.path.dirname(os.path.abspath(__file__))
# Specify the absolute file path
file_path = os.path.join(current_directory, 'intents.json')

with open(file_path, 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]


    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                a = text_to_speech(random.choice(intent['responses']))
                #print("The ouptu is", a)
                return a
    
    else:
        text_to_speech("I do not understand...")
    return "I do not understand..."

def text_to_speech(text):
    # Initialize the text-to-speech engine
    engine = pyttsx3.init()

    # Set properties (optional)
    engine.setProperty('rate', 150)  # Speed of speech

    # Convert the given text to speech
    engine.say(text)

    # Wait for the speech to finish
    engine.runAndWait()
    return text

def take_voice_input():
    # Create a speech recognition object
    recognizer = sr.Recognizer()

    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        print("Say something...")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise

        try:
            audio = recognizer.listen(source, timeout=5)  # Listen for up to 5 seconds

            # Recognize speech using Google Web Speech API
            text = recognizer.recognize_google(audio)
            print("You said: " + text)
            return text

        except sr.UnknownValueError:
            print("Sorry, could not understand audio.")
        except sr.RequestError as e:
            print(f"Could not request results from Google Web Speech API; {e}")


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = take_voice_input()
        #print(f"you said: f{sentence}")

        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)
        text_to_speech(resp)

