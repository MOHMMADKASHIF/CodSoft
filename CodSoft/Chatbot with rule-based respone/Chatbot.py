# Simple Rule-Based Chatbot

def chatbot_response(user_input):
    # Convert user input to lowercase to make the bot case-insensitive
    user_input = user_input.lower()

    # Predefined responses
    if "hello" in user_input or "hi" in user_input:
        return "Hello! How can I help you today?"
    elif "how are you" in user_input:
        return "I'm just a bot, but I'm doing great! How about you?"
    elif "your name" in user_input:
        return "I'm a simple chatbot created to assist you."
    elif "what can you do" in user_input:
        return "I can respond to simple questions and have basic conversations with you."
    elif "bye" in user_input or "goodbye" in user_input:
        return "Goodbye! Have a great day!"
    else:
        return "I'm sorry, I didn't understand that. Can you please rephrase?"

# Chatbot interaction loop
print("Chatbot: Hi! I'm a simple chatbot. Type 'bye' to exit.")
while True:
    user_input = input("You: ")
    if "bye" in user_input.lower():
        print("Chatbot: Goodbye! Have a great day!")
        break
    response = chatbot_response(user_input)
    print(f"Chatbot: {response}")