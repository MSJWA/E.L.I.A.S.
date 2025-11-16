from nlp_intent_core import parse_intent

print("ELIAS Core App Running...\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    output = parse_intent(user_input)
    print("Intent Detected:", output)

def route_intent(result):

    intent = result["intent"]
    entities = result["entities"]

    if intent == "open_website":
        print("-> would open website plugin here")

    elif intent == "search_youtube":
        print("-> would run youtube search plugin here")

    else:
        print("Unknown or unsupported intent.")

result = parse_intent(user_input)
route_intent(result)
