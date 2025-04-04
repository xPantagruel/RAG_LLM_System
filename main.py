import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import itertools
import threading
import time

# Make sure to install bitsandbytes by running: pip install bitsandbytes

def loading_spinner():
    spinner = itertools.cycle(['|', '/', '-', '\\'])
    while not stop_loading_event.is_set():
        print(f"\rGenerating response... {next(spinner)}", end="", flush=True)
        time.sleep(0.1)

def main():
    model_name = "mistralai/Mistral-7B-v0.1"

    # Define 4-bit quantization configuration
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float32
    )

    # Load the model and tokenizer
    print("Loading model... This might take a while.")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,   # Using new config object for 4-bit precision
        device_map="auto"                         # Automatically distribute model across available devices (GPU)
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # Confirm device
    print(f"Model is running on: {model.device}")

    while True:
        # Get user input
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("Exiting the chat. Goodbye!")
            break

        # Tokenize the input
        inputs = tokenizer(user_input, return_tensors="pt").to("cuda")  # Send inputs to GPU

        # Start the loading spinner in a separate thread
        global stop_loading_event
        stop_loading_event = threading.Event()
        spinner_thread = threading.Thread(target=loading_spinner)
        spinner_thread.start()

        # Generate response
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=150, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)

        # Stop the loading spinner
        stop_loading_event.set()
        spinner_thread.join()

        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Display the response
        print(f"\nMistral-7B: {response}\n")


if __name__ == "__main__":
    main()
