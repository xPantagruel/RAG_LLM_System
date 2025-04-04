import itertools
import threading
import time
import torch

# Define a threading event to control the spinner
stop_loading_event = threading.Event()

def loading_spinner():
    spinner = itertools.cycle(['|', '/', '-', '\\'])
    start_time = time.time()  # Track start time
    
    while not stop_loading_event.is_set():
        elapsed_time = time.time() - start_time
        print(f"\rGenerating response... {next(spinner)} (Elapsed Time: {elapsed_time:.2f} seconds)", end="", flush=True)
        time.sleep(0.1)

def generate_response(model, tokenizer, inputs):
    # Start the loading spinner
    global stop_loading_event
    stop_loading_event.clear()  # Make sure the event is reset
    spinner_thread = threading.Thread(target=loading_spinner)
    spinner_thread.start()

    try:
        print("\n[INFO] Starting generation process...")
        
        # Start generation process
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        print("[INFO] Generation process completed.")
    finally:
        # Stop the loading spinner properly
        stop_loading_event.set()
        spinner_thread.join()
    
    # Decode the output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
