import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os  # We need this to read the port from Render

# --- 1. Load the Model ---
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct", 
    dtype="auto", 
    trust_remote_code=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct", 
    trust_remote_code=True
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)
print("Model loaded successfully!")

# --- 2. Define the Chat Function ---
def get_studdy_lite_reply(prompt):
    print(f"Received prompt: {prompt}")

    messages = [
        {"role": "user", "content": prompt},
    ]

    formatted_prompt = pipe.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    generation_args = {
        "max_new_tokens": 150,
        "return_full_text": False,
        "temperature": 0.7,
        "do_sample": True,
    }

    try:
        output = pipe(formatted_prompt, **generation_args)
        reply_text = output[0]["generated_text"]
        print(f"Sending reply: {reply_text}")
        return {"reply": reply_text}

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

# --- 3. Get Port and Host from Environment ---
# This is the CRITICAL change for Render
port_number = int(os.environ.get('PORT', 7860)) # Render provides a 'PORT' variable
host_ip = '0.0.0.0' # This tells it to listen on all available IPs

# --- 4. Create and Launch the Gradio API ---
print(f"Launching Gradio interface on {host_ip}:{port_number}...")
demo = gr.Interface(
    fn=get_studdy_lite_reply,
    inputs=gr.Textbox(label="Prompt"),
    outputs=gr.JSON(label="Reply"),
    title="Studdy Lite API",
    description="A lightweight API for the Student Companion app, powered by Phi-3 Mini."
)

# Launch the app on the correct host and port
demo.launch(server_name=host_ip, server_port=port_number)
