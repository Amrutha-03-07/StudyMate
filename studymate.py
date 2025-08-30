import os
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
if torch.cuda.is_available():
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"🎉 GPU found: {gpu_name}")
else:
    device = "cpu"
    print("🐌 No GPU found, using CPU (slower but still works).")
class AIBuddy:
    def __init__(self, model_name="ibm-granite/granite-3.2b-instruct"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None

    def wake_up(self):
        """Load tokenizer + model safely"""
        try:
            print("📖 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )

            print("🧠 Loading model...")
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            self.model.eval()

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception:
                print("⚠️ torch.compile not supported here, skipping.")

            print("✅ Model loaded successfully!")
        except Exception as e:
            print("❌ Error loading model:", e)

    def chat(self, prompt: str, max_new_tokens: int = 150):
        """Generate a friendly response"""
        if self.model is None or self.tokenizer is None:
            return "⚠️ Please call wake_up() first!"

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        )
        for k in inputs:
            inputs[k] = inputs[k].to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        response_ids = outputs[0][prompt_len:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return response.strip()
buddy = AIBuddy()
buddy.wake_up()
def chat_with_buddy(user_input, history=[]):
    reply = buddy.chat(user_input)
    history.append((user_input, reply))
    return history, history

with gr.Blocks() as app:
    gr.Markdown("## 🤖 AI Buddy - Granite Powered")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Type your question here...")
    clear = gr.Button("Clear")

    def respond(user_input, chat_history):
        return chat_with_buddy(user_input, chat_history)

    msg.submit(respond, [msg, chatbot], [chatbot, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)
app.launch(
    share=True,
    server_name="0.0.0.0",
    server_port=7860,
    enable_queue=True
)

