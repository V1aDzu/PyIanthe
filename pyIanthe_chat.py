import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pyIanthe_config

# --- Параметри моделі ---
author, model_name = pyIanthe_config.MODEL_ID.split("/")
model_dir = os.path.join(pyIanthe_config.FOLDER_MODELS, author, model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Завантажуємо модель та токенізатор локально ---
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    dtype="auto",
    device_map="auto"
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# --- Клас чат-бота ---
class ChatBot:
    def __init__(self, model, tokenizer, device, max_history=10):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_history = max_history  # Максимальна кількість попередніх повідомлень у історії
        self.messages = [
            {"role": "system", "content": "Ви — корисний помічник."}
        ]

    def chat(self, user_message: str, max_new_tokens: int = 200,
             temperature: float = 0.7, top_p: float = 0.9):
        # Додаємо повідомлення користувача
        self.messages.append({"role": "user", "content": user_message})

        # Обрізаємо історію до max_history, залишаючи перше системне повідомлення
        if len(self.messages) > self.max_history + 1:
            self.messages = [self.messages[0]] + self.messages[-self.max_history:]

        # Формуємо prompt через шаблон чату
        prompt = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Перетворюємо у тензори
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        # Генеруємо відповідь
        output_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )[0]

        # Беремо лише згенеровану частину (відповідь бота)
        generated_ids = output_ids[model_inputs.input_ids.shape[-1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Додаємо відповідь бота в історію
        self.messages.append({"role": "assistant", "content": response})

        return response

    def reset(self):
        self.messages = [
            {"role": "system", "content": "Ви — корисний помічник."}
        ]

# --- Інтерактивний чат ---
def interactive_chat():
    bot = ChatBot(model, tokenizer, device, max_history=10)
    print("🤖 PyIanthe (Qwen) – чат. Введіть /exit щоб вийти.")
    while True:
        try:
            user_input = input("Ви: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["/exit", "/quit"]:
                break
            result = bot.chat(user_input, max_new_tokens=200, temperature=0.7, top_p=0.9)
            print("Бот:", result)
        except KeyboardInterrupt:
            print("\nВихід з чату...")
            break
        except Exception as e:
            print("Помилка:", e)

if __name__ == "__main__":
    interactive_chat()
