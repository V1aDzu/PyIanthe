# pyIanthe_chat.py
"""
🤖 PyIanthe Chatbot - Інтерактивний чат з Qwen
"""
import os
import torch
import re
from typing import Dict, Any
import pyIanthe_config
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Визначаємо де лежить модель ---
author, model_name = pyIanthe_config.MODEL_ID.split("/")
model_path = os.path.join(pyIanthe_config.FOLDER_MODELS, author, model_name)

# --- Завантажуємо модель та токенізатор ---
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# --- Спеціальні токени для математичних виразів ---
SPECIAL_TOKEN_MAP = {
    '<plu>': '+', '<min>': '-', '<mul>': '*', '<div>': '/',
    '<equ>': '=', '<obr>': '(', '<cbr>': ')', '<num>': '',
    '<que>': '?', '<exl>': '!', '<sym>': '', '<sep>': ' | ',
    '<bos>': '', '<eos>': ''
}

def decode_special_tokens(text: str) -> str:
    result = text
    for token, replacement in SPECIAL_TOKEN_MAP.items():
        result = result.replace(token, replacement)
    return result.strip()

def encode_math_expression(expr: str) -> str:
    replacements = {'+':'<plu>','-':'<min>','*':'<mul>','/':'<div>','=':'<equ>','(':'<obr>',')':'<cbr>'}
    for s,t in replacements.items(): expr = expr.replace(s,t)
    return expr

def preprocess_user_message(message: str) -> str:
    is_math = any(op in message for op in ['+','-','*','/','=','(',')'])
    if is_math: message = encode_math_expression(message)
    if message.endswith('?'): message = message.replace('?','<que>')
    elif message.endswith('!'): message = message.replace('!','<exl>')
    return message

# --- Клас чат-бота ---
class ChatBot:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.conversation_history = []
        self.max_history_turns = 3

    def format_prompt(self, user_message: str, use_history=True) -> str:
        processed = preprocess_user_message(user_message)
        if use_history and self.conversation_history:
            recent = self.conversation_history[-self.max_history_turns:]
            history_parts = [f"User: {u}<sep>Assistant: {a}" for u,a in recent]
            history_str = "<sep>".join(history_parts)
            prompt = f"<bos>{history_str}<sep>User: {processed}<sep>Assistant:"
        else:
            prompt = f"<bos>User: {processed}<sep>Assistant:"
        return prompt

    def chat(self, user_message: str, max_tokens=50, temperature=0.7, top_k=40, top_p=0.9, use_history=True):
        prompt = self.format_prompt(user_message, use_history)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        full_text = self.tokenizer.decode(output[0], skip_special_tokens=False)
        # берем только ответ после последнего Assistant:
        if "Assistant:" in full_text:
            answer = full_text.split("Assistant:")[-1]
        else:
            answer = full_text
        answer = decode_special_tokens(answer.replace("<eos>","").strip())
        self.conversation_history.append((preprocess_user_message(user_message), answer))
        if len(self.conversation_history) > 10: self.conversation_history = self.conversation_history[-10:]
        return answer

    def clear_history(self):
        self.conversation_history = []

    def get_history(self):
        return self.conversation_history

# --- Математичний eval ---
def evaluate_math_expression(bot: ChatBot, expression: str) -> Dict[str, Any]:
    response = bot.chat(f"Обчисли {expression}", max_tokens=30, temperature=0.3)
    match = re.search(r'<equ><num>(\d+)', response)
    result = {
        'expression': expression,
        'model_response': response,
        'decoded_response': decode_special_tokens(response),
        'extracted_result': int(match.group(1)) if match else None
    }
    try:
        expected = eval(expression.replace(' ',''))
        result['expected_result'] = expected
        result['is_correct'] = (result['extracted_result'] == expected)
    except:
        result['expected_result'] = None
        result['is_correct'] = None
    return result

# --- Інтерактивний чат ---
def interactive_chat():
    print("🤖 PyIanthe Chatbot - Інтерактивний режим")
    bot = ChatBot(model, tokenizer, device)
    settings = {'temperature':0.7, 'max_tokens':50, 'top_k':40, 'top_p':0.9, 'use_history':True}

    while True:
        try:
            user_input = input("\n💬 Ви: ").strip()
            if not user_input: continue
            if user_input.startswith('/'):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                if cmd in ['/quit','/exit','/q']: break
                elif cmd == '/clear': bot.clear_history(); print("✅ Історія очищена")
                elif cmd == '/history':
                    history = bot.get_history()
                    for i,(u,a) in enumerate(history,1):
                        print(f"{i}. Ви: {decode_special_tokens(u)} | Бот: {decode_special_tokens(a)}")
                elif cmd.startswith('/temp') and len(parts)==2: settings['temperature']=float(parts[1])
                elif cmd.startswith('/tokens') and len(parts)==2: settings['max_tokens']=int(parts[1])
                elif cmd.startswith('/context') and len(parts)==2:
                    settings['use_history'] = parts[1].lower() in ['on','yes','так']
                elif cmd.startswith('/math') and len(parts)==2:
                    result = evaluate_math_expression(bot, parts[1])
                    print(f"🧮 {result}")
                else: print("❌ Невідома команда")
                continue
            response = bot.chat(
                user_input,
                max_tokens=settings['max_tokens'],
                temperature=settings['temperature'],
                top_k=settings['top_k'],
                top_p=settings['top_p'],
                use_history=settings['use_history']
            )
            print(f"🤖 PyIanthe: {response}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Помилка: {e}")

# --- Тестування ---
def test_chatbot():
    print("🧪 Тестування PyIanthe Chatbot")
    bot = ChatBot(model, tokenizer, device)
    test_cases = ["Привіт!","Скільки буде 5+3?","Що таке AI?"]
    for q in test_cases:
        print(f"👤 Запит: {q}")
        print(f"🤖 Відповідь: {bot.chat(q)}")

if __name__=="__main__":
    import sys
    if len(sys.argv)>1 and sys.argv[1]=='test': test_chatbot()
    else: interactive_chat()
