# modules/pyIanthe_metrics.py
import torch

def compute_perplexity(logits, labels):
    """Обчислює perplexity"""
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return torch.exp(loss.mean()).item()

def compute_text_metrics(text):
    """Обчислює метрики тексту"""
    tokens = text.split()
    total_tokens = len(tokens)
    meaningful_tokens = sum(1 for t in tokens if t.isalnum())
    perc_meaningful = meaningful_tokens / total_tokens * 100 if total_tokens else 0
    return total_tokens, perc_meaningful

def test_text_generation(model, tokenizer, prompts, device, max_new_tokens, temperature, top_p):
    """Тестує генерацію тексту на прикладах"""
    model.eval()
    results = {}

    generation_config = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id
    }
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_config)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            logits = model(**inputs).logits
            ppl = compute_perplexity(logits, inputs["input_ids"])
            total_tokens, perc_meaningful = compute_text_metrics(generated_text)
            
            results[prompt] = {
                "generated_text": generated_text,
                "perplexity": ppl,
                "total_tokens": total_tokens,
                "perc_meaningful_tokens": perc_meaningful
            }
    
    model.train()
    return results

def test_eval_dataset(model, tokenizer, eval_dataset, device, max_samples, context_length, debug=False, logger=None):
    """Тестує модель на eval датасеті"""
    if eval_dataset is None:
        return None
        
    model.eval()
    
    test_samples = min(max_samples, len(eval_dataset))
    eval_subset = eval_dataset.select(range(test_samples))
    
    total_loss = 0
    total_perplexity = 0
    num_samples = 0
    
    with torch.no_grad():
        for sample in eval_subset:
            try:
                text = sample.get("text", "")
                if not text or not isinstance(text, str):
                    continue
                
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=context_length).to(device)
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
                ppl = torch.exp(outputs.loss).item()
                
                total_loss += loss
                total_perplexity += ppl
                num_samples += 1
                
            except Exception as e:
                if debug and logger:
                    logger.warning(f"Помилка обробки зразка: {e}")
                continue
    
    model.train()
    
    if num_samples == 0:
        return None
    
    return {
        "num_samples": num_samples,
        "avg_loss": total_loss / num_samples,
        "avg_perplexity": total_perplexity / num_samples
    }
