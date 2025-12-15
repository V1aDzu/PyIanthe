# modules/pyIanthe_test_callbacks.py
import os
import json
from datetime import datetime
from transformers import TrainerCallback
from modules.pyIanthe_metrics import test_text_generation, test_eval_dataset

class TestingCallback(TrainerCallback):
    """Callback для запуску тестів під час тренування"""
    
    def __init__(self, model, tokenizer, device, test_prompts, eval_dataset, config, logger):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.test_prompts = test_prompts
        self.eval_dataset = eval_dataset
        self.logger = logger
        self.current_epoch = 0
        self.CONTEXT_LENGTH = config.CONTEXT_LENGTH
        self.DEBUG = config.DEBUG
        self.REPORTS_DIR = config.REPORTS_DIR
        self.EVAL_STEPS = config.EVAL_STEPS
        self.TEST_ENABLED = config.TEST_ENABLED
        self.EVAL_ENABLED = config.EVAL_ENABLED
        self.EVAL_TESTS_COUNT = config.EVAL_TESTS_COUNT
        self.MAX_NEW_TOKENS = config.GEN_TEST_MNEW_TOKENS
        self.TEMPERATURE = config.GEN_TEST_TEMPERATURE
        self.TOP_P = config.GEN_TEST_TOP_P
     
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Відслідковує початок епохи"""
        self.current_epoch = int(state.epoch) if state.epoch is not None else 0

    def on_step_end(self, args, state, control, **kwargs):
        """Викликається після кожного кроку"""
        if state.global_step % self.EVAL_STEPS != 0 or state.global_step == 0:
            return

        if not self.TEST_ENABLED and not self.EVAL_ENABLED:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"ПРОМІЖНЕ ТЕСТУВАННЯ")
        self.logger.info(f"Епоха: {self.current_epoch+1}, Крок: {state.global_step}")
        self.logger.info(f"Час: {timestamp}")
        self.logger.info(f"{'='*60}")

        test_results = {
            "timestamp": timestamp,
            "epoch": self.current_epoch+1,
            "global_step": state.global_step,
            "text_generation": None,
            "eval_dataset": None
        }

        # Тест генерації тексту
        if self.TEST_ENABLED and self.test_prompts:
            self.logger.info("\n[TEST] Тестування генерації тексту...")
            text_results = test_text_generation(
                model=self.model,
                tokenizer=self.tokenizer,
                prompts=self.test_prompts,
                device=self.device,
                max_new_tokens = self.MAX_NEW_TOKENS,
                temperature = self.TEMPERATURE,
                top_p = self.TOP_P
            )
            test_results["text_generation"] = text_results

            # Виводимо приклад
            first_prompt = self.test_prompts[0]
            result = text_results[first_prompt]
            self.logger.info(f"  Промпт: {first_prompt}")
            self.logger.info(f"  Текст: {result['generated_text'][:100]}...")
            self.logger.info(f"  Perplexity: {result['perplexity']:.2f}")
            self.logger.info(f"  Токенів: {result['total_tokens']}")

        # Тест на eval датасеті
        if self.EVAL_ENABLED and self.eval_dataset:
            self.logger.info("\n[EVAL] Тестування на eval датасеті...")
            eval_results = test_eval_dataset(
                model=self.model,
                tokenizer=self.tokenizer,
                eval_dataset=self.eval_dataset,
                device=self.device,
                max_samples = self.EVAL_TESTS_COUNT,
                context_length = self.CONTEXT_LENGTH,
                debug = self.DEBUG,
                logger=self.logger
            )
            if eval_results:
                test_results["eval_dataset"] = eval_results
                self.logger.info(f"  Зразків: {eval_results['num_samples']}")
                self.logger.info(f"  Avg Loss: {eval_results['avg_loss']:.4f}")
                self.logger.info(f"  Avg Perplexity: {eval_results['avg_perplexity']:.2f}")

        # Зберігаємо JSON звіт тільки якщо є результати
        if test_results["text_generation"] or test_results["eval_dataset"]:
            report_filename = f"test_epoch{self.current_epoch}_step{state.global_step}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_path = os.path.join(self.REPORTS_DIR, report_filename)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(test_results, f, ensure_ascii=False, indent=2)
            self.logger.info(f"\n✓ Звіт збережено: {report_filename}")
        else:
            self.logger.info(f"\n[INFO] Тести вимкнені, звіт не створено")

        self.logger.info(f"{'='*60}\n")
