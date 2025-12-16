# modules/pyIanthe_trainer

import os
import torch
from torch.utils.data import DataLoader

class TrainerWrapper:
    def __init__(self, model, tokenizer, train_dataset, eval_dataset=None, device="cuda",
                 checkpoint_dir="checkpoints", main_model_dir="main_model"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.main_model_dir = main_model_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.main_model_dir, exist_ok=True)

    def train(self, epochs_steps, batch_size=4, warmup_steps=50,
              save_every_steps=50, save_main_at_steps=None):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        total_steps = sum(epochs_steps)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        if save_main_at_steps is None:
            save_main_at_steps = []

        global_step = 0
        for epoch_idx, steps_in_epoch in enumerate(epochs_steps):
            print(f"\n===== Epoch {epoch_idx+1} | Steps: {steps_in_epoch} =====")
            dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
            step_iter = iter(dataloader)

            for step in range(1, steps_in_epoch + 1):
                try:
                    batch = next(step_iter)
                except StopIteration:
                    step_iter = iter(dataloader)
                    batch = next(step_iter)

                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch, labels=batch["input_ids"])
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                global_step += 1

                # Чекпоинт для восстановления
                if global_step % save_every_steps == 0:
                    cp_dir = os.path.join(self.checkpoint_dir, f"checkpoint-{global_step}")
                    os.makedirs(cp_dir, exist_ok=True)
                    self.model.save_pretrained(cp_dir)
                    self.tokenizer.save_pretrained(cp_dir)
                    print(f"✓ Чекпоинт сохранён: {cp_dir}")

                # Обновление основной модели на определённых шагах
                if global_step in save_main_at_steps:
                    self.model.save_pretrained(self.main_model_dir)
                    self.tokenizer.save_pretrained(self.main_model_dir)
                    print(f"★ Основная модель обновлена на шаге {global_step}")

            print(f"Epoch {epoch_idx+1} завершена.")

        # Финальное сохранение основной модели
        self.model.save_pretrained(self.main_model_dir)
        self.tokenizer.save_pretrained(self.main_model_dir)
        print(f"\n✓ Основная модель финально сохранена: {self.main_model_dir}")
