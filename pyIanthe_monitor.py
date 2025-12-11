"""
Розширений монітор тренування з графіками та автооновленням
Використання: python training_monitor_advanced.py
"""
import os
import sys
import json
import time
from datetime import datetime, timedelta
import glob
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np

# Спроба імпорту конфігу
try:
    import pyIanthe_config
    MONITOR_INTERVAL = pyIanthe_config.MONITOR_INTERVAL
    CHECKPOINT_DIR = pyIanthe_config.FOLDER_CHECKPOINTS
    REPORTS_DIR = pyIanthe_config.FOLDER_REPORTS
except:
    MONITOR_INTERVAL = 2
    CHECKPOINT_DIR = "checkpoints"
    REPORTS_DIR = "reports"

# Налаштування matplotlib для інтерактивного режиму
plt.ion()

class TrainingMonitor:
    def __init__(self, max_history=1000):
        self.max_history = max_history
        self.loss_history = deque(maxlen=max_history)
        self.lr_history = deque(maxlen=max_history)
        self.grad_norm_history = deque(maxlen=max_history)
        self.step_history = deque(maxlen=max_history)
        self.time_history = deque(maxlen=max_history)
        
        # Створюємо фігуру з графіками
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle('🚀 PyIanthe Training Monitor', fontsize=16, fontweight='bold')
        
        # Налаштування осей
        self.ax_loss = self.axes[0, 0]
        self.ax_lr = self.axes[0, 1]
        self.ax_grad = self.axes[1, 0]
        self.ax_speed = self.axes[1, 1]
        
        # Лінії графіків
        self.line_loss, = self.ax_loss.plot([], [], 'b-', linewidth=2, label='Loss')
        self.line_lr, = self.ax_lr.plot([], [], 'g-', linewidth=2, label='Learning Rate')
        self.line_grad, = self.ax_grad.plot([], [], 'r-', linewidth=2, label='Grad Norm')
        self.line_speed, = self.ax_speed.plot([], [], 'm-', linewidth=2, label='Steps/sec')
        
        # Налаштування графіків
        self._setup_axes()
        
        # Текстова інформація
        self.text_info = self.fig.text(0.02, 0.02, '', fontsize=10, family='monospace',
                                       verticalalignment='bottom')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
    def _setup_axes(self):
        """Налаштування осей графіків"""
        # Loss
        self.ax_loss.set_title('📉 Loss', fontweight='bold')
        self.ax_loss.set_xlabel('Steps')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.grid(True, alpha=0.3)
        self.ax_loss.legend()
        
        # Learning Rate
        self.ax_lr.set_title('📊 Learning Rate', fontweight='bold')
        self.ax_lr.set_xlabel('Steps')
        self.ax_lr.set_ylabel('LR')
        self.ax_lr.grid(True, alpha=0.3)
        self.ax_lr.legend()
        
        # Grad Norm
        self.ax_grad.set_title('📈 Gradient Norm', fontweight='bold')
        self.ax_grad.set_xlabel('Steps')
        self.ax_grad.set_ylabel('Grad Norm')
        self.ax_grad.grid(True, alpha=0.3)
        self.ax_grad.legend()
        
        # Speed
        self.ax_speed.set_title('⚡ Training Speed', fontweight='bold')
        self.ax_speed.set_xlabel('Time')
        self.ax_speed.set_ylabel('Steps/sec')
        self.ax_speed.grid(True, alpha=0.3)
        self.ax_speed.legend()
    
    def get_latest_checkpoint(self):
        """Знаходить останній чекпоінт"""
        if not os.path.isdir(CHECKPOINT_DIR):
            return None
        
        checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint-*"))
        if not checkpoints:
            return None
        
        regular_checkpoints = [c for c in checkpoints if "interrupted" not in c]
        if not regular_checkpoints:
            regular_checkpoints = checkpoints
        
        try:
            latest = max(regular_checkpoints, key=lambda x: int(x.split("-")[-1]))
            return latest
        except:
            return None
    
    def read_trainer_state(self, checkpoint_path):
        """Читає стан тренера з чекпоінта"""
        state_path = os.path.join(checkpoint_path, "trainer_state.json")
        if not os.path.exists(state_path):
            return None
        
        with open(state_path, 'r') as f:
            return json.load(f)
    
    def update_data(self):
        """Оновлює дані з чекпоінта"""
        checkpoint = self.get_latest_checkpoint()
        if not checkpoint:
            return None
        
        state = self.read_trainer_state(checkpoint)
        if not state:
            return None
        
        log_history = state.get('log_history', [])
        if not log_history:
            return None
        
        # Оновлюємо історію (беремо останні записи)
        for log in log_history[-self.max_history:]:
            step = log.get('step', 0)
            if step not in self.step_history:
                self.step_history.append(step)
                self.loss_history.append(log.get('loss', 0))
                self.lr_history.append(log.get('learning_rate', 0))
                self.grad_norm_history.append(log.get('grad_norm', 0))
                self.time_history.append(time.time())
        
        return state
    
    def calculate_speed(self):
        """Розраховує швидкість тренування"""
        if len(self.step_history) < 2:
            return []
        
        speeds = []
        for i in range(1, len(self.step_history)):
            time_diff = self.time_history[i] - self.time_history[i-1]
            step_diff = self.step_history[i] - self.step_history[i-1]
            if time_diff > 0:
                speeds.append(step_diff / time_diff)
            else:
                speeds.append(0)
        
        return speeds
    
    def update_plots(self):
        """Оновлює графіки"""
        if not self.step_history:
            return
        
        steps = list(self.step_history)
        
        # Loss
        if self.loss_history:
            self.line_loss.set_data(steps, list(self.loss_history))
            self.ax_loss.relim()
            self.ax_loss.autoscale_view()
        
        # Learning Rate
        if self.lr_history:
            self.line_lr.set_data(steps, list(self.lr_history))
            self.ax_lr.relim()
            self.ax_lr.autoscale_view()
        
        # Grad Norm
        if self.grad_norm_history:
            self.line_grad.set_data(steps, list(self.grad_norm_history))
            self.ax_grad.relim()
            self.ax_grad.autoscale_view()
        
        # Speed
        speeds = self.calculate_speed()
        if speeds:
            self.line_speed.set_data(range(len(speeds)), speeds)
            self.ax_speed.relim()
            self.ax_speed.autoscale_view()
    
    def format_time(self, seconds):
        """Форматує секунди"""
        return str(timedelta(seconds=int(seconds)))
    
    def update_text_info(self, state):
        """Оновлює текстову інформацію"""
        if not state:
            self.text_info.set_text("⏳ Очікування даних...")
            return
        
        log_history = state.get('log_history', [])
        if not log_history:
            return
        
        latest = log_history[-1]
        epoch = state.get('epoch', 0)
        global_step = state.get('global_step', 0)
        max_steps = state.get('max_steps', 0)
        
        # Розраховуємо статистику
        if len(self.loss_history) >= 2:
            loss_change = self.loss_history[-1] - self.loss_history[-2]
            trend = "📉" if loss_change < 0 else "📈"
        else:
            loss_change = 0
            trend = "➡️"
        
        # Прогрес
        progress = (global_step / max_steps * 100) if max_steps > 0 else 0
        
        # Швидкість
        speeds = self.calculate_speed()
        avg_speed = np.mean(speeds[-10:]) if len(speeds) >= 10 else 0
        
        # ETA
        remaining_steps = max_steps - global_step
        eta_seconds = remaining_steps / avg_speed if avg_speed > 0 else 0
        
        info_text = f"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║  📊 ПОТОЧНИЙ СТАН                                                                     ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║  Епоха: {epoch:.2f}  │  Крок: {global_step:,} / {max_steps:,}  │  Прогрес: {progress:.1f}%       
║  Loss: {latest.get('loss', 0):.4f} {trend} ({loss_change:+.4f})  │  LR: {latest.get('learning_rate', 0):.6f}
║  Grad Norm: {latest.get('grad_norm', 0):.4f}  │  Швидкість: {avg_speed:.1f} steps/sec
║  ETA: {self.format_time(eta_seconds)}  │  Оновлено: {datetime.now().strftime('%H:%M:%S')}
╚══════════════════════════════════════════════════════════════════════════════════════╝
"""
        self.text_info.set_text(info_text)
    
    def run(self, interval=2):
        """Запускає монітор"""
        print(f"🚀 Запуск монітора тренування (оновлення кожні {interval} сек)")
        print("📊 Відкривається вікно з графіками...")
        print("⚠️  Закрийте вікно для виходу з монітора")
        
        def update_frame(frame):
            state = self.update_data()
            self.update_plots()
            self.update_text_info(state)
            return [self.line_loss, self.line_lr, self.line_grad, self.line_speed]
        
        # Анімація з автооновленням
        self.ani = animation.FuncAnimation(
            self.fig, 
            update_frame, 
            interval=interval * 1000,  # в мілісекундах
            blit=False,
            cache_frame_data=False
        )
        
        plt.show(block=True)

def main():
    """Головна функція"""
    print("="*80)
    print("🎯 PyIanthe Training Monitor (Advanced)")
    print("="*80)
    
    # Перевірка наявності checkpoints
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"❌ Папка checkpoints не знайдена: {CHECKPOINT_DIR}")
        print("   Почніть тренування перед запуском монітора")
        sys.exit(1)
    
    # Створюємо і запускаємо монітор
    monitor = TrainingMonitor()
    
    try:
        monitor.run(interval=MONITOR_INTERVAL)
    except KeyboardInterrupt:
        print("\n\n👋 Монітор зупинено користувачем")
        plt.close('all')
        sys.exit(0)

if __name__ == "__main__":
    main()
