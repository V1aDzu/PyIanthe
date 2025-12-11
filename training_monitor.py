"""
Монітор тренування - показує поточний стан навчання
Використання: python training_monitor.py
"""
import os
import json
import time
from datetime import datetime, timedelta
import glob

CHECKPOINT_DIR = "checkpoints"
REPORTS_DIR = "reports"

def format_time(seconds):
    """Форматує секунди в читабельний вигляд"""
    return str(timedelta(seconds=int(seconds)))

def get_latest_checkpoint():
    """Знаходить останній чекпоінт"""
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint-*"))
    if not checkpoints:
        return None
    
    # Фільтруємо interrupted чекпоінти
    regular_checkpoints = [c for c in checkpoints if "interrupted" not in c]
    if not regular_checkpoints:
        return None
    
    # Сортуємо за номером
    try:
        latest = max(regular_checkpoints, key=lambda x: int(x.split("-")[-1]))
        return latest
    except:
        return None

def read_trainer_state(checkpoint_path):
    """Читає стан тренера з чекпоінта"""
    state_path = os.path.join(checkpoint_path, "trainer_state.json")
    if not os.path.exists(state_path):
        return None
    
    with open(state_path, 'r') as f:
        return json.load(f)

def print_training_status():
    """Виводить поточний статус тренування"""
    print("\n" + "="*70)
    print("📊 МОНІТОРИНГ ТРЕНУВАННЯ PyIanthe")
    print("="*70)
    
    # Знаходимо останній чекпоінт
    latest_checkpoint = get_latest_checkpoint()
    
    if not latest_checkpoint:
        print("\n❌ Чекпоінти не знайдено")
        print("   Тренування ще не почалось або всі чекпоінти видалено")
        return
    
    print(f"\n📁 Останній чекпоінт: {os.path.basename(latest_checkpoint)}")
    
    # Читаємо стан
    state = read_trainer_state(latest_checkpoint)
    
    if not state:
        print("   ⚠ Не вдалося прочитати trainer_state.json")
        return
    
    # Основна інформація
    print(f"\n🎯 Прогрес:")
    print(f"   • Епоха: {state.get('epoch', 0):.2f}")
    print(f"   • Глобальний крок: {state.get('global_step', 0):,}")
    print(f"   • Макс. кроків: {state.get('max_steps', 0):,}")
    
    # Метрики
    log_history = state.get('log_history', [])
    if log_history:
        latest_log = log_history[-1]
        print(f"\n📈 Останні метрики:")
        print(f"   • Loss: {latest_log.get('loss', 'N/A'):.4f}")
        print(f"   • Learning Rate: {latest_log.get('learning_rate', 'N/A'):.6f}")
        print(f"   • Grad Norm: {latest_log.get('grad_norm', 'N/A'):.4f}")
        
        # Показуємо тренд loss
        if len(log_history) >= 2:
            prev_loss = log_history[-2].get('loss', 0)
            curr_loss = latest_log.get('loss', 0)
            change = curr_loss - prev_loss
            trend = "📉" if change < 0 else "📈"
            print(f"   • Loss тренд: {trend} ({change:+.4f})")
    
    # Час тренування
    best_metric = state.get('best_metric')
    best_model_checkpoint = state.get('best_model_checkpoint')
    
    if best_metric is not None:
        print(f"\n🏆 Найкраща модель:")
        print(f"   • Метрика: {best_metric:.4f}")
        if best_model_checkpoint:
            print(f"   • Чекпоінт: {os.path.basename(best_model_checkpoint)}")
    
    # Звіти
    reports = glob.glob(os.path.join(REPORTS_DIR, "report_epoch_*.json"))
    if reports:
        print(f"\n📝 Звіти по епохах: {len(reports)}")
        for report_path in sorted(reports)[-3:]:  # Останні 3
            print(f"   • {os.path.basename(report_path)}")
    
    # Оцінка часу до завершення
    total_steps = state.get('max_steps', 0)
    current_step = state.get('global_step', 0)
    
    if total_steps > 0 and current_step > 0:
        progress_pct = (current_step / total_steps) * 100
        remaining_steps = total_steps - current_step
        
        # Оцінка швидкості
        if len(log_history) >= 2:
            time_per_step = 1.0  # Приблизно, можна покращити
            remaining_time = remaining_steps * time_per_step
            
            print(f"\n⏱️  Прогрес:")
            print(f"   • Завершено: {progress_pct:.1f}%")
            print(f"   • Залишилось кроків: {remaining_steps:,}")
            print(f"   • Приблизний час: {format_time(remaining_time)}")
    
    print("\n" + "="*70)
    print("ℹ️  Оновіть командою: python training_monitor.py")
    print("="*70 + "\n")

def watch_training(interval=30):
    """Постійно моніторить тренування"""
    print("🔄 Режим моніторингу (оновлення кожні {} секунд)".format(interval))
    print("   Натисніть Ctrl+C для виходу\n")
    
    try:
        while True:
            print_training_status()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n👋 Моніторинг зупинено")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "watch":
        # Режим постійного моніторингу
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        watch_training(interval)
    else:
        # Одноразовий показ статусу
        print_training_status()
        print("\n💡 Підказка: Для постійного моніторингу:")
        print("   python training_monitor.py watch [інтервал_в_секундах]")
        print("   Приклад: python training_monitor.py watch 30")
