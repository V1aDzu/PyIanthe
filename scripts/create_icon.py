"""
Скрипт для створення іконки з круглою обрізкою
Використання: python create_icon.py
"""
from PIL import Image, ImageDraw, ImageFont
import os

def create_circular_icon(input_path, output_name="ianthe_icon"):
    """
    Створює кругову іконку з прозорим фоном
    """
    print(f"📂 Відкриваємо: {input_path}")
    
    # Відкриваємо картинку
    img = Image.open(input_path).convert("RGBA")
    
    # Визначаємо розмір (беремо мінімальний)
    width, height = img.size
    size = min(width, height)
    
    # Обрізаємо до квадрата (центруємо)
    left = (width - size) // 2
    top = (height - size) // 2
    right = left + size
    bottom = top + size
    img_square = img.crop((left, top, right, bottom))
    
    print(f"✂️  Обрізано до квадрата: {size}x{size}")
    
    # Створюємо кругову маску
    mask = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size, size), fill=255)
    
    # Застосовуємо маску
    img_circle = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    img_circle.paste(img_square, (0, 0), mask)
    
    print(f"⭕ Застосовано кругову маску")
    
    # Зберігаємо різні розміри
    sizes = [512, 256, 128, 64, 32, 16]
    
    for icon_size in sizes:
        img_resized = img_circle.resize((icon_size, icon_size), Image.Resampling.LANCZOS)
        output_file = f"{output_name}_{icon_size}.png"
        img_resized.save(output_file, "PNG")
        print(f"💾 Збережено: {output_file} ({icon_size}x{icon_size})")
    
    # Створюємо .ico файл (для Windows)
    try:
        icon_sizes_for_ico = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
        img_circle.save(
            f"{output_name}.ico",
            format='ICO',
            sizes=icon_sizes_for_ico
        )
        print(f"✅ Створено: {output_name}.ico (Windows іконка)")
    except Exception as e:
        print(f"⚠️  Не вдалося створити .ico: {e}")
        print(f"   Використай онлайн конвертер для .ico")
    
    # Зберігаємо головний PNG
    img_circle.save(f"{output_name}.png", "PNG")
    print(f"✅ Головна іконка: {output_name}.png")
    
    print(f"\n🎉 Готово!")
    print(f"📁 Файли збережено у поточній папці")
    return img_circle

def add_watermark(img, text="PyIanthe", opacity=128):
    """
    Додає легкий watermark (опціонально)
    """
    # Створюємо копію
    img_with_wm = img.copy()
    
    # Створюємо шар для watermark
    watermark = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(watermark)
    
    # Намагаємось завантажити шрифт
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Розмір тексту
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Позиція (внизу справа)
    width, height = img.size
    x = width - text_width - 10
    y = height - text_height - 10
    
    # Малюємо текст з прозорістю
    draw.text((x, y), text, fill=(255, 255, 255, opacity), font=font)
    
    # Накладаємо watermark
    img_with_wm = Image.alpha_composite(img_with_wm, watermark)
    
    return img_with_wm

def main():
    """
    Головна функція
    """
    print("=" * 60)
    print("🎨 Створення іконки PyIanthe")
    print("=" * 60)
    
    # Шукаємо картинку
    possible_names = [
        "Phoenix_10_Beautiful_nymph_Ianthe_from_Greek_mythology_etherea_1.jpg",
        "ianthe.jpg",
        "nymph.jpg",
        "input.jpg",
        "input.png"
    ]
    
    input_file = None
    for name in possible_names:
        if os.path.exists(name):
            input_file = name
            break
    
    if not input_file:
        # Запитуємо у користувача
        print("\n📂 Файл не знайдено автоматично")
        input_file = input("Введіть шлях до картинки: ").strip().strip('"')
        
        if not os.path.exists(input_file):
            print(f"❌ Файл не знайдено: {input_file}")
            return
    
    # Створюємо іконку
    icon = create_circular_icon(input_file, "PyIanthe_icon")
    
    # Питаємо чи додати watermark
    print("\n" + "=" * 60)
    add_wm = input("💧 Додати watermark 'PyIanthe'? (y/n): ").strip().lower()
    
    if add_wm == 'y':
        print("Додаємо watermark...")
        icon_wm = add_watermark(icon, "PyIanthe", opacity=100)
        icon_wm.save("ianthe_icon_watermark.png", "PNG")
        print("✅ Збережено з watermark: ianthe_icon_watermark.png")
    
    print("\n" + "=" * 60)
    print("🎉 Готово!")
    print("\nСтворені файли:")
    print("  • PyIanthe_icon.png - головна іконка")
    print("  • PyIanthe_icon_512.png - велика (512x512)")
    print("  • PyIanthe_icon_256.png - середня (256x256)")
    print("  • PyIanthe_icon_128.png, 64, 32, 16 - малі")
    print("  • PyIanthe_icon.ico - Windows іконка")
    if add_wm == 'y':
        print("  • ianthe_icon_watermark.png - з watermark")
    print("\nВикористай ianthe_icon.ico для PyInstaller!")
    print("=" * 60)

if __name__ == "__main__":
    main()
