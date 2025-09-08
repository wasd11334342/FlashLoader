import os
import random
import string
from captcha.image import ImageCaptcha

def generate_captcha_dataset(output_dir="captcha_dataset", num_images=9900, width=120, height=39, font_sizes=(25, 30, 34)):
    """
    生成驗證碼資料集 (4 碼: 大寫英文 + 數字)

    :param output_dir: 資料集輸出資料夾
    :param num_images: 生成圖片數量
    :param width: 圖片寬度
    :param height: 圖片高度
    :param font_sizes: 字體大小選項
    """
    # 建立輸出資料夾
    os.makedirs(output_dir, exist_ok=True)

    # 初始化 ImageCaptcha
    image_captcha = ImageCaptcha(width=width, height=height, font_sizes=font_sizes)

    for i in range(num_images):
        # 生成隨機 4 碼 (大寫英文 + 數字)
        text = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))

        # 生成圖片
        image = image_captcha.generate_image(text)

        # 儲存檔案 (檔名就是標籤，方便訓練)
        filename = os.path.join(output_dir, f"{text}.png")
        image.save(filename)

        if (i + 1) % 100 == 0:
            print(f"已生成 {i + 1}/{num_images} 張驗證碼")

    print(f"✅ 資料集生成完成，共 {num_images} 張，存放於: {output_dir}")

if __name__ == "__main__":
    generate_captcha_dataset(output_dir="captcha_dataset", num_images=1000)
