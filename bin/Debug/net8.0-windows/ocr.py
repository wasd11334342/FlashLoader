'''
流程:
1. 去雜訊
2. 對比度加強
3. OCR
'''
import numpy as np
from PIL import Image, ImageEnhance
import cv2
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pytesseract
import os

# 指定中文字體 (Windows)
rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 微軟正黑體

def remove_black_pixels_with_surrounding_color(PIL_image, threshold=10, radius=1):
    try:
        new_img = PIL_image.copy()
        pixels = PIL_image.load()
        new_pixels = new_img.load()
        width, height = PIL_image.size
        
        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]
                if r <= threshold and g <= threshold and b <= threshold:
                    total_r = total_g = total_b = count = 0
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < width and 0 <= ny < height:
                                nr, ng, nb = pixels[nx, ny]
                                if nr > threshold or ng > threshold or nb > threshold:
                                    total_r += nr
                                    total_g += ng
                                    total_b += nb
                                    count += 1
                    if count > 0:
                        new_pixels[x, y] = (total_r // count, total_g // count, total_b // count)
                    else:
                        new_pixels[x, y] = (255, 255, 255)
        
        return new_img
    
    except Exception as e:
        print(f"去雜訊處理失敗: {e}")
        raise

def enhance_contrast(PIL_image, factor=8.0):
    try:
        # 先做對比度增強
        enhancer = ImageEnhance.Contrast(PIL_image)
        enhance = enhancer.enhance(factor)
        
        # 轉換為灰階
        gray_enh = enhance.convert('L')
        gray_enh_cv = np.array(gray_enh)
        
        # 使用 Otsu's thresholding 進行二值化
        _, otsu_enhance = cv2.threshold(gray_enh_cv, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        return otsu_enhance
        
    except Exception as e:
        print(f"對比度增強處理失敗: {e}")
        raise

def display_images(images, titles):
    """使用 Matplotlib 在一個視窗中顯示多個圖片。"""
    num_images = len(images)
    if num_images == 0:
        return
    
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    
    # 如果只有一張圖片，axes 不是陣列，需特殊處理
    if num_images == 1:
        axes = [axes]
    
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray' if len(np.array(img).shape) == 2 else None)
        ax.set_title(title, fontsize=16)
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

# --- 主程式流程 ---
if __name__ == '__main__':
    image_path = 'verifycode_20250901_180510.png'
    
    if not os.path.exists(image_path):
        print(f"錯誤：找不到圖片檔案 '{image_path}'。請確認檔案名稱和路徑是否正確。")
    else:
        # 1. 載入原始圖片
        img = Image.open(image_path).convert('RGB')
        
        # 2. 去雜訊
        no_noise_img = remove_black_pixels_with_surrounding_color(img) # 建議提高半徑
        
        # 3. 對比度加強與二值化
        processed_img_cv2 = enhance_contrast(no_noise_img, factor=8.0) # 建議調整 factor
        processed_img_pil = Image.fromarray(processed_img_cv2)
        
        # 4. OCR 辨識
        # 將圖片傳遞給 pytesseract.image_to_string 進行文字辨識
        # config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        # --psm 6: 假設圖片中有一行統一的文字區塊。
        # -c tessedit_char_whitelist: 限制辨識範圍為大寫英文字母和數字。
        try:
            recognized_text = pytesseract.image_to_string(processed_img_pil, config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            
            # 移除文字兩側的空白
            recognized_text = recognized_text.strip()
            
            print(f"辨識到的文字是：{recognized_text}")
            
        except pytesseract.TesseractNotFoundError:
            print("錯誤：找不到 Tesseract OCR 引擎。請確認已正確安裝並設定 PATH。")
        except Exception as e:
            print(f"OCR 辨識失敗：{e}")

        # 5. 顯示處理流程中的圖片
        display_images(
            [img, no_noise_img, processed_img_pil],
            ['1. 原始圖片', '2. 去黑點', '3. 對比度加強 (二值化)']
        )