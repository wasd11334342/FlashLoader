import os
import cv2
import easyocr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageGrab
from matplotlib import rcParams

# 指定中文字體 (Windows)
rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 微軟正黑體

def check_image_path(image_path):
    """檢查圖片路徑是否有效"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"圖片文件不存在: {image_path}")
    
    if os.path.isdir(image_path):
        raise ValueError(f"路徑指向文件夾而不是圖片文件: {image_path}")
    
    # 檢查文件擴展名
    valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    if not any(image_path.lower().endswith(ext) for ext in valid_extensions):
        print(f"警告: {image_path} 可能不是支援的圖片格式")
    
    return True

def remove_black_pixels_with_surrounding_color(image_path, threshold=10, radius=1):
    try:
        check_image_path(image_path)
        img = Image.open(image_path).convert('RGB')
        new_img = img.copy()
        pixels = img.load()
        new_pixels = new_img.load()
        width, height = img.size
        
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
        
        output_path = os.path.splitext(image_path)[0] + "-inpaint.png"
        new_img.save(output_path)
        return output_path
    
    except Exception as e:
        print(f"去雜訊處理失敗: {e}")
        raise

def enhance_contrast(image_path, factor=8.0):
    try:
        check_image_path(image_path)
        img = Image.open(image_path).convert('RGB')
        # 先做對比度增強
        enhancer = ImageEnhance.Contrast(img)
        enhanced = enhancer.enhance(factor)
        enhanced_path = os.path.splitext(image_path)[0] + "_enhance.png"
        enhanced.save(enhanced_path, "PNG")  # 明確指定PNG格式
        
        # 再做灰階處理 - 使用PIL處理並儲存為PNG
        enhanced_pil = Image.open(enhanced_path)
        gray_pil = enhanced_pil.convert('L')  # 轉換為灰階
        gray_path = os.path.splitext(image_path)[0] + "_gray.png"
        gray_pil.save(gray_path, "PNG")
        
        # 轉換為OpenCV格式進行二值化處理
        gray_cv = np.array(gray_pil)
        
        # 多種二值化方法
        base_name = os.path.splitext(image_path)[0]
        
        # 1. 高斯模糊後OTSU (減少雜訊影響)
        blurred = cv2.GaussianBlur(gray_cv, (3, 3), 0)
        _, otsu_blur = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        otsu_blur_path = base_name + "_otsu_blur.png"
        # 使用PIL儲存為PNG
        Image.fromarray(otsu_blur).save(otsu_blur_path, "PNG")
        
        # 2. 自適應閾值處理 (對不均勻光照更好)
        adaptive = cv2.adaptiveThreshold(gray_cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        adaptive_path = base_name + "_adaptive.png"
        Image.fromarray(adaptive).save(adaptive_path, "PNG")
        
        # 3. 形態學處理改善OTSU結果
        _, otsu_raw = cv2.threshold(gray_cv, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((2,2), np.uint8)
        otsu_morph = cv2.morphologyEx(otsu_raw, cv2.MORPH_CLOSE, kernel)
        otsu_morph = cv2.morphologyEx(otsu_morph, cv2.MORPH_OPEN, kernel)
        otsu_morph_path = base_name + "_otsu_morph.png"
        Image.fromarray(otsu_morph).save(otsu_morph_path, "PNG")
        
        # 4. 原始OTSU
        otsu_path = base_name + "_otsu.png"
        Image.fromarray(otsu_raw).save(otsu_path, "PNG")
        
        print(f"已生成PNG文件:")
        print(f"  - 對比度增強: {enhanced_path}")
        print(f"  - 灰階處理: {gray_path}")
        print(f"  - OTSU二值化: {otsu_path}")
        print(f"  - 高斯+OTSU: {otsu_blur_path}")
        print(f"  - 自適應閾值: {adaptive_path}")
        print(f"  - OTSU+形態學: {otsu_morph_path}")
        
        return enhanced_path, gray_path, otsu_path, otsu_blur_path, adaptive_path, otsu_morph_path
    
    except Exception as e:
        print(f"對比度增強處理失敗: {e}")
        raise

def test_multiple_preprocessing_methods(image_paths, description):
    """測試多種預處理方法的OCR效果"""
    reader = easyocr.Reader(['en'], gpu=False)
    best_result = ""
    best_confidence = 0
    best_path = ""
    
    print(f"\n=== {description} ===")
    for i, path in enumerate(image_paths):
        if not os.path.exists(path):
            continue
            
        results = reader.readtext(path, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        text = ''.join([r[1] for r in results])
        avg_confidence = np.mean([r[2] for r in results]) if results else 0
        
        method_name = os.path.basename(path).split('_')[-1].replace('.png', '')
        print(f"方法 {i+1} ({method_name}): '{text}' (信心度: {avg_confidence:.3f})")
        
        if avg_confidence > best_confidence:
            best_result = text
            best_confidence = avg_confidence
            best_path = path
    
    print(f"最佳結果: '{best_result}' (信心度: {best_confidence:.3f})")
    return best_path, best_result, best_confidence

def recognize_text_with_easyocr(enhanced_path, otsu_path, otsu_blur_path, adaptive_path, otsu_morph_path):
    # 測試不同預處理方法
    test_paths = [enhanced_path, otsu_path, otsu_blur_path, adaptive_path, otsu_morph_path]
    best_path, best_text, best_confidence = test_multiple_preprocessing_methods(
        test_paths, "比較不同預處理方法"
    )
    
    # 在最佳結果上標註
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(best_path, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    
    image = cv2.imread(best_path)
    if len(image.shape) == 2:  # 如果是灰階圖，轉為彩色
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    for (bbox, text, confidence) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(image, text, (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    # 保存辨識結果圖片為PNG
    recognized_path = os.path.splitext(best_path)[0] + "_recognized.png"
    # 轉換為PIL格式並儲存為PNG
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    Image.fromarray(image_rgb).save(recognized_path, "PNG")
    
    print(f"辨識結果已儲存為PNG: {recognized_path}")
    
    return recognized_path, results, best_path

def display_all_steps(original_path, inpainted_path, enhanced_path, gray_path, otsu_path, 
                     otsu_blur_path, adaptive_path, otsu_morph_path, recognized_path, best_method_path):
    try:
        # 檢查所有路徑
        paths_to_check = [original_path, inpainted_path, enhanced_path, gray_path, 
                         otsu_path, otsu_blur_path, adaptive_path, otsu_morph_path, recognized_path]
        
        for path in paths_to_check:
            if not os.path.exists(path):
                print(f"警告: 文件不存在，跳過顯示: {path}")
                return
        
        # 讀取所有圖片
        original = cv2.imread(original_path)
        inpainted = cv2.imread(inpainted_path)
        enhanced = cv2.imread(enhanced_path)
        gray = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
        
        # 讀取不同二值化方法的結果
        otsu = cv2.imread(otsu_path, cv2.IMREAD_GRAYSCALE)
        otsu_blur = cv2.imread(otsu_blur_path, cv2.IMREAD_GRAYSCALE)
        adaptive = cv2.imread(adaptive_path, cv2.IMREAD_GRAYSCALE)
        otsu_morph = cv2.imread(otsu_morph_path, cv2.IMREAD_GRAYSCALE)
        recognized = cv2.imread(recognized_path)
        
        # 檢查圖片是否成功讀取
        images = [original, inpainted, enhanced, gray, otsu, otsu_blur, adaptive, otsu_morph, recognized]
        for i, img in enumerate(images):
            if img is None:
                print(f"警告: 圖片 {i+1} 讀取失敗")
                return
        
        # 創建大的子圖顯示
        fig, axes = plt.subplots(3, 3, figsize=(18, 16))
        fig.suptitle('OCR圖像處理流程 - 多種二值化方法比較', fontsize=16, fontweight='bold')
        
        # 第一行：基本處理流程
        axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('1. 原始圖片', fontsize=11)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('2. 去雜訊後', fontsize=11)
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title('3. 對比度增強', fontsize=11)
        axes[0, 2].axis('off')
        
        # 第二行：灰階和不同二值化方法
        axes[1, 0].imshow(gray, cmap='gray')
        axes[1, 0].set_title('4. 灰階化', fontsize=11)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(otsu, cmap='gray')
        axes[1, 1].set_title('5a. OTSU二值化', fontsize=11)
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(otsu_blur, cmap='gray')
        axes[1, 2].set_title('5b. 高斯模糊+OTSU', fontsize=11)
        axes[1, 2].axis('off')
        
        # 第三行：其他二值化方法和最終結果
        axes[2, 0].imshow(adaptive, cmap='gray')
        axes[2, 0].set_title('5c. 自適應閾值', fontsize=11)
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(otsu_morph, cmap='gray')
        axes[2, 1].set_title('5d. OTSU+形態學', fontsize=11)
        axes[2, 1].axis('off')
        
        axes[2, 2].imshow(cv2.cvtColor(recognized, cv2.COLOR_BGR2RGB))
        best_method_name = os.path.basename(best_method_path).split('_')[-1].replace('.png', '')
        axes[2, 2].set_title(f'6. 最佳結果 ({best_method_name})', fontsize=11)
        axes[2, 2].axis('off')
        
        # 標註最佳方法
        for i, (ax, path) in enumerate(zip([axes[1,1], axes[1,2], axes[2,0], axes[2,1]], 
                                          [otsu_path, otsu_blur_path, adaptive_path, otsu_morph_path])):
            if path == best_method_path:
                ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, 
                                         fill=False, edgecolor='red', linewidth=3))
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"顯示圖片時發生錯誤: {e}")
        print("請檢查所有圖片文件是否正確生成")

def save_processing_summary(original_path, final_text, best_method_path, results):
    """儲存處理摘要為PNG圖片"""
    try:
        # 創建摘要文字
        summary_text = f"""
OCR處理摘要報告
================

原始圖片: {os.path.basename(original_path)}
處理時間: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

最佳預處理方法: {os.path.basename(best_method_path).replace('.png', '')}

辨識結果: "{final_text}"

詳細結果:
"""
        for i, (bbox, text, confidence) in enumerate(results, 1):
            summary_text += f"  {i}. '{text}' (信心度: {confidence:.3f})\n"
        
        # 創建圖片
        from PIL import ImageDraw, ImageFont
        
        # 設定圖片大小和背景
        img_width, img_height = 800, 600
        img = Image.new('RGB', (img_width, img_height), 'white')
        draw = ImageDraw.Draw(img)
        
        # 嘗試使用系統字體，如果失敗則使用默認字體
        try:
            # Windows系統字體
            font_title = ImageFont.truetype("arial.ttf", 24)
            font_normal = ImageFont.truetype("arial.ttf", 14)
        except:
            try:
                # Linux系統字體
                font_title = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 24)
                font_normal = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 14)
            except:
                # 使用默認字體
                font_title = ImageFont.load_default()
                font_normal = ImageFont.load_default()
        
        # 繪製文字
        y_position = 30
        lines = summary_text.split('\n')
        
        for line in lines:
            if line.strip():
                if line.startswith('OCR處理摘要報告') or line.startswith('==='):
                    if not line.startswith('==='):
                        draw.text((50, y_position), line, fill='black', font=font_title)
                        y_position += 40
                else:
                    draw.text((50, y_position), line, fill='black', font=font_normal)
                    y_position += 25
            else:
                y_position += 10
        
        # 儲存摘要圖片
        summary_path = os.path.splitext(original_path)[0] + "_summary.png"
        img.save(summary_path, "PNG")
        print(f"處理摘要已儲存為PNG: {summary_path}")
        
        return summary_path
        
    except Exception as e:
        print(f"儲存摘要時發生錯誤: {e}")
        return None

# 主程式執行
# 定義區域 (left, top, right, bottom)
bbox = (901,559,1019,610)
img = ImageGrab.grab(bbox)
img.save("screenshot.png")
print('截圖成功')
original_image_path = 'verifycode_20250901_175905.png'

try:
    # 檢查原始圖片路徑
    check_image_path(original_image_path)
    print(f"處理圖片: {original_image_path}")
    
    # 步驟1：去除黑色雜訊
    print("\n步驟1：去除黑色雜訊...")
    inpainted_path = remove_black_pixels_with_surrounding_color(original_image_path)
    print(f"完成，輸出PNG文件: {inpainted_path}")
    
    # 步驟2：對比度增強、灰階化、多種二值化處理
    print("\n步驟2：對比度增強、灰階化、多種二值化處理...")
    enhanced_path, gray_path, otsu_path, otsu_blur_path, adaptive_path, otsu_morph_path = enhance_contrast(inpainted_path)
    
    # 步驟3：測試多種方法並選擇最佳結果
    print("\n步驟3：測試多種預處理方法...")
    recognized_path, results, best_method_path = recognize_text_with_easyocr(
        enhanced_path, otsu_path, otsu_blur_path, adaptive_path, otsu_morph_path
    )
    
    # 步驟4：儲存處理摘要
    print("\n步驟4：生成處理摘要...")
    final_text = ''.join([r[1] for r in results])
    summary_path = save_processing_summary(original_image_path, final_text, best_method_path, results)
    
    # 顯示所有處理步驟和比較結果
    print("\n顯示完整處理流程...")
    display_all_steps(original_image_path, inpainted_path, enhanced_path, gray_path, 
                     otsu_path, otsu_blur_path, adaptive_path, otsu_morph_path, 
                     recognized_path, best_method_path)
    
    # 輸出最終結果和文件清單
    print(f"\n{'='*50}")
    print(f"最終辨識結果：{final_text}")
    for bbox, text, confidence in results:
        print(f"辨識：{text}，信心度：{confidence:.3f}")
    
    print(f"\n使用的最佳預處理方法：{os.path.basename(best_method_path)}")
    
    # 列出所有生成的PNG文件
    print(f"\n{'='*50}")
    print("已生成的PNG文件:")
    base_name = os.path.splitext(original_image_path)[0]
    png_files = [
        f"{base_name}-inpaint.png",
        f"{base_name}_enhance.png", 
        f"{base_name}_gray.png",
        f"{base_name}_otsu.png",
        f"{base_name}_otsu_blur.png", 
        f"{base_name}_adaptive.png",
        f"{base_name}_otsu_morph.png",
        recognized_path,
    ]
    if summary_path:
        png_files.append(summary_path)
    
    for i, png_file in enumerate(png_files, 1):
        if os.path.exists(png_file):
            file_size = os.path.getsize(png_file)
            print(f"  {i:2d}. {os.path.basename(png_file)} ({file_size:,} bytes)")
        else:
            print(f"  {i:2d}. {os.path.basename(png_file)} (未生成)")

except FileNotFoundError as e:
    print(f"錯誤: {e}")
    print("請確認圖片文件路徑是否正確，例如:")
    print("- 'test8.png'")
    print("- './images/test8.png'") 
    print("- '/full/path/to/test8.png'")
    
except ValueError as e:
    print(f"錯誤: {e}")
    print("請確認路徑指向的是圖片文件而不是文件夾")
    
except Exception as e:
    print(f"處理過程中發生未預期的錯誤: {e}")
    print("請檢查:")
    print("1. 圖片文件是否存在且可讀取")
    print("2. 圖片格式是否支援 (png, jpg, jpeg, bmp, tiff)")
    print("3. 是否有足夠的磁盤空間保存處理後的圖片")