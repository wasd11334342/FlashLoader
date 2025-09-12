import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image, ImageEnhance
import string
import os
import sys
import json
import argparse

# 字符集定义
CHARS = string.ascii_uppercase + string.digits  # A-Z + 0-9
CHAR_TO_NUM = {char: i for i, char in enumerate(CHARS)}
NUM_TO_CHAR = {i: char for i, char in enumerate(CHARS)}
NUM_CLASSES = len(CHARS)

# 驗證用的圖像變換
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def remove_black_pixels_with_surrounding_color(PIL_image, threshold=10, radius=1):
    """去除黑线的函数"""
    try:
        if PIL_image.mode != 'RGB':
            PIL_image = PIL_image.convert('RGB')
            
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
    except:
        return PIL_image

def enhance_contrast(PIL_image, factor=8.0):
    """对比度增强函数"""
    try:
        enhancer = ImageEnhance.Contrast(PIL_image)
        return enhancer.enhance(factor)
    except:
        return PIL_image

class MobileNetV2Captcha(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, captcha_length=4, freeze_backbone=False):
        super(MobileNetV2Captcha, self).__init__()
        self.num_classes = num_classes
        self.captcha_length = captcha_length
        
        self.backbone = models.mobilenet_v2(pretrained=True)
        feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(feature_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, num_classes)
            ) for _ in range(captcha_length)
        ])
    
    def forward(self, x):
        features = self.backbone(x)
        outputs = []
        for classifier in self.classifiers:
            outputs.append(classifier(features))
        return outputs

def predict_captcha(model, image_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """预测单张验证码图片"""
    model.eval()
    
    try:
        # 預處理
        image = Image.open(image_path).convert('RGB')
        image = remove_black_pixels_with_surrounding_color(image)
        processed_image = enhance_contrast(image)
        
        # 轉換為tensor
        image_tensor = transform_val(processed_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            predicted_chars = []
            confidences = []
            
            for output in outputs:
                prob = F.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(prob, 1)
                predicted_chars.append(NUM_TO_CHAR[predicted_idx.item()])
                confidences.append(confidence.item())
        
        result = ''.join(predicted_chars)
        avg_confidence = sum(confidences) / len(confidences)
        
        return result, avg_confidence
    except Exception as e:
        return None, 0.0

def load_model(weight_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """載入模型"""
    try:
        model = MobileNetV2Captcha(num_classes=NUM_CLASSES, captcha_length=4)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.to(device)
        return model, device
    except Exception as e:
        print(f"ERROR:載入模型失敗: {e}", file=sys.stderr)
        return None, None

def predict_single(image_path, weight_path):
    """預測單張圖片並輸出結構化結果"""
    model, device = load_model(weight_path)
    if model is None:
        sys.exit(1)
    
    result, confidence = predict_captcha(model, image_path, device)
    
    if result is not None:
        # 只輸出結果供C#解析
        print(f"{result}")
        # 如果需要信心度，取消下行註解
        # print(f"CONFIDENCE:{confidence}")
    else:
        sys.exit(1)

def predict_batch(image_folder, weight_path):
    """批量預測並輸出JSON結果"""
    model, device = load_model(weight_path)
    if model is None:
        sys.exit(1)
    
    # 支援的圖片格式
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']
    image_files = []
    
    try:
        for ext in image_extensions:
            image_files.extend([f for f in os.listdir(image_folder) if f.lower().endswith(ext)])
        
        if not image_files:
            print("ERROR:沒有找到圖片檔案", file=sys.stderr)
            sys.exit(1)
        
        results = []
        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            result, confidence = predict_captcha(model, image_path, device)
            
            if result is not None:
                results.append({
                    'filename': image_file,
                    'prediction': result,
                    'confidence': confidence
                })
        
        # 輸出JSON格式結果
        print(json.dumps(results, ensure_ascii=False))
        
    except Exception as e:
        print(f"ERROR:批量預測失敗: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='CAPTCHA預測程序')
    parser.add_argument('image_path', help='圖片路徑或資料夾路徑')
    parser.add_argument('weight_path', help='模型權重檔案路徑')
    parser.add_argument('--batch', action='store_true', help='批量預測模式')
    
    # 如果沒有參數，嘗試從命令行讀取
    if len(sys.argv) == 1:
        # 兼容舊版本調用方式
        if len(sys.argv) >= 3:
            image_path = sys.argv[1]
            weight_path = sys.argv[2] if len(sys.argv) > 2 else 'best_mobilenet_captcha_model.pth'
            predict_single(image_path, weight_path)
        else:
            print("ERROR:參數不足", file=sys.stderr)
            sys.exit(1)
    else:
        args = parser.parse_args()
        
        if args.batch:
            predict_batch(args.image_path, args.weight_path)
        else:
            predict_single(args.image_path, args.weight_path)

if __name__ == "__main__":
    # 兼容兩種調用方式
    if len(sys.argv) >= 2 and not sys.argv[1].startswith('-'):
        # 直接參數調用: python script.py image_path [weight_path]
        image_path = sys.argv[1]
        weight_path = sys.argv[2] if len(sys.argv) > 2 else 'best_mobilenet_captcha_model.pth'
        
        if '--batch' in sys.argv:
            predict_batch(image_path, weight_path)
        else:
            predict_single(image_path, weight_path)
    else:
        # argparse調用
        main()