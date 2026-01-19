# 自动图片初步分类脚本
# 用已有模型（如灯具识别、灯具类型）对vcg_lamp_dataset下图片分类
import os
import shutil

# 原始图片文件夹
SRC_DIR = 'vcg_lamp_dataset/未分类'
# 目标根目录
DST_ROOT = 'vcg_lamp_dataset'

# 支持的图片扩展名
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

def get_all_classify_rules():
    """
    自动遍历 vcg_lamp_dataset 下所有类别子文件夹，生成 {类别名: 目标路径}
    """
    rules = {}
    for root, dirs, files in os.walk(DST_ROOT):
        # 跳过未分类文件夹
        if SRC_DIR in root:
            continue
        for d in dirs:
            # 只加最底层类别文件夹
            dpath = os.path.join(root, d)
            # 判断该文件夹下是否还有子文件夹
            if any(os.path.isdir(os.path.join(dpath, sub)) for sub in os.listdir(dpath)):
                continue
            rules[d] = dpath
    return rules

def classify_images():
    if not os.path.exists(SRC_DIR):
        print(f'原始图片文件夹不存在: {SRC_DIR}')
        return
    classify_rules = get_all_classify_rules()
    print('自动生成的分类规则:')
    for k, v in classify_rules.items():
        print(f'  关键词: {k} -> {v}')
    files = os.listdir(SRC_DIR)
    count = 0
    for fname in files:
        fpath = os.path.join(SRC_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext not in IMG_EXTS:
            continue
        moved = False
        for keyword, target_dir in classify_rules.items():
            if keyword in fname:
                os.makedirs(target_dir, exist_ok=True)
                shutil.move(fpath, os.path.join(target_dir, fname))
                print(f'已分类: {fname} -> {target_dir}')
                count += 1
                moved = True
                break
        if not moved:
            print(f'未分类: {fname}')
    print(f'分类完成，共移动 {count} 张图片。')

if __name__ == '__main__':
    classify_images()
from PIL import Image
import torch
from torchvision import transforms, models
import shutil

# 选择模型类型：'lamp_recognition' 或 'lamp_type'
MODEL_TYPE = 'lamp_recognition'  # 或 'lamp_type'

# 路径配置
DATA_DIR = 'vcg_lamp_dataset'  # 爬虫图片目录
MODEL_PATH = 'models/lamp_recognition/lamp_model_resnet50.pth'  # 或灯具类型模型

# 标签定义
LABELS = {
    'lamp_recognition': ['非灯具', '灯具'],
    'lamp_type': ['吊灯', '台灯', '壁挂灯', '吸顶灯', '落地灯', '射灯', '路灯', '艺术灯']
}

# 预处理
TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载模型
model = models.resnet50(weights=None)
if MODEL_TYPE == 'lamp_recognition':
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
else:
    model.fc = torch.nn.Linear(model.fc.in_features, len(LABELS['lamp_type']))
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# 分类并移动图片
def classify_and_move():
    for fname in os.listdir(DATA_DIR):
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            img = Image.open(fpath).convert('RGB')
            input_tensor = TRANSFORMS(img).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                pred = output.argmax(dim=1).item()
                label = LABELS[MODEL_TYPE][pred]
            # 目标文件夹
            target_dir = os.path.join(DATA_DIR, label)
            os.makedirs(target_dir, exist_ok=True)
            shutil.move(fpath, os.path.join(target_dir, fname))
            print(f"{fname} -> {label}")
        except Exception as e:
            print(f"跳过 {fname}: {e}")

if __name__ == '__main__':
    classify_and_move()
    print('分类完成！')
