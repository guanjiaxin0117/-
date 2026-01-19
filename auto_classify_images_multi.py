# 多模型自动分类脚本
# 用所有可用模型对vcg_lamp_dataset下图片进行多类别分类，分别移动到各自类别子文件夹
import os
import shutil
import torch
from torchvision import transforms
from PIL import Image

# 模型配置（模型路径、类别列表、主目录名）
MODEL_CONFIGS = [
    {
        'name': 'lamp_recognition',
        'model_path': 'models/lamp_recognition/lamp_model_resnet50.pth',
        'class_names': ['灯具', '非灯具'],
        'dst_root': 'vcg_lamp_dataset/lamp_recognition',
    },
    {
        'name': 'lamp_type',
        'model_path': 'models/lamp_type/resnet50_finetuned.pth',
        'class_names': ['壁挂灯', '吊灯', '路灯', '落地灯', '射灯', '台灯', '吸顶灯', '艺术灯'],
        'dst_root': 'vcg_lamp_dataset/lamp_type',
    },
    {
        'name': 'lamp_quality',
        'model_path': 'models/quality_assessment/vit_model.pth',
        'class_names': ['尺寸标注', '核心元素残缺', '模糊', '水印', '正常'],
        'dst_root': 'vcg_lamp_dataset/lamp_quality',
    },
    {
        'name': 'quality_grading',
        'model_path': 'models/quality_grading/best_model.pth',
        'class_names': ['A', 'B', 'C', 'D'],
        'dst_root': 'vcg_lamp_dataset/quality_grading',
    },
    {
        'name': 'status_detection',
        'model_path': 'models/status_detection/vit_model.pth',
        'class_names': ['正常', '异常'],
        'dst_root': 'vcg_lamp_dataset/status_detection',
    },
    {
        'name': 'style_material',
        'model_path': 'models/style_material/resnet50_classifier.pth',
        'class_names': ['现代', '欧式', '中式', '美式', '北欧', '铁艺', '水晶', '布艺', '木艺', '玻璃', '铜艺', '塑料'],
        'dst_root': 'vcg_lamp_dataset/style_material',
    },
]

SRC_DIR = 'vcg_lamp_dataset/未分类'
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

# 通用图片预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def load_model(model_path, num_classes):
    # 这里只以 resnet50/vit 为例，实际需根据你的模型结构调整
    if 'vit' in model_path:
        from torchvision.models import vit_b_16
        model = vit_b_16(pretrained=False)
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
    else:
        from torchvision.models import resnet50
        model = resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def predict(model, img):
    with torch.no_grad():
        input_tensor = transform(img).unsqueeze(0)
        output = model(input_tensor)
        pred = output.argmax(dim=1).item()
    return pred

def classify_all_models():
    if not os.path.exists(SRC_DIR):
        print(f'原始图片文件夹不存在: {SRC_DIR}')
        return
    files = os.listdir(SRC_DIR)
    files = [f for f in files if os.path.splitext(f)[1].lower() in IMG_EXTS]
    if not files:
        print('未分类文件夹下没有图片。')
        return
    # 预加载所有模型
    models = []
    for cfg in MODEL_CONFIGS:
        print(f'加载模型: {cfg["name"]} ...')
        model = load_model(cfg['model_path'], len(cfg['class_names']))
        models.append((cfg, model))
    for fname in files:
        fpath = os.path.join(SRC_DIR, fname)
        try:
            img = Image.open(fpath).convert('RGB')
        except Exception as e:
            print(f'无法打开图片: {fname}, 错误: {e}')
            continue
        for cfg, model in models:
            pred_idx = predict(model, img)
            pred_class = cfg['class_names'][pred_idx]
            target_dir = os.path.join(cfg['dst_root'], pred_class)
            os.makedirs(target_dir, exist_ok=True)
            shutil.copy(fpath, os.path.join(target_dir, fname))
            print(f'{cfg["name"]}: {fname} -> {pred_class}')
    print('全部模型分类完成。')

if __name__ == '__main__':
    classify_all_models()
from PIL import Image
import torch
from torchvision import transforms, models
import timm
import shutil

# 各模型标签
LABELS = {
    "style_material": ['复古玻璃', '波西米亚', '现代水晶', '乳白色调', '莫兰迪', '古铜白'],
    "quality_assessment": ['正常', '非正常'],
    "status_detection": ['亮灯', '暗灯'],
    "quality_grading": ['水印', '尺寸标注', '模糊', '核心元素残缺', '正常'],
    "lamp_recognition": ['非灯具', '灯具'],
    "lamp_type": ['吊灯', '台灯', '壁挂灯', '吸顶灯', '落地灯', '射灯', '路灯', '艺术灯'],
}

# 各模型权重路径
MODEL_PATHS = {
    "style_material": "models/style_material/resnet50_classifier.pth",
    "quality_assessment": "models/quality_assessment/vit_model.pth",
    "status_detection": "models/status_detection/vit_model.pth",
    "quality_grading": "models/quality_grading/best_model.pth",
    "lamp_recognition": "models/lamp_recognition/lamp_model_resnet50.pth",
    "lamp_type": "models/lamp_type/resnet50_finetuned.pth",
}

# 各模型预处理
TRANSFORMS = {
    "style_material": transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ]),
    "quality_assessment": transforms.Compose([
        transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ]),
    "status_detection": transforms.Compose([
        transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ]),
    "quality_grading": transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ]),
    "lamp_recognition": transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ]),
    "lamp_type": transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ]),
}

# 加载模型
MODELS = {}
def load_all_models():
    for key in MODEL_PATHS:
        path = MODEL_PATHS[key]
        if not os.path.exists(path):
            print(f"模型未找到: {key}")
            continue
        if key in ["style_material", "lamp_recognition", "lamp_type"]:
            model = models.resnet50(weights=None)
            if key == "style_material":
                model.fc = torch.nn.Linear(model.fc.in_features, len(LABELS[key]))
            elif key == "lamp_recognition":
                model.fc = torch.nn.Linear(model.fc.in_features, 2)
            elif key == "lamp_type":
                model.fc = torch.nn.Linear(model.fc.in_features, len(LABELS[key]))
            model.load_state_dict(torch.load(path, map_location='cpu'))
        else:
            # ViT模型
            num_classes = len(LABELS[key])
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model.load_state_dict(torch.load(path, map_location='cpu'), strict=False)
        model.eval()
        MODELS[key] = model

# 分类并移动图片
def classify_and_move_all():
    DATA_DIR = 'vcg_lamp_dataset'
    files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
    for fname in files:
        fpath = os.path.join(DATA_DIR, fname)
        try:
            img = Image.open(fpath).convert('RGB')
            for key in MODELS:
                input_tensor = TRANSFORMS[key](img).unsqueeze(0)
                with torch.no_grad():
                    output = MODELS[key](input_tensor)
                    pred = output.argmax(dim=1).item()
                    label = LABELS[key][pred]
                # 目标文件夹
                target_dir = os.path.join(DATA_DIR, f"{key}_{label}")
                os.makedirs(target_dir, exist_ok=True)
                shutil.copy(fpath, os.path.join(target_dir, fname))
                print(f"{fname} -> {key}: {label}")
        except Exception as e:
            print(f"跳过 {fname}: {e}")

if __name__ == '__main__':
    load_all_models()
    classify_and_move_all()
    print('多模型分类完成！')
