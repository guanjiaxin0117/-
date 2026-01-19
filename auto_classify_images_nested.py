# 多模型多级嵌套自动分类脚本
# 按原有子文件夹为基准，输出到 vcg_lamp_dataset/原子文件夹/模型名_类别/图片.jpg
import os
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
def classify_and_move_nested():
    DATA_DIR = 'vcg_lamp_dataset'
    for subdir in os.listdir(DATA_DIR):
        subdir_path = os.path.join(DATA_DIR, subdir)
        if not os.path.isdir(subdir_path):
            continue
        files = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]
        for fname in files:
            fpath = os.path.join(subdir_path, fname)
            try:
                img = Image.open(fpath).convert('RGB')
                for key in MODELS:
                    input_tensor = TRANSFORMS[key](img).unsqueeze(0)
                    with torch.no_grad():
                        output = MODELS[key](input_tensor)
                        pred = output.argmax(dim=1).item()
                        label = LABELS[key][pred]
                    # 目标文件夹：原子文件夹/模型_类别
                    target_dir = os.path.join(DATA_DIR, subdir, f"{key}_{label}")
                    os.makedirs(target_dir, exist_ok=True)
                    shutil.copy(fpath, os.path.join(target_dir, fname))
                    print(f"{subdir}/{fname} -> {key}: {label}")
            except Exception as e:
                print(f"跳过 {subdir}/{fname}: {e}")

if __name__ == '__main__':
    load_all_models()
    classify_and_move_nested()
    print('多模型多级嵌套分类完成！')
