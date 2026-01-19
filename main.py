
# 导入FastAPI相关模块
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
# 导入uvicorn用于启动服务
import uvicorn
# 导入PyTorch及相关深度学习库
import torch
from torchvision import transforms, models
import timm
# 图像处理库
from PIL import Image
import io
import torch.nn as nn
import torch.nn.functional as F
# 路径处理
from pathlib import Path
import os


# 获取项目根目录
BASE_DIR = Path(__file__).parent
# 模型文件夹路径
MODELS_DIR = BASE_DIR / "models"


# 创建FastAPI应用实例
app = FastAPI(
    title="灯具图像分类系统",
    description="基于深度学习的灯具图像分类API",
)


# 跨域配置，允许所有来源访问API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 设备选择，优先使用GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")


# 各模型对应的标签定义
LABELS = {
    "style_material": ['复古玻璃', '波西米亚', '现代水晶', '乳白色调', '莫兰迪', '古铜白'],
    "quality_assessment": ['正常', '非正常'],
    "status_detection": ['亮灯', '暗灯'],
    "quality_grading": ['水印', '尺寸标注', '模糊', '核心元素残缺', '正常'],
    "lamp_recognition": ['非灯具', '灯具'],
    "lamp_type": ['吊灯', '台灯', '壁挂灯', '吸顶灯', '落地灯', '射灯', '路灯', '艺术灯'],
}


# 各模型的功能描述
MODEL_DESCRIPTIONS = {
    "style_material": "风格材质分类 (6类)",
    "quality_assessment": "质量评估 (正常/非正常)",
    "status_detection": "状态检测 (亮灯/暗灯)",
    "quality_grading": "质量分级 (5类)",
    "lamp_recognition": "灯具识别 (是否为灯具)",
    "lamp_type": "灯具类型分类 (8类)",
}

 
# 各模型的图像预处理流程定义
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

 
# 各模型的权重文件路径（本地models文件夹）
MODEL_PATHS = {
    "style_material": str(MODELS_DIR / "style_material" / "resnet50_classifier.pth"),
    "quality_assessment": str(MODELS_DIR / "quality_assessment" / "vit_model.pth"),
    "status_detection": str(MODELS_DIR / "status_detection" / "vit_model.pth"),
    "quality_grading": str(MODELS_DIR / "quality_grading" / "best_model.pth"),
    "lamp_recognition": str(MODELS_DIR / "lamp_recognition" / "lamp_model_resnet50.pth"),
    "lamp_type": str(MODELS_DIR / "lamp_type" / "resnet50_finetuned.pth"),
}

 
# 加载风格材质分类模型
def load_style_material(path):
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(LABELS['style_material']))
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model.to(DEVICE).eval()


 
# 加载质量评估模型（ViT）
def load_quality_assessment(path):
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
    if os.path.exists(path):
        try:
            state_dict = torch.load(path, map_location=DEVICE)
            # 尝试完整加载
            model.load_state_dict(state_dict, strict=True)
        except:
            # 如果失败，尝试部分加载
            model.load_state_dict(state_dict, strict=False)
    return model.to(DEVICE).eval()


 
# 加载状态检测模型（ViT）
def load_status_detection(path):
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
    if os.path.exists(path):
        try:
            state_dict = torch.load(path, map_location=DEVICE)
            # 尝试完整加载
            model.load_state_dict(state_dict, strict=True)
        except:
            # 如果失败，尝试部分加载
            model.load_state_dict(state_dict, strict=False)
    return model.to(DEVICE).eval()


 
# 加载质量分级模型（ViT）
def load_quality_grading(path):
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=len(LABELS['quality_grading']))
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE), strict=False)
    return model.to(DEVICE).eval()


 
# 加载灯具识别模型（ResNet50）
def load_lamp_recognition(path):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model.to(DEVICE).eval()


 
# 加载灯具类型分类模型（ResNet50）
def load_lamp_type(path):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(LABELS['lamp_type']))
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model.to(DEVICE).eval()


 
# 各模型加载函数的映射表
MODEL_LOADERS = {
    'style_material': load_style_material,
    'quality_assessment': load_quality_assessment,
    'status_detection': load_status_detection,
    'quality_grading': load_quality_grading,
    'lamp_recognition': load_lamp_recognition,
    'lamp_type': load_lamp_type,
}


# 启动时加载所有模型到内存，便于后续快速推理
print("\n" + "=" * 50)
print("加载模型中...")
print("=" * 50)
MODELS = {}
for key, loader in MODEL_LOADERS.items():
    try:
        path = MODEL_PATHS[key]
        if os.path.exists(path):
            MODELS[key] = loader(path)
            print(f"[OK] 已加载: {key}")
        else:
            print(f"[--] 未找到: {key}")
    except Exception as e:
        print(f"[ERR] 加载失败 {key}: {e}")
print("=" * 50 + "\n")


 
# 测试时增强（TTA）预测函数，提高模型鲁棒性
def predict_with_tta(model_key: str, image: Image.Image):
    """使用测试时增强(TTA)提高预测准确性"""
    if model_key not in MODELS:
        raise HTTPException(status_code=404, detail=f"模型未找到: {model_key}")
    
    model = MODELS[model_key]
    base_transform = TRANSFORMS[model_key]
    
    # TTA变换列表
    tta_transforms = []
    
    # 原图
    tta_transforms.append(base_transform)
    
    # 水平翻转
    flip_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        base_transform.transforms[0],  # Resize
        base_transform.transforms[1],  # ToTensor
        base_transform.transforms[2],  # Normalize
    ])
    tta_transforms.append(flip_transform)
    
    # 轻微旋转 +10度
    rotate_transform1 = transforms.Compose([
        transforms.RandomRotation((10, 10)),
        base_transform.transforms[0],
        base_transform.transforms[1],
        base_transform.transforms[2],
    ])
    tta_transforms.append(rotate_transform1)
    
    # 轻微旋转 -10度
    rotate_transform2 = transforms.Compose([
        transforms.RandomRotation((-10, -10)),
        base_transform.transforms[0],
        base_transform.transforms[1],
        base_transform.transforms[2],
    ])
    tta_transforms.append(rotate_transform2)
    
    # 亮度调整
    brightness_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2),
        base_transform.transforms[0],
        base_transform.transforms[1],
        base_transform.transforms[2],
    ])
    tta_transforms.append(brightness_transform)
    
    # 收集所有预测结果
    all_outputs = []
    with torch.no_grad():
        for transform in tta_transforms:
            try:
                tensor = transform(image).unsqueeze(0).to(DEVICE)
                outputs = model(tensor)
                all_outputs.append(outputs)
            except:
                # 如果某个变换失败，跳过
                continue
    
    # 平均所有输出
    if len(all_outputs) > 0:
        avg_outputs = torch.mean(torch.stack(all_outputs), dim=0)
    else:
        # 如果所有TTA都失败，使用原图
        tensor = base_transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            avg_outputs = model(tensor)
    
    return avg_outputs


 
# 通用预测函数，支持TTA和置信度阈值
def predict(model_key: str, image: Image.Image, use_tta: bool = False):
    """模型预测函数，支持TTA和置信度阈值"""
    if model_key not in MODELS:
        raise HTTPException(status_code=404, detail=f"模型未找到: {model_key}")
    
    # 使用TTA或标准预测
    if use_tta and model_key in ['quality_assessment', 'status_detection']:
        outputs = predict_with_tta(model_key, image)
    else:
        model = MODELS[model_key]
        transform = TRANSFORMS[model_key]
        tensor = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(tensor)
    
    # 后处理
    if model_key == 'lamp_recognition':
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
        label_idx = int(probs.argmax())
        label = LABELS[model_key][label_idx]
        confidence = float(probs[label_idx])
    elif model_key == 'quality_assessment':
        # 质量评估：使用温度缩放 + 置信度阈值
        temperature = 1.3
        probs = F.softmax(outputs / temperature, dim=1).cpu().numpy()[0]
        label_idx = int(probs.argmax())
        confidence = float(probs[label_idx])
        
        # 置信度阈值过滤
        if confidence < 0.65:
            label = "不确定"
            label_idx = -1
        else:
            label = LABELS[model_key][label_idx]
    elif model_key == 'status_detection':
        # 状态检测：使用温度缩放 + 置信度阈值
        temperature = 1.0
        probs = F.softmax(outputs / temperature, dim=1).cpu().numpy()[0]
        label_idx = int(probs.argmax())
        confidence = float(probs[label_idx])
        
        # 置信度阈值过滤
        if confidence < 0.70:
            label = "不确定"
            label_idx = -1
        else:
            label = LABELS[model_key][label_idx]
    else:
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
        label_idx = int(probs.argmax())
        label = LABELS[model_key][label_idx]
        confidence = float(probs[label_idx])
    
    return {
        'model': model_key,
        'predicted_index': label_idx,
        'predicted_label': label,
        'confidence': confidence,
    }



# 首页路由，返回前端页面或简单提示
@app.get("/", response_class=HTMLResponse)
async def root():
    """返回HTML界面"""
    html_path = BASE_DIR / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding='utf-8')
    return "<h1>灯具图像分类系统</h1><p>请创建 index.html 文件</p>"



# 获取所有可用模型及其描述、标签、加载状态
@app.get("/api/models")
async def list_models():
    """获取所有可用模型"""
    models_info = []
    for key in MODEL_LOADERS.keys():
        models_info.append({
            "key": key,
            "description": MODEL_DESCRIPTIONS.get(key, key),
            "labels": LABELS.get(key, []),
            "loaded": key in MODELS
        })
    return {"models": models_info}



# 单模型预测接口，支持TTA增强
@app.post("/api/predict/{model_key}")
async def predict_image(model_key: str, file: UploadFile = File(...), use_tta: bool = False):
    """对上传的图片进行预测，支持TTA增强"""
    content = await file.read()
    try:
        img = Image.open(io.BytesIO(content)).convert('RGB')
    except Exception:
        raise HTTPException(status_code=400, detail="无效的图片文件")
    
    # 对质量评估和状态检测自动启用TTA
    auto_tta = model_key in ['quality_assessment', 'status_detection']
    result = predict(model_key, img, use_tta=use_tta or auto_tta)
    return {'success': True, 'data': result}



# 多模型批量预测接口，自动为部分模型启用TTA
@app.post("/api/predict_all")
async def predict_all(file: UploadFile = File(...)):
    """使用所有模型对图片进行预测，质量评估和状态检测自动使用TTA"""
    content = await file.read()
    try:
        img = Image.open(io.BytesIO(content)).convert('RGB')
    except Exception:
        raise HTTPException(status_code=400, detail="无效的图片文件")
    
    results = {}
    for model_key in MODELS.keys():
        try:
            # 对质量评估和状态检测启用TTA
            use_tta = model_key in ['quality_assessment', 'status_detection']
            results[model_key] = predict(model_key, img, use_tta=use_tta)
        except Exception as e:
            results[model_key] = {"error": str(e)}
    
    return {'success': True, 'data': results}



# 以脚本方式运行时，启动FastAPI服务
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8001)
