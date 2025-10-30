# pip install torch torchvision

import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

'''
Значение от -1 до 1 (обычно от 0 до 1 для изображений).
Чем ближе к 1 — тем семантически похожее содержание.
Даже если изображения разного размера, ракурса, освещения — сеть может "понять", что это, например, "портрет женщины".
'''

# 1. Загрузка предобученной модели (без головы классификации)
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Убираем последний слой
model.eval()

# 2. Преобразования для изображений (как при обучении ImageNet)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)  # Добавляем batch dimension
    with torch.no_grad():
        features = model(img_tensor)
    return features.flatten()

def cosine_similarity_pytorch(img1_path, img2_path):
    feat1 = extract_features(img1_path)
    feat2 = extract_features(img2_path)
    # Cosine similarity
    cos_sim = F.cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0))
    return cos_sim.item()

# Использование
sim = cosine_similarity_pytorch('vermeer.jpg', 'amelie_lens.jpg')
print(f"Semantic similarity (ResNet50 + Cosine): {sim:.4f}")

# Semantic similarity (ResNet50 + Cosine): 0.7499


# pip install git+https://github.com/openai/CLIP.git

import clip
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def clip_similarity(img1_path, img2_path):
    image1 = preprocess(Image.open(img1_path)).unsqueeze(0).to(device)
    image2 = preprocess(Image.open(img2_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        feat1 = model.encode_image(image1)
        feat2 = model.encode_image(image2)
        similarity = F.cosine_similarity(feat1, feat2).item()
    return similarity

sim = clip_similarity('vermeer.jpg', 'amelie_lens.jpg')
print(f"Clip similarity (ViT-B/32): {sim:.4f}")

# Clip similarity (ViT-B/32): 0.4933

'''
1. ResNet50 + Cosine: 0.7499 

    Это довольно высокая семантическая похожесть.
    ResNet50 обучалась на классификации объектов (ImageNet), поэтому она хорошо понимает:
        Есть ли на изображении человек, лицо, одежда, фон, цвета, композиция.
         
    Значение 0.75 говорит, что:
        Оба изображения, скорее всего, содержат похожие объекты (например, портреты людей).
        Возможно, похожая композиция (человек по центру, мягкий фон).
        Но это не идентичные изображения — иначе было бы ближе к 0.95–1.0.
         
     
Интерпретация: «Изображения визуально и семантически схожи — оба, вероятно, портреты, с похожей структурой сцены». 
     
 
🔹 2. CLIP (ViT-B/32): 0.4933 

    Это умеренно низкое значение.
    CLIP обучена на парах "изображение–текст", поэтому она понимает высокоуровневый смысл, а не просто объекты.
        Например: «женщина в шляпе», «ретро-стиль», «мечтательный взгляд», «картина Вермеера» и т.д.
         
    Значение ~0.5 означает:
        Изображения не очень похожи по смыслу, несмотря на визуальное сходство.
        Например: одно — картина старого мастера (Vermeer), другое — современное фото в стиле кино (Amélie).
        CLIP "видит", что это разные эпохи, стили, медиумы (живопись vs фотография), даже если оба — портреты женщин.
         
     

Интерпретация: «Хотя оба изображения — портреты, их контекст, стиль и происхождение сильно различаются, поэтому CLIP считает их слабо связанными».
'''