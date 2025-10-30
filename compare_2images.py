# pip install Pillow opencv-python scikit-image numpy matplotlib

'''
Выводится значение RMS, которое показывает, как сильно различаются два изображения 
по своим цветовым характеристикам. Низкое значение RMS указывает на то, что изображения 
очень похожи, в то время как высокое значение сигнализирует о том, что они сильно отличаются.

RMS (Root Mean Square) — это статистическая мера, которая используется для оценки величины отклонения или изменения между двумя набором данных. В данном контексте RMS применяется для сравнения гистограмм двух изображений.
Как RMS рассчитывается

    Разница между элементами: Для каждого пикселя (или цветовой компоненты) в гистограммах двух изображений определяется разница.
    Квадрат отклонения: Каждая разница возводится в квадрат, чтобы избежать отрицательных значений и подчеркнуть более крупные отклонения.
    Среднее значение: Сумма всех квадратов делится на количество элементов, чтобы получить среднее значение различий.
    Квадратный корень: Из полученного среднего значения извлекается квадратный корень.
'''

from PIL import Image
from math import sqrt
from functools import reduce
import operator

# Сравнение гистограмм

import numpy as np
import matplotlib.pyplot as plt

def compare_histograms(image1_path, image2_path):
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)

    hist1 = img1.histogram()
    hist2 = img2.histogram()

    # RMS для гистограмм
    rms = np.sqrt(np.mean([(a - b) ** 2 for a, b in zip(hist1, hist2)]))
    
    print(f'RMS of histograms: {rms}')

compare_histograms('vermeer.jpg', 'amelie_lens.jpg')


# Метрики расстояния: Euclidean Distance

def euclidean_distance(image1_path, image2_path):
    img1 = Image.open(image1_path).resize((800, 800)).convert('L')  # Приведение к градациям серого
    img2 = Image.open(image2_path).resize((800, 800)).convert('L')  # Приведение к градациям серого

    data1 = np.array(img1).flatten()
    data2 = np.array(img2).flatten()

    distance = np.sqrt(np.sum((data1 - data2) ** 2))
    
    print(f'Euclidean Distance: {distance}')

# Пример использования
euclidean_distance('vermeer.jpg', 'amelie_lens.jpg')

# Сравнение по ключевым точкам

import cv2

def compare_keypoints(image1_path, image2_path):
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    # Преобразуем в градации серого
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Обнаружение графических ключевых точек
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Сравнение ключевых точек
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    print(f'Number of matches: {len(matches)}')

compare_keypoints('vermeer.jpg', 'amelie_lens.jpg')


# Метод SSIM (Structural Similarity Index)

from skimage.metrics import structural_similarity as ssim

def compare_ssim(image1_path, image2_path):
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    img1 = cv2.resize(img1, (800, 800))
    img2 = cv2.resize(img2, (800, 800))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    ssim_value = ssim(gray1, gray2)
    
    print(f'SSIM: {ssim_value}')

compare_ssim('vermeer.jpg', 'amelie_lens.jpg')


# Пиксельное сравнение

def pixel_comparison(image1_path, image2_path):
    img1 = Image.open(image1_path).resize((800, 800))#.convert('L')
    img2 = Image.open(image2_path).resize((800, 800))#.convert('L')
    
    data1 = np.array(img1).flatten()
    data2 = np.array(img2).flatten()

    diff = np.sum(data1 != data2)
    
    print(f'Different pixels: {diff}')

pixel_comparison('vermeer.jpg', 'amelie_lens.jpg')


'''
RMS of histograms: 20986.94605278233
Euclidean Distance: 8471.4778521814
Number of matches: 117
SSIM: 0.09005005850167969
Different pixels: 1914727
'''