# pip install scikit-learn

'''
Cosine Similarity (Косинусное сходство)

Косинусное сходство измеряет угол между векторами. Эта метрика полезна для оценки схожести направлений векторов пикселей.
'''

from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cosine_similarity_images(image1_path, image2_path):
    img1 = Image.open(image1_path).resize((800, 800)).convert('L')
    img2 = Image.open(image2_path).resize((800, 800)).convert('L')

    data1 = np.array(img1).flatten().reshape(1, -1)
    data2 = np.array(img2).flatten().reshape(1, -1)

    similarity = cosine_similarity(data1, data2)[0][0]
    print(f'Cosine Similarity: {similarity}')

cosine_similarity_images('vermeer.jpg', 'amelie_lens.jpg')

'''
MSE (Mean Squared Error)

Среднеквадратичная ошибка вычисляет среднее значение квадратов различий между пикселями двух изображений.
'''

def mean_squared_error(image1_path, image2_path):
    img1 = Image.open(image1_path).resize((800, 800)).convert('L')
    img2 = Image.open(image2_path).resize((800, 800)).convert('L')

    data1 = np.array(img1).flatten()
    data2 = np.array(img2).flatten()

    mse = np.mean((data1 - data2) ** 2)
    print(f'Mean Squared Error (MSE): {mse}')

mean_squared_error('vermeer.jpg', 'amelie_lens.jpg')

'''
PSNR (Peak Signal-to-Noise Ratio)

Пиковое отношение сигнал/шум используется для оценки качества изображений, в том числе сжатых и искаженных.
'''

def peak_signal_to_noise_ratio(image1_path, image2_path):
    img1 = Image.open(image1_path).resize((800, 800)).convert('L')
    img2 = Image.open(image2_path).resize((800, 800)).convert('L')

    data1 = np.array(img1).flatten()
    data2 = np.array(img2).flatten()

    mse = np.mean((data1 - data2) ** 2)
    if mse == 0:
        print("Images are identical")
        return

    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    print(f'Peak Signal-to-Noise Ratio (PSNR): {psnr} dB')

peak_signal_to_noise_ratio('vermeer.jpg', 'amelie_lens.jpg')

'''
NCC (Normalized Cross-Correlation)

Нормализованная кросс-корреляция измеряет совместное изменение двух изображений.
'''

def normalized_cross_correlation(image1_path, image2_path):
    img1 = Image.open(image1_path).resize((800, 800)).convert('L')
    img2 = Image.open(image2_path).resize((800, 800)).convert('L')

    data1 = np.array(img1).flatten().astype(np.float32)
    data2 = np.array(img2).flatten().astype(np.float32)

    ncc = np.dot(data1 - np.mean(data1), data2 - np.mean(data2)) / (
        np.sqrt(np.sum((data1 - np.mean(data1)) ** 2) * np.sum((data2 - np.mean(data2)) ** 2))
    )
    
    print(f'Normalized Cross-Correlation (NCC): {ncc}')

normalized_cross_correlation('vermeer.jpg', 'amelie_lens.jpg')

'''
Cosine Similarity: 0.3194198872064478
Mean Squared Error (MSE): 112.1342765625
Peak Signal-to-Noise Ratio (PSNR): 27.63341975332988 dB
Normalized Cross-Correlation (NCC): -0.4248916506767273
'''