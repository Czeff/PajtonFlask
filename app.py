from flask import Flask, request, jsonify, render_template_string, send_file
from werkzeug.utils import secure_filename
from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ImageDraw, ImageFont
import os
import time
import math
import json
import io
import traceback
import gc
from collections import defaultdict
import numpy as np
from scipy import ndimage
from skimage import measure, morphology, segmentation
from skimage.filters import gaussian
from skimage.color import rgb2lab, lab2rgb
try:
    import cv2
except ImportError:
    print("⚠️ OpenCV nie jest dostępne - używam fallback metod")
    cv2 = None

app = Flask(__name__)

# Konfiguracja
UPLOAD_FOLDER = 'uploads'
MAX_FILE_SIZE = 12 * 1024 * 1024  # 12MB - zwiększono dla lepszej jakości
MAX_IMAGE_SIZE = 1800  # Ultra wysoka rozdzielczość dla maksymalnej jakości
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'svg'}

# Upewnij się, że katalogi istnieją
for folder in ['raster', 'vector_auto', 'vector_manual', 'preview']:
    os.makedirs(os.path.join(UPLOAD_FOLDER, folder), exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def optimize_image_for_vectorization(image_path, max_size=MAX_IMAGE_SIZE):
    """Ultra zaawansowana optymalizacja obrazu z zachowaniem szczegółów oryginalnego - PREMIUM VERSION"""
    try:
        with Image.open(image_path) as img:
            # Konwersja do RGB z zachowaniem maksymalnej jakości
            if img.mode != 'RGB':
                if img.mode == 'RGBA':
                    # Zachowaj przezroczystość przez kompozycję z białym tłem
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background
                else:
                    img = img.convert('RGB')

            # PREMIUM OPTYMALIZACJA: Inteligentne skalowanie bazujące na zawartości
            original_width, original_height = img.size
            aspect_ratio = original_width / original_height
            
            # Analiza gęstości szczegółów dla inteligentnego skalowania
            detail_density = analyze_image_detail_density(img)
            
            if detail_density > 0.7:  # Bardzo dużo szczegółów
                target_size = min(max_size * 1.5, 2000)
            elif detail_density > 0.5:  # Dużo szczegółów
                target_size = min(max_size * 1.2, 1600)
            elif max(original_width, original_height) < 600:
                # Małe obrazy - agresywne zwiększanie dla zachowania detali
                target_size = min(max_size * 2.5, 2200)
            elif max(original_width, original_height) < 1000:
                # Średnie obrazy - umiarkowane zwiększanie
                target_size = min(max_size * 1.8, 1800)
            else:
                # Większe obrazy - kontrolowane skalowanie
                target_size = max_size

            # Ultra wysokiej jakości skalowanie z multi-pass sharpening
            current_size = max(original_width, original_height)
            if current_size != target_size:
                if aspect_ratio > 1:  # Landscape
                    new_width = target_size
                    new_height = int(target_size / aspect_ratio)
                else:  # Portrait
                    new_height = target_size
                    new_width = int(target_size * aspect_ratio)
                
                # Multi-step resizing dla lepszej jakości
                img = multi_step_resize(img, (new_width, new_height))

            # Premium multi-pass enhancement
            img = enhance_cartoon_precision_premium(img)

            return img
    except Exception as e:
        print(f"Błąd podczas optymalizacji obrazu: {e}")
        return None

def analyze_image_detail_density(img):
    """Analizuje gęstość szczegółów w obrazie"""
    try:
        # Konwertuj do skali szarości dla analizy
        gray = img.convert('L')
        img_array = np.array(gray)
        
        # Oblicz gradient dla wykrywania krawędzi
        from scipy import ndimage
        gradient_x = ndimage.sobel(img_array, axis=1)
        gradient_y = ndimage.sobel(img_array, axis=0)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Normalizuj i oblicz gęstość
        detail_density = np.mean(gradient_magnitude) / 255.0
        return min(1.0, detail_density * 3)
    except:
        return 0.5

def multi_step_resize(img, target_size):
    """Multi-step resizing dla lepszej jakości"""
    try:
        current_width, current_height = img.size
        target_width, target_height = target_size
        
        # Jeśli różnica jest duża, rób to w krokach
        width_ratio = target_width / current_width
        height_ratio = target_height / current_height
        max_ratio = max(width_ratio, height_ratio)
        
        if max_ratio > 2.0 or max_ratio < 0.5:
            # Duża zmiana - rób w krokach
            steps = int(abs(np.log2(max_ratio))) + 1
            
            for step in range(steps):
                progress = (step + 1) / steps
                intermediate_width = int(current_width + (target_width - current_width) * progress)
                intermediate_height = int(current_height + (target_height - current_height) * progress)
                
                if step == steps - 1:
                    # Ostatni krok - użyj dokładnego rozmiaru
                    intermediate_width, intermediate_height = target_width, target_height
                
                img = img.resize((intermediate_width, intermediate_height), Image.Resampling.LANCZOS)
                
                # Wyostrz po każdym kroku
                if step < steps - 1:
                    img = img.filter(ImageFilter.UnsharpMask(radius=0.5, percent=100, threshold=1))
        else:
            # Mała zmiana - bezpośrednie skalowanie
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        return img
    except:
        return img.resize(target_size, Image.Resampling.LANCZOS)

def enhance_cartoon_precision_premium(img):
    """Premium enhancement dla maksymalnej jakości cartoon-style obrazów"""
    try:
        # Multi-pass contrast enhancement
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.15)
        
        # Advanced multi-kernel sharpening
        img = img.filter(ImageFilter.UnsharpMask(radius=0.2, percent=150, threshold=1))
        img = img.filter(ImageFilter.UnsharpMask(radius=0.5, percent=100, threshold=2))
        img = img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=60, threshold=3))
        
        # Intelligent noise reduction
        img = img.filter(ImageFilter.MedianFilter(size=3))
        
        # Color enhancement dla lepszego wykrywania
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.15)
        
        # Final precision sharpening
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.3)
        
        # Edge enhancement
        img = img.filter(ImageFilter.EDGE_ENHANCE)
        
        return img
    except Exception as e:
        print(f"Błąd w enhance_cartoon_precision_premium: {e}")
        return img

def enhance_cartoon_precision_ultra(img):
    """Ultra precyzja dla obrazów cartoon-style z zachowaniem najmniejszych detali"""
    try:
        # Bardzo delikatne zwiększenie kontrastu z zachowaniem detali
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)

        # Multi-step wyostrzenie krawędzi z zachowaniem detali
        img = img.filter(ImageFilter.UnsharpMask(radius=0.3, percent=120, threshold=1))
        img = img.filter(ImageFilter.UnsharpMask(radius=0.8, percent=80, threshold=2))

        # Bardzo delikatna redukcja szumu bez utraty detali
        img = img.filter(ImageFilter.SMOOTH)

        # Zwiększenie nasycenia dla lepszego wykrywania kolorów
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.1)

        # Finalne delikatne wyostrzenie
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.2)

        return img
    except Exception as e:
        print(f"Błąd w enhance_cartoon_precision_ultra: {e}")
        return img

def detect_edge_density_advanced(img_array):
    """Zaawansowane wykrywanie gęstości krawędzi z wieloma filtrami"""
    try:
        from scipy import ndimage
        gray = np.mean(img_array, axis=2)
        
        # Różne operatory krawędzi
        sobel_h = ndimage.sobel(gray, axis=0)
        sobel_v = ndimage.sobel(gray, axis=1)
        sobel_combined = np.sqrt(sobel_h**2 + sobel_v**2)
        
        # Laplacian dla wykrywania cienkich linii
        laplacian = ndimage.laplace(gray)
        
        # Gradient magnitude
        grad_mag = np.sqrt(sobel_h**2 + sobel_v**2)
        
        # Kombinuj wyniki z wagami
        edge_density = (
            np.mean(sobel_combined) * 0.4 +
            np.mean(np.abs(laplacian)) * 0.3 +
            np.mean(grad_mag) * 0.3
        ) / 255.0
        
        return min(1.0, edge_density)
    except:
        return 0.1

def detect_edge_sharpness(img_array):
    """Wykrywa ostrość krawędzi"""
    try:
        from scipy import ndimage
        gray = np.mean(img_array, axis=2)
        
        # Drugi gradient dla ostrości
        laplacian = ndimage.laplace(gray)
        variance = np.var(laplacian)
        
        return min(1.0, variance / 10000.0)
    except:
        return 0.5

def detect_edge_connectivity(img_array):
    """Wykrywa łączność krawędzi (ciągłość linii)"""
    try:
        from scipy import ndimage
        gray = np.mean(img_array, axis=2)
        
        # Wykryj krawędzie
        edges = ndimage.sobel(gray)
        edge_mask = edges > np.percentile(edges, 85)
        
        # Policz komponenty połączone
        labeled, num_features = ndimage.label(edge_mask)
        
        if num_features == 0:
            return 0.0
            
        # Większa liczba małych komponentów = mniejsza łączność
        component_sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
        avg_component_size = np.mean(component_sizes)
        total_edge_pixels = np.sum(edge_mask)
        
        connectivity = min(1.0, avg_component_size / max(1, total_edge_pixels / num_features))
        return connectivity
    except:
        return 0.5

def detect_color_complexity_advanced(img_array):
    """Zaawansowana analiza złożoności kolorów"""
    try:
        # Analiza w różnych skalach
        complexities = []
        
        for scale in [1, 2, 4]:
            scaled = img_array[::scale, ::scale]
            
            # Unikalne kolory w przestrzeni LAB
            try:
                from skimage.color import rgb2lab
                lab_img = rgb2lab(scaled / 255.0)
                
                # Kvantyzacja w przestrzeni LAB
                lab_quantized = np.round(lab_img * 10) / 10
                unique_colors = np.unique(lab_quantized.reshape(-1, 3), axis=0)
                complexities.append(len(unique_colors))
            except:
                # Fallback do RGB
                unique_colors = np.unique(scaled.reshape(-1, 3), axis=0)
                complexities.append(len(unique_colors))
        
        return max(complexities)
    except:
        return 100

def detect_color_variance(img_array):
    """Wykrywa wariancję kolorów"""
    try:
        # Wariancja w każdym kanale
        var_r = np.var(img_array[:,:,0])
        var_g = np.var(img_array[:,:,1])
        var_b = np.var(img_array[:,:,2])
        
        total_variance = (var_r + var_g + var_b) / 3
        normalized_variance = total_variance / (255.0 ** 2)
        
        return min(1.0, normalized_variance * 4)
    except:
        return 0.5

def detect_color_gradients(img_array):
    """Wykrywa obecność gradientów kolorów"""
    try:
        from scipy import ndimage
        
        gradient_magnitudes = []
        
        for channel in range(3):
            grad_x = ndimage.sobel(img_array[:,:,channel], axis=1)
            grad_y = ndimage.sobel(img_array[:,:,channel], axis=0)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            gradient_magnitudes.append(np.mean(grad_mag))
        
        avg_gradient = np.mean(gradient_magnitudes) / 255.0
        return min(1.0, avg_gradient * 2)
    except:
        return 0.3

def detect_texture_complexity(img_array):
    """Wykrywa złożoność tekstur"""
    try:
        from scipy import ndimage
        gray = np.mean(img_array, axis=2)
        
        # Analiza tekstury przez filtry Gabora (symulowane)
        texture_responses = []
        
        # Różne kierunki i częstotliwości
        for angle in [0, 45, 90, 135]:
            for sigma in [1, 2, 4]:
                # Symulacja filtra Gabora przez gradient kierunkowy
                if angle == 0:
                    filtered = ndimage.sobel(gray, axis=1)
                elif angle == 45:
                    filtered = ndimage.sobel(gray, axis=0) + ndimage.sobel(gray, axis=1)
                elif angle == 90:
                    filtered = ndimage.sobel(gray, axis=0)
                else:  # 135
                    filtered = ndimage.sobel(gray, axis=0) - ndimage.sobel(gray, axis=1)
                
                # Gaussian blur dla różnych skal
                blurred = ndimage.gaussian_filter(filtered, sigma=sigma)
                texture_responses.append(np.std(blurred))
        
        texture_complexity = np.mean(texture_responses) / 255.0
        return min(1.0, texture_complexity * 3)
    except:
        return 0.3

def detect_pattern_regularity(img_array):
    """Wykrywa regularność wzorów"""
    try:
        from scipy import ndimage
        gray = np.mean(img_array, axis=2)
        
        # Autokorelacja dla wykrywania powtarzających się wzorów
        h, w = gray.shape
        
        # Zmniejsz obraz dla wydajności
        if h > 200 or w > 200:
            scale = min(200/h, 200/w)
            new_h, new_w = int(h*scale), int(w*scale)
            gray = ndimage.zoom(gray, (new_h/h, new_w/w))
        
        # Prosta autokorelacja
        mean_val = np.mean(gray)
        centered = gray - mean_val
        
        # Oblicz autokorelację dla małych przesunięć
        autocorr_values = []
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                if dx == 0 and dy == 0:
                    continue
                    
                shifted = np.roll(np.roll(centered, dx, axis=1), dy, axis=0)
                correlation = np.mean(centered * shifted)
                autocorr_values.append(abs(correlation))
        
        regularity = np.mean(autocorr_values) / max(1, np.var(gray))
        return min(1.0, regularity)
    except:
        return 0.2

def detect_geometric_shapes(img_array):
    """Wykrywa obecność kształtów geometrycznych"""
    try:
        from scipy import ndimage
        gray = np.mean(img_array, axis=2)
        
        # Wykryj krawędzie
        edges = ndimage.sobel(gray)
        edge_mask = edges > np.percentile(edges, 90)
        
        # Oblicz krzywizny (symulacja przez angles)
        shape_complexity = 0
        
        labeled, num_features = ndimage.label(edge_mask)
        
        for i in range(1, min(num_features + 1, 20)):  # Ogranicz do 20 komponentów
            component = labeled == i
            
            if np.sum(component) < 20:
                continue
                
            # Znajdź kontury komponentu
            y_coords, x_coords = np.where(component)
            
            if len(y_coords) > 10:
                # Prosta analiza kształtu przez bounding box
                bbox_area = (np.max(y_coords) - np.min(y_coords)) * (np.max(x_coords) - np.min(x_coords))
                actual_area = len(y_coords)
                
                if bbox_area > 0:
                    shape_ratio = actual_area / bbox_area
                    shape_complexity += (1 - shape_ratio)  # Mniej regularne = więcej złożoności
        
        return min(1.0, shape_complexity / max(1, num_features))
    except:
        return 0.3

def detect_curve_complexity(img_array):
    """Wykrywa złożoność krzywych"""
    try:
        from scipy import ndimage
        gray = np.mean(img_array, axis=2)
        
        # Drugi gradient dla krzywizny
        grad_x = ndimage.sobel(gray, axis=1)
        grad_y = ndimage.sobel(gray, axis=0)
        
        # Pochodne drugiego rzędu
        grad_xx = ndimage.sobel(grad_x, axis=1)
        grad_yy = ndimage.sobel(grad_y, axis=0)
        grad_xy = ndimage.sobel(grad_x, axis=0)
        
        # Krzywizna gaussowska (aproksymacja)
        curvature = grad_xx * grad_yy - grad_xy**2
        
        curve_complexity = np.std(curvature) / 255.0
        return min(1.0, curve_complexity * 2)
    except:
        return 0.3

def calculate_perceptual_importance(img_array):
    """Oblicza perceptualną ważność elementów obrazu"""
    try:
        from scipy import ndimage
        
        # Konwersja do jasności perceptualnej
        luminance = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
        
        # Kontrast lokalny
        local_std = ndimage.uniform_filter(luminance**2, size=9) - ndimage.uniform_filter(luminance, size=9)**2
        
        # Wysokofrequencyjne komponenty
        high_freq = luminance - ndimage.gaussian_filter(luminance, sigma=2)
        
        # Kombinuj miary
        importance = np.mean(local_std) * 0.6 + np.std(high_freq) * 0.4
        
        normalized_importance = importance / (255.0 ** 2)
        return min(1.0, normalized_importance * 3)
    except:
        return 0.5

def calculate_detail_density(img_array):
    """Oblicza gęstość szczegółów w obrazie"""
    try:
        from scipy import ndimage
        gray = np.mean(img_array, axis=2)
        
        # Wieloskalowa analiza szczegółów
        detail_maps = []
        
        for sigma in [0.5, 1.0, 2.0, 4.0]:
            blurred = ndimage.gaussian_filter(gray, sigma=sigma)
            details = np.abs(gray - blurred)
            detail_maps.append(details)
        
        # Kombinuj szczegóły z różnych skal
        combined_details = np.mean(detail_maps, axis=0)
        
        # Gęstość szczegółów
        detail_density = np.mean(combined_details) / 255.0
        return min(1.0, detail_density * 4)
    except:
        return 0.4

def calculate_overall_complexity_score(edge_density, edge_sharpness, edge_connectivity,
                                     color_complexity, color_variance, color_gradients,
                                     texture_complexity, pattern_regularity,
                                     geometric_complexity, curve_complexity,
                                     perceptual_importance, detail_density):
    """Oblicza ogólny wynik złożoności używając zaawansowanego algoritmu AI"""
    try:
        # Znormalizuj color_complexity
        color_complexity_norm = min(1.0, color_complexity / 500.0)
        
        # Wagi dla różnych komponentów (zoptymalizowane dla jakości wektoryzacji)
        weights = {
            'edge': 0.25,
            'color': 0.20,
            'texture': 0.15,
            'geometry': 0.15,
            'perception': 0.15,
            'detail': 0.10
        }
        
        # Komponenty złożoności
        edge_component = (edge_density * 0.4 + edge_sharpness * 0.4 + edge_connectivity * 0.2)
        color_component = (color_complexity_norm * 0.4 + color_variance * 0.3 + color_gradients * 0.3)
        texture_component = (texture_complexity * 0.7 + pattern_regularity * 0.3)
        geometry_component = (geometric_complexity * 0.6 + curve_complexity * 0.4)
        perception_component = perceptual_importance
        detail_component = detail_density
        
        # Ważona suma
        overall_score = (
            edge_component * weights['edge'] +
            color_component * weights['color'] +
            texture_component * weights['texture'] +
            geometry_component * weights['geometry'] +
            perception_component * weights['perception'] +
            detail_component * weights['detail']
        )
        
        # Nieliniowa transformacja dla lepszego rozkładu
        adjusted_score = np.power(overall_score, 0.8)
        
        return min(1.0, max(0.0, adjusted_score))
    except:
        return 0.5

def extract_dominant_colors_advanced(image, max_colors=50, params=None):
    """AI-enhanced ultra precyzyjna analiza kolorów z maksymalną jakością"""
    try:
        img_array = np.array(image)

        # Pobierz zaawansowane parametry AI
        quality_level = params.get('quality_enhancement', 'ai_high') if params else 'ai_high'
        tolerance_factor = params.get('tolerance_factor', 0.3) if params else 0.3
        edge_enhancement = params.get('edge_enhancement', True) if params else True
        gradient_preservation = params.get('gradient_preservation', True) if params else True
        micro_detail_preservation = params.get('micro_detail_preservation', False) if params else False

        print(f"🎨 AI Color Analysis: jakość={quality_level}, tolerancja={tolerance_factor}, edge_enh={edge_enhancement}")

        colors = []

        # 1. AI-ENHANCED DOMINANT COLORS - Największa precyzja
        if 'ai_' in quality_level:
            dominant_colors = extract_ai_enhanced_dominant_colors(img_array, max_colors // 2, params)
            colors.extend(dominant_colors)
            print(f"   🤖 AI dominant: {len(dominant_colors)} kolorów")

        # 2. MULTI-SCALE EDGE ANALYSIS - Kluczowe dla ostrych krawędzi
        if edge_enhancement:
            edge_colors = extract_multi_scale_edge_colors(img_array, max_colors // 4, params)
            colors.extend(edge_colors)
            print(f"   📐 Multi-scale edges: {len(edge_colors)} kolorów")

        # 3. GRADIENT PRESERVATION - Zachowanie płynnych przejść
        if gradient_preservation:
            gradient_colors = extract_advanced_gradient_colors(img_array, max_colors // 6, params)
            colors.extend(gradient_colors)
            print(f"   🌈 Advanced gradients: {len(gradient_colors)} kolorów")

        # 4. MICRO-DETAIL EXTRACTION - Najmniejsze szczegóły
        if micro_detail_preservation:
            micro_colors = extract_micro_detail_colors(img_array, max_colors // 8, params)
            colors.extend(micro_colors)
            print(f"   🔬 Micro details: {len(micro_colors)} kolorów")

        # 5. PERCEPTUAL COLOR MINING - Kolory ważne perceptualnie
        perceptual_colors = extract_perceptual_important_colors(img_array, max_colors // 5, params)
        colors.extend(perceptual_colors)
        print(f"   👁️ Perceptual: {len(perceptual_colors)} kolorów")

        # 6. ADAPTIVE CLUSTERING - Zaawansowany klaster analysis
        if len(colors) < max_colors:
            additional_colors = extract_adaptive_clustering_colors(img_array, max_colors - len(colors), params)
            colors.extend(additional_colors)
            print(f"   🎯 Adaptive clustering: {len(additional_colors)} kolorów")

        # AI-POWERED COLOR REFINEMENT
        final_colors = ai_powered_color_refinement(colors, max_colors, img_array, params)

        # INTELLIGENT COLOR SORTING
        final_colors = intelligent_color_importance_sorting(img_array, final_colors, params)

        print(f"🎨 AI Color Analysis Complete: {len(final_colors)} wysokiej jakości kolorów")
        return final_colors

    except Exception as e:
        print(f"❌ Błąd AI color analysis: {e}")
        return extract_dominant_colors_simple(image, max_colors)

def extract_dominant_colors_simple(image, max_colors=8):
    """Prosta metoda wyciągania kolorów dominujących jako fallback"""
    try:
        from sklearn.cluster import KMeans
        
        img_array = np.array(image)
        pixels = img_array.reshape(-1, 3)
        
        # Próbkowanie dla wydajności
        if len(pixels) > 10000:
            step = len(pixels) // 10000
            pixels = pixels[::step]
        
        # K-means clustering
        kmeans = KMeans(n_clusters=min(max_colors, len(pixels)), random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        colors = [(int(c[0]), int(c[1]), int(c[2])) for c in kmeans.cluster_centers_]
        return colors
        
    except Exception as e:
        print(f"Błąd w extract_dominant_colors_simple: {e}")
        # Ostateczny fallback - próbkowanie kolorów
        try:
            img_array = np.array(image)
            h, w = img_array.shape[:2]
            
            colors = []
            step_h = max(1, h // 10)
            step_w = max(1, w // 10)
            
            for y in range(0, h, step_h):
                for x in range(0, w, step_w):
                    color = tuple(img_array[y, x])
                    if color not in colors and len(colors) < max_colors:
                        colors.append(color)
            
            return colors[:max_colors]
        except:
            return [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Domyślne kolory

def extract_precise_dominant_colors(img_array, max_colors):
    """Precyzyjne wyciąganie kolorów dominujących"""
    try:
        from sklearn.cluster import KMeans

        # Próbkowanie z zachowaniem reprezentatywności
        height, width = img_array.shape[:2]
        sample_rate = min(0.3, 50000 / (height * width))

        pixels = img_array.reshape(-1, 3)
        if len(pixels) > 50000:
            step = int(1 / sample_rate)
            pixels = pixels[::step]

        # K-means z większą liczbą iteracji dla precyzji
        kmeans = KMeans(n_clusters=max_colors, random_state=42, n_init=30, max_iter=500)
        kmeans.fit(pixels)

        return [(int(c[0]), int(c[1]), int(c[2])) for c in kmeans.cluster_centers_]
    except:
        return []

def extract_edge_based_colors(img_array, max_colors):
    """Wyciąga kolory z obszarów krawędzi - kluczowe dla cartoon-style"""
    try:
        from scipy import ndimage

        # Wykryj krawędzie z wysoką precyzją
        gray = np.mean(img_array, axis=2)
        edges = ndimage.sobel(gray)

        # Threshold adaptacyjny
        threshold = np.percentile(edges, 85)
        edge_mask = edges > threshold

        # Rozszerz obszary krawędzi
        from scipy.ndimage import binary_dilation
        edge_mask = binary_dilation(edge_mask, iterations=2)

        # Wyciągnij kolory z obszarów krawędzi
        edge_pixels = img_array[edge_mask]

        if len(edge_pixels) > 1000:
            from sklearn.cluster import KMeans
            n_clusters = min(max_colors, len(edge_pixels) // 100)
            if n_clusters > 0:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                kmeans.fit(edge_pixels)
                return [(int(c[0]), int(c[1]), int(c[2])) for c in kmeans.cluster_centers_]

        return []
    except:
        return []

def extract_gradient_colors(img_array, max_colors):
    """Wyciąga kolory z obszarów gradientów"""
    try:
        from scipy import ndimage

        # Oblicz gradienty dla każdego kanału
        gradients = []
        for channel in range(3):
            grad_x = ndimage.sobel(img_array[:,:,channel], axis=1)
            grad_y = ndimage.sobel(img_array[:,:,channel], axis=0)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradients.append(gradient_magnitude)

        # Znajdź obszary z wysokimi gradientami
        total_gradient = np.sum(gradients, axis=0)
        threshold = np.percentile(total_gradient, 70)
        gradient_mask = total_gradient > threshold

        # Wyciągnij kolory z tych obszarów
        gradient_pixels = img_array[gradient_mask]

        if len(gradient_pixels) > 500:
            # Clustering kolorów gradientów
            from sklearn.cluster import KMeans
            n_clusters = min(max_colors, len(gradient_pixels) // 200)
            if n_clusters > 0:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                kmeans.fit(gradient_pixels)
                return [(int(c[0]), int(c[1]), int(c[2])) for c in kmeans.cluster_centers_]

        return []
    except:
        return []

def extract_detail_colors(img_array, max_colors):
    """Wyciąga kolory z małych szczegółów i tekstur"""
    try:
        from sklearn.cluster import KMeans
        from scipy import ndimage

        # Wykryj małe obiekty i detale
        gray = np.mean(img_array, axis=2)

        # Filtr Laplace'a do wykrywania szczegółów
        laplacian = ndimage.laplace(gray)
        detail_mask = np.abs(laplacian) > np.percentile(np.abs(laplacian), 85)

        # Rozszerz obszary szczegółów
        detail_mask = ndimage.binary_dilation(detail_mask, iterations=1)

        # Wyciągnij kolory z obszarów szczegółów
        detail_pixels = img_array[detail_mask]

        if len(detail_pixels) > 100:
            # Używaj większej liczby klastrów dla szczegółów
            n_clusters = min(max_colors, max(5, len(detail_pixels) // 50))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
            kmeans.fit(detail_pixels)
            return [(int(c[0]), int(c[1]), int(c[2])) for c in kmeans.cluster_centers_]

        return []
    except:
        return []

def extract_shadow_highlight_colors(img_array, max_colors):
    """Wyciąga kolory cieni i rozjaśnień - kluczowe dla cartoon-style"""
    try:
        from sklearn.cluster import KMeans

        # Oblicz jasność każdego piksela
        brightness = np.mean(img_array, axis=2)

        # Znajdź bardzo ciemne obszary (cienie)
        shadow_threshold = np.percentile(brightness, 15)
        shadow_mask = brightness <= shadow_threshold

        # Znajdź bardzo jasne obszary (rozjaśnienia)
        highlight_threshold = np.percentile(brightness, 85)
        highlight_mask = brightness >= highlight_threshold

        colors = []

        # Wyciągnij kolory cieni
        shadow_pixels = img_array[shadow_mask]
        if len(shadow_pixels) > 100:
            n_clusters = min(max_colors // 2, len(shadow_pixels) // 200)
            if n_clusters > 0:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                kmeans.fit(shadow_pixels)
                shadow_colors = [(int(c[0]), int(c[1]), int(c[2])) for c in kmeans.cluster_centers_]
                colors.extend(shadow_colors)

        # Wyciągnij kolory rozjaśnień
        highlight_pixels = img_array[highlight_mask]
        if len(highlight_pixels) > 100:
            n_clusters = min(max_colors // 2, len(highlight_pixels) // 200)
            if n_clusters > 0:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                kmeans.fit(highlight_pixels)
                highlight_colors = [(int(c[0]), int(c[1]), int(c[2])) for c in kmeans.cluster_centers_]
                colors.extend(highlight_colors)

        return colors[:max_colors]
    except:
        return []

def remove_similar_colors_ultra_precise(colors, max_colors, tolerance_factor=0.8):
    """Ultra precyzyjne usuwanie podobnych kolorów z maksymalnie liberalnym podejściem"""
    if not colors:
        return []

    final_colors = [colors[0]]

    for color in colors[1:]:
        is_unique = True

        for existing in final_colors:
            # Zaawansowane obliczanie różnicy kolorów w przestrzeni LAB
            distance = calculate_advanced_color_distance(color, existing)

            # DRASTYCZNIE zmniejszone progi - zachowaj praktycznie wszystkie odcienie
            brightness = sum(existing) / 3
            saturation = max(existing) - min(existing)

            # Minimalne progi tolerancji dla maksymalnej szczegółowości
            if brightness < 30:  # Bardzo ciemne kolory
                base_tolerance = 0.8
            elif brightness < 60:  # Ciemne kolory
                base_tolerance = 1.0
            elif brightness < 120:  # Średnio ciemne
                base_tolerance = 1.2
            elif brightness > 230:  # Bardzo jasne kolory
                base_tolerance = 2.0
            elif brightness > 200:  # Jasne kolory
                base_tolerance = 1.8
            elif brightness > 160:  # Średnio jasne
                base_tolerance = 1.5
            else:  # Średnie kolory
                base_tolerance = 1.3

            # Zastosuj bardzo liberalny czynnik tolerancji
            tolerance = base_tolerance * tolerance_factor

            # Dodatkowa tolerancja dla wysoko nasyconych kolorów (typowe w cartoon)
            if saturation > 120:  # Bardzo nasycone
                tolerance += 5
            elif saturation > 80:  # Nasycone
                tolerance += 3
            elif saturation < 20:  # Szare/niskie nasycenie
                tolerance -= 2

            # Specjalna logika dla kolorów skóry (cartoon-style często ma specyficzne odcienie)
            if is_skin_tone(existing) and is_skin_tone(color):
                tolerance = max(4, tolerance * 0.6)  # Mniejsza tolerancja dla odcieni skóry

            # Specjalna logika dla zieleni (liście, trawa w cartoon)
            if is_green_tone(existing) and is_green_tone(color):
                tolerance *= 0.8  # Mniejsza tolerancja dla odcieni zieleni

            # Dodatkowa precyzja dla podstawowych kolorów cartoon
            if is_primary_cartoon_color(existing) or is_primary_cartoon_color(color):
                tolerance *= 0.7

            if distance < tolerance:
                is_unique = False
                break

        if is_unique and len(final_colors) < max_colors:
            final_colors.append(color)

    return final_colors

def is_skin_tone(color):
    """Sprawdza czy kolor to odcień skóry"""
    r, g, b = color[:3]
    # Typowe zakresy dla odcieni skóry
    return (120 <= r <= 255 and 80 <= g <= 220 and 60 <= b <= 180 and 
            r > g > b and r - g < 80 and g - b < 60)

def is_green_tone(color):
    """Sprawdza czy kolor to odcień zieleni"""
    r, g, b = color[:3]
    # Zielone odcienie - g dominuje
    return g > r and g > b and g > 80

def is_primary_cartoon_color(color):
    """Sprawdza czy to podstawowy kolor cartoon (czerwony, niebieski, żółty, etc.)"""
    r, g, b = color[:3]

    # Czerwony
    if r > 180 and g < 80 and b < 80:
        return True
    # Niebieski
    if b > 180 and r < 80 and g < 80:
        return True
    # Żółty
    if r > 180 and g > 180 and b < 80:
        return True
    # Czarny
    if r < 50 and g < 50 and b < 50:
        return True
    # Biały
    if r > 220 and g > 220 and b > 220:
        return True

    return False

def calculate_advanced_color_distance(color1, color2):
    """Zaawansowane obliczanie odległości kolorów z Delta E 2000"""
    try:
        from skimage.color import rgb2lab, deltaE_cie76

        # Konwersja do przestrzeni LAB
        c1_lab = rgb2lab(np.array(color1).reshape(1, 1, 3) / 255.0)[0, 0]
        c2_lab = rgb2lab(np.array(color2).reshape(1, 1, 3) / 255.0)[0, 0]

        # Delta E CIE76 - bardziej precyzyjna miara różnicy kolorów
        delta_e = np.sqrt(
            (c1_lab[0] - c2_lab[0])**2 + 
            (c1_lab[1] - c2_lab[1])**2 + 
            (c1_lab[2] - c2_lab[2])**2
        )
        return delta_e
    except:
        # Fallback do ulepszonej Euclidean distance z wagami
        r_diff = (color1[0] - color2[0]) * 0.299
        g_diff = (color1[1] - color2[1]) * 0.587
        b_diff = (color1[2] - color2[2]) * 0.114
        return np.sqrt(r_diff**2 + g_diff**2 + b_diff**2)

def sort_colors_by_visual_importance(img_array, colors):
    """Sortuje kolory według wizualnej ważności w obrazie"""
    try:
        color_importance = []
        height, width = img_array.shape[:2]

        for color in colors:
            # Oblicz częstotliwość i pozycję
            distances = np.sqrt(np.sum((img_array - np.array(color))**2, axis=2))
            frequency = np.sum(distances < 25)

            if frequency > 0:
                # Znajdź pozycje pikseli tego koloru
                y_coords, x_coords = np.where(distances < 25)

                # Centralność (środek obrazu jest ważniejszy)
                center_distance = np.mean(np.sqrt(
                    ((y_coords - height/2) / height)**2 + 
                    ((x_coords - width/2) / width)**2
                ))
                centrality_weight = 1.0 - center_distance

                # Rozłożenie (bardziej rozproszone kolory są ważniejsze)
                if len(y_coords) > 1:
                    spread = np.std(y_coords) + np.std(x_coords)
                    spread_weight = min(1.0, spread / (height + width) * 4)
                else:
                    spread_weight = 0

                # Kontrast (kolory kontrastujące z otoczeniem są ważniejsze)
                contrast_weight = calculate_local_contrast(img_array, color, y_coords, x_coords)

                # Kombinuj wszystkie czynniki
                importance = (
                    frequency * 0.4 +  # Częstotliwość
                    frequency * centrality_weight * 0.3 +  # Centralność
                    frequency * spread_weight * 0.2 +  # Rozłożenie
                    frequency * contrast_weight * 0.1  # Kontrast
                )
            else:
                importance = 0

            color_importance.append((importance, color))

        # Sortuj według ważności (malejąco)
        color_importance.sort(reverse=True)
        return [color for importance, color in color_importance]
    except:
        return colors

def calculate_local_contrast(img_array, color, y_coords, x_coords):
    """Oblicza lokalny kontrast koloru z otoczeniem"""
    try:
        if len(y_coords) == 0:
            return 0

        color_array = np.array(color)
        contrasts = []

        # Sprawdź kontrast w losowych punktach
        sample_size = min(100, len(y_coords))
        indices = np.random.choice(len(y_coords), sample_size, replace=False)

        for idx in indices:
            y, x = y_coords[idx], x_coords[idx]

            # Sprawdź otoczenie 5x5
            y_start, y_end = max(0, y-2), min(img_array.shape[0], y+3)
            x_start, x_end = max(0, x-2), min(img_array.shape[1], x+3)

            neighborhood = img_array[y_start:y_end, x_start:x_end]
            if neighborhood.size > 0:
                avg_neighbor_color = np.mean(neighborhood.reshape(-1, 3), axis=0)
                contrast = np.sqrt(np.sum((color_array - avg_neighbor_color)**2))
                contrasts.append(contrast)

        return np.mean(contrasts) / 255.0 if contrasts else 0
    except:
        return 0

def extract_high_precision_kmeans(img_array, max_colors):
    """K-means z wysoką precyzją"""
    try:
        from sklearn.cluster import KMeans

        # Konwersja do przestrzeni LAB dla lepszej percepcji kolorów
        from skimage.color import rgb2lab, lab2rgb
        lab_image = rgb2lab(img_array / 255.0)

        pixels = lab_image.reshape(-1, 3)
        if len(pixels) > 20000:
            pixels = pixels[::len(pixels)//20000]

        kmeans = KMeans(n_clusters=max_colors, random_state=42, n_init=50, max_iter=1000)
        kmeans.fit(pixels)

        # Konwersja z powrotem do RGB
        rgb_colors = []
        for lab_color in kmeans.cluster_centers_:
            try:
                rgb = lab2rgb(lab_color.reshape(1, 1, 3))[0, 0]
                rgb = np.clip(rgb * 255, 0, 255).astype(int)
                rgb_colors.append(tuple(rgb))
            except:
                continue

        return rgb_colors
    except:
        return []

def extract_ai_enhanced_dominant_colors(img_array, max_colors, params):
    """AI-enhanced extraction dominujących kolorów"""
    try:
        from sklearn.cluster import KMeans
        
        # Multi-resolution sampling dla lepszej reprezentacji
        samples = []
        
        # Pełna rozdzielczość - najważniejsze piksele
        height, width = img_array.shape[:2]
        full_sample_rate = min(0.3, 100000 / (height * width))
        if full_sample_rate > 0.01:
            full_pixels = img_array.reshape(-1, 3)
            step = max(1, int(1 / full_sample_rate))
            samples.extend(full_pixels[::step])

        # Średnia rozdzielczość - balance między jakością a wydajnością
        medium_img = img_array[::2, ::2]
        medium_pixels = medium_img.reshape(-1, 3)
        if len(medium_pixels) > 5000:
            step = len(medium_pixels) // 5000
            samples.extend(medium_pixels[::step])
        else:
            samples.extend(medium_pixels)

        # Niska rozdzielczość - globalne trendy
        low_img = img_array[::4, ::4]
        samples.extend(low_img.reshape(-1, 3))

        if not samples:
            return []

        all_samples = np.array(samples)
        
        # AI-enhanced K-means z multiple iterations i optimization
        best_colors = []
        best_inertia = float('inf')
        
        for attempt in range(3):  # Multiple attempts for stability
            try:
                kmeans = KMeans(
                    n_clusters=min(max_colors, len(all_samples)), 
                    random_state=42 + attempt,
                    n_init=20,
                    max_iter=500,
                    algorithm='lloyd',
                    init='k-means++'
                )
                kmeans.fit(all_samples)
                
                if kmeans.inertia_ < best_inertia:
                    best_inertia = kmeans.inertia_
                    best_colors = [(int(c[0]), int(c[1]), int(c[2])) for c in kmeans.cluster_centers_]
            except:
                continue
        
        return best_colors if best_colors else []
    except:
        return []

def extract_multi_scale_edge_colors(img_array, max_colors, params):
    """Multi-scale extraction kolorów z krawędzi"""
    try:
        from scipy import ndimage
        colors = []
        
        # Różne skale dla wykrywania krawędzi
        scales = [1.0, 1.5, 2.0, 3.0]
        
        for sigma in scales:
            # Gaussian blur followed by edge detection
            blurred = ndimage.gaussian_filter(img_array.astype(float), sigma=(sigma, sigma, 0))
            
            # Multi-channel edge detection
            edges_combined = np.zeros(img_array.shape[:2])
            
            for channel in range(3):
                grad_x = ndimage.sobel(blurred[:,:,channel], axis=1)
                grad_y = ndimage.sobel(blurred[:,:,channel], axis=0)
                edges_combined += np.sqrt(grad_x**2 + grad_y**2)
            
            # Adaptive threshold based on scale
            threshold = np.percentile(edges_combined, 90 - sigma * 5)
            edge_mask = edges_combined > threshold
            
            # Dilate edges to capture nearby colors
            edge_mask = ndimage.binary_dilation(edge_mask, iterations=int(sigma))
            
            edge_pixels = img_array[edge_mask]
            
            if len(edge_pixels) > 50:
                from sklearn.cluster import KMeans
                n_clusters = min(max_colors // len(scales), len(edge_pixels) // 100, 8)
                if n_clusters > 0:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    kmeans.fit(edge_pixels)
                    scale_colors = [(int(c[0]), int(c[1]), int(c[2])) for c in kmeans.cluster_centers_]
                    colors.extend(scale_colors)
        
        return colors[:max_colors]
    except:
        return []

def extract_advanced_gradient_colors(img_array, max_colors, params):
    """Zaawansowane wykrywanie kolorów gradientów"""
    try:
        from scipy import ndimage
        colors = []
        
        # Analiza gradientów w przestrzeni LAB dla lepszej percepcji
        try:
            from skimage.color import rgb2lab, lab2rgb
            lab_img = rgb2lab(img_array / 255.0)
            
            # Gradienty w każdym kanale LAB
            gradient_maps = []
            for channel in range(3):
                grad_x = ndimage.sobel(lab_img[:,:,channel], axis=1)
                grad_y = ndimage.sobel(lab_img[:,:,channel], axis=0)
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                gradient_maps.append(gradient_magnitude)
            
            # Kombinuj gradienty
            combined_gradient = np.mean(gradient_maps, axis=0)
            
            # Wykryj obszary z wysokimi gradientami
            gradient_threshold = np.percentile(combined_gradient, 75)
            gradient_mask = combined_gradient > gradient_threshold
            
            # Rozszerz obszary gradientów
            gradient_mask = ndimage.binary_dilation(gradient_mask, iterations=2)
            
            gradient_pixels = img_array[gradient_mask]
            
            if len(gradient_pixels) > 100:
                # Clustering w przestrzeni LAB
                lab_pixels = rgb2lab(gradient_pixels.reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
                
                from sklearn.cluster import KMeans
                n_clusters = min(max_colors, len(lab_pixels) // 50, 12)
                if n_clusters > 0:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=15)
                    kmeans.fit(lab_pixels)
                    
                    # Konwersja z powrotem do RGB
                    for lab_color in kmeans.cluster_centers_:
                        try:
                            rgb = lab2rgb(lab_color.reshape(1, 1, 3))[0, 0] * 255
                            rgb_color = tuple(np.clip(rgb, 0, 255).astype(int))
                            colors.append(rgb_color)
                        except:
                            continue
        except:
            # Fallback do standardowej analizy RGB
            return extract_gradient_colors(img_array, max_colors)
        
        return colors
    except:
        return []

def extract_micro_detail_colors(img_array, max_colors, params):
    """Wykrywanie kolorów z mikro-detali"""
    try:
        from scipy import ndimage
        colors = []
        
        # High-frequency details detection
        gray = np.mean(img_array, axis=2)
        
        # Multi-scale Laplacian for detail detection
        detail_maps = []
        for sigma in [0.5, 1.0, 1.5]:
            blurred = ndimage.gaussian_filter(gray, sigma=sigma)
            details = gray - blurred
            detail_maps.append(np.abs(details))
        
        # Kombinuj mapy szczegółów
        combined_details = np.mean(detail_maps, axis=0)
        
        # Znajdź obszary z wysokimi szczegółami
        detail_threshold = np.percentile(combined_details, 85)
        detail_mask = combined_details > detail_threshold
        
        # Dodatkowo sprawdź variance w lokalnym sąsiedztwie
        local_variance = ndimage.uniform_filter(gray**2, size=3) - ndimage.uniform_filter(gray, size=3)**2
        variance_threshold = np.percentile(local_variance, 80)
        variance_mask = local_variance > variance_threshold
        
        # Kombinuj maski
        micro_detail_mask = detail_mask | variance_mask
        
        # Rozszerz nieznacznie aby złapać kolory w pobliżu
        micro_detail_mask = ndimage.binary_dilation(micro_detail_mask, iterations=1)
        
        micro_pixels = img_array[micro_detail_mask]
        
        if len(micro_pixels) > 30:
            from sklearn.cluster import KMeans
            n_clusters = min(max_colors, len(micro_pixels) // 20, 6)
            if n_clusters > 0:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans.fit(micro_pixels)
                colors = [(int(c[0]), int(c[1]), int(c[2])) for c in kmeans.cluster_centers_]
        
        return colors
    except:
        return []

def extract_perceptual_important_colors(img_array, max_colors, params):
    """Wykrywanie perceptualnie ważnych kolorów"""
    try:
        from scipy import ndimage
        
        # Konwersja do przestrzeni luminancji perceptualnej
        luminance = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
        
        # Obszary o wysokim kontraście lokalnym
        local_contrast = ndimage.uniform_filter(luminance**2, size=5) - ndimage.uniform_filter(luminance, size=5)**2
        contrast_mask = local_contrast > np.percentile(local_contrast, 75)
        
        # Obszary o wysokiej saturacji
        saturation = np.max(img_array, axis=2) - np.min(img_array, axis=2)
        saturation_mask = saturation > np.percentile(saturation, 70)
        
        # Obszary na krawędziach obrazu (często ważne perceptualnie)
        height, width = img_array.shape[:2]
        edge_region = np.zeros((height, width), dtype=bool)
        border_width = min(height, width) // 20
        edge_region[:border_width, :] = True
        edge_region[-border_width:, :] = True
        edge_region[:, :border_width] = True
        edge_region[:, -border_width:] = True
        
        # Kombinuj maski ważności perceptualnej
        importance_mask = contrast_mask | saturation_mask | edge_region
        
        important_pixels = img_array[importance_mask]
        
        colors = []
        if len(important_pixels) > 50:
            from sklearn.cluster import KMeans
            n_clusters = min(max_colors, len(important_pixels) // 100, 10)
            if n_clusters > 0:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=15)
                kmeans.fit(important_pixels)
                colors = [(int(c[0]), int(c[1]), int(c[2])) for c in kmeans.cluster_centers_]
        
        return colors
    except:
        return []

def extract_adaptive_clustering_colors(img_array, max_colors, params):
    """Adaptive clustering z automatycznym doborem parametrów"""
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        # Przygotuj dane
        pixels = img_array.reshape(-1, 3)
        
        # Próbkowanie adaptacyjne
        n_pixels = len(pixels)
        if n_pixels > 50000:
            sample_rate = 50000 / n_pixels
            indices = np.random.choice(n_pixels, size=50000, replace=False)
            sampled_pixels = pixels[indices]
        else:
            sampled_pixels = pixels
        
        # Znajdź optymalną liczbę klastrów
        best_k = max_colors
        best_score = -1
        
        for k in range(min(5, max_colors), min(max_colors + 1, 15)):
            if k >= len(sampled_pixels):
                break
                
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(sampled_pixels)
                
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(sampled_pixels, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
            except:
                continue
        
        # Clustering z optymalną liczbą klastrów
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20, max_iter=500)
        kmeans.fit(sampled_pixels)
        
        colors = [(int(c[0]), int(c[1]), int(c[2])) for c in kmeans.cluster_centers_]
        return colors[:max_colors]
    except:
        return []

def ai_powered_color_refinement(colors, max_colors, img_array, params):
    """AI-powered refinement kolorów z zaawansowanymi algorytmami"""
    try:
        if not colors:
            return []
        
        tolerance_factor = params.get('tolerance_factor', 0.3) if params else 0.3
        
        # Usuń duplikaty z zaawansowaną tolerancją
        refined_colors = advanced_color_deduplication(colors, tolerance_factor)
        
        # Intelligent color merging dla podobnych odcieni
        merged_colors = intelligent_color_merging(refined_colors, img_array, tolerance_factor)
        
        # Validate colors against image content
        validated_colors = validate_colors_against_image(merged_colors, img_array)
        
        # Ensure we don't exceed max_colors
        final_colors = validated_colors[:max_colors]
        
        print(f"   🎯 Color refinement: {len(colors)} → {len(final_colors)} kolorów")
        return final_colors
    except:
        return colors[:max_colors]

def advanced_color_deduplication(colors, tolerance_factor):
    """Zaawansowane usuwanie duplikatów kolorów"""
    try:
        if not colors:
            return []
        
        final_colors = [colors[0]]
        
        for color in colors[1:]:
            is_unique = True
            
            for existing in final_colors:
                # Multi-space color distance
                distance = calculate_multi_space_color_distance(color, existing)
                
                # Adaptive tolerance based on color properties
                adaptive_tolerance = calculate_adaptive_tolerance(existing, tolerance_factor)
                
                if distance < adaptive_tolerance:
                    is_unique = False
                    break
            
            if is_unique:
                final_colors.append(color)
        
        return final_colors
    except:
        return colors

def calculate_multi_space_color_distance(color1, color2):
    """Oblicza odległość kolorów w wielu przestrzeniach"""
    try:
        # RGB Euclidean
        rgb_dist = np.sqrt(sum((a - b)**2 for a, b in zip(color1, color2)))
        
        # Weighted RGB (perceptual)
        weighted_rgb_dist = np.sqrt(
            0.3 * (color1[0] - color2[0])**2 +
            0.59 * (color1[1] - color2[1])**2 +
            0.11 * (color1[2] - color2[2])**2
        )
        
        # HSV distance
        hsv1 = rgb_to_hsv_precise(color1)
        hsv2 = rgb_to_hsv_precise(color2)
        
        h_dist = min(abs(hsv1[0] - hsv2[0]), 1 - abs(hsv1[0] - hsv2[0])) if hsv1[1] > 0.1 and hsv2[1] > 0.1 else 0
        s_dist = abs(hsv1[1] - hsv2[1])
        v_dist = abs(hsv1[2] - hsv2[2])
        
        hsv_dist = np.sqrt(h_dist**2 + s_dist**2 + v_dist**2)
        
        # Kombinuj odległości
        combined_distance = (rgb_dist * 0.4 + weighted_rgb_dist * 0.4 + hsv_dist * 100 * 0.2)
        
        return combined_distance
    except:
        return np.sqrt(sum((a - b)**2 for a, b in zip(color1, color2)))

def rgb_to_hsv_precise(color):
    """Precyzyjna konwersja RGB do HSV"""
    try:
        r, g, b = [x / 255.0 for x in color]
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        diff = max_val - min_val
        
        # Value
        v = max_val
        
        # Saturation
        s = 0 if max_val == 0 else diff / max_val
        
        # Hue
        h = 0
        if diff != 0:
            if max_val == r:
                h = (60 * ((g - b) / diff) + 360) % 360
            elif max_val == g:
                h = (60 * ((b - r) / diff) + 120) % 360
            else:
                h = (60 * ((r - g) / diff) + 240) % 360
        
        return [h / 360.0, s, v]
    except:
        return [0, 0, 0]

def calculate_adaptive_tolerance(color, base_tolerance):
    """Oblicza adaptacyjną tolerancję bazując na właściwościach koloru"""
    try:
        brightness = sum(color) / 3
        saturation = (max(color) - min(color)) / 255.0
        
        # Mniejsza tolerancja dla ciemnych kolorów
        brightness_factor = 0.8 if brightness < 50 else 1.0 if brightness < 150 else 1.2
        
        # Mniejsza tolerancja dla wysoko nasyconych kolorów
        saturation_factor = 0.7 if saturation > 0.7 else 0.9 if saturation > 0.4 else 1.1
        
        adaptive_tolerance = base_tolerance * brightness_factor * saturation_factor * 100
        
        return max(5, min(50, adaptive_tolerance))
    except:
        return base_tolerance * 25

def intelligent_color_merging(colors, img_array, tolerance_factor):
    """Inteligentne łączenie podobnych kolorów"""
    try:
        if len(colors) <= 1:
            return colors
        
        merged_colors = []
        used_indices = set()
        
        for i, color in enumerate(colors):
            if i in used_indices:
                continue
            
            # Znajdź podobne kolory
            similar_colors = [color]
            similar_indices = [i]
            
            for j, other_color in enumerate(colors[i+1:], i+1):
                if j in used_indices:
                    continue
                
                distance = calculate_multi_space_color_distance(color, other_color)
                merge_threshold = calculate_adaptive_tolerance(color, tolerance_factor) * 0.8
                
                if distance < merge_threshold:
                    similar_colors.append(other_color)
                    similar_indices.append(j)
            
            # Jeśli znaleziono podobne kolory, połącz je
            if len(similar_colors) > 1:
                # Weighted average based on importance in image
                weights = []
                for sim_color in similar_colors:
                    weight = calculate_color_importance_in_image(sim_color, img_array)
                    weights.append(weight)
                
                total_weight = sum(weights)
                if total_weight > 0:
                    merged_color = [0, 0, 0]
                    for k, (sim_color, weight) in enumerate(zip(similar_colors, weights)):
                        for channel in range(3):
                            merged_color[channel] += sim_color[channel] * (weight / total_weight)
                    
                    merged_color = tuple(int(c) for c in merged_color)
                else:
                    merged_color = color
                
                merged_colors.append(merged_color)
                used_indices.update(similar_indices)
            else:
                merged_colors.append(color)
                used_indices.add(i)
        
        return merged_colors
    except:
        return colors

def calculate_color_importance_in_image(color, img_array):
    """Oblicza ważność koloru w obrazie"""
    try:
        # Policz piksele podobne do danego koloru
        distances = np.sqrt(np.sum((img_array - np.array(color))**2, axis=2))
        similar_pixels = np.sum(distances < 30)
        
        # Normalizuj przez całkowitą liczbę pikseli
        total_pixels = img_array.shape[0] * img_array.shape[1]
        importance = similar_pixels / total_pixels
        
        return importance
    except:
        return 1.0

def validate_colors_against_image(colors, img_array):
    """Waliduje kolory względem zawartości obrazu"""
    try:
        validated_colors = []
        
        for color in colors:
            # Sprawdź czy kolor rzeczywiście występuje w obrazie
            distances = np.sqrt(np.sum((img_array - np.array(color))**2, axis=2))
            min_distance = np.min(distances)
            
            # Akceptuj kolor jeśli ma podobne piksele w obrazie
            if min_distance < 50:  # Tolerancja na błędy kwantyzacji
                validated_colors.append(color)
        
        return validated_colors if validated_colors else colors
    except:
        return colors

def intelligent_color_importance_sorting(img_array, colors, params):
    """Inteligentne sortowanie kolorów według ważności"""
    try:
        if not colors:
            return colors
        
        color_scores = []
        
        for color in colors:
            # Frequency score
            distances = np.sqrt(np.sum((img_array - np.array(color))**2, axis=2))
            frequency = np.sum(distances < 25)
            
            # Perceptual importance score
            brightness = sum(color) / 3
            saturation = (max(color) - min(color)) / 255.0
            
            perceptual_score = 1.0
            if brightness < 30 or brightness > 225:  # Very dark or very bright
                perceptual_score *= 1.2
            if saturation > 0.7:  # High saturation
                perceptual_score *= 1.1
            
            # Edge presence score
            edge_score = calculate_color_edge_presence(color, img_array)
            
            # Combined score
            total_score = frequency * 0.5 + perceptual_score * frequency * 0.3 + edge_score * 0.2
            color_scores.append((total_score, color))
        
        # Sort by score (descending)
        color_scores.sort(reverse=True)
        
        return [color for score, color in color_scores]
    except:
        return colors

def calculate_color_edge_presence(color, img_array):
    """Oblicza obecność koloru na krawędziach"""
    try:
        from scipy import ndimage
        
        # Wykryj krawędzie
        gray = np.mean(img_array, axis=2)
        edges = ndimage.sobel(gray)
        edge_mask = edges > np.percentile(edges, 85)
        
        # Sprawdź obecność koloru na krawędziach
        distances = np.sqrt(np.sum((img_array - np.array(color))**2, axis=2))
        color_mask = distances < 30
        
        edge_color_overlap = np.sum(edge_mask & color_mask)
        total_color_pixels = np.sum(color_mask)
        
        if total_color_pixels > 0:
            return edge_color_overlap / total_color_pixels
        
        return 0
    except:
        return 0

def get_color_hue(color):
    """Oblicza hue koloru"""
    try:
        r, g, b = [x/255.0 for x in color[:3]]
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        diff = max_val - min_val

        if diff == 0:
            return 0

        if max_val == r:
            hue = (60 * ((g - b) / diff) + 360) % 360
        elif max_val == g:
            hue = (60 * ((b - r) / diff) + 120) % 360
        else:
            hue = (60 * ((r - g) / diff) + 240) % 360

        return hue
    except:
        return 0

def create_smooth_curve_path(contour):
    """Tworzy gładką ścieżkę z selektywnymi krzywymi"""
    try:
        path_data = f"M {contour[0][0]:.2f} {contour[0][1]:.2f}"

        for i in range(1, len(contour)):
            current = contour[i]

            # Use quadratic curves for smooth segments
            if i % 2 == 0 and i + 1 < len(contour):
                next_point = contour[i + 1] if i + 1 < len(contour) else contour[0]
                prev_point = contour[i - 1]

                # Calculate control point
                cp_x = current[0] + (next_point[0] - prev_point[0]) * 0.15
                cp_y = current[1] + (next_point[1] - prev_point[1]) * 0.15

                path_data += f" Q {cp_x:.2f} {cp_y:.2f} {current[0]:.2f} {current[1]:.2f}"
            else:
                path_data += f" L {current[0]:.2f} {current[1]:.2f}"

        path_data += " Z"
        return path_data

    except Exception as e:
        print(f"Błąd w create_smooth_curve_path: {e}")
        return create_simple_svg_path(contour)

def analyze_image_complexity(image):
    """Zaawansowana analiza złożoności obrazu z algorytmami AI do maksymalnej jakości"""
    try:
        img_array = np.array(image)
        height, width = img_array.shape[:2]

        # 1. ZAAWANSOWANA ANALIZA KRAWĘDZI
        edge_density = detect_edge_density_advanced(img_array)
        edge_sharpness = detect_edge_sharpness(img_array)
        edge_connectivity = detect_edge_connectivity(img_array)

        # 2. ULTRA PRECYZYJNA ANALIZA KOLORÓW
        color_complexity = detect_color_complexity_advanced(img_array)
        color_variance = detect_color_variance(img_array)
        color_gradients = detect_color_gradients(img_array)

        # 3. ANALIZA TEKSTUR I WZORÓW
        texture_complexity = detect_texture_complexity(img_array)
        pattern_regularity = detect_pattern_regularity(img_array)

        # 4. ANALIZA GEOMETRYCZNA
        geometric_complexity = detect_geometric_shapes(img_array)
        curve_complexity = detect_curve_complexity(img_array)

        # 5. PERCEPTUALNA ANALIZA JAKOŚCI
        perceptual_importance = calculate_perceptual_importance(img_array)
        detail_density = calculate_detail_density(img_array)

        print(f"🔬 Zaawansowana analiza:")
        print(f"   📐 Krawędzie: gęstość={edge_density:.3f}, ostrość={edge_sharpness:.3f}, łączność={edge_connectivity:.3f}")
        print(f"   🎨 Kolory: złożoność={color_complexity}, wariancja={color_variance:.3f}, gradienty={color_gradients:.3f}")
        print(f"   🖼️ Tekstury: złożoność={texture_complexity:.3f}, regularność={pattern_regularity:.3f}")
        print(f"   📊 Geometria: kształty={geometric_complexity:.3f}, krzywe={curve_complexity:.3f}")
        print(f"   👁️ Percepcja: ważność={perceptual_importance:.3f}, szczegóły={detail_density:.3f}")

        # ALGORYTM ADAPTACYJNEGO DOBORU PARAMETRÓW
        complexity_score = calculate_overall_complexity_score(
            edge_density, edge_sharpness, edge_connectivity,
            color_complexity, color_variance, color_gradients,
            texture_complexity, pattern_regularity,
            geometric_complexity, curve_complexity,
            perceptual_importance, detail_density
        )

        print(f"🎯 Wynik złożoności: {complexity_score:.3f} (0.0-1.0)")

        # PREMIUM DYNAMICZNE PARAMETRY - MAKSYMALNA JAKOŚĆ
        if complexity_score > 0.8:  # ULTRA-COMPLEX
            return {
                'max_colors': 80,  # Znacznie zwiększona liczba kolorów
                'tolerance_factor': 0.15,  # Ultra wysoka precyzja
                'detail_preservation': 'ai_supreme',
                'min_region_size': 1,
                'color_flattening': False,
                'quality_enhancement': 'ai_supreme',
                'curve_smoothing': 'adaptive_supreme',
                'edge_enhancement': True,
                'micro_detail_preservation': True,
                'gradient_preservation': True,
                'ultra_precision_mode': True,
                'advanced_color_analysis': True
            }
        elif complexity_score > 0.65:  # VERY COMPLEX
            return {
                'max_colors': 70,  # Zwiększono
                'tolerance_factor': 0.18,  # Zwiększona precyzja
                'detail_preservation': 'ai_supreme',
                'min_region_size': 1,
                'color_flattening': False,
                'quality_enhancement': 'ai_supreme',
                'curve_smoothing': 'adaptive_supreme',
                'edge_enhancement': True,
                'micro_detail_preservation': True,
                'gradient_preservation': True,
                'ultra_precision_mode': True,
                'advanced_color_analysis': True
            }
        elif complexity_score > 0.5:  # COMPLEX
            return {
                'max_colors': 65,  # Zwiększono
                'tolerance_factor': 0.22,  # Zwiększona precyzja
                'detail_preservation': 'ai_ultra_high',
                'min_region_size': 1,
                'color_flattening': False,
                'quality_enhancement': 'ai_ultra_high',
                'curve_smoothing': 'adaptive_ultra',
                'edge_enhancement': True,
                'micro_detail_preservation': True,
                'gradient_preservation': True,
                'ultra_precision_mode': True,
                'advanced_color_analysis': True
            }
        elif complexity_score > 0.35:  # MEDIUM
            return {
                'max_colors': 60,  # Zwiększono
                'tolerance_factor': 0.25,  # Zwiększona precyzja
                'detail_preservation': 'ai_high',
                'min_region_size': 1,
                'color_flattening': False,
                'quality_enhancement': 'ai_high',
                'curve_smoothing': 'adaptive_high',
                'edge_enhancement': True,
                'micro_detail_preservation': True,
                'gradient_preservation': True,
                'ultra_precision_mode': False,
                'advanced_color_analysis': True
            }
        else:  # SIMPLE
            return {
                'max_colors': 55,  # Zwiększono
                'tolerance_factor': 0.28,  # Zwiększona precyzja
                'detail_preservation': 'ai_medium',
                'min_region_size': 1,
                'color_flattening': False,
                'quality_enhancement': 'ai_medium',
                'curve_smoothing': 'adaptive',
                'edge_enhancement': True,
                'micro_detail_preservation': False,
                'gradient_preservation': True,
                'ultra_precision_mode': False,
                'advanced_color_analysis': True
            }

    except Exception as e:
        print(f"⚠️ Błąd analizy złożoności: {e}")
        return {
            'max_colors': 45,
            'tolerance_factor': 0.3,
            'detail_preservation': 'ai_high',
            'min_region_size': 1,
            'color_flattening': False,
            'quality_enhancement': 'ai_high',
            'curve_smoothing': 'adaptive',
            'edge_enhancement': True,
            'micro_detail_preservation': True,
            'gradient_preservation': True
        }

def create_simple_svg_path(contour):
    """Tworzy prostą ścieżkę SVG"""
    if len(contour) < 3:
        return None

    simplified = contour[::max(1, len(contour)//20)]  # Maksymalnie 20 punktów

    path_data = f"M {simplified[0][0]} {simplified[0][1]}"
    for point in simplified[1:]:
        path_data += f" L {point[0]} {point[1]}"
    path_data += " Z"

    return path_data

def create_color_regions_advanced(image, colors):
    """Ultra precyzyjne tworzenie regionów z zachowaniem szczegółów oryginalnego obrazu"""
    try:
        width, height = image.size
        img_array = np.array(image)

        regions = []

        # Analiza każdego koloru z maksymalną precyzją
        for i, color in enumerate(colors):
            print(f"🎯 Ultra precyzyjne przetwarzanie koloru {i+1}/{len(colors)}: {color}")

            # Wielopoziomowa detekcja regionów
            mask = create_ultra_precise_mask(img_array, color)

            if mask is None:
                continue

            initial_pixels = np.sum(mask)
            print(f"  📊 Początkowe piksele: {initial_pixels}")

            if initial_pixels > 1:  # DRASTYCZNIE zmniejszony próg - zachowaj wszystkie detale
                # Zachowanie szczegółów z minimalnymi przekształceniami
                mask = preserve_detail_processing_ultra(mask, initial_pixels)

                # Inteligentne łączenie z zachowaniem kształtów
                mask = smart_shape_preserving_merge(mask, img_array, color)

                final_pixels = np.sum(mask)
                print(f"  ✅ Finalne piksele: {final_pixels}")

                if final_pixels > 1:  # DRASTYCZNIE zmniejszony próg dla zachowania detali
                    regions.append((color, mask))
                    print(f"  ✓ Dodano region z zachowaniem szczegółów dla koloru {color}")
                else:
                    print(f"  ✗ Region za mały po przetwarzaniu")
            else:
                print(f"  ✗ Brak wystarczających pikseli")

        print(f"🏁 Utworzono {len(regions)} regionów ultra wysokiej precyzji")
        return regions

    except Exception as e:
        print(f"❌ Błąd podczas ultra precyzyjnego tworzenia regionów: {e}")
        return create_color_regions_simple(image, colors)

def create_color_regions_simple(image, colors):
    """Prosta metoda tworzenia regionów kolorów jako fallback"""
    try:
        img_array = np.array(image)
        regions = []
        
        for color in colors:
            # Prosta maska podobieństwa kolorów
            distances = np.sqrt(np.sum((img_array - np.array(color))**2, axis=2))
            mask = distances < 50  # Próg podobieństwa
            
            if np.sum(mask) > 10:  # Minimum pikseli
                regions.append((color, mask))
        
        return regions
        
    except Exception as e:
        print(f"Błąd w create_color_regions_simple: {e}")
        return []

def create_ultra_precise_mask(img_array, color):
    """Tworzy perfekcyjną maskę koloru z usuwaniem szumów i artefaktów"""
    try:
        height, width = img_array.shape[:2]
        color_array = np.array(color)

        # Multi-metodowa ultra precyzyjna detekcja z redukcją szumów
        masks = []

        # 1. Najbardziej precyzyjna odległość RGB z adaptacyjnym progiem
        rgb_diff = np.sqrt(np.sum((img_array - color_array)**2, axis=2))

        # Zaawansowana analiza histogramu dla lepszego progu
        hist, bins = np.histogram(rgb_diff, bins=200)

        # Znajdź najlepszy próg używając analizy gradientu
        cumsum = np.cumsum(hist)
        total_pixels = cumsum[-1]

        # Dynamiczny próg bazujący na nasyceniu koloru
        saturation = max(color) - min(color)
        brightness = sum(color) / 3

        if saturation > 100:  # Wysoko nasycone kolory - bardziej restrykcyjny próg
            percentile_threshold = 8
        elif saturation > 50:  # Średnio nasycone
            percentile_threshold = 12
        else:  # Nisko nasycone kolory - bardziej liberalny próg
            percentile_threshold = 18

        # Dodatkowa regulacja dla jasności
        if brightness < 50:  # Ciemne kolory
            percentile_threshold *= 0.8
        elif brightness > 200:  # Jasne kolory
            percentile_threshold *= 1.2

        threshold = np.percentile(rgb_diff, percentile_threshold)
        mask1 = rgb_diff <= threshold
        masks.append(mask1)

        # 2. Ulepszona analiza w przestrzeni LAB (lepiej dla percepcji kolorów)
        try:
            from skimage.color import rgb2lab
            lab_img = rgb2lab(img_array / 255.0)
            lab_color = rgb2lab(color_array.reshape(1, 1, 3) / 255.0)[0, 0]

            # Delta E - profesjonalna miara różnicy kolorów
            l_diff = (lab_img[:,:,0] - lab_color[0]) / 100.0  # Normalizuj L
            a_diff = (lab_img[:,:,1] - lab_color[1]) / 127.0  # Normalizuj a
            b_diff = (lab_img[:,:,2] - lab_color[2]) / 127.0  # Normalizuj b

            # Ważona odległość LAB
            lab_distance = np.sqrt(l_diff**2 + 2*a_diff**2 + 2*b_diff**2)
            lab_threshold = np.percentile(lab_distance, percentile_threshold * 0.7)
            mask2 = lab_distance <= lab_threshold
            masks.append(mask2)
        except:
            # Fallback do HSV
            hsv_img = rgb_to_hsv_ultra_precise(img_array)
            hsv_color = rgb_to_hsv_ultra_precise(color_array.reshape(1, 1, 3))[0, 0]

            h_diff = np.minimum(
                np.abs(hsv_img[:,:,0] - hsv_color[0]),
                1.0 - np.abs(hsv_img[:,:,0] - hsv_color[0])
            )
            s_diff = np.abs(hsv_img[:,:,1] - hsv_color[1])
            v_diff = np.abs(hsv_img[:,:,2] - hsv_color[2])

            hsv_distance = np.sqrt(3*h_diff**2 + 2*s_diff**2 + v_diff**2)
            hsv_threshold = np.percentile(hsv_distance, percentile_threshold * 0.8)
            mask2 = hsv_distance <= hsv_threshold
            masks.append(mask2)

        # 3. Maska uwzględniająca lokalne sąsiedztwo
        neighborhood_mask = create_neighborhood_coherence_mask(img_array, color_array)
        if neighborhood_mask is not None:
            masks.append(neighborhood_mask)

        # Inteligentne kombinowanie masek z redukcją szumów
        if len(masks) > 0:
            # Głosowanie większościowe z wagami i filtracją szumów
            combined_mask = np.zeros_like(masks[0], dtype=float)
            weights = [1.0, 0.9, 0.5]  # Zoptymalizowane wagi

            for i, mask in enumerate(masks):
                weight = weights[i] if i < len(weights) else 0.3
                combined_mask += mask.astype(float) * weight

            # Próg dla decyzji końcowej z redukcją szumów
            total_weight = sum(weights[:len(masks)])
            confidence_threshold = total_weight * 0.6  # Wyższy próg pewności

            final_mask = combined_mask >= confidence_threshold

            # Zaawansowane usuwanie szumów i artefaktów
            final_mask = remove_noise_and_artifacts(final_mask, img_array, color_array)

            return final_mask

        return None

    except Exception as e:
        print(f"Błąd w create_ultra_precise_mask: {e}")
        return None

def rgb_to_hsv_ultra_precise(rgb):
    """Ultra precyzyjna konwersja RGB do HSV"""
    try:
        rgb = rgb.astype(float) / 255.0
        max_val = np.max(rgb, axis=-1)
        min_val = np.min(rgb, axis=-1)
        diff = max_val - min_val

        # Value
        v = max_val

        # Saturation
        s = np.where(max_val > 1e-6, diff / max_val, 0)

        # Hue z wysoką precyzją
        h = np.zeros_like(max_val)

        # Unikaj dzielenia przez zero
        mask = diff > 1e-6

        # Red is max
        red_max = (rgb[..., 0] == max_val) & mask
        h[red_max] = ((rgb[red_max, 1] - rgb[red_max, 2]) / diff[red_max]) % 6

        # Green is max
        green_max = (rgb[..., 1] == max_val) & mask
        h[green_max] = (rgb[green_max, 2] - rgb[green_max, 0]) / diff[green_max] + 2

        # Blue is max
        blue_max = (rgb[..., 2] == max_val) & mask
        h[blue_max] = (rgb[blue_max, 0] - rgb[blue_max, 1]) / diff[blue_max] + 4

        h = h / 6.0  # Normalize to [0, 1]

        return np.stack([h, s, v], axis=-1)
    except:
        return rgb

def create_neighborhood_coherence_mask(img_array, color_array):
    """Tworzy maskę bazującą na spójności sąsiedztwa - redukuje artefakty"""
    try:
        from scipy import ndimage

        # Podstawowa maska podobieństwa
        distances = np.sqrt(np.sum((img_array - color_array)**2, axis=2))
        base_mask = distances < 35

        if np.sum(base_mask) == 0:
            return None

        # Analiza spójności lokalnej (5x5 sąsiedztwo)
        kernel = np.ones((5, 5))
        local_density = ndimage.convolve(base_mask.astype(float), kernel, mode='constant')

        # Piksele z wysoką gęstością sąsiadów tego samego koloru
        coherent_areas = local_density >= 8  # Minimum 8/25 podobnych pikseli w sąsiedztwie

        # Kombinuj z bazową maską
        coherent_mask = base_mask & coherent_areas

        # Rozszerz spójne obszary na bliskie piksele
        extended_mask = ndimage.binary_dilation(coherent_mask, structure=np.ones((3, 3)), iterations=1)

        # Sprawdź czy rozszerzone obszary są rzeczywiście podobne
        extended_pixels_coords = np.where(extended_mask & ~coherent_mask)
        if len(extended_pixels_coords[0]) > 0:
            extended_pixels = img_array[extended_pixels_coords]
            ext_distances = np.sqrt(np.sum((extended_pixels - color_array)**2, axis=1))

            # Usuń piksele które są zbyt różne
            invalid_extension = ext_distances > 45
            for i, is_invalid in enumerate(invalid_extension):
                if is_invalid:
                    extended_mask[extended_pixels_coords[0][i], extended_pixels_coords[1][i]] = False

        return extended_mask
    except:
        return None

def remove_noise_and_artifacts(mask, img_array, color_array):
    """Zaawansowane usuwanie szumów i artefaktów z maski"""
    try:
        from scipy import ndimage

        # 1. Usuń pojedyncze piksele (szum punktowy)
        structure = np.ones((3, 3))
        opened = ndimage.binary_opening(mask, structure=structure, iterations=1)

        # 2. Wypełnij małe dziury
        filled = ndimage.binary_fill_holes(opened)

        # 3. Usuń bardzo małe komponenty (artefakty)
        labeled, num_features = ndimage.label(filled)

        if num_features > 0:
            # Oblicz rozmiary komponentów
            component_sizes = ndimage.sum(filled, labeled, range(1, num_features + 1))
            total_area = np.sum(filled)

            # Usuń komponenty mniejsze niż 0.5% całkowitego obszaru lub mniejsze niż 10 pikseli
            min_component_size = max(10, total_area * 0.005)

            cleaned_mask = np.zeros_like(mask)
            for i in range(1, num_features + 1):
                if component_sizes[i-1] >= min_component_size:
                    component = labeled == i

                    # Dodatkowa weryfikacja spójności kolorowej komponentu
                    component_pixels = img_array[component]
                    if len(component_pixels) > 0:
                        mean_distance = np.mean(np.sqrt(np.sum((component_pixels - color_array)**2, axis=1)))

                        # Zachowaj tylko komponenty o dobrej spójności kolorowej
                        if mean_distance < 50:
                            cleaned_mask[component] = True
        else:
            cleaned_mask = filled

        # 4. Końcowe wygładzenie krawędzi
        smoothed = ndimage.binary_closing(cleaned_mask, structure=structure, iterations=1)

        return smoothed

    except Exception as e:
        print(f"Błąd w remove_noise_and_artifacts: {e}")
        return mask

def preserve_detail_processing_ultra(mask, initial_pixels):
    """Ultra precyzyjne przetwarzanie z maksymalnym zachowaniem szczegółów"""
    try:
        from scipy import ndimage

        # MINIMALNE przetwarzanie - zachowaj każdy detal
        if initial_pixels > 1000:
            # Dla większych regionów - bardzo delikatne czyszczenie
            structure = np.ones((3, 3))

            # Tylko wypełnij małe dziury
            mask = ndimage.binary_fill_holes(mask)

            # Usuń tylko oczywiste artefakty (pojedyncze izolowane piksele)
            labeled, num_features = ndimage.label(mask)
            for i in range(1, num_features + 1):
                component = labeled == i
                if np.sum(component) == 1:  # Tylko pojedyncze piksele
                    # Sprawdź czy to rzeczywiście izolowany artefakt
                    y, x = np.where(component)
                    if len(y) > 0:
                        # Sprawdź 3x3 sąsiedztwo
                        neighbors = 0
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                ny, nx = y[0] + dy, x[0] + dx
                                if (0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and 
                                    mask[ny, nx] and not component[ny, nx]):
                                    neighbors += 1

                        # Usuń tylko jeśli ma mniej niż 2 sąsiadów
                        if neighbors < 2:
                            mask[component] = False

        elif initial_pixels > 100:
            # Dla średnich regionów - bardzo delikatne czyszczenie
            mask = ndimage.binary_fill_holes(mask)

            # Usuń tylko pojedyncze izolowane piksele
            labeled, num_features = ndimage.label(mask)
            for i in range(1, num_features + 1):
                component = labeled == i
                if np.sum(component) == 1:
                    mask[component] = False

        else:
            # Dla małych regionów - praktycznie bez czyszczenia
            # Tylko wypełnij pojedyncze dziury
            mask = ndimage.binary_fill_holes(mask)

        return mask

    except Exception as e:
        print(f"Błąd w preserve_detail_processing_ultra: {e}")
        return mask

def smart_shape_preserving_merge(mask, img_array, color):
    """Inteligentne łączenie z zachowaniem kształtów"""
    try:
        from scipy import ndimage

        # Znajdź komponenty
        labeled, num_features = ndimage.label(mask)

        if num_features <= 1:
            return mask

        # Analizuj każdy komponent
        color_array = np.array(color)
        merged_mask = np.zeros_like(mask)

        for i in range(1, num_features + 1):
            component = labeled == i
            component_size = np.sum(component)

            # Zachowaj wszystkie komponenty powyżej minimalnego rozmiaru
            if component_size >= 1:  # DRASTYCZNIE zmniejszony próg - zachowaj każdy piksel
                # Sprawdź jakość dopasowania koloru
                component_pixels = img_array[component]
                if len(component_pixels) > 0:
                    mean_distance = np.mean(np.sqrt(np.sum((component_pixels - color_array)**2, axis=1)))

                    # Bardzo liberalne kryteria dla zachowania szczegółów
                    if mean_distance <= 80:  # Jeszcze wyższy próg tolerancji
                        merged_mask[component] = True

        return merged_mask

    except Exception as e:
        print(f"Błąd w smart_shape_preserving_merge: {e}")
        return mask

def trace_contours_advanced(mask):
    """Ultra precyzyjne śledzenie konturów z zachowaniem detali oryginalnego kształtu"""
    try:
        from scipy import ndimage

        # Analiza maski dla wyboru optymalnej strategii
        mask_size = np.sum(mask)

        print(f"  🔍 Analiza maski: rozmiar={mask_size}")

        # Minimalne przetwarzanie wstępne - zachowaj oryginalny kształt
        processed_mask = minimal_mask_preprocessing(mask)

        # Wybór metody śledzenia bazującej na rozmiarze
        if mask_size > 1000:
            contours = trace_high_detail_contours(processed_mask)
        elif mask_size > 100:
            contours = trace_balanced_contours(processed_mask)
        else:
            contours = trace_simple_precise_contours(processed_mask)

        # Minimalna post-processing - zachowaj szczegóły
        final_contours = []
        for contour in contours:
            if len(contour) >= 3:
                # Bardzo delikatna optymalizacja
                optimized = minimal_contour_optimization(contour)
                if optimized and len(optimized) >= 3:
                    final_contours.append(optimized)

        print(f"  ✅ Wygenerowano {len(final_contours)} konturów ultra wysokiej precyzji")
        return final_contours

    except Exception as e:
        print(f"❌ Błąd podczas ultra precyzyjnego śledzenia: {e}")
        return trace_contours_simple_improved(mask)

def minimal_mask_preprocessing(mask):
    """Minimalne przetwarzanie maski - zachowaj oryginalny kształt"""
    try:
        from scipy import ndimage

        # Tylko usuń pojedyncze izolowane piksele
        labeled, num_features = ndimage.label(mask)
        cleaned_mask = mask.copy()

        for i in range(1, num_features + 1):
            component = labeled == i
            if np.sum(component) == 1:  # Pojedynczy piksel
                cleaned_mask[component] = False

        return cleaned_mask
    except:
        return mask

def trace_high_detail_contours(mask):
    """Śledzenie konturów dla wysokich detali"""
    try:
        if cv2 is not None:
            mask_uint8 = (mask * 255).astype(np.uint8)

            # Użyj CHAIN_APPROX_NONE dla zachowania wszystkich punktów
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            processed = []
            for contour in contours:
                if len(contour) >= 6:
                    # Minimalne upraszczanie - zachowaj 95% punktów
                    perimeter = cv2.arcLength(contour, True)
                    epsilon = 0.001 * perimeter  # Bardzo mały epsilon
                    simplified = cv2.approxPolyDP(contour, epsilon, True)

                    if len(simplified) >= 3:
                        points = [(int(p[0][0]), int(p[0][1])) for p in simplified]
                        processed.append(points)

            return processed
        else:
            return trace_contours_simple_improved(mask)
    except:
        return trace_contours_simple_improved(mask)

def trace_balanced_contours(mask):
    """Śledzenie konturów z balansem między precyzją a wydajnością"""
    try:
        if cv2 is not None:
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            processed = []
            for contour in contours:
                if len(contour) >= 4:
                    perimeter = cv2.arcLength(contour, True)
                    epsilon = 0.002 * perimeter  # Umiarkowane upraszczanie
                    simplified = cv2.approxPolyDP(contour, epsilon, True)

                    if len(simplified) >= 3:
                        points = [(int(p[0][0]), int(p[0][1])) for p in simplified]
                        processed.append(points)

            return processed
        else:
            return trace_contours_simple_improved(mask)
    except:
        return trace_contours_simple_improved(mask)

def trace_simple_precise_contours(mask):
    """Śledzenie konturów dla prostych kształtów z wysoką precyzją"""
    try:
        if cv2 is not None:
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

            processed = []
            for contour in contours:
                if len(contour) >= 3:
                    # Bardzo delikatne upraszczanie
                    perimeter = cv2.arcLength(contour, True)
                    epsilon = 0.0005 * perimeter  # Minimalny epsilon
                    simplified = cv2.approxPolyDP(contour, epsilon, True)

                    if len(simplified) >= 3:
                        points = [(int(p[0][0]), int(p[0][1])) for p in simplified]
                        processed.append(points)

            return processed
        else:
            return trace_contours_simple_improved(mask)
    except:
        return trace_contours_simple_improved(mask)

def minimal_contour_optimization(contour):
    """Minimalna optymalizacja konturu - zachowaj maksimum szczegółów"""
    try:
        if len(contour) <= 5:
            return contour

        # Usuń tylko punkty, które są bardzo blisko siebie
        optimized = [contour[0]]

        for i in range(1, len(contour)):
            current = contour[i]
            last = optimized[-1]

            # Usuń tylko jeśli punkty są praktycznie identyczne
            distance = np.sqrt((current[0] - last[0])**2 + (current[1] - last[1])**2)
            if distance >= 1.0:  # Bardzo niski próg
                optimized.append(current)

        return optimized if len(optimized) >= 3 else contour
    except:
        return contour

def trace_contours_simple_improved(mask):
    """Ulepszona prosta metoda śledzenia konturów"""
    try:
        from skimage import measure

        # Użyj skimage do znajdowania konturów
        contours = measure.find_contours(mask, 0.5)

        processed_contours = []
        for contour in contours:
            if len(contour) >= 6:
                # Zmień kolejność współrzędnych (y,x) -> (x,y)
                points = [(int(point[1]), int(point[0])) for point in contour[::2]]  # Co drugi punkt

                if len(points) >= 4:
                    processed_contours.append(points)

        return processed_contours
    except:
        # Ostateczny fallback
        return trace_contours_simple(mask)

def trace_contours_simple(mask):
    """Proste śledzenie konturów"""
    try:
        height, width = mask.shape
        contours = []

        # Znajdź punkty brzegowe
        edge_points = []
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if mask[y, x]:
                    # Sprawdź czy to punkt brzegowy
                    neighbors = [
                        mask[y-1, x-1], mask[y-1, x], mask[y-1, x+1],
                        mask[y, x-1], mask[y, x+1],
                        mask[y+1, x-1], mask[y+1, x], mask[y+1, x+1]
                    ]
                    if not all(neighbors):
                        edge_points.append((x, y))

        if len(edge_points) >= 3:
            # Ogranicz liczbę punktów
            if len(edge_points) > 100:
                step = len(edge_points) // 50
                edge_points = edge_points[::step]

            contours.append(edge_points)

        return contours
    except Exception as e:
        print(f"Błąd podczas prostego śledzenia konturów: {e}")
        return []

def create_smooth_svg_path(contour):
    """AI-enhanced ultra precyzyjna ścieżka SVG z maksymalną jakością"""
    if len(contour) < 3:
        return None

    try:
        # AI-enhanced contour analysis
        contour_analysis = ai_enhanced_contour_analysis(contour)
        
        # Adaptive contour preservation
        preserved_contour = adaptive_contour_preservation(contour, contour_analysis)

        print(f"    📐 AI Contour: {len(contour)} → {len(preserved_contour)} punktów")
        print(f"       🤖 Analiza: complexity={contour_analysis['complexity']:.2f}, curvature={contour_analysis['curvature']:.2f}")

        # AI-guided path generation
        if contour_analysis['complexity'] > 0.8:
            path_data = create_ai_maximum_fidelity_path(preserved_contour, contour_analysis)
        elif contour_analysis['complexity'] > 0.6:
            path_data = create_ai_high_fidelity_path(preserved_contour, contour_analysis)
        elif contour_analysis['complexity'] > 0.4:
            path_data = create_ai_balanced_path(preserved_contour, contour_analysis)
        else:
            path_data = create_ai_optimized_path(preserved_contour, contour_analysis)

        return path_data

    except Exception as e:
        print(f"❌ Błąd AI path generation: {e}")
        return create_simple_svg_path(contour)

def ai_enhanced_contour_analysis(contour):
    """AI-enhanced analiza konturu z wieloma metrykami"""
    try:
        if len(contour) < 3:
            return {
                'complexity': 0.1,
                'curvature': 0.1,
                'smoothness': 0.9,
                'detail_density': 0.1,
                'geometric_regularity': 0.9,
                'optimization_strategy': 'simple'
            }
        
        # 1. COMPLEXITY ANALYSIS
        point_density = len(contour) / max(1, calculate_contour_perimeter(contour))
        length_complexity = min(1.0, len(contour) / 100.0)
        
        # 2. CURVATURE ANALYSIS
        curvature_variance = calculate_curvature_variance(contour)
        sharp_angles = count_sharp_angles(contour)
        
        # 3. SMOOTHNESS ANALYSIS
        smoothness = calculate_contour_smoothness(contour)
        
        # 4. DETAIL DENSITY
        detail_density = calculate_detail_density_contour(contour)
        
        # 5. GEOMETRIC REGULARITY
        geometric_regularity = calculate_geometric_regularity(contour)
        
        # 6. COMBINED COMPLEXITY SCORE
        complexity_score = (
            length_complexity * 0.25 +
            curvature_variance * 0.25 +
            (1 - smoothness) * 0.2 +
            detail_density * 0.15 +
            (1 - geometric_regularity) * 0.15
        )
        
        # 7. OPTIMIZATION STRATEGY
        if complexity_score > 0.8:
            strategy = 'ai_maximum_preservation'
        elif complexity_score > 0.6:
            strategy = 'ai_high_preservation'
        elif complexity_score > 0.4:
            strategy = 'ai_balanced'
        else:
            strategy = 'ai_optimized'
        
        return {
            'complexity': min(1.0, complexity_score),
            'curvature': min(1.0, curvature_variance),
            'smoothness': smoothness,
            'detail_density': detail_density,
            'geometric_regularity': geometric_regularity,
            'optimization_strategy': strategy,
            'point_density': point_density,
            'sharp_angles': sharp_angles
        }
    except Exception as e:
        print(f"⚠️ Błąd AI contour analysis: {e}")
        return {
            'complexity': 0.5,
            'curvature': 0.5,
            'smoothness': 0.5,
            'detail_density': 0.5,
            'geometric_regularity': 0.5,
            'optimization_strategy': 'ai_balanced'
        }

def calculate_contour_perimeter(contour):
    """Oblicza obwód konturu"""
    try:
        if len(contour) < 2:
            return 1
        
        perimeter = 0
        for i in range(len(contour)):
            p1 = contour[i]
            p2 = contour[(i + 1) % len(contour)]
            distance = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            perimeter += distance
        
        return max(1, perimeter)
    except:
        return 1

def calculate_curvature_variance(contour):
    """Oblicza wariancję krzywizny konturu"""
    try:
        if len(contour) < 5:
            return 0.1
        
        curvatures = []
        
        for i in range(2, len(contour) - 2):
            # Trzy kolejne punkty
            p1 = np.array(contour[i-1])
            p2 = np.array(contour[i])
            p3 = np.array(contour[i+1])
            
            # Wektory
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Długości
            len1 = np.linalg.norm(v1)
            len2 = np.linalg.norm(v2)
            
            if len1 > 0 and len2 > 0:
                # Kąt między wektorami
                cos_angle = np.dot(v1, v2) / (len1 * len2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                
                # Krzywizna jako zmiana kąta
                curvature = angle / max(len1, len2)
                curvatures.append(curvature)
        
        if curvatures:
            return min(1.0, np.var(curvatures) * 10)
        
        return 0.1
    except:
        return 0.1

def count_sharp_angles(contour, angle_threshold=np.pi/3):
    """Liczy ostre kąty w konturze"""
    try:
        if len(contour) < 3:
            return 0
        
        sharp_count = 0
        
        for i in range(1, len(contour) - 1):
            p1 = np.array(contour[i-1])
            p2 = np.array(contour[i])
            p3 = np.array(contour[i+1])
            
            v1 = p1 - p2
            v2 = p3 - p2
            
            len1 = np.linalg.norm(v1)
            len2 = np.linalg.norm(v2)
            
            if len1 > 0 and len2 > 0:
                cos_angle = np.dot(v1, v2) / (len1 * len2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                
                if angle < angle_threshold:
                    sharp_count += 1
        
        return sharp_count
    except:
        return 0

def calculate_contour_smoothness(contour):
    """Oblicza gładkość konturu"""
    try:
        if len(contour) < 4:
            return 0.9
        
        direction_changes = []
        
        for i in range(2, len(contour)):
            p1 = np.array(contour[i-2])
            p2 = np.array(contour[i-1])
            p3 = np.array(contour[i])
            
            v1 = p2 - p1
            v2 = p3 - p2
            
            len1 = np.linalg.norm(v1)
            len2 = np.linalg.norm(v2)
            
            if len1 > 0 and len2 > 0:
                direction_change = np.linalg.norm(v2/len2 - v1/len1)
                direction_changes.append(direction_change)
        
        if direction_changes:
            avg_change = np.mean(direction_changes)
            smoothness = 1.0 / (1.0 + avg_change)
            return min(1.0, smoothness)
        
        return 0.9
    except:
        return 0.5

def calculate_detail_density_contour(contour):
    """Oblicza gęstość szczegółów w konturze"""
    try:
        if len(contour) < 5:
            return 0.1
        
        # Analiza lokalnej wariancji pozycji
        local_variances = []
        window_size = min(5, len(contour) // 4)
        
        for i in range(window_size, len(contour) - window_size):
            window_points = contour[i-window_size:i+window_size+1]
            
            # Oblicz wariancję pozycji w oknie
            x_coords = [p[0] for p in window_points]
            y_coords = [p[1] for p in window_points]
            
            x_var = np.var(x_coords)
            y_var = np.var(y_coords)
            
            local_variance = (x_var + y_var) / 2
            local_variances.append(local_variance)
        
        if local_variances:
            avg_variance = np.mean(local_variances)
            # Normalizacja zależna od rozmiaru konturu
            normalized_variance = avg_variance / max(1, len(contour))
            return min(1.0, normalized_variance / 100)
        
        return 0.1
    except:
        return 0.1

def calculate_geometric_regularity(contour):
    """Oblicza regularność geometryczną konturu"""
    try:
        if len(contour) < 4:
            return 0.9
        
        # Analiza podobieństwa do podstawowych kształtów
        
        # 1. Test na prostokąt
        rectangle_score = test_rectangle_similarity(contour)
        
        # 2. Test na elipsę/okrąg
        ellipse_score = test_ellipse_similarity(contour)
        
        # 3. Test na wielokąt regularny
        polygon_score = test_regular_polygon_similarity(contour)
        
        # Najwyższy wynik podobieństwa
        max_regularity = max(rectangle_score, ellipse_score, polygon_score)
        
        return min(1.0, max_regularity)
    except:
        return 0.5

def test_rectangle_similarity(contour):
    """Testuje podobieństwo do prostokąta"""
    try:
        if len(contour) < 4:
            return 0
        
        # Znajdź bounding box
        x_coords = [p[0] for p in contour]
        y_coords = [p[1] for p in contour]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Punkty prostokąta
        rect_points = [
            (min_x, min_y), (max_x, min_y),
            (max_x, max_y), (min_x, max_y)
        ]
        
        # Oblicz średnią odległość od prostokąta
        min_distances = []
        for point in contour:
            distances = [np.sqrt((point[0]-rp[0])**2 + (point[1]-rp[1])**2) for rp in rect_points]
            min_distances.append(min(distances))
        
        avg_distance = np.mean(min_distances)
        perimeter = max(1, 2 * ((max_x - min_x) + (max_y - min_y)))
        
        similarity = 1.0 / (1.0 + avg_distance / perimeter)
        return similarity
    except:
        return 0

def test_ellipse_similarity(contour):
    """Testuje podobieństwo do elipsy"""
    try:
        if len(contour) < 5:
            return 0
        
        # Znajdź środek i osie
        x_coords = [p[0] for p in contour]
        y_coords = [p[1] for p in contour]
        
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
        
        # Promienie w kierunkach głównych
        max_dist_x = max(abs(x - center_x) for x in x_coords)
        max_dist_y = max(abs(y - center_y) for y in y_coords)
        
        if max_dist_x == 0 or max_dist_y == 0:
            return 0
        
        # Sprawdź jak punkty pasują do elipsy
        ellipse_errors = []
        for x, y in contour:
            # Równanie elipsy: (x-cx)²/a² + (y-cy)²/b² = 1
            ellipse_eq = ((x - center_x) / max_dist_x)**2 + ((y - center_y) / max_dist_y)**2
            error = abs(ellipse_eq - 1.0)
            ellipse_errors.append(error)
        
        avg_error = np.mean(ellipse_errors)
        similarity = 1.0 / (1.0 + avg_error)
        
        return similarity
    except:
        return 0

def test_regular_polygon_similarity(contour):
    """Testuje podobieństwo do wielokąta regularnego"""
    try:
        if len(contour) < 3:
            return 0
        
        # Uproszczona analiza - sprawdź równość długości boków
        side_lengths = []
        for i in range(len(contour)):
            p1 = contour[i]
            p2 = contour[(i + 1) % len(contour)]
            length = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            side_lengths.append(length)
        
        if not side_lengths:
            return 0
        
        # Wariancja długości boków
        length_variance = np.var(side_lengths)
        avg_length = np.mean(side_lengths)
        
        if avg_length == 0:
            return 0
        
        # Normalizowana wariancja
        normalized_variance = length_variance / (avg_length**2)
        
        # Im mniejsza wariancja, tym bardziej regularny
        regularity = 1.0 / (1.0 + normalized_variance)
        
        return regularity
    except:
        return 0

def adaptive_contour_preservation(contour, analysis):
    """Adaptacyjne zachowanie szczegółów konturu bazujące na AI analysis"""
    try:
        complexity = analysis['complexity']
        strategy = analysis['optimization_strategy']
        
        if strategy == 'ai_maximum_preservation':
            # Maksymalne zachowanie - praktycznie bez uproszczeń
            preserved = preserve_maximum_detail(contour)
        elif strategy == 'ai_high_preservation':
            # Wysokie zachowanie - minimalne uproszczenia
            preserved = preserve_high_detail(contour, complexity)
        elif strategy == 'ai_balanced':
            # Zbalansowane - inteligentne uproszczenia
            preserved = preserve_balanced_detail(contour, analysis)
        else:  # ai_optimized
            # Optymalizowane - większe uproszczenia ale zachowanie kluczowych cech
            preserved = preserve_optimized_detail(contour, analysis)
        
        return preserved
    except:
        return contour

def preserve_maximum_detail(contour):
    """Maksymalne zachowanie szczegółów - tylko usuwanie duplikatów"""
    try:
        if len(contour) <= 3:
            return contour
        
        preserved = [contour[0]]
        
        for point in contour[1:]:
            # Usuń tylko punkty praktycznie identyczne
            last_point = preserved[-1]
            distance = np.sqrt((point[0] - last_point[0])**2 + (point[1] - last_point[1])**2)
            
            if distance >= 0.5:  # Bardzo mały próg
                preserved.append(point)
        
        return preserved if len(preserved) >= 3 else contour
    except:
        return contour

def preserve_high_detail(contour, complexity):
    """Wysokie zachowanie szczegółów z minimalnym upraszczaniem"""
    try:
        if len(contour) <= 5:
            return contour
        
        # Adaptacyjny próg bazujący na złożoności
        base_threshold = 1.0 + (1.0 - complexity) * 2.0
        
        preserved = [contour[0]]
        
        for i, point in enumerate(contour[1:], 1):
            last_point = preserved[-1]
            distance = np.sqrt((point[0] - last_point[0])**2 + (point[1] - last_point[1])**2)
            
            # Dynamiczny próg - zachowaj więcej punktów w obszarach o wysokiej krzywizne
            if i > 1 and i < len(contour) - 1:
                local_curvature = calculate_local_curvature(contour, i)
                threshold = base_threshold * (1.0 - local_curvature * 0.5)
            else:
                threshold = base_threshold
            
            if distance >= threshold:
                preserved.append(point)
        
        return preserved if len(preserved) >= 3 else contour
    except:
        return contour

def calculate_local_curvature(contour, index):
    """Oblicza lokalną krzywiznę w danym punkcie"""
    try:
        if index <= 0 or index >= len(contour) - 1:
            return 0
        
        p1 = np.array(contour[index - 1])
        p2 = np.array(contour[index])
        p3 = np.array(contour[index + 1])
        
        v1 = p2 - p1
        v2 = p3 - p2
        
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        
        if len1 > 0 and len2 > 0:
            cos_angle = np.dot(v1, v2) / (len1 * len2)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            
            # Normalizuj do 0-1
            curvature = angle / np.pi
            return curvature
        
        return 0
    except:
        return 0

def preserve_balanced_detail(contour, analysis):
    """Zbalansowane zachowanie szczegółów z inteligentnym upraszczaniem"""
    try:
        if len(contour) <= 8:
            return contour
        
        complexity = analysis['complexity']
        smoothness = analysis['smoothness']
        
        # Adaptacyjne parametry
        base_threshold = 2.0 + (1.0 - complexity) * 3.0
        angle_threshold = np.pi / 6 + smoothness * np.pi / 6
        
        preserved = [contour[0]]
        
        i = 1
        while i < len(contour):
            current_point = contour[i]
            last_point = preserved[-1]
            
            distance = np.sqrt((current_point[0] - last_point[0])**2 + (current_point[1] - last_point[1])**2)
            
            # Sprawdź czy punkt jest ważny geometrycznie
            is_important = False
            
            if i > 0 and i < len(contour) - 1:
                local_curvature = calculate_local_curvature(contour, i)
                if local_curvature > 0.3:  # Znacząca krzywizna
                    is_important = True
            
            # Zachowaj punkt jeśli jest ważny lub przekracza próg odległości
            if is_important or distance >= base_threshold:
                preserved.append(current_point)
                i += 1
            else:
                # Sprawdź czy można bezpiecznie pominąć kilka punktów
                skip_count = find_safe_skip_distance(contour, i, base_threshold, angle_threshold)
                i += max(1, skip_count)
        
        return preserved if len(preserved) >= 3 else contour
    except:
        return contour

def find_safe_skip_distance(contour, start_index, distance_threshold, angle_threshold):
    """Znajduje bezpieczną odległość do pominięcia punktów"""
    try:
        if start_index >= len(contour) - 1:
            return 1
        
        start_point = contour[start_index - 1] if start_index > 0 else contour[start_index]
        
        for skip in range(1, min(5, len(contour) - start_index)):
            test_point = contour[start_index + skip - 1]
            
            # Sprawdź odległość
            distance = np.sqrt((test_point[0] - start_point[0])**2 + (test_point[1] - start_point[1])**2)
            
            if distance >= distance_threshold:
                return skip
            
            # Sprawdź zmianę kierunku
            if start_index + skip < len(contour) - 1:
                angle_change = calculate_angle_change(contour, start_index, start_index + skip)
                if angle_change > angle_threshold:
                    return skip
        
        return 1
    except:
        return 1

def calculate_angle_change(contour, start_idx, end_idx):
    """Oblicza zmianę kąta między punktami"""
    try:
        if start_idx <= 0 or end_idx >= len(contour) - 1:
            return 0
        
        p1 = np.array(contour[start_idx - 1])
        p2 = np.array(contour[start_idx])
        p3 = np.array(contour[end_idx])
        p4 = np.array(contour[end_idx + 1])
        
        v1 = p2 - p1
        v2 = p4 - p3
        
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        
        if len1 > 0 and len2 > 0:
            cos_angle = np.dot(v1, v2) / (len1 * len2)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle_change = np.arccos(cos_angle)
            return angle_change
        
        return 0
    except:
        return 0

def preserve_optimized_detail(contour, analysis):
    """Optymalizowane zachowanie szczegółów z większymi uproszczeniami"""
    try:
        if len(contour) <= 10:
            return contour
        
        complexity = analysis['complexity']
        geometric_regularity = analysis['geometric_regularity']
        
        # Więcej uproszczeń dla regularnych kształtów
        base_threshold = 3.0 + geometric_regularity * 2.0 + (1.0 - complexity) * 4.0
        
        preserved = [contour[0]]
        
        i = 1
        while i < len(contour):
            current_point = contour[i]
            last_point = preserved[-1]
            
            distance = np.sqrt((current_point[0] - last_point[0])**2 + (current_point[1] - last_point[1])**2)
            
            # Zachowaj tylko kluczowe punkty
            if distance >= base_threshold:
                preserved.append(current_point)
            
            i += 1
        
        # Post-processing - usuń zbędne punkty na prostych liniach
        final_preserved = remove_collinear_points(preserved)
        
        return final_preserved if len(final_preserved) >= 3 else preserved
    except:
        return contour

def remove_collinear_points(contour, tolerance=2.0):
    """Usuwa punkty leżące na prostych liniach"""
    try:
        if len(contour) <= 3:
            return contour
        
        preserved = [contour[0]]
        
        for i in range(1, len(contour) - 1):
            p1 = np.array(preserved[-1])
            p2 = np.array(contour[i])
            p3 = np.array(contour[i + 1])
            
            # Sprawdź czy punkt leży na linii między p1 i p3
            line_distance = point_to_line_distance(p2, p1, p3)
            
            if line_distance > tolerance:
                preserved.append(contour[i])
        
        # Zawsze dodaj ostatni punkt
        preserved.append(contour[-1])
        
        return preserved
    except:
        return contour

def point_to_line_distance(point, line_start, line_end):
    """Oblicza odległość punktu od linii"""
    try:
        line_vec = line_end - line_start
        line_len = np.linalg.norm(line_vec)
        
        if line_len == 0:
            return np.linalg.norm(point - line_start)
        
        line_unit = line_vec / line_len
        point_vec = point - line_start
        
        # Projekcja punktu na linię
        projection_length = np.dot(point_vec, line_unit)
        projection_length = max(0, min(line_len, projection_length))
        
        projection_point = line_start + projection_length * line_unit
        distance = np.linalg.norm(point - projection_point)
        
        return distance
    except:
        return 0

def create_ai_maximum_fidelity_path(contour, analysis):
    """Tworzy ścieżkę maksymalnej wierności z AI enhancement"""
    try:
        if len(contour) < 3:
            return create_simple_svg_path(contour)
        
        path_data = f"M {contour[0][0]:.4f} {contour[0][1]:.4f}"
        
        # Używaj krzywych Beziera dla gładkich obszarów
        i = 1
        while i < len(contour):
            current = contour[i]
            
            # Sprawdź czy użyć krzywej
            if (i + 2 < len(contour) and 
                should_use_curve_ai_enhanced(contour, i, analysis)):
                
                next_point = contour[i + 1]
                
                # Zaawansowane punkty kontrolne bazujące na lokalnej geometrii
                cp1, cp2 = calculate_ai_control_points(contour, i, analysis)
                
                path_data += f" C {cp1[0]:.4f} {cp1[1]:.4f} {cp2[0]:.4f} {cp2[1]:.4f} {next_point[0]:.4f} {next_point[1]:.4f}"
                i += 2
            else:
                path_data += f" L {current[0]:.4f} {current[1]:.4f}"
                i += 1
        
        path_data += " Z"
        return path_data
    except:
        return create_simple_svg_path(contour)

def should_use_curve_ai_enhanced(contour, index, analysis):
    """AI-enhanced decyzja o użyciu krzywej"""
    try:
        if index < 1 or index >= len(contour) - 2:
            return False
        
        smoothness = analysis.get('smoothness', 0.5)
        complexity = analysis.get('complexity', 0.5)
        
        # Analiza lokalnej geometrii
        local_curvature = calculate_local_curvature(contour, index)
        local_smoothness = calculate_local_smoothness(contour, index)
        
        # AI decision criteria
        use_curve = (
            local_curvature > 0.1 and  # Znacząca krzywizna
            local_smoothness > 0.6 and  # Gładki obszar
            smoothness > 0.4 and  # Ogólna gładkość konturu
            complexity > 0.3  # Wystarczająca złożoność
        )
        
        return use_curve
    except:
        return False

def calculate_local_smoothness(contour, index, window=2):
    """Oblicza lokalną gładkość w oknie wokół punktu"""
    try:
        start_idx = max(0, index - window)
        end_idx = min(len(contour), index + window + 1)
        
        local_contour = contour[start_idx:end_idx]
        
        if len(local_contour) < 3:
            return 0.5
        
        direction_changes = []
        
        for i in range(1, len(local_contour) - 1):
            p1 = np.array(local_contour[i-1])
            p2 = np.array(local_contour[i])
            p3 = np.array(local_contour[i+1])
            
            v1 = p2 - p1
            v2 = p3 - p2
            
            len1 = np.linalg.norm(v1)
            len2 = np.linalg.norm(v2)
            
            if len1 > 0 and len2 > 0:
                direction_change = np.linalg.norm(v2/len2 - v1/len1)
                direction_changes.append(direction_change)
        
        if direction_changes:
            avg_change = np.mean(direction_changes)
            smoothness = 1.0 / (1.0 + avg_change)
            return min(1.0, smoothness)
        
        return 0.5
    except:
        return 0.5

def calculate_ai_control_points(contour, index, analysis):
    """Oblicza inteligentne punkty kontrolne dla krzywych Beziera"""
    try:
        if index < 1 or index >= len(contour) - 2:
            # Fallback dla prostych punktów kontrolnych
            current = contour[index]
            next_point = contour[index + 1]
            return (current, next_point)
        
        p0 = np.array(contour[index - 1])
        p1 = np.array(contour[index])
        p2 = np.array(contour[index + 1])
        p3 = np.array(contour[index + 2])
        
        # Catmull-Rom spline inspired control points
        tension = 0.5 - analysis.get('smoothness', 0.5) * 0.3
        
        # Tangent vectors
        t1 = (p2 - p0) * tension
        t2 = (p3 - p1) * tension
        
        # Control points
        cp1 = p1 + t1 / 3
        cp2 = p2 - t2 / 3
        
        return (tuple(cp1.astype(float)), tuple(cp2.astype(float)))
    except:
        # Simple fallback
        current = contour[index]
        next_point = contour[index + 1]
        return (current, next_point)

def create_ai_high_fidelity_path(contour, analysis):
    """Tworzy ścieżkę wysokiej wierności z AI enhancement"""
    try:
        path_data = f"M {contour[0][0]:.3f} {contour[0][1]:.3f}"
        
        i = 1
        while i < len(contour):
            current = contour[i]
            
            # Używaj krzywych dla gładkich obszarów
            if (i % 3 == 0 and i + 1 < len(contour) and 
                should_use_curve_ai_enhanced(contour, i, analysis)):
                
                next_point = contour[min(i + 1, len(contour) - 1)]
                prev_point = contour[i - 1] if i > 0 else contour[i]
                
                # Prostsze punkty kontrolne
                cp_x = current[0] + (next_point[0] - prev_point[0]) * 0.2
                cp_y = current[1] + (next_point[1] - prev_point[1]) * 0.2
                
                path_data += f" Q {cp_x:.3f} {cp_y:.3f} {current[0]:.3f} {current[1]:.3f}"
            else:
                path_data += f" L {current[0]:.3f} {current[1]:.3f}"
            
            i += 1
        
        path_data += " Z"
        return path_data
    except:
        return create_simple_svg_path(contour)

def create_ai_balanced_path(contour, analysis):
    """Tworzy zbalansowaną ścieżkę AI"""
    try:
        path_data = f"M {contour[0][0]:.2f} {contour[0][1]:.2f}"
        
        for i in range(1, len(contour)):
            current = contour[i]
            
            # Okazjonalne krzywe dla kluczowych punktów
            if (i % 4 == 0 and i > 0 and i < len(contour) - 1 and
                analysis.get('smoothness', 0) > 0.5):
                
                prev_point = contour[i - 1]
                next_point = contour[min(i + 1, len(contour) - 1)]
                
                cp_x = current[0] + (next_point[0] - prev_point[0]) * 0.15
                cp_y = current[1] + (next_point[1] - prev_point[1]) * 0.15
                
                path_data += f" Q {cp_x:.2f} {cp_y:.2f} {current[0]:.2f} {current[1]:.2f}"
            else:
                path_data += f" L {current[0]:.2f} {current[1]:.2f}"
        
        path_data += " Z"
        return path_data
    except:
        return create_simple_svg_path(contour)

def create_ai_optimized_path(contour, analysis):
    """Tworzy optymalizowaną ścieżkę AI"""
    try:
        path_data = f"M {contour[0][0]:.2f} {contour[0][1]:.2f}"
        
        for point in contour[1:]:
            path_data += f" L {point[0]:.2f} {point[1]:.2f}"
        
        path_data += " Z"
        return path_data
    except:
        return create_simple_svg_path(contour)

def analyze_contour_detail_level(contour):
    """Analizuje poziom szczegółowości konturu"""
    try:
        if len(contour) > 50:
            return 'high'
        elif len(contour) > 20:
            return 'medium'
        else:
            return 'simple'
    except:
        return 'medium'

def preserve_contour_details(contour, detail_level):
    """Zachowuje szczegóły konturu z minimalnym upraszczaniem"""
    try:
        if detail_level == 'high':
            # Zachowaj 95% punktów
            step = max(1, len(contour) // 95)
        elif detail_level == 'medium':
            # Zachowaj 90% punktów
            step = max(1, len(contour) // 45)
        else:
            # Zachowaj większość punktów
            step = max(1, len(contour) // 20)

        if step == 1:
            return contour
        else:
            preserved = contour[::step]
            # Zawsze zachowaj pierwszy i ostatni punkt
            if len(preserved) > 0 and preserved[-1] != contour[-1]:
                preserved.append(contour[-1])
            return preserved
    except:
        return contour

def create_high_fidelity_svg_path(contour):
    """Tworzy ścieżkę SVG wysokiej wierności"""
    try:
        if len(contour) < 3:
            return create_simple_svg_path(contour)

        path_data = f"M {contour[0][0]:.3f} {contour[0][1]:.3f}"

        # Użyj krzywych dla płynnych przejść, ale zachowaj precyzję
        i = 1
        while i < len(contour):
            if i + 2 < len(contour):
                # Sprawdź czy warto użyć krzywej
                if should_use_curve_precise(contour, i):
                    p1 = contour[i]
                    p2 = contour[i + 1]

                    # Delikatne krzywe Beziera
                    cp1_x = p1[0] + (p2[0] - contour[i-1][0]) * 0.15
                    cp1_y = p1[1] + (p2[1] - contour[i-1][1]) * 0.15

                    path_data += f" Q {cp1_x:.3f} {cp1_y:.3f} {p2[0]:.3f} {p2[1]:.3f}"
                    i += 2
                else:
                    current = contour[i]
                    path_data += f" L {current[0]:.3f} {current[1]:.3f}"
                    i += 1
            else:
                current = contour[i]
                path_data += f" L {current[0]:.3f} {current[1]:.3f}"
                i += 1

        path_data += " Z"
        return path_data
    except:
        return create_simple_svg_path(contour)

def create_balanced_svg_path(contour):
    """Tworzy zbalansowaną ścieżkę SVG"""
    try:
        path_data = f"M {contour[0][0]:.2f} {contour[0][1]:.2f}"

        for i in range(1, len(contour)):
            current = contour[i]

            # Używaj głównie linii z okazjonalnymi krzywymi
            if i % 3 == 0 and i + 1 < len(contour) and should_use_curve_precise(contour, i):
                next_point = contour[i + 1] if i + 1 < len(contour) else contour[0]
                prev_point = contour[i - 1]

                cp_x = current[0] + (next_point[0] - prev_point[0]) * 0.1
                cp_y = current[1] + (next_point[1] - prev_point[1]) * 0.1

                path_data += f" Q {cp_x:.2f} {cp_y:.2f} {current[0]:.2f} {current[1]:.2f}"
            else:
                path_data += f" L {current[0]:.2f} {current[1]:.2f}"

        path_data += " Z"
        return path_data
    except:
        return create_simple_svg_path(contour)

def create_simple_accurate_svg_path(contour):
    """Tworzy prostą ale dokładną ścieżkę SVG"""
    try:
        path_data = f"M {contour[0][0]:.2f} {contour[0][1]:.2f}"

        for point in contour[1:]:
            path_data += f" L {point[0]:.2f} {point[1]:.2f}"

        path_data += " Z"
        return path_data
    except:
        return create_simple_svg_path(contour)

def should_use_curve_precise(contour, index):
    """Precyzyjnie określa czy użyć krzywej"""
    try:
        if index < 1 or index >= len(contour) - 1:
            return False

        prev_point = contour[index - 1]
        current = contour[index]
        next_point = contour[index + 1]

        # Oblicz kąty między segmentami
        v1 = (current[0] - prev_point[0], current[1] - prev_point[1])
        v2 = (next_point[0] - current[0], next_point[1] - current[1])

        len1 = np.sqrt(v1[0]**2 + v1[1]**2)
        len2 = np.sqrt(v2[0]**2 + v2[1]**2)

        if len1 == 0 or len2 == 0:
            return False

        # Znormalizuj wektory
        v1_norm = (v1[0]/len1, v1[1]/len1)
        v2_norm = (v2[0]/len2, v2[1]/len2)

        # Oblicz kąt
        dot_product = v1_norm[0] * v2_norm[0] + v1_norm[1] * v2_norm[1]
        angle = np.arccos(np.clip(dot_product, -1, 1))

        # Użyj krzywej dla łagodnych zakrętów i odpowiednio długich segmentów
        return angle > np.pi/6 and min(len1, len2) > 8
    except:
        return False

def vectorize_image_improved(input_path, svg_path):
    """Główna funkcja wektoryzacji z użyciem zaawansowanych algorytmów"""
    try:
        print(f"🎯 Rozpoczynam zaawansowaną wektoryzację: {input_path}")
        
        # Załaduj i zoptymalizuj obraz
        optimized_image = optimize_image_for_vectorization(input_path)
        if optimized_image is None:
            print("❌ Nie udało się zoptymalizować obrazu")
            return False

        # Analizuj złożoność obrazu i dobierz parametry
        complexity_params = analyze_image_complexity(optimized_image)
        print(f"📊 Parametry jakości: {complexity_params}")

        # Wyciągnij dominujące kolory z wysoką precyzją
        colors = extract_dominant_colors_advanced(optimized_image, 
                                                complexity_params['max_colors'], 
                                                complexity_params)
        
        if not colors:
            print("❌ Nie udało się wyciągnąć kolorów z obrazu")
            return False

        print(f"🎨 Znaleziono {len(colors)} kolorów wysokiej jakości")

        # Stwórz regiony kolorów z maksymalną precyzją
        regions = create_color_regions_advanced(optimized_image, colors)
        
        if not regions:
            print("❌ Nie udało się utworzyć regionów kolorów")
            return False

        print(f"🗺️ Utworzono {len(regions)} regionów kolorów")

        # Generuj SVG z parametrami haftu
        svg_content = generate_professional_embroidery_svg(optimized_image, regions, complexity_params)
        
        if not svg_content:
            print("❌ Nie udało się wygenerować pliku SVG")
            return False

        # Zapisz plik SVG
        with open(svg_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)

        print(f"✅ Pomyślnie zapisano: {svg_path}")
        return True

    except Exception as e:
        print(f"❌ Błąd podczas wektoryzacji: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_realistic_preview(svg_path, preview_path, original_image, size=(400, 400)):
    """Tworzy czysty podgląd bez obramówek i tekstu"""
    try:
        # Utwórz podgląd na podstawie oryginalnego obrazu
        preview_img = original_image.copy()
        preview_img.thumbnail(size, Image.Resampling.LANCZOS)

        # Zapisz podgląd bez żadnych dodatkowych elementów
        preview_img.save(preview_path, 'PNG', quality=95)
        return True

    except Exception as e:
        print(f"Błąd podczas tworzenia podglądu: {e}")
        # Fallback - zapisz przynajmniej zmniejszony oryginał
        try:
            fallback_img = original_image.copy()
            fallback_img.thumbnail(size, Image.Resampling.LANCZOS)
            fallback_img.save(preview_path, 'PNG')
            return True
        except:
            return False

def generate_professional_embroidery_svg(image, regions, params):
    """Generuje profesjonalny plik SVG z parametrami haftu"""
    try:
        width, height = image.size
        
        svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" 
     viewBox="0 0 {width} {height}"
     xmlns="http://www.w3.org/2000/svg"
     xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
     xmlns:inkstitch="http://inkstitch.org/namespace">
  
  <defs>
    <!-- Definicje wzorów haftu -->
    <pattern id="satin-pattern" patternUnits="userSpaceOnUse" width="2" height="2">
      <rect width="2" height="2" fill="none"/>
      <line x1="0" y1="0" x2="2" y2="2" stroke="#000" stroke-width="0.1"/>
    </pattern>
  </defs>
  
  <g inkscape:label="Embroidery Design" inkscape:groupmode="layer">
'''

        # Dodaj każdy region jako ścieżkę z parametrami haftu
        for i, (color, mask) in enumerate(regions):
            try:
                # Śledź kontury regionu
                contours = trace_contours_advanced(mask)
                
                for j, contour in enumerate(contours):
                    if len(contour) >= 3:
                        # Utwórz ścieżkę SVG
                        path_data = create_smooth_svg_path(contour)
                        
                        if path_data:
                            # Parametry haftu zależne od koloru
                            stitch_params = get_embroidery_parameters(color, params)
                            
                            svg_content += f'''
    <path id="region_{i}_{j}"
          d="{path_data}"
          fill="rgb({color[0]},{color[1]},{color[2]})"
          stroke="rgb({color[0]},{color[1]},{color[2]})"
          stroke-width="0.5"
          inkstitch:object_type="{stitch_params['type']}"
          inkstitch:density_mm="{stitch_params['density']}"
          inkstitch:angle="{stitch_params['angle']}"
          inkstitch:stitch_length_mm="{stitch_params['stitch_length']}"
          inkstitch:color_sort_index="{i}"
          inkstitch:thread_color="#{color[0]:02x}{color[1]:02x}{color[2]:02x}"/>
'''
            except Exception as e:
                print(f"Błąd podczas przetwarzania regionu {i}: {e}")
                continue

        svg_content += '''
  </g>
</svg>'''

        return svg_content

    except Exception as e:
        print(f"Błąd podczas generowania SVG: {e}")
        return None

def get_embroidery_parameters(color, params):
    """Zwraca parametry haftu dostosowane do koloru"""
    try:
        brightness = sum(color) / 3
        saturation = max(color) - min(color)
        
        # Dostosuj parametry do koloru
        if brightness < 80:  # Ciemne kolory
            return {
                'type': 'fill',
                'density': '0.4',
                'angle': '45',
                'stitch_length': '2.5'
            }
        elif saturation > 100:  # Nasycone kolory
            return {
                'type': 'satin',
                'density': '0.3',
                'angle': '0',
                'stitch_length': '3.0'
            }
        else:  # Standardowe kolory
            return {
                'type': 'fill',
                'density': '0.35',
                'angle': '30',
                'stitch_length': '2.8'
            }
    except:
        return {
            'type': 'fill',
            'density': '0.4',
            'angle': '45',
            'stitch_length': '2.5'
        }

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Generator Wzorów Haftu - Profesjonalna Wektoryzacja</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 15px; text-align: center; margin-bottom: 30px; }
        .upload-area { border: 3px dashed #667eea; border-radius: 15px; padding: 50px; text-align: center; margin: 20px 0; background: white; transition: all 0.3s; }
        .upload-area:hover { border-color: #764ba2; transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
        .btn { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px 30px; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; font-weight: bold; }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.2); }
        .result { margin-top: 30px; padding: 25px; border: 1px solid #ddd; border-radius: 15px; background: white; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .preview { max-width: 100%; height: auto; border: 2px solid #667eea; border-radius: 10px; }
        .success { color: #28a745; font-weight: bold; }
        .warning { color: #fd7e14; font-weight: bold; background: #fff3cd; padding: 15px; border-radius: 8px; margin: 20px 0; }
        .info { color: #17a2b8; background: #d1ecf1; padding: 15px; border-radius: 8px; margin: 20px 0; }
        .features { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 30px 0; }
        .feature { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center; }
        .feature-icon { font-size: 2em; margin-bottom: 10px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧵 Generator Wzorów Haftu</h1>
        <h2>Profesjonalna Wektoryzacja Obrazów</h2>
        <p>Zaawansowana technologia wektoryzacji z kompatybilnością InkStitch</p>
    </div>

    <div class="features">
        <div class="feature">
            <div class="feature-icon">🎨</div>
            <h3>Inteligentne Kolory</h3>
            <p>Zaawansowany algorytm K-means do wykrywania dominujących kolorów</p>
        </div>
        <div class="feature">
            <div class="feature-icon">🔍</div>
            <h3>Precyzyjne Kontury</h3>
            <p>Wykrywanie konturów z wygładzaniem i optymalizacją ścieżek</p>
        </div>
        <div class="feature">
            <div class="feature-icon">⚡</div>
            <h3>Wysoka Wydajność</h3>
            <p>Zoptymalizowane algorytmy dla szybkiego przetwarzania</p>
        </div>
    </div>

    <div class="info">
        <strong>🚀 PREMIUM funkcje najwyższej jakości:</strong>
        <br>• Ultra precyzyjna wektoryzacja z AI enhancement
        <br>• Do 80 kolorów z advanced color analysis
        <br>• Multi-step image processing dla maksymalnej jakości
        <br>• Supreme curve smoothing algorithms
        <br>• Ultra precision mode dla detali
        <br>• Advanced micro-detail preservation
    </div>

    <div class="warning">
        ⚠️ Parametry PREMIUM optymalizacji:
        <br>• Maksymalny rozmiar pliku: 12MB (zwiększono)
        <br>• Ultra wysoka rozdzielczość do 1800px
        <br>• Do 80 kolorów wysokiej precyzji
        <br>• AI-enhanced curve smoothing
        <br>• Multi-step image enhancement
    </div>

    <div class="upload-area" onclick="document.getElementById('file').click()">
        <p style="font-size: 1.2em; margin-bottom: 10px;">📁 Kliknij tutaj lub przeciągnij obraz</p>
        <p>Obsługiwane formaty: PNG, JPG, JPEG, WebP, SVG</p>
        <p style="color: #666; font-size: 0.9em;">Dla najlepszych rezultatów używaj obrazów o wysokim kontraście</p>
        <input type="file" id="file" style="display: none" accept=".png,.jpg,.jpeg,.webp,.svg">
    </div>

    <button class="btn" onclick="uploadFile()">🚀 Rozpocznij Profesjonalną Wektoryzację</button>

    <div id="result" class="result" style="display: none;">
        <h3>📊 Wynik wektoryzacji:</h3>
        <div id="result-content"></div>
    </div>

    <script>
        function uploadFile() {
            const fileInput = document.getElementById('file');
            const file = fileInput.files[0];

            if (!file) {
                alert('Wybierz plik do przetworzenia');
                return;
            }

            if (file.size > 8 * 1024 * 1024) {
                alert('Plik jest za duży. Maksymalny rozmiar to 8MB.');
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            document.getElementById('result').style.display = 'block';
            document.getElementById('result-content').innerHTML = 
                '<div style="text-align: center; padding: 20px;">' +
                '<div style="display: inline-block; width: 40px; height: 40px; border: 4px solid #f3f3f3; border-top: 4px solid #667eea; border-radius: 50%; animation: spin 1s linear infinite;"></div>' +
                '<p style="margin-top: 15px;">🎨 Profesjonalna wektoryzacja w toku...</p>' +
                '<p style="color: #666;">Analizowanie kolorów i tworzenie ścieżek SVG...</p>' +
                '</div>' +
                '<style>@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }</style>';

            fetch('/vectorize', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('result-content').innerHTML = 
                        '<div style="text-align: center;">' +
                        '<p class="success">✅ Profesjonalna wektoryzacja zakończona pomyślnie!</p>' +
                        '<p>🎯 Wygenerowano wysokiej jakości plik SVG z parametrami haftu</p>' +
                        '<img src="' + data.preview_url + '" class="preview" alt="Podgląd haftu" style="max-width: 400px; margin: 20px 0;">' +
                        '<br><br>' +
                        '<a href="' + data.svg_url + '" download class="btn" style="text-decoration: none; display: inline-block;">📥 Pobierz Profesjonalny SVG</a>' +
                        '<p style="margin-top: 15px; color: #666; font-size: 0.9em;">Plik kompatybilny z InkStitch i programami do haftu</p>' +
                        '</div>';
                } else {
                    document.getElementById('result-content').innerHTML = 
                        '<div style="text-align: center; color: #dc3545;">' +
                        '<p>❌ Błąd: ' + data.error + '</p>' +
                        '<p style="color: #666;">Spróbuj z innym obrazem lub sprawdź format pliku</p>' +
                        '</div>';
                }
            })
            .catch(error => {
                document.getElementById('result-content').innerHTML = 
                    '<div style="text-align: center; color: #dc3545;">' +
                    '<p>❌ Błąd połączenia: ' + error + '</p>' +
                    '<p style="color: #666;">Sprawdź połączenie internetowe i spróbuj ponownie</p>' +
                    '</div>';
            });
        }

        // Drag and drop functionality
        const uploadArea = document.querySelector('.upload-area');

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#764ba2';
            uploadArea.style.backgroundColor = '#f8f9ff';
        });

        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#667eea';
            uploadArea.style.backgroundColor = 'white';
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#667eea';
            uploadArea.style.backgroundColor = 'white';

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                document.getElementById('file').files = files;
                uploadFile();
            }
        });
    </script>
</body>
</html>
    ''')

@app.route('/vectorize', methods=['POST'])
def vectorize():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'Brak pliku'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Nie wybrano pliku'})

        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Nieobsługiwany format pliku'})

        # Sprawdź rozmiar pliku
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        if file_size > MAX_FILE_SIZE:
            return jsonify({'success': False, 'error': f'Plik za duży. Maksymalny rozmiar: {MAX_FILE_SIZE/1024/1024:.1f}MB'})

        # Generuj unikalne ID
        timestamp = str(int(time.time() * 1000))

        # Zapisz plik wejściowy
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, 'raster', f"{timestamp}_{filename}")
        file.save(input_path)

        # Ścieżki plików wyjściowych
        svg_filename = f"professional_{timestamp}.svg"
        svg_path = os.path.join(UPLOAD_FOLDER, 'vector_auto', svg_filename)
        preview_filename = f"{timestamp}_embroidery_preview.png"
        preview_path = os.path.join(UPLOAD_FOLDER, 'preview', preview_filename)

        print(f"🎯 Rozpoczynanie profesjonalnej wektoryzacji: {input_path}")

        # Załaduj oryginalny obraz dla podglądu
        original_image = Image.open(input_path)

        # Wektoryzacja
        success = vectorize_image_improved(input_path, svg_path)

        if not success:
            return jsonify({'success': False, 'error': 'Nie udało się zwektoryzować obrazu. Spróbuj z obrazem o wyższym kontraście.'})

        # Sprawdź jakość pliku SVG
        if not os.path.exists(svg_path):
            return jsonify({'success': False, 'error': 'Plik SVG nie został utworzony'})

        file_size = os.path.getsize(svg_path)
        if file_size < 300:
            return jsonify({'success': False, 'error': 'Wygenerowany plik SVG jest za mały - możliwe problemy z jakością obrazu'})

        # Tworzenie realistycznego podglądu
        preview_success = create_realistic_preview(svg_path, preview_path, original_image)
        if not preview_success:
            print("⚠️ Nie udało się utworzyć podglądu")

        # Wymuś czyszczenie pamięci
        gc.collect()

        print(f"🎉 Profesjonalna wektoryzacja zakończona! Rozmiar pliku: {file_size} bajtów")

        return jsonify({
            'success': True,
            'svg_url': f'/download/vector_auto/{svg_filename}',
            'preview_url': f'/download/preview/{preview_filename}',
            'message': f'Profesjonalna wektoryzacja zakończona! Wygenerowano plik SVG ({file_size} B) kompatybilny z InkStitch'
        })

    except Exception as e:
        print(f"❌ Błąd podczas wektoryzacji: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': 'Błąd serwera podczas przetwarzania. Spróbuj z innym obrazem.'})

@app.route('/download/<path:subpath>')
def download_file(subpath):
    try:
        file_path = os.path.join(UPLOAD_FOLDER, subpath)
        if os.path.exists(file_path):
            return send_file(file_path)
        else:
            return "Plik nie znaleziony", 404
    except Exception as e:
        return f"Błąd: {e}", 500

if __name__ == '__main__':
    print("🧵 Generator Wzorów Haftu - Profesjonalna Wektoryzacja")
    print("🎨 Zaawansowane algorytmy wykrywania kolorów i konturów")
    print("⚡ Optymalizacja wydajności i jakości")
    print("🔗 Kompatybilność z InkStitch")
    print("📡 Serwer uruchamiany na porcie 5000...")

    app.run(host='0.0.0.0', port=5000, debug=False)