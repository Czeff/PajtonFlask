python
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
    """ULTRA PREMIUM analiza kolorów z AI enhancement"""
    try:
        img_array = np.array(image)

        # Pobierz parametry
        tolerance_factor = params.get('tolerance_factor', 0.3) if params else 0.3
        cartoon_optimization = params.get('cartoon_optimization', False) if params else False
        line_art_optimization = params.get('line_art_optimization', False) if params else False
        ultra_precision_mode = params.get('ultra_precision_mode', False) if params else False

        print(f"🎨 ULTRA PREMIUM Color Analysis:")
        print(f"   🔧 Tolerancja: {tolerance_factor}, Cartoon: {cartoon_optimization}, LineArt: {line_art_optimization}")

        colors = []

        # 1. AI-ENHANCED PERCEPTUAL COLOR EXTRACTION
        if ultra_precision_mode:
            perceptual_colors = extract_perceptual_important_colors_ultra(img_array, max_colors // 3, params)
            colors.extend(perceptual_colors)
            print(f"   🧠 AI Perceptual: {len(perceptual_colors)} kolorów")

        # 2. CARTOON/ANIME OPTIMIZED COLORS
        if cartoon_optimization:
            cartoon_colors = extract_cartoon_optimized_colors(img_array, max_colors // 4, params)
            colors.extend(cartoon_colors)
            print(f"   🎭 Cartoon: {len(cartoon_colors)} kolorów")

        # 3. LINE ART OPTIMIZED COLORS
        if line_art_optimization:
            line_art_colors = extract_line_art_colors(img_array, max_colors // 4, params)
            colors.extend(line_art_colors)
            print(f"   ✏️ Line Art: {len(line_art_colors)} kolorów")

        # 4. MULTI-SCALE DOMINANT COLORS with AI clustering
        multi_scale_colors = extract_multi_scale_dominant_colors(img_array, max_colors // 3, params)
        colors.extend(multi_scale_colors)
        print(f"   🔍 Multi-scale: {len(multi_scale_colors)} kolorów")

        # 5. EDGE-AWARE COLOR EXTRACTION
        edge_aware_colors = extract_edge_aware_colors(img_array, max_colors // 4, params)
        colors.extend(edge_aware_colors)
        print(f"   📐 Edge-aware: {len(edge_aware_colors)} kolorów")

        # 6. GRADIENT & TRANSITION COLORS
        if params.get('gradient_preservation', False):
            gradient_colors = extract_gradient_transition_colors(img_array, max_colors // 6, params)
            colors.extend(gradient_colors)
            print(f"   🌈 Gradient: {len(gradient_colors)} kolorów")

        # AI-POWERED COLOR REFINEMENT & MERGING
        final_colors = ai_powered_color_refinement_ultra(colors, max_colors, img_array, params)

        # PERCEPTUAL IMPORTANCE SORTING
        final_colors = sort_colors_by_perceptual_importance(img_array, final_colors, params)

        print(f"🎨 ULTRA PREMIUM Color Analysis Complete: {len(final_colors)} najwyższej jakości kolorów")
        return final_colors

    except Exception as e:
        print(f"❌ Błąd ULTRA color analysis: {e}")
        import traceback
        traceback.print_exc()
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
    """ULTRA PREMIUM analiza złożoności obrazu z najnowszymi algorytmami AI"""
    try:
        img_array = np.array(image)
        height, width = img_array.shape[:2]

        # 1. ZAAWANSOWANA ANALIZA KRAWĘDZI Z DEEP LEARNING INSIGHTS
        edge_density = detect_edge_density_advanced(img_array)
        edge_sharpness = detect_edge_sharpness(img_array)
        edge_connectivity = detect_edge_connectivity(img_array)
        edge_continuity = analyze_edge_continuity(img_array)

        # 2. ULTRA PRECYZYJNA ANALIZA KOLORÓW Z PERCEPTUAL SCIENCE
        color_complexity = detect_color_complexity_advanced(img_array)
        color_variance = detect_color_variance(img_array)
        color_gradients = detect_color_gradients(img_array)
        color_harmony = analyze_color_harmony(img_array)

        # 3. ANALIZA TEKSTUR I WZORÓW Z MACHINE LEARNING
        texture_complexity = detect_texture_complexity(img_array)
        pattern_regularity = detect_pattern_regularity(img_array)
        
        try:
            texture_directionality = analyze_texture_directionality(img_array)
        except:
            texture_directionality = 0.5

        # 4. ZAAWANSOWANA ANALIZA GEOMETRYCZNA
        geometric_complexity = detect_geometric_shapes(img_array)
        curve_complexity = detect_curve_complexity(img_array)
        shape_regularity = analyze_shape_regularity(img_array)

        # 5. PERCEPTUALNA ANALIZA JAKOŚCI + VISUAL SALIENCY
        perceptual_importance = calculate_perceptual_importance(img_array)
        detail_density = calculate_detail_density(img_array)
        visual_saliency = calculate_visual_saliency(img_array)

        # 6. NOWE: ANALIZA CARTOON/ANIME STYLE
        cartoon_score = detect_cartoon_style_features(img_array)
        line_art_score = detect_line_art_quality(img_array)

        print(f"🔬 ULTRA PREMIUM Analiza AI:")
        print(f"   📐 Krawędzie: gęstość={edge_density:.3f}, ostrość={edge_sharpness:.3f}, ciągłość={edge_continuity:.3f}")
        print(f"   🎨 Kolory: złożoność={color_complexity}, harmonia={color_harmony:.3f}, gradienty={color_gradients:.3f}")
        print(f"   🖼️ Tekstury: złożoność={texture_complexity:.3f}, kierunkowość={texture_directionality:.3f}")
        print(f"   📊 Geometria: kształty={geometric_complexity:.3f}, regularność={shape_regularity:.3f}")
        print(f"   👁️ Percepcja: ważność={perceptual_importance:.3f}, saliency={visual_saliency:.3f}")
        print(f"   🎭 Styl: cartoon={cartoon_score:.3f}, line_art={line_art_score:.3f}")

        # ZAAWANSOWANY ALGORYTM AI DO DOBORU PARAMETRÓW
        complexity_score = calculate_advanced_complexity_score(
            edge_density, edge_sharpness, edge_connectivity, edge_continuity,
            color_complexity, color_variance, color_gradients, color_harmony,
            texture_complexity, pattern_regularity, texture_directionality,
            geometric_complexity, curve_complexity, shape_regularity,
            perceptual_importance, detail_density, visual_saliency,
            cartoon_score, line_art_score
        )

        print(f"🎯 AI Complexity Score: {complexity_score:.3f} (0.0-1.0)")

        # ULTRA PREMIUM ADAPTIVE PARAMETERS
        return generate_ultra_premium_parameters(complexity_score, cartoon_score, line_art_score, 
                                                edge_density, color_complexity, detail_density)

    except Exception as e:
        print(f"⚠️ Błąd analizy złożoności: {e}")
        return get_fallback_premium_parameters()

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

def create_color_regions_advanced(image, colors, params=None):
    """Zoptymalizowane tworzenie regionów skupiające się na głównych obszarach"""
    try:
        width, height = image.size
        img_array = np.array(image)

        # Pobierz parametry
        min_region_size = params.get('min_region_size', 100) if params else 100
        tolerance_factor = params.get('tolerance_factor', 0.5) if params else 0.5

        regions = []

        # Analiza każdego koloru z fokusem na większe obszary
        for i, color in enumerate(colors):
            print(f"🎯 Przetwarzanie głównego koloru {i+1}/{len(colors)}: {color}")

            # Tworzenie maski z większą tolerancją
            mask = create_main_area_mask(img_array, color, tolerance_factor)

            if mask is None:
                continue

            initial_pixels = np.sum(mask)
            print(f"  📊 Początkowe piksele: {initial_pixels}")

            if initial_pixels >= min_region_size:
                # Minimalne przetwarzanie dla zachowania dużych obszarów
                mask = clean_main_regions(mask, min_region_size)

                final_pixels = np.sum(mask)
                print(f"  ✅ Finalne piksele: {final_pixels}")

                if final_pixels >= min_region_size:
                    regions.append((color, mask))
                    print(f"  ✓ Dodano główny region ({final_pixels} px) dla koloru {color}")
                else:
                    print(f"  ✗ Region za mały po czyszczeniu ({final_pixels} px < {min_region_size} px)")
            else:
                print(f"  ✗ Region za mały ({initial_pixels} px < {min_region_size} px)")

        print(f"🏁 Utworzono {len(regions)} głównych regionów")
        return regions

    except Exception as e:
        print(f"❌ Błąd podczas tworzenia regionów: {e}")
        return create_color_regions_simple(image, colors)

def extract_main_area_colors(img_array, max_colors, params):
    """Wyciąga kolory z głównych obszarów obrazu"""
    try:
        from sklearn.cluster import KMeans

        # Większe próbkowanie dla głównych kolorów
        height, width = img_array.shape[:2]
        sample_rate = min(0.5, 200000 / (height * width))

        pixels = img_array.reshape(-1, 3)
        if len(pixels) > 100000:
            step = max(1, int(1 / sample_rate))
            pixels = pixels[::step]

        def analyze_edge_continuity(img_array):
            """Analizuje ciągłość krawędzi - kluczowe dla jakości cartoon/anime"""
            try:
                from scipy import ndimage
                gray = np.mean(img_array, axis=2)

                # Gradients in different directions
                grad_x = ndimage.sobel(gray, axis=1)
                grad_y = ndimage.sobel(gray, axis=0)
                grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

                # Analyze continuity by connecting neighboring edges
                edge_mask = grad_magnitude > np.percentile(grad_magnitude, 85)

                # Morphological operations for continuity
                from scipy.ndimage import binary_closing, binary_opening
                continuous_edges = binary_closing(edge_mask, structure=np.ones((3, 3)))
                continuous_edges = binary_opening(continuous_edges, structure=np.ones((2, 2)))

                # Ratio of continuous edges to all
                continuity_ratio = np.sum(continuous_edges) / max(1, np.sum(edge_mask))

                return min(1.0, continuity_ratio * 1.2)
            except Exception as e:
                print(f"Błąd w analyze_edge_continuity: {e}")
                return 0.5

        def analyze_color_harmony(img_array):
            """Analizuje harmonię kolorów - ważne dla estetyki wektoryzacji"""
            try:
                # Konwersja do HSV dla analizy harmonii
                from skimage.color import rgb2hsv
                hsv_img = rgb2hsv(img_array / 255.0)

                # Pobierz dominujące odcienie
                hue_values = hsv_img[:,:,0].flatten()
                saturation_values = hsv_img[:,:,1].flatten()

                # Usuń nienasycone kolory z analizy
                saturated_mask = saturation_values > 0.3
                if np.sum(saturated_mask) == 0:
                    return 0.8  # Obrazy monochromatyczne mają dobrą harmonię

                saturated_hues = hue_values[saturated_mask]

                # Analiza dystrybucji odcieni
                hue_hist, _ = np.histogram(saturated_hues, bins=36, range=(0, 1))

                # Znajdź wskaźniki do analizy harmonii
                peak_indices = np.where(hue_hist > np.percentile(hue_hist, 75))[0]

                if len(peak_indices) == 0:
                    return 0.5

                # Oblicz średnie odległości między pikami
                if len(peak_indices) > 1:
                    peak_distances = []
                    for i in range(len(peak_indices) - 1):
                        dist = min(abs(peak_indices[i+1] - peak_indices[i]), 
                                  36 - abs(peak_indices[i+1] - peak_indices[i]))
                        peak_distances.append(dist)

                    # Zwróć funkcję ostateczną na podstawie odległości
                    return max(0.0, 1.0 - (np.mean(peak_distances) / 36.0))

                return 1.0  # Wysoka harmonia

            except Exception as e:
                print(f"Błąd w analyze_color_harmony: {e}")
                return 0.5



def analyze_shape_regularity(img_array):
    """Analizuje regularność kształtów"""
    try:
        from scipy import ndimage
        from skimage import measure

        gray = np.mean(img_array, axis=2)

        # Threshold dla wykrywania kształtów
        threshold = np.percentile(gray, 50)
        binary = gray > threshold

        # Znajdź kontury
        contours = measure.find_contours(binary, 0.5)

        if len(contours) == 0:
            return 0.5

        regularities = []

        for contour in contours[:10]:  # Ogranicz do 10 największych
            if len(contour) < 10:
                continue

            # Oblicz regularność przez analiza obwodu vs powierzchni
            area = 0.5 * abs(sum(contour[i,0] * contour[i+1,1] - contour[i+1,0] * contour[i,1] 
                                for i in range(-1, len(contour)-1)))

            if area == 0:
                continue

            perimeter = np.sum(np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1)))

            if perimeter == 0:
                continue

            # Współczynnik regularności (im bliżej okręgu, tym wyższy)
            circularity = 4 * np.pi * area / (perimeter ** 2)
            regularities.append(min(1.0, circularity))

        if regularities:
            return np.mean(regularities)

        return 0.5
    except:
        return 0.5

def calculate_visual_saliency(img_array):
    """Oblicza wizualną saliency - gdzie skupia się uwaga"""
    try:
        from scipy import ndimage

        # Konwersja do przestrzeni LAB dla lepszej percepcji
        try:
            from skimage.color import rgb2lab
            lab_img = rgb2lab(img_array / 255.0)
        except:
            # Fallback do grayscale
            lab_img = np.mean(img_array, axis=2)
            lab_img = lab_img[:,:,np.newaxis]

        saliency_maps = []

        # 1. Saliency bazująca na kontraście
        if lab_img.ndim == 3 and lab_img.shape[2] >= 3:
            for channel in range(min(3, lab_img.shape[2])):
                channel_data = lab_img[:,:,channel]

                # Lokalne kontrast
                mean_filtered = ndimage.uniform_filter(channel_data, size=9)
                contrast_map = np.abs(channel_data - mean_filtered)
                saliency_maps.append(contrast_map)
        else:
            # Fallback dla grayscale
            gray = lab_img[:,:,0] if lab_img.ndim == 3 else lab_img
            mean_filtered = ndimage.uniform_filter(gray, size=9)
            contrast_map = np.abs(gray - mean_filtered)
            saliency_maps.append(contrast_map)

        # 2. Saliency bazująca na krawędziach
        if lab_img.ndim == 3:
            gray = np.mean(lab_img, axis=2)
        else:
            gray = lab_img

        sobel_h = ndimage.sobel(gray, axis=0)
        sobel_v = ndimage.sobel(gray, axis=1)
        edge_saliency = np.sqrt(sobel_h**2 + sobel_v**2)
        saliency_maps.append(edge_saliency)

        # Kombinuj mapy saliency
        combined_saliency = np.mean(saliency_maps, axis=0)

        # Normalizuj i oblicz średnią saliency
        normalized_saliency = (combined_saliency - np.min(combined_saliency)) / \
                             (np.max(combined_saliency) - np.min(combined_saliency) + 1e-8)

        return np.mean(normalized_saliency)
    except:
        return 0.5

def detect_cartoon_style_features(img_array):
    """Wykrywa cechy stylu cartoon/anime"""
    try:
        from scipy import ndimage

        # 1. Analiza płaskich obszarów kolorów (typowe dla cartoon)
        flat_areas_score = analyze_flat_color_areas(img_array)

        # 2. Analiza ostrych krawędzi (typowe dla cartoon)
        sharp_edges_score = analyze_sharp_edges(img_array)

        # 3. Analiza ograniczonej palety kolorów
        limited_palette_score = analyze_limited_palette(img_array)

        # 4. Analiza wysokiego kontrastu
        high_contrast_score = analyze_high_contrast(img_array)

        # Kombinuj wyniki
        cartoon_score = (
            flat_areas_score * 0.3 +
            sharp_edges_score * 0.3 +
            limited_palette_score * 0.25 +
            high_contrast_score * 0.15
        )

        return min(1.0, cartoon_score)
    except:
        return 0.3

def detect_line_art_quality(img_array):
    """Wykrywa jakość line art - ważne dla dokładnej wektoryzacji"""
    try:
        from scipy import ndimage
        gray = np.mean(img_array, axis=2)

        # 1. Analiza grubości linii
        line_thickness_score = analyze_line_thickness_consistency(gray)

        # 2. Analiza ciągłości linii
        line_continuity_score = analyze_line_continuity(gray)

        # 3. Analiza czystości linii (brak artefaktów)
        line_cleanliness_score = analyze_line_cleanliness(gray)

        # Kombinuj wyniki
        line_art_score = (
            line_thickness_score * 0.4 +
            line_continuity_score * 0.4 +
            line_cleanliness_score * 0.2
        )

        return min(1.0, line_art_score)
    except:
        return 0.3

def analyze_flat_color_areas(img_array):
    """Analizuje obecność płaskich obszarów kolorów"""
    try:
        from scipy import ndimage

        # Wygładź obraz dla lepszej analizy
        smoothed = ndimage.gaussian_filter(img_array.astype(float), sigma=1.0)

        # Oblicz lokalne odchylenie standardowe
        local_variance = np.zeros_like(smoothed[:,:,0])

        for channel in range(3):
            channel_data = smoothed[:,:,channel]
            mean_filtered = ndimage.uniform_filter(channel_data, size=5)
            var_filtered = ndimage.uniform_filter(channel_data**2, size=5)
            local_var = var_filtered - mean_filtered**2
            local_variance += local_var

        local_variance /= 3

        # Obszary z małą wariancją = płaskie obszary
        flat_threshold = np.percentile(local_variance, 30)
        flat_areas = local_variance <= flat_threshold

        flat_ratio = np.sum(flat_areas) / flat_areas.size
        return min(1.0, flat_ratio * 2)
    except:
        return 0.3

def analyze_sharp_edges(img_array):
    """Analizuje ostrość krawędzi"""
    try:
        from scipy import ndimage
        gray = np.mean(img_array, axis=2)

        # Laplacian dla wykrywania ostrych krawędzi
        laplacian = ndimage.laplace(gray)
        sharp_edges = np.abs(laplacian) > np.percentile(np.abs(laplacian), 85)

        # Gradient magnitude
        grad_x = ndimage.sobel(gray, axis=1)
        grad_y = ndimage.sobel(gray, axis=0)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Wysoki gradient = ostre krawędzie
        sharp_gradient = grad_magnitude > np.percentile(grad_magnitude, 80)

        # Kombinuj wskaźniki
        combined_sharp = sharp_edges | sharp_gradient
        sharpness_ratio = np.sum(combined_sharp) / combined_sharp.size

        return min(1.0, sharpness_ratio * 10)
    except:
        return 0.3

def analyze_limited_palette(img_array):
    """Analizuje ograniczoną paletę kolorów (typowe dla cartoon)"""
    try:
        # Kwantyzuj kolory do sprawdzenia różnorodności
        quantized = (img_array // 32) * 32  # Redukcja do 8 poziomów na kanał

        # Policza unikalne kolory
        unique_colors = np.unique(quantized.reshape(-1, 3), axis=0)
        num_unique = len(unique_colors)

        total_pixels = img_array.shape[0] * img_array.shape[1]

        # Im mniej unikalnych kolorów, tym wyższy wynik
        if total_pixels > 0:
            color_diversity = num_unique / total_pixels
            limited_score = 1.0 - min(1.0, color_diversity * 100)
            return limited_score

        return 0.5
    except:
        return 0.3

def analyze_high_contrast(img_array):
    """Analizuje wysoki kontrast obrazu"""
    try:
        # Konwersja do jasności
        luminance = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]

        # Histogram jasności
        hist, bins = np.histogram(luminance, bins=50, range=(0, 255))

        # Sprawdź czy są wyraźne piki na końcach (wysoki kontrast)
        dark_pixels = np.sum(hist[:10])  # Pierwsze 20%
        bright_pixels = np.sum(hist[-10:])  # Ostatnie 20%
        middle_pixels = np.sum(hist[15:35])  # Środek

        total_pixels = luminance.size

        # Wysoki kontrast = dużo bardzo ciemnych i bardzo jasnych pikseli
        contrast_score = (dark_pixels + bright_pixels) / total_pixels

        # Dodatkowo sprawdź rozrzut
        std_luminance = np.std(luminance)
        normalized_std = std_luminance / 255.0

        final_score = (contrast_score * 0.7 + normalized_std * 0.3)
        return min(1.0, final_score * 1.5)
    except:
        return 0.3

def analyze_line_thickness_consistency(gray):
    """Analizuje konsystencję grubości linii"""
    try:
        from scipy import ndimage

        # Wykryj krawędzie
        edges = ndimage.sobel(gray)
        edge_mask = edges > np.percentile(edges, 90)

        if np.sum(edge_mask) == 0:
            return 0.3

        # Morfologiczne analizy grubości
        from scipy.ndimage import binary_erosion, binary_dilation

        # Różne grubości erozji
        thickness_scores = []
        for erosion_size in range(1, 4):
            eroded = binary_erosion(edge_mask, iterations=erosion_size)
            remaining_ratio = np.sum(eroded) / np.sum(edge_mask)
            thickness_scores.append(remaining_ratio)

        # Konsystentna grubość = równomierne zmniejszanie
        if len(thickness_scores) > 1:
            consistency = 1.0 - np.std(thickness_scores)
            return max(0.0, consistency)

        return 0.5
    except:
        return 0.3

def analyze_line_continuity(gray):
    """Analizuje ciągłość linii"""
    try:
        from scipy import ndimage

        edges = ndimage.sobel(gray)
        edge_mask = edges > np.percentile(edges, 85)

        # Morfologiczne zamknięcie dla ciągłości
        from scipy.ndimage import binary_closing
        closed_edges = binary_closing(edge_mask, structure=np.ones((3, 3)))

        # Stosunek zamkniętych do oryginalnych
        if np.sum(edge_mask) > 0:
            continuity_ratio = np.sum(closed_edges) / np.sum(edge_mask)
            return min(1.0, continuity_ratio)

        return 0.5
    except:
        return 0.3

def analyze_line_cleanliness(gray):
    """Analizuje czystość linii (brak artefaktów)"""
    try:
        from scipy import ndimage

        # Porównaj oryginał z wygładzonym
        smoothed = ndimage.gaussian_filter(gray, sigma=0.5)
        difference = np.abs(gray - smoothed)

        # Małe różnice = czyste linie
        noise_level = np.mean(difference)
        cleanliness = 1.0 - min(1.0, noise_level / 50.0)

        return max(0.0, cleanliness)
    except:
        return 0.5

def calculate_advanced_complexity_score(*args):
    """Zaawansowany algorytm obliczania złożoności z wszystkimi parametrami"""
    try:
        (edge_density, edge_sharpness, edge_connectivity, edge_continuity,
         color_complexity, color_variance, color_gradients, color_harmony,
         texture_complexity, pattern_regularity, texture_directionality,
         geometric_complexity, curve_complexity, shape_regularity,
         perceptual_importance, detail_density, visual_saliency,
         cartoon_score, line_art_score) = args

        # Normalizuj color_complexity
        color_complexity_norm = min(1.0, color_complexity / 500.0)

        # Zaawansowane wagi bazujące na stylu obrazu
        if cartoon_score > 0.6:  # Cartoon style
            weights = {
                'edge': 0.30,      # Krawędzie bardzo ważne dla cartoon
                'color': 0.25,     # Kolory ważne
                'style': 0.20,     # Styl cartoon
                'geometry': 0.15,  # Geometria
                'perception': 0.10 # Percepcja
            }
        elif line_art_score > 0.6:  # Line art style
            weights = {
                'edge': 0.35,      # Krawędzie najważniejsze
                'style': 0.25,     # Jakość line art
                'geometry': 0.20,  # Geometria
                'color': 0.15,     # Kolory mniej ważne
                'perception': 0.05
            }
        else:  # Standard/photo style
            weights = {
                'edge': 0.25,
                'color': 0.25,
                'perception': 0.20,
                'geometry': 0.15,
                'style': 0.15
            }

        # Komponenty złożoności
        edge_component = (
            edge_density * 0.3 + 
            edge_sharpness * 0.3 + 
            edge_connectivity * 0.2 + 
            edge_continuity * 0.2
        )

        color_component = (
            color_complexity_norm * 0.3 + 
            color_variance * 0.25 + 
            color_gradients * 0.25 + 
            color_harmony * 0.2
        )

        style_component = (cartoon_score + line_art_score) / 2

        geometry_component = (
            geometric_complexity * 0.3 + 
            curve_complexity * 0.3 + 
            shape_regularity * 0.2 + 
            texture_directionality * 0.2
        )

        perception_component = (
            perceptual_importance * 0.4 + 
            detail_density * 0.3 + 
            visual_saliency * 0.3
        )

        # Ważona suma z zaawansowanymi wagami
        overall_score = (
            edge_component * weights['edge'] +
            color_component * weights['color'] +
            style_component * weights['style'] +
            geometry_component * weights['geometry'] +
            perception_component * weights['perception']
        )

        # Nieliniowa transformacja z bonus za wysoką jakość
        if cartoon_score > 0.7 or line_art_score > 0.7:
            overall_score *= 1.1  # Bonus za wysoką jakość stylu

        adjusted_score = np.power(overall_score, 0.85)
        return min(1.0, max(0.0, adjusted_score))
    except:
        return 0.5

def generate_ultra_premium_parameters(complexity_score, cartoon_score, line_art_score, 
                                    edge_density, color_complexity, detail_density):
    """Generuje ultra premium parametry bazując na AI analysis"""
    try:
        # Bazowe parametry
        base_params = {
            'max_colors': 30,
            'tolerance_factor': 0.3,
            'detail_preservation': 'ultra_high',
            'min_region_size': 50,
            'color_flattening': False,
            'quality_enhancement': 'ai_ultra',
            'curve_smoothing': 'ai_adaptive',
            'edge_enhancement': True,
            'micro_detail_preservation': True,
            'gradient_preservation': True,
            'ultra_precision_mode': True,
            'advanced_color_analysis': True,
            'focus_main_areas': False,
            'cartoon_optimization': False,
            'line_art_optimization': False
        }

        # Adaptacja dla cartoon style
        if cartoon_score > 0.6:
            base_params.update({
                'max_colors': min(35, int(25 + cartoon_score * 15)),
                'tolerance_factor': 0.25,  # Mniejsza tolerancja dla ostrych krawędzi
                'min_region_size': max(30, int(100 - cartoon_score * 50)),
                'cartoon_optimization': True,
                'edge_enhancement': True,
                'curve_smoothing': 'cartoon_adaptive'
            })

        # Adaptacja dla line art
        if line_art_score > 0.6:
            base_params.update({
                'max_colors': min(40, int(20 + line_art_score * 25)),
                'tolerance_factor': 0.2,  # Bardzo mała tolerancja dla precyzji
                'min_region_size': max(20, int(80 - line_art_score * 40)),
                'line_art_optimization': True,
                'ultra_precision_mode': True,
                'curve_smoothing': 'line_art_adaptive'
            })

        # Adaptacja bazująca na complexity_score
        if complexity_score > 0.8:  # ULTRA COMPLEX
            base_params.update({
                'max_colors': min(45, base_params['max_colors'] + 10),
                'tolerance_factor': max(0.15, base_params['tolerance_factor'] - 0.1),
                'min_region_size': max(10, base_params['min_region_size'] - 20)
            })
        elif complexity_score > 0.6:  # HIGH COMPLEX
            base_params.update({
                'max_colors': min(40, base_params['max_colors'] + 5),
                'tolerance_factor': max(0.2, base_params['tolerance_factor'] - 0.05)
            })
        elif complexity_score < 0.3:  # SIMPLE
            base_params.update({
                'max_colors': max(15, base_params['max_colors'] - 10),
                'tolerance_factor': min(0.6, base_params['tolerance_factor'] + 0.2),
                'min_region_size': min(200, base_params['min_region_size'] + 100)
            })

        # Adaptacja bazująca na edge_density
        if edge_density > 0.3:
            base_params['edge_enhancement'] = True
            base_params['curve_smoothing'] = 'edge_preserving'

        # Adaptacja bazująca na color_complexity
        if color_complexity > 1000:
            base_params['max_colors'] = min(50, base_params['max_colors'] + 8)
            base_params['advanced_color_analysis'] = True

        # Adaptacja bazująca na detail_density
        if detail_density > 0.6:
            base_params['micro_detail_preservation'] = True
            base_params['min_region_size'] = max(5, base_params['min_region_size'] - 30)

        return base_params
    except:
        return get_fallback_premium_parameters()

def get_fallback_premium_parameters():
    """Fallback premium parameters"""
    return {
        'max_colors': 35,
        'tolerance_factor': 0.3,
        'detail_preservation': 'ultra_high',
        'min_region_size': 50,
        'color_flattening': False,
        'quality_enhancement': 'ai_ultra',
        'curve_smoothing': 'ai_adaptive',
        'edge_enhancement': True,
        'micro_detail_preservation': True,
        'gradient_preservation': True,
        'ultra_precision_mode': True,
        'advanced_color_analysis': True,
        'focus_main_areas': False,
        'cartoon_optimization': False,
        'line_art_optimization': False
    }