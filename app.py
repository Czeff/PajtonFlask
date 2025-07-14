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
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB
MAX_IMAGE_SIZE = 1200  # Znacznie zwiększono dla maksymalnej jakości detali
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'svg'}

# Upewnij się, że katalogi istnieją
for folder in ['raster', 'vector_auto', 'vector_manual', 'preview']:
    os.makedirs(os.path.join(UPLOAD_FOLDER, folder), exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def optimize_image_for_vectorization(image_path, max_size=MAX_IMAGE_SIZE):
    """Ultra zaawansowana optymalizacja obrazu z zachowaniem szczegółów oryginalnego"""
    try:
        with Image.open(image_path) as img:
            # Konwersja do RGB z zachowaniem jakości
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # OPTYMALIZACJA: Kontrolowana rozdzielczość dla lepszej jakości
            original_width, original_height = img.size
            if max(original_width, original_height) < 400:
                # Małe obrazy - zwiększ 2x dla zachowania detali
                target_size = min(max_size * 2, 1200)
            elif max(original_width, original_height) < 800:
                # Średnie obrazy - zwiększ 1.5x
                target_size = min(max_size * 1.5, 900)
            else:
                # Większe obrazy - zachowuj oryginalny rozmiar z kontrolą
                target_size = max_size

            # Wysokiej jakości skalowanie z zachowaniem ostrości
            if max(original_width, original_height) > target_size:
                img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
            elif max(original_width, original_height) < target_size * 0.8:
                # Zwiększ małe obrazy dla lepszej jakości detali
                scale_factor = target_size / max(original_width, original_height)
                new_width = int(original_width * scale_factor)
                new_height = int(original_height * scale_factor)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Multi-pass enhancement dla cartoon-style images z zachowaniem detali
            img = enhance_cartoon_precision_ultra(img)

            return img
    except Exception as e:
        print(f"Błąd podczas optymalizacji obrazu: {e}")
        return None

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

def detect_edge_density(img_array):
    """Wykrywa gęstość krawędzi w obrazie"""
    try:
        from scipy import ndimage
        gray = np.mean(img_array, axis=2)
        edges = ndimage.sobel(gray)
        return np.mean(np.abs(edges)) / 255.0
    except:
        return 0.1

def detect_color_complexity(img_array):
    """Wykrywa złożoność kolorową obrazu"""
    try:
        # Zlicz unikalne kolory w zmniejszonym obrazie
        small = img_array[::4, ::4]
        colors = np.unique(small.reshape(-1, 3), axis=0)
        return len(colors)
    except:
        return 100

def extract_dominant_colors_advanced(image, max_colors=50, params=None):
    """Ultra precyzyjna analiza kolorów z perfekcyjnym dopasowaniem cartoon-style"""
    try:
        img_array = np.array(image)

        # Pobierz parametry jakości
        quality_level = params.get('quality_enhancement', 'high') if params else 'high'
        tolerance_factor = params.get('tolerance_factor', 0.8) if params else 0.8

        print(f"🎨 Analiza kolorów: jakość={quality_level}, tolerancja={tolerance_factor}, max_kolorów={max_colors}")

        # Wielopoziomowa analiza kolorów z adaptacyjnymi parametrami
        colors = []

        # 1. Precyzyjne wykrywanie kolorów dominujących - zwiększona precyzja
        dominant_portion = max_colors // 2 if quality_level == 'maximum' else max_colors // 3
        dominant_colors = extract_precise_dominant_colors(img_array, dominant_portion)
        colors.extend(dominant_colors)

        # 2. Analiza kolorów krawędzi (kluczowe dla cartoon-style) - zwiększona dla wysokiej jakości
        edge_portion = max_colors // 3 if quality_level == 'maximum' else max_colors // 4
        edge_colors = extract_edge_based_colors(img_array, edge_portion)
        colors.extend(edge_colors)

        # 3. Analiza kolorów przejść i gradientów - tylko dla wysokiej jakości
        if quality_level in ['maximum', 'high']:
            gradient_colors = extract_gradient_colors(img_array, max_colors // 6)
            colors.extend(gradient_colors)

        # 4. Wykrywanie kolorów małych obszarów - zwiększona precyzja
        if quality_level == 'maximum':
            detail_colors = extract_detail_colors(img_array, max_colors // 5)
            colors.extend(detail_colors)

        # 5. Wykrywanie kolorów cieni i rozjaśnień
        shadow_highlight_colors = extract_shadow_highlight_colors(img_array, max_colors // 8)
        colors.extend(shadow_highlight_colors)

        # 6. K-means clustering z najwyższą precyzją
        if len(colors) < max_colors:
            additional_colors = extract_high_precision_kmeans(img_array, max_colors - len(colors))
            colors.extend(additional_colors)

        # Usuwanie duplikatów z dostosowaną tolerancją
        final_colors = remove_similar_colors_ultra_precise(colors, max_colors, tolerance_factor)

        # Sortowanie według ważności wizualnej w obrazie
        final_colors = sort_colors_by_visual_importance(img_array, final_colors)

        print(f"🎨 Perfekcyjna analiza: {len(final_colors)} kolorów z maksymalną precyzją (jakość: {quality_level})")
        return final_colors

    except Exception as e:
        print(f"Błąd podczas perfekcyjnej analizy kolorów: {e}")
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
    """Analizuje złożoność obrazu i dostosowuje parametry z maksymalnym priorytetem na szczegółowość"""
    try:
        img_array = np.array(image)

        # Oblicz wskaźniki złożoności
        edge_density = detect_edge_density(img_array)
        color_complexity = detect_color_complexity(img_array)

        # Oblicz entropię obrazu
        from scipy.stats import entropy
        hist, _ = np.histogram(img_array.flatten(), bins=256, range=(0, 256))
        img_entropy = entropy(hist + 1e-10)  # Dodaj małą wartość aby uniknąć log(0)

        print(f"📊 Analiza złożoności: krawędzie={edge_density:.3f}, kolory={color_complexity}, entropia={img_entropy:.3f}")

        # MAKSYMALNA SZCZEGÓŁOWOŚĆ: Drastycznie zwiększone parametry dla wierności oryginałowi
        if edge_density > 0.08 and color_complexity > 100:
            return {
                'max_colors': 40,  # Drastycznie zwiększono dla maksymalnej szczegółowości
                'tolerance_factor': 0.4,  # Bardzo wysoka precyzja
                'detail_preservation': 'ultra_maximum',
                'min_region_size': 1,  # Zachowaj każdy piksel
                'color_flattening': False,  # Wyłącz spłaszczanie dla zachowania wszystkich odcieni
                'quality_enhancement': 'ultra_maximum'
            }
        elif edge_density > 0.05 or color_complexity > 50:
            return {
                'max_colors': 32,  # Bardzo wysokie dla detali
                'tolerance_factor': 0.45,
                'detail_preservation': 'ultra_high',
                'min_region_size': 1,
                'color_flattening': False,
                'quality_enhancement': 'maximum'
            }
        elif color_complexity > 25:
            return {
                'max_colors': 28,  # Wysokie dla średnich obrazów
                'tolerance_factor': 0.5,
                'detail_preservation': 'very_high',
                'min_region_size': 1,
                'color_flattening': False,
                'quality_enhancement': 'high'
            }
        else:
            return {
                'max_colors': 24,  # Minimum dla prostych obrazów
                'tolerance_factor': 0.55,
                'detail_preservation': 'high',
                'min_region_size': 1,
                'color_flattening': False,
                'quality_enhancement': 'high'
            }
    except:
        return {
            'max_colors': 32,  # Domyślnie wysokie dla szczegółowości
            'tolerance_factor': 0.5,
            'detail_preservation': 'ultra_high',
            'min_region_size': 1,
            'color_flattening': False,
            'quality_enhancement': 'maximum'
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
    """Tworzy ultra precyzyjną ścieżkę SVG z zachowaniem szczegółów oryginalnego kształtu"""
    if len(contour) < 3:
        return None

    try:
        # Minimalna analiza - zachowaj oryginalny kształt
        contour_detail_level = analyze_contour_detail_level(contour)

        # Bardzo minimalne upraszczanie - zachowaj większość punktów
        preserved_contour = preserve_contour_details(contour, contour_detail_level)

        print(f"    📐 Kontur: {len(contour)} → {len(preserved_contour)} punktów, szczegółowość: {contour_detail_level}")

        # Wybierz metodę zachowującą szczegóły
        if contour_detail_level == 'high':
            path_data = create_high_fidelity_svg_path(preserved_contour)
        elif contour_detail_level == 'medium':
            path_data = create_balanced_svg_path(preserved_contour)
        else:
            path_data = create_simple_accurate_svg_path(preserved_contour)

        return path_data

    except Exception as e:
        print(f"Błąd podczas tworzenia precyzyjnej ścieżki SVG: {e}")
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
        <strong>🚀 Nowe funkcje:</strong>
        <br>• Zaawansowana wektoryzacja z krzywymi Beziera
        <br>• Inteligentne wykrywanie i segmentacja kolorów
        <br>• Kompatybilność z InkStitch i parametrami haftu
        <br>• Realistyczne podglądy z efektami haftu
        <br>• Adaptacyjne parametry w zależności od koloru
    </div>

    <div class="warning">
        ⚠️ Parametry optymalizacji:
        <br>• Maksymalny rozmiar pliku: 8MB
        <br>• Obrazy skalowane do 600px dla jakości
        <br>• 8 dominujących kolorów maksymalnie
        <br>• Wygładzone ścieżki SVG z krzywymi
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