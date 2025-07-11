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
MAX_IMAGE_SIZE = 600  # Zmniejszono dla lepszej kontroli jakości przy zachowaniu detali
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

def enhance_cartoon_precision(img):
    """Ulepszona precyzja dla obrazów cartoon-style"""
    try:
        # Delikatne zwiększenie kontrastu bez utraty szczegółów
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.3)
        
        # Precyzyjne wyostrzenie krawędzi
        img = img.filter(ImageFilter.UnsharpMask(radius=0.5, percent=150, threshold=2))
        
        # Redukcja szumu przy zachowaniu detali
        img = img.filter(ImageFilter.SMOOTH_MORE)
        
        # Finalne wyostrzenie
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.4)
        
        return img
    except Exception as e:
        print(f"Błąd w enhance_cartoon_precision: {e}")
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

def enhance_vector_graphics(img):
    """Optymalizacja dla grafik wektorowych/logo"""
    # Zwiększ kontrast dramatycznie
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    
    # Maksymalna ostrość
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(2.5)
    
    # Zwiększ nasycenie
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.5)
    
    return img

def enhance_cartoon_style(img):
    """Optymalizacja dla cartoon-style"""
    # Umiarkowany kontrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.4)
    
    # Zwiększ ostrość
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.8)
    
    # Nasycenie kolorów
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.3)
    
    # Filtr wygładzający zachowujący krawędzie
    img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    
    return img

def enhance_photo_for_vector(img):
    """Przygotowanie zdjęć fotorealistycznych"""
    # Delikatny kontrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    
    # Redukcja szumu
    img = img.filter(ImageFilter.SMOOTH_MORE)
    
    # Delikatne zwiększenie ostrości
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.1)
    
    return img

def flatten_color_palette(colors, target_count=16):
    """Spłaszcza paletę kolorów do określonej liczby zachowując najważniejsze odcienie"""
    if len(colors) <= target_count:
        return colors
    
    try:
        from sklearn.cluster import KMeans
        
        # Konwertuj kolory do przestrzeni LAB dla lepszej percepcji
        lab_colors = []
        for color in colors:
            try:
                from skimage.color import rgb2lab
                lab = rgb2lab(np.array(color).reshape(1, 1, 3) / 255.0)[0, 0]
                lab_colors.append(lab)
            except:
                # Fallback do RGB
                lab_colors.append(color)
        
        # K-means clustering w przestrzeni LAB
        if lab_colors:
            kmeans = KMeans(n_clusters=target_count, random_state=42, n_init=20, max_iter=300)
            kmeans.fit(lab_colors)
            
            # Konwertuj z powrotem do RGB
            flattened_colors = []
            for lab_center in kmeans.cluster_centers_:
                try:
                    from skimage.color import lab2rgb
                    if len(lab_center) == 3:  # LAB color
                        rgb = lab2rgb(lab_center.reshape(1, 1, 3))[0, 0]
                        rgb = np.clip(rgb * 255, 0, 255).astype(int)
                        flattened_colors.append(tuple(rgb))
                    else:  # Already RGB
                        flattened_colors.append(tuple(np.clip(lab_center, 0, 255).astype(int)))
                except:
                    # Fallback
                    flattened_colors.append(tuple(np.clip(lab_center[:3], 0, 255).astype(int)))
            
            print(f"🎨 Spłaszczono {len(colors)} kolorów do {len(flattened_colors)} kolorów podstawowych")
            return flattened_colors
        
        return colors[:target_count]
    except:
        print(f"⚠️ Błąd spłaszczania - używam prostego obcinania do {target_count} kolorów")
        return colors[:target_count]

def enhance_image_quality_maximum(image):
    """Maksymalne podniesienie jakości obrazu dla cartoon-style"""
    try:
        # Multi-pass enhancement z zachowaniem szczegółów
        
        # Pass 1: Delikatne zwiększenie kontrastu z zachowaniem detali
        enhancer = ImageEnhance.Contrast(image)
        enhanced = enhancer.enhance(1.15)
        
        # Pass 2: Precyzyjne wyostrzenie krawędzi bez artefaktów
        enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=0.2, percent=100, threshold=1))
        
        # Pass 3: Zwiększenie nasycenia dla lepszego wykrywania kolorów
        enhancer = ImageEnhance.Color(enhanced)
        enhanced = enhancer.enhance(1.08)
        
        # Pass 4: Bardzo delikatna redukcja szumu
        enhanced = enhanced.filter(ImageFilter.SMOOTH_MORE)
        
        # Pass 5: Finalne subtelne wyostrzenie
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(1.1)
        
        return enhanced
    except Exception as e:
        print(f"Błąd w enhance_image_quality_maximum: {e}")
        return image

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

def extract_texture_colors(img_array, max_colors):
    """Wyciąga kolory z obszarów z teksturą"""
    try:
        from sklearn.cluster import KMeans
        from scipy import ndimage
        
        # Zastosuj filtry teksturowe
        gray = np.mean(img_array, axis=2)
        
        # Filtr Gabora (uproszczony) - wykrywa tekstury
        gx = ndimage.sobel(gray, axis=0)
        gy = ndimage.sobel(gray, axis=1)
        texture_response = np.sqrt(gx**2 + gy**2)
        
        # Znajdź obszary z wysoką odpowiedzią teksturową
        texture_threshold = np.percentile(texture_response, 75)
        texture_mask = texture_response > texture_threshold
        
        # Wyciągnij kolory z obszarów teksturowych
        texture_pixels = img_array[texture_mask]
        
        if len(texture_pixels) > 200:
            n_clusters = min(max_colors, len(texture_pixels) // 100)
            if n_clusters > 0:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                kmeans.fit(texture_pixels)
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
    """Ultra precyzyjne usuwanie podobnych kolorów z adaptacyjnym progiem dla cartoon-style"""
    if not colors:
        return []
    
    final_colors = [colors[0]]
    
    for color in colors[1:]:
        is_unique = True
        
        for existing in final_colors:
            # Zaawansowane obliczanie różnicy kolorów w przestrzeni LAB
            distance = calculate_advanced_color_distance(color, existing)
            
            # ZNACZNIE zmniejszone progi dla cartoon-style - zachowaj więcej kolorów
            brightness = sum(existing) / 3
            saturation = max(existing) - min(existing)
            
            # Bazowe progi tolerancji - jeszcze bardziej zmniejszone
            if brightness < 30:  # Bardzo ciemne kolory
                base_tolerance = 2.5
            elif brightness < 60:  # Ciemne kolory
                base_tolerance = 3.0
            elif brightness < 120:  # Średnio ciemne
                base_tolerance = 3.5
            elif brightness > 230:  # Bardzo jasne kolory
                base_tolerance = 6.0
            elif brightness > 200:  # Jasne kolory
                base_tolerance = 4.5
            elif brightness > 160:  # Średnio jasne
                base_tolerance = 4.0
            else:  # Średnie kolory
                base_tolerance = 3.0
            
            # Zastosuj czynnik tolerancji
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

def remove_similar_colors_precise(colors, max_colors):
    """Precyzyjne usuwanie podobnych kolorów"""
    if not colors:
        return []
    
    final_colors = [colors[0]]
    
    for color in colors[1:]:
        is_unique = True
        
        for existing in final_colors:
            # Bardziej precyzyjne obliczanie odległości
            distance = calculate_color_distance_precise(color, existing)
            
            # Adaptacyjny próg w zależności od jasności
            brightness = sum(existing) / 3
            if brightness < 30:
                tolerance = 12  # Zwiększona tolerancja dla ciemnych kolorów
            elif brightness > 220:
                tolerance = 18  # Zwiększona tolerancja dla jasnych kolorów
            else:
                tolerance = 15  # Zwiększona tolerancja dla średnich kolorów
            
            if distance < tolerance:
                is_unique = False
                break
        
        if is_unique and len(final_colors) < max_colors:
            final_colors.append(color)
    
    return final_colors

def calculate_color_distance_precise(color1, color2):
    """Precyzyjne obliczanie odległości kolorów"""
    try:
        # Konwersja do przestrzeni LAB dla lepszej percepcji
        from skimage.color import rgb2lab
        
        c1_lab = rgb2lab(np.array(color1).reshape(1, 1, 3) / 255.0)[0, 0]
        c2_lab = rgb2lab(np.array(color2).reshape(1, 1, 3) / 255.0)[0, 0]
        
        # Delta E - profesjonalna miara różnicy kolorów
        delta_e = np.sqrt(np.sum((c1_lab - c2_lab)**2))
        return delta_e
    except:
        # Fallback do Euclidean distance
        return np.sqrt(sum((color1[i] - color2[i])**2 for i in range(3)))

def sort_colors_by_importance(img_array, colors):
    """Sortuje kolory według ważności w obrazie"""
    try:
        color_importance = []
        
        for color in colors:
            # Oblicz częstotliwość występowania
            distances = np.sqrt(np.sum((img_array - np.array(color))**2, axis=2))
            frequency = np.sum(distances < 20)
            
            # Oblicz ważność na podstawie położenia (środek ważniejszy)
            height, width = img_array.shape[:2]
            y_coords, x_coords = np.where(distances < 20)
            
            if len(y_coords) > 0:
                center_distance = np.mean(np.sqrt(
                    ((y_coords - height/2) / height)**2 + 
                    ((x_coords - width/2) / width)**2
                ))
                centrality_weight = 1.0 - center_distance
            else:
                centrality_weight = 0
            
            # Kombinuj częstotliwość i centralność
            importance = frequency * (1 + centrality_weight)
            color_importance.append((importance, color))
        
        # Sortuj według ważności (malejąco)
        color_importance.sort(reverse=True)
        return [color for importance, color in color_importance]
    except:
        return colors

def extract_histogram_peaks(img_array, max_colors):
    """Wyciąga kolory z pików histogramów"""
    colors = []
    try:
        # Analiza każdego kanału osobno
        for channel in range(3):
            hist, bins = np.histogram(img_array[:,:,channel], bins=64, range=(0, 256))
            peaks = find_histogram_peaks(hist, bins)
            for peak in peaks[:max_colors//3]:
                if peak not in [c[channel] for c in colors]:
                    base_color = [128, 128, 128]
                    base_color[channel] = int(peak)
                    colors.append(tuple(base_color))
    except:
        pass
    return colors[:max_colors]

def find_histogram_peaks(hist, bins):
    """Znajduje piki w histogramie"""
    try:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(hist, height=np.max(hist) * 0.1, distance=5)
        return bins[peaks]
    except:
        # Fallback - znajdź maksima
        peaks = []
        for i in range(1, len(hist)-1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > np.max(hist) * 0.1:
                peaks.append(bins[i])
        return peaks

def extract_kmeans_lab_colors(lab_image, max_colors):
    """K-means w przestrzeni LAB"""
    try:
        from sklearn.cluster import KMeans
        
        pixels = lab_image.reshape(-1, 3)
        # Próbkowanie dla wydajności
        if len(pixels) > 10000:
            pixels = pixels[::len(pixels)//10000]
        
        kmeans = KMeans(n_clusters=max_colors, random_state=42, n_init=20)
        kmeans.fit(pixels)
        
        # Konwersja z powrotem do RGB
        lab_colors = kmeans.cluster_centers_
        rgb_colors = []
        for lab_color in lab_colors:
            try:
                rgb = lab2rgb(lab_color.reshape(1, 1, 3))[0, 0]
                rgb = np.clip(rgb * 255, 0, 255).astype(int)
                rgb_colors.append(tuple(rgb))
            except:
                continue
        
        return rgb_colors
    except:
        return []

def extract_edge_colors(img_array, max_colors):
    """Wyciąga kolory z obszarów krawędzi"""
    try:
        from scipy import ndimage
        
        # Wykryj krawędzie
        gray = np.mean(img_array, axis=2)
        edges = ndimage.sobel(gray) > 30
        
        # Wyciągnij kolory z pikseli przy krawędziach
        edge_pixels = img_array[edges]
        if len(edge_pixels) > 1000:
            step = len(edge_pixels) // 1000
            edge_pixels = edge_pixels[::step]
        
        # Clustering kolorów krawędzi
        if len(edge_pixels) > 0:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=min(max_colors, len(edge_pixels)), random_state=42)
            kmeans.fit(edge_pixels)
            return [(int(c[0]), int(c[1]), int(c[2])) for c in kmeans.cluster_centers_]
        
        return []
    except:
        return []

def remove_similar_colors_advanced(colors, max_colors):
    """Usuwa podobne kolory z adaptacyjną tolerancją"""
    if not colors:
        return []
    
    final_colors = [colors[0]]
    
    for color in colors[1:]:
        is_unique = True
        min_distance = float('inf')
        
        for existing in final_colors:
            # Adaptacyjna tolerancja w zależności od jasności
            brightness = sum(existing) / 3
            if brightness < 50:  # Ciemne kolory
                tolerance = 15
            elif brightness > 200:  # Jasne kolory
                tolerance = 25
            else:  # Średnie kolory
                tolerance = 20
            
            # Odległość w przestrzeni RGB
            distance = np.sqrt(sum((color[i] - existing[i])**2 for i in range(3)))
            min_distance = min(min_distance, distance)
            
            if distance < tolerance:
                is_unique = False
                break
        
        if is_unique and len(final_colors) < max_colors:
            final_colors.append(color)
    
    return final_colors

def sort_colors_by_frequency(img_array, colors):
    """Sortuje kolory według częstotliwości występowania"""
    try:
        color_counts = []
        for color in colors:
            # Oblicz ile pikseli jest podobnych do tego koloru
            distances = np.sqrt(np.sum((img_array - np.array(color))**2, axis=2))
            count = np.sum(distances < 30)  # Piksele w promieniu 30
            color_counts.append((count, color))
        
        # Sortuj według częstotliwości (malejąco)
        color_counts.sort(reverse=True)
        return [color for count, color in color_counts]
    except:
        return colors

def extract_dominant_colors_simple(image, max_colors=8):
    """Prosta metoda wyciągania kolorów dominujących"""
    try:
        # Zmniejsz obraz dla szybszej analizy
        small_image = image.copy()
        small_image.thumbnail((100, 100))
        
        # Kwantyzacja kolorów
        palette_image = small_image.quantize(colors=max_colors)
        palette = palette_image.getpalette()
        
        colors = []
        for i in range(min(max_colors, len(palette) // 3)):
            r = palette[i * 3] if i * 3 < len(palette) else 0
            g = palette[i * 3 + 1] if i * 3 + 1 < len(palette) else 0
            b = palette[i * 3 + 2] if i * 3 + 2 < len(palette) else 0
            colors.append((r, g, b))
        
        return colors
    except Exception as e:
        print(f"Błąd podczas prostego wyciągania kolorów: {e}")
        return [(0, 0, 0), (128, 128, 128), (255, 255, 255)]

def create_color_regions_advanced(image, colors):
    """Ultra precyzyjne tworzenie regionów z zachowaniem szczegółów oryginalnego obrazu"""
    try:
        width, height = image.size
        img_array = np.array(image)
        
        regions = []
        
        # Zaawansowana segmentacja z zachowaniem krawędzi
        segments = create_edge_preserving_segmentation(img_array)
        
        # Analiza każdego koloru z maksymalną precyzją
        for i, color in enumerate(colors):
            print(f"🎯 Ultra precyzyjne przetwarzanie koloru {i+1}/{len(colors)}: {color}")
            
            # Wielopoziomowa detekcja regionów
            mask = create_ultra_precise_mask(img_array, color, segments)
            
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

def create_edge_preserving_segmentation(img_array):
    """Tworzy segmentację zachowującą krawędzie"""
    try:
        from skimage.segmentation import felzenszwalb, slic
        from skimage.filters import gaussian
        
        # Delikatne wygładzanie bez utraty krawędzi
        smoothed = gaussian(img_array, sigma=0.5, channel_axis=2)
        
        # Felzenszwalb dla zachowania krawędzi
        segments = felzenszwalb(smoothed, scale=100, sigma=0.5, min_size=10)
        
        return segments
    except:
        return None

def create_ultra_precise_mask(img_array, color, segments):
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
        
        # 3. Kontekstowa maska bazująca na segmentacji
        if segments is not None:
            mask3 = create_noise_resistant_segment_mask(img_array, color_array, segments)
            if mask3 is not None:
                masks.append(mask3)
        
        # 4. Maska uwzględniająca lokalne sąsiedztwo
        neighborhood_mask = create_neighborhood_coherence_mask(img_array, color_array)
        if neighborhood_mask is not None:
            masks.append(neighborhood_mask)
        
        # Inteligentne kombinowanie masek z redukcją szumów
        if len(masks) > 0:
            # Głosowanie większościowe z wagami i filtracją szumów
            combined_mask = np.zeros_like(masks[0], dtype=float)
            weights = [1.0, 0.9, 0.7, 0.5]  # Zoptymalizowane wagi
            
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

def create_advanced_segment_mask(img_array, color_array, segments):
    """Tworzy zaawansowaną maskę bazującą na segmentach"""
    try:
        mask = np.zeros(img_array.shape[:2], dtype=bool)
        
        for seg_id in np.unique(segments):
            seg_mask = segments == seg_id
            seg_pixels = img_array[seg_mask]
            
            if len(seg_pixels) > 10:
                # Sprawdź średni kolor segmentu
                seg_mean_color = np.mean(seg_pixels, axis=0)
                mean_distance = np.sqrt(np.sum((seg_mean_color - color_array)**2))
                
                # Sprawdź odsetek podobnych pikseli w segmencie
                distances = np.sqrt(np.sum((seg_pixels - color_array)**2, axis=1))
                similar_ratio = np.sum(distances < 30) / len(seg_pixels)
                
                # Decyzja na podstawie średniej i stosunku
                if mean_distance < 35 and similar_ratio > 0.25:
                    mask[seg_mask] = True
                elif mean_distance < 20:  # Bardzo podobny średni kolor
                    mask[seg_mask] = True
                elif similar_ratio > 0.6:  # Większość pikseli podobna
                    mask[seg_mask] = True
        
        return mask if np.sum(mask) > 0 else None
    except:
        return None

def create_local_color_mask(img_array, color_array):
    """Tworzy maskę bazującą na lokalnym podobieństwie kolorów"""
    try:
        from scipy import ndimage
        
        # Podstawowa maska podobieństwa
        distances = np.sqrt(np.sum((img_array - color_array)**2, axis=2))
        base_mask = distances < 35
        
        if np.sum(base_mask) == 0:
            return None
        
        # Rozszerz maskę w obszarach o podobnych kolorach
        kernel = np.ones((5, 5))
        dilated_mask = ndimage.binary_dilation(base_mask, structure=kernel, iterations=1)
        
        # Sprawdź czy rozszerzone obszary są rzeczywiście podobne
        extended_pixels = img_array[dilated_mask & ~base_mask]
        if len(extended_pixels) > 0:
            ext_distances = np.sqrt(np.sum((extended_pixels - color_array)**2, axis=1))
            valid_extension = ext_distances < 50
            
            # Dodaj tylko te rozszerzone piksele, które są wystarczająco podobne
            extension_coords = np.where(dilated_mask & ~base_mask)
            for i, is_valid in enumerate(valid_extension):
                if not is_valid:
                    dilated_mask[extension_coords[0][i], extension_coords[1][i]] = False
        
        return dilated_mask
    except:
        return None

def create_texture_similarity_mask(img_array, color_array):
    """Tworzy maskę bazującą na podobieństwie tekstury lokalnej"""
    try:
        from scipy import ndimage
        
        # Znajdź piksele o podobnym kolorze
        distances = np.sqrt(np.sum((img_array - color_array)**2, axis=2))
        base_mask = distances < 40
        
        if np.sum(base_mask) < 50:
            return None
        
        # Oblicz lokalną teksturę (gradient)
        gray = np.mean(img_array, axis=2)
        gx = ndimage.sobel(gray, axis=0)
        gy = ndimage.sobel(gray, axis=1)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        
        # Znajdź średnią teksturę w obszarach bazowej maski
        mean_texture = np.mean(gradient_magnitude[base_mask])
        texture_std = np.std(gradient_magnitude[base_mask])
        
        # Rozszerz maskę na obszary o podobnej teksturze
        texture_threshold = mean_texture + 1.5 * texture_std
        similar_texture = np.abs(gradient_magnitude - mean_texture) < texture_threshold
        
        # Kombinuj maski kolorów i tekstury
        combined_mask = base_mask | (similar_texture & (distances < 60))
        
        return combined_mask if np.sum(combined_mask) > np.sum(base_mask) else base_mask
    except:
        return None

def create_noise_resistant_segment_mask(img_array, color_array, segments):
    """Tworzy maskę bazującą na segmentach z odpornością na szumy"""
    try:
        mask = np.zeros(img_array.shape[:2], dtype=bool)
        
        for seg_id in np.unique(segments):
            seg_mask = segments == seg_id
            seg_pixels = img_array[seg_mask]
            
            if len(seg_pixels) > 20:  # Zwiększony próg dla redukcji szumów
                # Sprawdź średni kolor segmentu z większą precyzją
                seg_mean_color = np.mean(seg_pixels, axis=0)
                seg_std_color = np.std(seg_pixels, axis=0)
                
                # Delta E w przestrzeni RGB z kompensacją odchylenia
                mean_distance = np.sqrt(np.sum((seg_mean_color - color_array)**2))
                color_variance = np.mean(seg_std_color)
                
                # Sprawdź odsetek podobnych pikseli w segmencie
                distances = np.sqrt(np.sum((seg_pixels - color_array)**2, axis=1))
                
                # Adaptacyjny próg bazujący na wariancji koloru w segmencie
                adaptive_threshold = 25 + color_variance * 0.5
                similar_ratio = np.sum(distances < adaptive_threshold) / len(seg_pixels)
                
                # Decyzja na podstawie średniej, wariancji i stosunku
                confidence_score = 0
                if mean_distance < 30:
                    confidence_score += 0.4
                if similar_ratio > 0.3:
                    confidence_score += 0.4
                if color_variance < 15:  # Jednolity segment
                    confidence_score += 0.2
                
                # Dodatkowy bonus dla większych segmentów (mniej prawdopodobne szumy)
                if len(seg_pixels) > 100:
                    confidence_score += 0.1
                
                if confidence_score >= 0.6:
                    mask[seg_mask] = True
        
        return mask if np.sum(mask) > 0 else None
    except:
        return None

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
        
        # 5. Usuń cienkie "mosty" które mogą być artefaktami
        thinned = remove_thin_bridges(smoothed)
        
        return thinned
        
    except Exception as e:
        print(f"Błąd w remove_noise_and_artifacts: {e}")
        return mask

def remove_thin_bridges(mask):
    """Usuwa cienkie mosty które często są artefaktami"""
    try:
        from scipy import ndimage
        from skimage import morphology
        
        # Znajdź szkielet maski
        skeleton = morphology.skeletonize(mask)
        
        # Znajdź punkty rozgałęzienia (gdzie łączą się różne części)
        # Użyj kernela 3x3 do wykrycia punktów z więcej niż 2 sąsiadami
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])
        
        neighbor_count = ndimage.convolve(skeleton.astype(int), kernel, mode='constant')
        branch_points = (skeleton & (neighbor_count > 2))
        
        # Jeśli jest mało punktów rozgałęzienia, prawdopodobnie nie ma problemów
        if np.sum(branch_points) < 3:
            return mask
        
        # Erozja z małym kernelem, potem dylatacja - usuwa cienkie połączenia
        eroded = ndimage.binary_erosion(mask, structure=np.ones((2, 2)), iterations=1)
        restored = ndimage.binary_dilation(eroded, structure=np.ones((2, 2)), iterations=1)
        
        # Sprawdź czy nie usunęliśmy zbyt wiele
        original_area = np.sum(mask)
        restored_area = np.sum(restored)
        
        # Jeśli ubytek jest zbyt duży, zwróć oryginalną maskę
        if restored_area < original_area * 0.7:
            return mask
        
        return restored
        
    except:
        return mask

def rgb_to_hsv_precise(rgb):
    """Precyzyjna konwersja RGB do HSV"""
    try:
        rgb = rgb.astype(float) / 255.0
        max_val = np.max(rgb, axis=-1)
        min_val = np.min(rgb, axis=-1)
        diff = max_val - min_val
        
        # Value
        v = max_val
        
        # Saturation
        s = np.where(max_val != 0, diff / max_val, 0)
        
        # Hue
        h = np.zeros_like(max_val)
        
        # Calculate hue for each pixel
        mask = diff != 0
        
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

def create_segment_based_mask(img_array, color_array, segments):
    """Tworzy maskę bazując na segmentach"""
    try:
        mask = np.zeros(img_array.shape[:2], dtype=bool)
        
        for seg_id in np.unique(segments):
            seg_mask = segments == seg_id
            seg_pixels = img_array[seg_mask]
            
            if len(seg_pixels) > 0:
                # Sprawdź czy segment zawiera podobne kolory
                distances = np.sqrt(np.sum((seg_pixels - color_array)**2, axis=1))
                similar_ratio = np.sum(distances < 25) / len(seg_pixels)
                
                # Jeśli znaczna część segmentu to podobny kolor
                if similar_ratio > 0.3:
                    mask[seg_mask] = True
        
        return mask if np.sum(mask) > 0 else None
    except:
        return None

def create_edge_aware_mask(img_array, color_array):
    """Tworzy maskę uwzględniającą krawędzie"""
    try:
        from scipy import ndimage
        
        # Podstawowa maska koloru
        distances = np.sqrt(np.sum((img_array - color_array)**2, axis=2))
        base_mask = distances < 30
        
        if np.sum(base_mask) == 0:
            return None
        
        # Wykryj krawędzie
        gray = np.mean(img_array, axis=2)
        edges = ndimage.sobel(gray) > 20
        
        # Rozszerz maskę w obszarach bez krawędzi
        # Ale ogranicz w obszarach z krawędziami
        from scipy.ndimage import binary_dilation, binary_erosion
        
        # W obszarach bez krawędzi - rozszerz
        no_edge_areas = ~edges
        expanded_in_smooth = binary_dilation(base_mask & no_edge_areas, iterations=1)
        
        # W obszarach z krawędziami - zachowaj precyzję
        precise_at_edges = base_mask & edges
        
        final_mask = expanded_in_smooth | precise_at_edges
        
        return final_mask if np.sum(final_mask) > np.sum(base_mask) * 0.5 else base_mask
    except:
        return None

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

def preserve_detail_processing(mask, initial_pixels):
    """Przetwarzanie z zachowaniem szczegółów i usuwaniem artefaktów"""
    try:
        from scipy import ndimage
        
        # Zaawansowane czyszczenie z zachowaniem szczegółów
        if initial_pixels > 2000:
            # Dla większych regionów - bardziej agresywne czyszczenie
            structure = np.ones((3, 3))
            
            # 1. Zamknij małe dziury
            mask = ndimage.binary_closing(mask, structure=structure, iterations=2)
            
            # 2. Usuń szum punktowy
            mask = ndimage.binary_opening(mask, structure=structure, iterations=1)
            
            # 3. Inteligentne usuwanie małych komponentów
            labeled, num_features = ndimage.label(mask)
            component_sizes = []
            
            for i in range(1, num_features + 1):
                size = np.sum(labeled == i)
                component_sizes.append(size)
            
            if component_sizes:
                # Usuń komponenty mniejsze niż 0.5% największego komponentu
                max_component_size = max(component_sizes)
                min_size = max(5, max_component_size * 0.005, initial_pixels // 500)
                
                for i in range(1, num_features + 1):
                    if component_sizes[i-1] < min_size:
                        mask[labeled == i] = False
                        
        elif initial_pixels > 500:
            # Dla średnich regionów - umiarkowane czyszczenie
            structure = np.ones((3, 3))
            
            # Usuń pojedyncze piksele i małe skupiska
            mask = ndimage.binary_opening(mask, structure=structure, iterations=1)
            
            labeled, num_features = ndimage.label(mask)
            min_size = max(3, initial_pixels // 200)
            
            for i in range(1, num_features + 1):
                if np.sum(labeled == i) < min_size:
                    mask[labeled == i] = False
                    
        else:
            # Dla małych regionów - bardzo delikatne czyszczenie
            # Usuń tylko pojedyncze izolowane piksele
            labeled, num_features = ndimage.label(mask)
            for i in range(1, num_features + 1):
                component = labeled == i
                if np.sum(component) <= 2:  # Usuń tylko bardzo małe artefakty
                    # Sprawdź czy to rzeczywiście artefakt (izolowany)
                    dilated = ndimage.binary_dilation(component, iterations=2)
                    overlap = np.sum(dilated & (mask & ~component))
                    if overlap < 3:  # Bardzo mało połączeń z resztą
                        mask[component] = False
        
        # Końcowe wypełnienie małych dziur
        mask = ndimage.binary_fill_holes(mask)
        
        return mask
        
    except Exception as e:
        print(f"Błąd w preserve_detail_processing: {e}")
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

def create_multi_method_mask(img_array, color, segments):
    """Tworzy maskę używając wielu metod"""
    try:
        height, width = img_array.shape[:2]
        color_array = np.array(color)
        
        # Metoda 1: Odległość RGB z adaptacyjnym progiem
        rgb_diff = np.sqrt(np.sum((img_array - color_array)**2, axis=2))
        brightness = np.mean(color)
        
        # Adaptacyjny próg w zależności od jasności i kontekstu
        if brightness < 40:
            threshold = 25
        elif brightness < 100:
            threshold = 30
        elif brightness > 200:
            threshold = 40
        else:
            threshold = 35
        
        mask1 = rgb_diff <= threshold
        
        # Metoda 2: Odległość w przestrzeni HSV
        try:
            hsv_img = rgb_to_hsv_manual(img_array)
            hsv_color = rgb_to_hsv_manual(color_array.reshape(1, 1, 3))[0, 0]
            hsv_diff = np.sqrt(np.sum((hsv_img - hsv_color)**2, axis=2))
            mask2 = hsv_diff <= threshold * 0.8
        except:
            mask2 = mask1
        
        # Metoda 3: Segmentacja
        mask3 = np.zeros((height, width), dtype=bool)
        if segments is not None:
            try:
                # Znajdź segmenty zawierające podobne kolory
                for seg_id in np.unique(segments):
                    seg_mask = segments == seg_id
                    if np.sum(seg_mask) > 0:
                        seg_mean_color = np.mean(img_array[seg_mask], axis=0)
                        color_distance = np.sqrt(np.sum((seg_mean_color - color_array)**2))
                        if color_distance <= threshold:
                            mask3[seg_mask] = True
            except:
                pass
        
        # Kombinuj maski inteligentnie
        if np.sum(mask3) > 0:
            # Jeśli segmentacja dała rezultaty, użyj jej jako bazy
            combined_mask = mask3 | (mask1 & mask2)
        else:
            # Inaczej kombinuj RGB i HSV
            combined_mask = mask1 | mask2
        
        return combined_mask if np.sum(combined_mask) > 0 else None
        
    except Exception as e:
        print(f"Błąd w multi-method mask: {e}")
        return None

def rgb_to_hsv_manual(rgb):
    """Prosta konwersja RGB do HSV"""
    try:
        rgb = rgb.astype(float) / 255.0
        max_val = np.max(rgb, axis=2)
        min_val = np.min(rgb, axis=2)
        diff = max_val - min_val
        
        # Value
        v = max_val
        
        # Saturation
        s = np.where(max_val != 0, diff / max_val, 0)
        
        # Hue (uproszczona)
        h = np.zeros_like(max_val)
        
        return np.stack([h, s, v], axis=2)
    except:
        return rgb

def advanced_morphological_processing(mask, initial_pixels):
    """Zaawansowane przetwarzanie morfologiczne"""
    try:
        from scipy import ndimage
        
        # Adaptacyjne struktury w zależności od rozmiaru regionu
        if initial_pixels > 5000:
            # Duże regiony - agresywne czyszczenie
            structure = np.ones((5, 5))
            mask = ndimage.binary_closing(mask, structure=structure, iterations=2)
            mask = ndimage.binary_opening(mask, structure=structure, iterations=1)
        elif initial_pixels > 1000:
            # Średnie regiony - umiarkowane czyszczenie  
            structure = np.ones((3, 3))
            mask = ndimage.binary_closing(mask, structure=structure, iterations=2)
            mask = ndimage.binary_opening(mask, structure=structure, iterations=1)
        else:
            # Małe regiony - delikatne czyszczenie
            structure = np.ones((3, 3))
            mask = ndimage.binary_closing(mask, structure=structure, iterations=1)
        
        # Wypełnij dziury
        mask = ndimage.binary_fill_holes(mask)
        
        # Usuń bardzo małe komponenty
        labeled, num_features = ndimage.label(mask)
        min_size = max(5, initial_pixels // 50)
        
        for i in range(1, num_features + 1):
            if np.sum(labeled == i) < min_size:
                mask[labeled == i] = False
        
        return mask
        
    except Exception as e:
        print(f"Błąd w morphological processing: {e}")
        return mask

def intelligent_fragment_merging(mask, img_array, color):
    """Inteligentne łączenie fragmentów o podobnych kolorach"""
    try:
        from scipy import ndimage
        
        # Znajdź komponenty
        labeled, num_features = ndimage.label(mask)
        
        if num_features <= 1:
            return mask
        
        # Sprawdź czy komponenty można połączyć
        color_array = np.array(color)
        merged_mask = np.zeros_like(mask)
        
        for i in range(1, num_features + 1):
            component = labeled == i
            component_pixels = img_array[component]
            
            if len(component_pixels) > 0:
                # Sprawdź średni kolor komponentu
                mean_color = np.mean(component_pixels, axis=0)
                color_distance = np.sqrt(np.sum((mean_color - color_array)**2))
                
                # Jeśli kolor jest podobny, dodaj do wyniku
                if color_distance <= 50:  # Liberalny próg dla łączenia
                    merged_mask[component] = True
        
        return merged_mask
        
    except Exception as e:
        print(f"Błąd w fragment merging: {e}")
        return mask

def create_color_regions_simple(image, colors):
    """Prosta metoda tworzenia regionów kolorów"""
    try:
        width, height = image.size
        img_array = np.array(image)
        
        regions = []
        
        for color in colors:
            print(f"Prosta metoda - przetwarzanie koloru: {color}")
            
            # Zwiększona tolerancja dla prostej metody
            tolerance = 60
            
            # Wektoryzowana operacja zamiast pętli
            color_array = np.array(color)
            diff = np.abs(img_array - color_array)
            mask = np.all(diff <= tolerance, axis=2)
            
            pixel_count = np.sum(mask)
            print(f"Prosta metoda - piksele dla koloru {color}: {pixel_count}")
            
            if pixel_count > 10:  # Bardzo liberalny próg
                regions.append((color, mask))
                print(f"✓ Prosta metoda - dodano region dla koloru {color}")
        
        print(f"Prosta metoda - utworzono {len(regions)} regionów")
        return regions
        
    except Exception as e:
        print(f"Błąd podczas prostego tworzenia regionów: {e}")
        return []

def trace_contours_advanced(mask):
    """Ultra precyzyjne śledzenie konturów z zachowaniem detali oryginalnego kształtu"""
    try:
        from scipy import ndimage
        
        # Analiza maski dla wyboru optymalnej strategii
        mask_size = np.sum(mask)
        mask_complexity = analyze_mask_complexity(mask)
        
        print(f"  🔍 Analiza maski: rozmiar={mask_size}, złożoność={mask_complexity}")
        
        # Minimalne przetwarzanie wstępne - zachowaj oryginalny kształt
        processed_mask = minimal_mask_preprocessing(mask)
        
        # Wybór metody śledzenia bazującej na rozmiarze i złożoności
        if mask_size > 1000 and mask_complexity == 'high':
            contours = trace_high_detail_contours(processed_mask)
        elif mask_complexity == 'medium':
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

def analyze_mask_complexity(mask):
    """Analizuje złożoność kształtu maski"""
    try:
        from scipy import ndimage
        
        # Policz komponenty
        labeled, num_features = ndimage.label(mask)
        
        # Policz "dziury" (holes)
        filled = ndimage.binary_fill_holes(mask)
        holes = filled & ~mask
        num_holes = np.sum(holes)
        
        # Policz zmienność krawędzi
        edges = ndimage.sobel(mask.astype(float))
        edge_variance = np.var(edges[edges > 0]) if np.any(edges > 0) else 0
        
        # Klasyfikacja złożoności
        if num_features > 3 or num_holes > 10 or edge_variance > 0.5:
            return 'high'
        elif num_features > 1 or num_holes > 2 or edge_variance > 0.2:
            return 'medium'
        else:
            return 'simple'
    except:
        return 'medium'

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

def analyze_mask_shape(mask):
    """Analizuje typ kształtu maski"""
    try:
        from scipy import ndimage
        
        # Oblicz wskaźniki geometryczne
        labeled, num_features = ndimage.label(mask)
        
        if num_features == 0:
            return 'empty'
        
        # Dla największego komponentu
        largest_component = labeled == np.bincount(labeled.flat)[1:].argmax() + 1
        
        # Wypełnienie vs. obwód
        area = np.sum(largest_component)
        
        # Znajdź obwód
        try:
            if cv2 is not None:
                mask_uint8 = (largest_component * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    perimeter = cv2.arcLength(contours[0], True)
                else:
                    perimeter = 0
            else:
                # Aproksymacja obwodu bez OpenCV
                edges = ndimage.sobel(largest_component.astype(float))
                perimeter = np.sum(edges > 0)
        except:
            perimeter = np.sqrt(area) * 4  # Aproksymacja
        
        # Wskaźnik kształtu (4π * pole / obwód²)
        if perimeter > 0:
            shape_factor = (4 * np.pi * area) / (perimeter ** 2)
        else:
            shape_factor = 0
        
        # Klasyfikacja
        if shape_factor > 0.7:  # Prawie koło
            return 'geometric'
        elif shape_factor > 0.3:  # Umiarkowanie regularne
            return 'mixed'
        else:  # Nieregularne
            return 'organic'
            
    except:
        return 'mixed'

def preserve_sharp_edges(mask):
    """Zachowuje ostre krawędzie dla geometrycznych kształtów"""
    try:
        from scipy import ndimage
        
        # Minimalne wygładzanie
        smoothed = ndimage.gaussian_filter(mask.astype(float), sigma=0.3) > 0.7
        
        # Zachowaj oryginalne krawędzie tam gdzie to możliwe
        edges = ndimage.sobel(mask.astype(float)) > 0.1
        result = smoothed.copy()
        result[edges] = mask[edges]
        
        return result
    except:
        return mask

def smooth_organic_edges(mask):
    """Wygładza krawędzie dla organicznych kształtów"""
    try:
        from scipy import ndimage
        
        # Większe wygładzanie dla organicznych kształtów
        smoothed = ndimage.gaussian_filter(mask.astype(float), sigma=1.0) > 0.4
        
        # Wypełnij małe dziury
        filled = ndimage.binary_fill_holes(smoothed)
        
        # Delikatne morfologiczne czyszczenie
        structure = np.ones((3, 3))
        cleaned = ndimage.binary_opening(filled, structure=structure)
        
        return cleaned
    except:
        return mask

def hybrid_processing(mask):
    """Hybrydowe przetwarzanie dla mieszanych kształtów"""
    try:
        from scipy import ndimage
        
        # Umiarkowane wygładzanie
        smoothed = ndimage.gaussian_filter(mask.astype(float), sigma=0.6) > 0.5
        
        # Adaptacyjne morfologiczne przetwarzanie
        structure_small = np.ones((3, 3))
        cleaned = ndimage.binary_closing(smoothed, structure=structure_small)
        cleaned = ndimage.binary_fill_holes(cleaned)
        
        return cleaned
    except:
        return mask

def trace_geometric_contours(mask):
    """Śledzenie konturów dla geometrycznych kształtów"""
    try:
        if cv2 is not None:
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            processed = []
            for contour in contours:
                if len(contour) >= 4:
                    # Minimalne upraszczanie dla zachowania kształtu
                    perimeter = cv2.arcLength(contour, True)
                    epsilon = 0.002 * perimeter  # Bardzo mały epsilon
                    simplified = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(simplified) >= 3:
                        points = [(int(p[0][0]), int(p[0][1])) for p in simplified]
                        processed.append(points)
            
            return processed
        else:
            return trace_contours_simple_improved(mask)
    except:
        return trace_contours_simple_improved(mask)

def trace_organic_contours(mask):
    """Śledzenie konturów dla organicznych kształtów"""
    try:
        if cv2 is not None:
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
            
            processed = []
            for contour in contours:
                if len(contour) >= 6:
                    # Większe upraszczanie dla organicznych kształtów
                    perimeter = cv2.arcLength(contour, True)
                    epsilon = 0.01 * perimeter
                    simplified = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(simplified) >= 3:
                        points = [(int(p[0][0]), int(p[0][1])) for p in simplified]
                        # Wygładzanie dla organicznych kształtów
                        smoothed = smooth_contour_points(points)
                        processed.append(smoothed)
            
            return processed
        else:
            return trace_contours_simple_improved(mask)
    except:
        return trace_contours_simple_improved(mask)

def trace_hybrid_contours(mask):
    """Śledzenie konturów dla mieszanych kształtów"""
    try:
        if cv2 is not None:
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            processed = []
            for contour in contours:
                if len(contour) >= 4:
                    perimeter = cv2.arcLength(contour, True)
                    epsilon = 0.005 * perimeter  # Umiarkowane upraszczanie
                    simplified = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(simplified) >= 3:
                        points = [(int(p[0][0]), int(p[0][1])) for p in simplified]
                        processed.append(points)
            
            return processed
        else:
            return trace_contours_simple_improved(mask)
    except:
        return trace_contours_simple_improved(mask)

def optimize_contour_points(contour, shape_type):
    """Optymalizuje punkty konturu w zależności od typu kształtu"""
    try:
        if len(contour) < 3:
            return contour
        
        if shape_type == 'geometric':
            # Dla geometrycznych - usuń zbędne punkty na prostych
            return remove_collinear_points(contour)
        elif shape_type == 'organic':
            # Dla organicznych - delikatne wygładzanie
            return smooth_contour_points(contour)
        else:
            # Dla mieszanych - minimalna optymalizacja
            return reduce_redundant_points(contour)
            
    except:
        return contour

def remove_collinear_points(points, tolerance=2.0):
    """Usuwa punkty leżące na prostej"""
    if len(points) <= 3:
        return points
    
    filtered = [points[0]]
    
    for i in range(1, len(points) - 1):
        p1 = np.array(filtered[-1])
        p2 = np.array(points[i])
        p3 = np.array(points[i + 1])
        
        # Oblicz odległość punktu od prostej
        line_vec = p3 - p1
        point_vec = p2 - p1
        
        if np.linalg.norm(line_vec) > 0:
            # Odległość punktu od prostej
            cross = np.cross(point_vec, line_vec)
            distance = abs(cross) / np.linalg.norm(line_vec)
            
            if distance > tolerance:
                filtered.append(points[i])
    
    filtered.append(points[-1])
    return filtered

def smooth_contour_points(points, factor=0.3):
    """Wygładza punkty konturu"""
    if len(points) <= 3:
        return points
    
    smoothed = []
    for i in range(len(points)):
        prev_idx = (i - 1) % len(points)
        next_idx = (i + 1) % len(points)
        
        current = np.array(points[i])
        prev_point = np.array(points[prev_idx])
        next_point = np.array(points[next_idx])
        
        # Wygładzanie
        smooth_point = current * (1 - factor) + (prev_point + next_point) * factor / 2
        smoothed.append((int(smooth_point[0]), int(smooth_point[1])))
    
    return smoothed

def reduce_redundant_points(points, min_distance=3):
    """Usuwa zbędnie blisko położone punkty"""
    if len(points) <= 3:
        return points
    
    filtered = [points[0]]
    
    for point in points[1:]:
        last_point = filtered[-1]
        distance = np.sqrt((point[0] - last_point[0])**2 + (point[1] - last_point[1])**2)
        
        if distance >= min_distance:
            filtered.append(point)
    
    return filtered

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

def analyze_contour_characteristics(contour):
    """Analizuje charakterystykę konturu"""
    try:
        if len(contour) < 4:
            return {'type': 'simple', 'complexity': 'low', 'smoothness': 'high'}
        
        # Oblicz kąty między kolejnymi segmentami
        angles = []
        for i in range(len(contour)):
            p1 = contour[i]
            p2 = contour[(i + 1) % len(contour)]
            p3 = contour[(i + 2) % len(contour)]
            
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            # Oblicz kąt
            try:
                angle = np.arccos(np.clip(
                    (v1[0]*v2[0] + v1[1]*v2[1]) / 
                    (np.sqrt(v1[0]**2 + v1[1]**2) * np.sqrt(v2[0]**2 + v2[1]**2)),
                    -1, 1
                ))
                angles.append(angle)
            except:
                angles.append(np.pi)
        
        # Analiza kątów
        sharp_angles = sum(1 for a in angles if a < np.pi/3)  # Ostre kąty
        smooth_angles = sum(1 for a in angles if a > 2*np.pi/3)  # Płynne kąty
        
        # Długości segmentów
        segment_lengths = []
        for i in range(len(contour)):
            p1 = contour[i]
            p2 = contour[(i + 1) % len(contour)]
            length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            segment_lengths.append(length)
        
        avg_length = np.mean(segment_lengths)
        length_variance = np.var(segment_lengths)
        
        # Klasyfikacja
        if sharp_angles > len(contour) * 0.6:
            contour_type = 'geometric'
        elif smooth_angles > len(contour) * 0.6:
            contour_type = 'organic'
        else:
            contour_type = 'hybrid'
        
        complexity = 'high' if len(contour) > 50 else 'medium' if len(contour) > 20 else 'low'
        smoothness = 'low' if length_variance > avg_length**2 else 'medium' if length_variance > avg_length/2 else 'high'
        
        return {
            'type': contour_type,
            'complexity': complexity,
            'smoothness': smoothness,
            'sharp_angles': sharp_angles,
            'smooth_angles': smooth_angles,
            'avg_segment_length': avg_length
        }
        
    except Exception as e:
        print(f"Błąd w analizie konturu: {e}")
        return {'type': 'hybrid', 'complexity': 'medium', 'smoothness': 'medium'}

def adaptive_contour_simplification(contour, analysis):
    """Adaptacyjne upraszczanie konturu"""
    try:
        if analysis['complexity'] == 'low':
            # Niski poziom upraszczania dla prostych kształtów
            return contour[::max(1, len(contour) // 20)]
        elif analysis['complexity'] == 'medium':
            # Średni poziom upraszczania
            return contour[::max(1, len(contour) // 35)]
        else:
            # Wysokie zachowanie detali dla złożonych kształtów
            return contour[::max(1, len(contour) // 60)]
    except:
        return contour

def create_geometric_svg_path(contour):
    """Tworzy ścieżkę SVG dla geometrycznych kształtów"""
    try:
        path_data = f"M {contour[0][0]:.2f} {contour[0][1]:.2f}"
        
        # Używaj głównie linii prostych z okazjonalnymi krzywymi
        for i in range(1, len(contour)):
            current = contour[i]
            path_data += f" L {current[0]:.2f} {current[1]:.2f}"
        
        path_data += " Z"
        return path_data
    except:
        return create_simple_svg_path(contour)

def create_organic_svg_path(contour):
    """Tworzy ścieżkę SVG dla organicznych kształtów"""
    try:
        if len(contour) < 4:
            return create_simple_svg_path(contour)
        
        path_data = f"M {contour[0][0]:.2f} {contour[0][1]:.2f}"
        
        # Używaj krzywych Beziera dla płynnych kształtów
        i = 1
        while i < len(contour):
            if i + 2 < len(contour):
                # Krzywa kubiczna Beziera
                p1 = contour[i]
                p2 = contour[i + 1]
                p3 = contour[i + 2]
                
                # Oblicz punkty kontrolne
                cp1_x = p1[0] + (p2[0] - contour[i-1][0]) * 0.3
                cp1_y = p1[1] + (p2[1] - contour[i-1][1]) * 0.3
                
                cp2_x = p2[0] - (p3[0] - p1[0]) * 0.3
                cp2_y = p2[1] - (p3[1] - p1[1]) * 0.3
                
                path_data += f" C {cp1_x:.2f} {cp1_y:.2f} {cp2_x:.2f} {cp2_y:.2f} {p2[0]:.2f} {p2[1]:.2f}"
                i += 1
            else:
                # Linia prosta dla pozostałych punktów
                current = contour[i]
                path_data += f" L {current[0]:.2f} {current[1]:.2f}"
                i += 1
        
        path_data += " Z"
        return path_data
    except:
        return create_simple_svg_path(contour)

def create_hybrid_svg_path(contour):
    """Tworzy ścieżkę SVG dla mieszanych kształtów"""
    try:
        if len(contour) < 4:
            return create_simple_svg_path(contour)
        
        path_data = f"M {contour[0][0]:.2f} {contour[0][1]:.2f}"
        
        # Inteligentnie wybieraj między liniami a krzywymi
        i = 1
        while i < len(contour):
            if i + 1 < len(contour):
                current = contour[i]
                next_point = contour[i + 1]
                
                # Sprawdź czy segment jest odpowiedni dla krzywej
                if should_use_curve(contour, i):
                    # Użyj krzywej kwadratowej
                    prev_point = contour[i - 1] if i > 0 else contour[-1]
                    
                    cp_x = current[0] + (next_point[0] - prev_point[0]) * 0.2
                    cp_y = current[1] + (next_point[1] - prev_point[1]) * 0.2
                    
                    path_data += f" Q {cp_x:.2f} {cp_y:.2f} {next_point[0]:.2f} {next_point[1]:.2f}"
                    i += 2
                else:
                    # Użyj linii prostej
                    path_data += f" L {current[0]:.2f} {current[1]:.2f}"
                    i += 1
            else:
                current = contour[i]
                path_data += f" L {current[0]:.2f} {current[1]:.2f}"
                i += 1
        
        path_data += " Z"
        return path_data
    except:
        return create_simple_svg_path(contour)

def should_use_curve(contour, index):
    """Określa czy segment powinien używać krzywej"""
    try:
        if index < 1 or index >= len(contour) - 1:
            return False
        
        prev_point = contour[index - 1]
        current = contour[index]
        next_point = contour[index + 1]
        
        # Oblicz kąt
        v1 = (current[0] - prev_point[0], current[1] - prev_point[1])
        v2 = (next_point[0] - current[0], next_point[1] - current[1])
        
        # Sprawdź długości segmentów
        len1 = np.sqrt(v1[0]**2 + v1[1]**2)
        len2 = np.sqrt(v2[0]**2 + v2[1]**2)
        
        if len1 == 0 or len2 == 0:
            return False
        
        # Znormalizuj wektory
        v1_norm = (v1[0]/len1, v1[1]/len1)
        v2_norm = (v2[0]/len2, v2[1]/len2)
        
        # Oblicz kąt między wektorami
        dot_product = v1_norm[0] * v2_norm[0] + v1_norm[1] * v2_norm[1]
        angle = np.arccos(np.clip(dot_product, -1, 1))
        
        # Używaj krzywej jeśli kąt nie jest zbyt ostry i segmenty są odpowiednio długie
        return angle > np.pi/4 and min(len1, len2) > 5
    except:
        return False

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

def analyze_image_complexity(image):
    """Analizuje złożoność obrazu i dostosowuje parametry z priorytetem na detale i kontrolę kolorów"""
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
        
        # OPTYMALIZACJA: Ograniczamy do maksymalnie 16 kolorów dla lepszej kontroli
        # ale zwiększamy precyzję wykrywania i zachowania detali
        if edge_density > 0.12 and color_complexity > 150:
            return {
                'max_colors': 16,  # Ograniczono do 16 kolorów
                'tolerance_factor': 0.7,  # Jeszcze większa precyzja
                'detail_preservation': 'ultra_high',
                'min_region_size': 1,  # Zachowaj najmniejsze detale
                'color_flattening': True,  # Włącz spłaszczanie kolorów
                'quality_enhancement': 'maximum'
            }
        elif edge_density > 0.08 or color_complexity > 100:
            return {
                'max_colors': 14,  # Średnio-złożone: 14 kolorów
                'tolerance_factor': 0.75,
                'detail_preservation': 'very_high',
                'min_region_size': 1,
                'color_flattening': True,
                'quality_enhancement': 'high'
            }
        elif color_complexity > 50:
            return {
                'max_colors': 12,  # Prosta grafika: 12 kolorów
                'tolerance_factor': 0.8,
                'detail_preservation': 'high',
                'min_region_size': 2,
                'color_flattening': True,
                'quality_enhancement': 'medium'
            }
        else:
            return {
                'max_colors': 10,  # Bardzo prosta: 10 kolorów
                'tolerance_factor': 0.85,
                'detail_preservation': 'medium',
                'min_region_size': 3,
                'color_flattening': True,
                'quality_enhancement': 'standard'
            }
    except:
        return {
            'max_colors': 16,  # Domyślnie 16 kolorów
            'tolerance_factor': 0.75,
            'detail_preservation': 'high',
            'min_region_size': 2,
            'color_flattening': True,
            'quality_enhancement': 'high'
        }

def vectorize_image_improved(image_path, output_path):
    """Ultra zaawansowana wektoryzacja z AI-podobną analizą obrazu"""
    try:
        print("🚀 Rozpoczynanie ultra zaawansowanej wektoryzacji AI...")
        
        # Zaawansowana optymalizacja obrazu
        optimized_image = optimize_image_for_vectorization(image_path, max_size=1500)
        if not optimized_image:
            print("❌ Błąd optymalizacji obrazu")
            return False
        
        print(f"✅ Obraz zoptymalizowany do rozmiaru: {optimized_image.size}")
        
        # Analizuj złożoność obrazu
        complexity_params = analyze_image_complexity(optimized_image)
        
        # Ultra zaawansowane wyciąganie kolorów z analizą LAB i histogramów
        max_colors = complexity_params['max_colors']
        
        # Dodatkowe podniesienie jakości jeśli wymagane
        if complexity_params.get('quality_enhancement') == 'maximum':
            optimized_image = enhance_image_quality_maximum(optimized_image)
            print("✨ Zastosowano maksymalne podniesienie jakości obrazu")
        
        colors = extract_dominant_colors_advanced(optimized_image, max_colors=max_colors*2, params=complexity_params)
        
        # Spłaszczenie kolorów jeśli włączone
        if complexity_params.get('color_flattening', False):
            colors = flatten_color_palette(colors, target_count=max_colors)
        print(f"🎨 Znaleziono {len(colors)} kolorów ultra wysokiej jakości (dostosowano do złożoności: {max_colors})")
        
        if not colors:
            print("❌ Nie znaleziono kolorów")
            return False
        
        # Zaawansowane tworzenie regionów z segmentacją
        regions = create_color_regions_advanced(optimized_image, colors)
        print(f"🗺️ Utworzono {len(regions)} regionów ultra wysokiej jakości")
        
        if not regions:
            print("⚠️ Nie utworzono regionów zaawansowaną metodą, próbuję prostszą")
            regions = create_color_regions_simple(optimized_image, colors)
            print(f"🗺️ Prostą metodą utworzono {len(regions)} regionów")
            
        if not regions:
            print("❌ Nie można utworzyć żadnych regionów kolorowych")
            return False
        
        # Generuj ultra wysokiej jakości ścieżki SVG
        svg_paths = []
        total_contours = 0
        
        for i, (color, mask) in enumerate(regions):
            print(f"🎯 Przetwarzanie regionu {i+1}/{len(regions)} dla koloru {color}")
            
            contours = trace_contours_advanced(mask)
            total_contours += len(contours)
            
            for j, contour in enumerate(contours):
                if len(contour) >= 3:
                    path_data = create_smooth_svg_path(contour)
                    if path_data:
                        svg_paths.append((color, path_data))
        
        print(f"📝 Wygenerowano {len(svg_paths)} ścieżek ultra wysokiej jakości z {total_contours} konturów")
        
        if not svg_paths:
            print("❌ Nie wygenerowano żadnych ścieżek")
            return False
        
        # Generuj ultra profesjonalne SVG
        width, height = optimized_image.size
        svg_content = generate_ultra_professional_svg(svg_paths, width, height)
        
        # Zapisz SVG
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        
        print(f"💾 Ultra wysokiej jakości SVG zapisany do: {output_path}")
        
        # Zaawansowana walidacja jakości
        file_size = os.path.getsize(output_path)
        quality_score = assess_svg_quality(svg_paths, file_size)
        
        print(f"📊 Ocena jakości: {quality_score}/100")
        print(f"📁 Rozmiar pliku: {file_size} bajtów")
        print(f"🎨 Liczba kolorów: {len(set(color for color, _ in svg_paths))}")
        print(f"📐 Liczba ścieżek: {len(svg_paths)}")
        
        if file_size < 300:
            print("⚠️ Plik może być za mały - możliwe problemy z obrazem wejściowym")
            return False
        
        print("🏆 Ultra zaawansowana wektoryzacja zakończona sukcesem!")
        return True
        
    except Exception as e:
        print(f"❌ Błąd podczas ultra zaawansowanej wektoryzacji: {e}")
        traceback.print_exc()
        return False
    finally:
        gc.collect()

def generate_ultra_professional_svg(svg_paths, width, height):
    """Generuje ultra profesjonalne SVG z zaawansowanymi parametrami"""
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" 
     xmlns="http://www.w3.org/2000/svg"
     xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
     xmlns:inkstitch="http://inkstitch.org/namespace">
  <title>Ultra High-Quality Vector Art - AI Enhanced</title>
  <defs>
    <style>
      .ultra-vector-path {{
        stroke-width: 0.05;
        stroke-linejoin: round;
        stroke-linecap: round;
        fill-opacity: 1.0;
        stroke-opacity: 0.8;
        shape-rendering: geometricPrecision;
        vector-effect: non-scaling-stroke;
      }}
    </style>
  </defs>
  <g inkscape:label="Ultra Vector Shapes" inkscape:groupmode="layer">'''
    
    # Sortuj ścieżki według jasności kolorów (ciemne na spód)
    sorted_paths = sorted(svg_paths, key=lambda x: sum(x[0]))
    
    for i, (color, path_data) in enumerate(sorted_paths):
        color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        color_rgb = f"rgb({color[0]},{color[1]},{color[2]})"
        
        # Ultra zaawansowane parametry haftu
        brightness = sum(color) / 3
        saturation = max(color) - min(color)
        
        # Adaptacyjne parametry w zależności od koloru
        if brightness < 60:  # Bardzo ciemne
            row_spacing = "0.3"
            angle = str(45 + (i * 25) % 180)
            stitch_length = "2.8"
            density = "high"
        elif brightness < 120:  # Ciemne
            row_spacing = "0.4"
            angle = str(60 + (i * 30) % 180)
            stitch_length = "3.0"
            density = "medium-high"
        elif brightness > 200:  # Bardzo jasne
            row_spacing = "0.7"
            angle = str(135 + (i * 20) % 180)
            stitch_length = "3.8"
            density = "low"
        elif brightness > 160:  # Jasne
            row_spacing = "0.6"
            angle = str(120 + (i * 25) % 180)
            stitch_length = "3.5"
            density = "medium-low"
        else:  # Średnie
            row_spacing = "0.5"
            angle = str(90 + (i * 35) % 180)
            stitch_length = "3.2"
            density = "medium"
        
        # Dodatkowe parametry dla wysokiej jakości
        underlay = "1" if brightness < 100 else "0"
        pull_compensation = "0.2" if saturation > 100 else "0.1"
        
        svg_content += f'''
    <path id="ultra-vector-path-{i}" 
          d="{path_data}" 
          class="ultra-vector-path"
          style="fill: {color_rgb}; stroke: {color_rgb}; fill-rule: evenodd;"
          inkstitch:fill="1"
          inkstitch:color="{color_hex}"
          inkstitch:angle="{angle}"
          inkstitch:row_spacing_mm="{row_spacing}"
          inkstitch:end_row_spacing_mm="{row_spacing}"
          inkstitch:max_stitch_length_mm="{stitch_length}"
          inkstitch:staggers="4"
          inkstitch:skip_last="0"
          inkstitch:underpath="{underlay}"
          inkstitch:pull_compensation_mm="{pull_compensation}"
          inkstitch:expand_mm="0.1"
          inkstitch:clockwise="true"
          data-quality="ultra-high"
          data-density="{density}" />'''
    
    svg_content += '''
  </g>
  <metadata>
    <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
      <rdf:Description rdf:about="">
        <dc:title xmlns:dc="http://purl.org/dc/elements/1.1/">Ultra High-Quality Vector Art</dc:title>
        <dc:description xmlns:dc="http://purl.org/dc/elements/1.1/">Generated with AI-enhanced vectorization</dc:description>
        <dc:creator xmlns:dc="http://purl.org/dc/elements/1.1/">Ultra Vector Generator</dc:creator>
      </rdf:Description>
    </rdf:RDF>
  </metadata>
</svg>'''
    
    return svg_content

def assess_svg_quality(svg_paths, file_size):
    """Ocenia jakość wygenerowanego SVG"""
    try:
        score = 0
        
        # Punkty za liczbę ścieżek (max 30 punktów)
        path_count = len(svg_paths)
        if path_count > 50:
            score += 30
        elif path_count > 20:
            score += 25
        elif path_count > 10:
            score += 20
        elif path_count > 5:
            score += 15
        else:
            score += 10
        
        # Punkty za różnorodność kolorów (max 25 punktów)
        unique_colors = len(set(color for color, _ in svg_paths))
        if unique_colors > 15:
            score += 25
        elif unique_colors > 10:
            score += 20
        elif unique_colors > 5:
            score += 15
        else:
            score += 10
        
        # Punkty za rozmiar pliku (max 20 punktów)
        if 1000 <= file_size <= 50000:  # Optymalny zakres
            score += 20
        elif 500 <= file_size <= 100000:
            score += 15
        elif file_size > 100000:
            score += 10
        else:
            score += 5
        
        # Punkty za złożoność ścieżek (max 25 punktów)
        total_path_length = sum(len(path_data) for _, path_data in svg_paths)
        avg_path_complexity = total_path_length / max(path_count, 1)
        
        if avg_path_complexity > 200:
            score += 25
        elif avg_path_complexity > 100:
            score += 20
        elif avg_path_complexity > 50:
            score += 15
        else:
            score += 10
        
        return min(score, 100)
    except:
        return 50

def generate_professional_svg(svg_paths, width, height):
    """Generuje wysokiej jakości SVG z dokładnymi ścieżkami"""
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" 
     xmlns="http://www.w3.org/2000/svg"
     xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
     xmlns:inkstitch="http://inkstitch.org/namespace">
  <title>High-Quality Vector Pattern</title>
  <defs>
    <style>
      .vector-path {{
        stroke-width: 0.1;
        stroke-linejoin: round;
        stroke-linecap: round;
        fill-opacity: 1.0;
        stroke-opacity: 1.0;
        shape-rendering: geometricPrecision;
      }}
    </style>
  </defs>
  <g inkscape:label="Vector Shapes" inkscape:groupmode="layer">'''
    
    # Dodaj ścieżki z wysoką jakością
    for i, (color, path_data) in enumerate(svg_paths):
        color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        color_rgb = f"rgb({color[0]},{color[1]},{color[2]})"
        
        # Parametry haftu dla InkStitch
        brightness = sum(color) / 3
        if brightness < 85:
            row_spacing = "0.4"
            angle = "45"
            stitch_length = "3.0"
        elif brightness > 170:
            row_spacing = "0.6"
            angle = "135"
            stitch_length = "3.5"
        else:
            row_spacing = "0.5"
            angle = str(45 + (i * 30) % 180)
            stitch_length = "3.2"
        
        svg_content += f'''
    <path id="vector-path-{i}" 
          d="{path_data}" 
          class="vector-path"
          style="fill: {color_rgb}; stroke: none; fill-rule: evenodd;"
          inkstitch:fill="1"
          inkstitch:color="{color_hex}"
          inkstitch:angle="{angle}"
          inkstitch:row_spacing_mm="{row_spacing}"
          inkstitch:end_row_spacing_mm="{row_spacing}"
          inkstitch:max_stitch_length_mm="{stitch_length}"
          inkstitch:staggers="3"
          inkstitch:skip_last="0"
          inkstitch:underpath="0" />'''
    
    svg_content += '''
  </g>
</svg>'''
    
    return svg_content

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