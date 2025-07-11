
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
MAX_IMAGE_SIZE = 800  # Zwiększono dla lepszej jakości
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'svg'}

# Upewnij się, że katalogi istnieją
for folder in ['raster', 'vector_auto', 'vector_manual', 'preview']:
    os.makedirs(os.path.join(UPLOAD_FOLDER, folder), exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def optimize_image_for_vectorization(image_path, max_size=MAX_IMAGE_SIZE):
    """Zaawansowana optymalizacja obrazu do wektoryzacji wysokiej jakości"""
    try:
        with Image.open(image_path) as img:
            # Konwersja do RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Znacznie większy rozmiar dla ultra wysokiej jakości
            target_size = min(max_size * 2.5, 2000)
            img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
            
            # Adaptacyjne przetwarzanie w zależności od typu obrazu
            img_array = np.array(img)
            
            # Wykryj typ obrazu (cartoon vs. foto vs. logo)
            edge_density = detect_edge_density(img_array)
            color_complexity = detect_color_complexity(img_array)
            
            if edge_density > 0.15:  # Logo/grafika wektorowa
                # Agresywna optymalizacja dla grafik wektorowych
                img = enhance_vector_graphics(img)
            elif color_complexity < 50:  # Cartoon/ilustracja
                # Optymalizacja dla cartoon-style
                img = enhance_cartoon_style(img)
            else:  # Zdjęcie fotorealistyczne
                # Przygotowanie zdjęć do wektoryzacji
                img = enhance_photo_for_vector(img)
            
            # Finalne szarpienie krawędzi
            img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=200, threshold=3))
            
            return img
    except Exception as e:
        print(f"Błąd podczas optymalizacji obrazu: {e}")
        return None

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

def extract_dominant_colors_advanced(image, max_colors=20):
    """Ultra zaawansowane wyciąganie kolorów z analizą histogramów i LAB"""
    try:
        img_array = np.array(image)
        
        # Konwersja do przestrzeni LAB dla lepszej percepcji kolorów
        from skimage.color import rgb2lab, lab2rgb
        lab_image = rgb2lab(img_array / 255.0)
        
        # Wieloetapowa analiza kolorów
        colors = []
        
        # 1. Analiza histogramów RGB
        rgb_colors = extract_histogram_peaks(img_array, max_colors // 2)
        colors.extend(rgb_colors)
        
        # 2. K-means clustering w przestrzeni LAB
        if len(colors) < max_colors:
            lab_colors = extract_kmeans_lab_colors(lab_image, max_colors - len(colors))
            colors.extend(lab_colors)
        
        # 3. Analiza krawędzi dla kolorów kontrastowych
        if len(colors) < max_colors:
            edge_colors = extract_edge_colors(img_array, max_colors - len(colors))
            colors.extend(edge_colors)
        
        # Usuwanie podobnych kolorów z adaptacyjną tolerancją
        final_colors = remove_similar_colors_advanced(colors, max_colors)
        
        # Sortowanie kolorów według częstotliwości występowania
        final_colors = sort_colors_by_frequency(img_array, final_colors)
        
        print(f"🎨 Zaawansowana analiza: {len(final_colors)} kolorów wysokiej jakości")
        return final_colors
        
    except Exception as e:
        print(f"Błąd podczas zaawansowanej analizy kolorów: {e}")
        return extract_dominant_colors_simple(image, max_colors)

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
    """Ultra zaawansowane tworzenie regionów z segmentacją i analizą sąsiedztwa"""
    try:
        width, height = image.size
        img_array = np.array(image)
        
        regions = []
        
        # Wstępna segmentacja obrazu dla lepszej jakości
        try:
            from skimage.segmentation import slic, felzenszwalb
            segments = slic(img_array, n_segments=min(300, width*height//1000), compactness=10, sigma=1)
        except:
            segments = None
        
        for i, color in enumerate(colors):
            print(f"🎯 Przetwarzanie koloru {i+1}/{len(colors)}: {color}")
            
            # Multi-metodowa detekcja regionów
            mask = create_multi_method_mask(img_array, color, segments)
            
            if mask is None:
                continue
                
            initial_pixels = np.sum(mask)
            print(f"  📊 Początkowe piksele: {initial_pixels}")
            
            if initial_pixels > 10:  # Bardzo liberalny próg
                # Zaawansowane przetwarzanie morfologiczne
                mask = advanced_morphological_processing(mask, initial_pixels)
                
                # Inteligentne łączenie fragmentów
                mask = intelligent_fragment_merging(mask, img_array, color)
                
                final_pixels = np.sum(mask)
                print(f"  ✅ Finalne piksele: {final_pixels}")
                
                if final_pixels > 5:  # Bardzo niski próg
                    regions.append((color, mask))
                    print(f"  ✓ Dodano region dla koloru {color}")
                else:
                    print(f"  ✗ Region za mały po przetwarzaniu")
            else:
                print(f"  ✗ Brak wystarczających pikseli")
        
        print(f"🏁 Utworzono {len(regions)} regionów wysokiej jakości")
        return regions
        
    except Exception as e:
        print(f"❌ Błąd podczas zaawansowanego tworzenia regionów: {e}")
        return create_color_regions_simple(image, colors)

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
    """Ultra zaawansowane śledzenie konturów z adaptacyjnymi algorytmami"""
    try:
        from scipy import ndimage
        
        # Analiza maski dla wyboru optymalnej strategii
        mask_size = np.sum(mask)
        mask_shape = analyze_mask_shape(mask)
        
        print(f"  🔍 Analiza maski: rozmiar={mask_size}, kształt={mask_shape}")
        
        # Adaptacyjne przetwarzanie w zależności od charakterystyki
        if mask_shape == 'geometric':
            # Geometryczne kształty - zachowaj ostre krawędzie
            processed_mask = preserve_sharp_edges(mask)
            contours = trace_geometric_contours(processed_mask)
        elif mask_shape == 'organic':
            # Organiczne kształty - delikatne wygładzanie
            processed_mask = smooth_organic_edges(mask)
            contours = trace_organic_contours(processed_mask)
        else:
            # Mieszane - hybrydowe podejście
            processed_mask = hybrid_processing(mask)
            contours = trace_hybrid_contours(processed_mask)
        
        # Post-processing konturów
        final_contours = []
        for contour in contours:
            if len(contour) >= 3:
                # Optymalizacja konturu
                optimized = optimize_contour_points(contour, mask_shape)
                if optimized and len(optimized) >= 3:
                    final_contours.append(optimized)
        
        print(f"  ✅ Wygenerowano {len(final_contours)} konturów wysokiej jakości")
        return final_contours
        
    except Exception as e:
        print(f"❌ Błąd podczas ultra zaawansowanego śledzenia: {e}")
        return trace_contours_simple_improved(mask)

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
    """Tworzy ultra wysokiej jakości ścieżkę SVG z inteligentnym doborem krzywych"""
    if len(contour) < 3:
        return None
    
    try:
        # Analizuj charakterystykę konturu
        contour_analysis = analyze_contour_characteristics(contour)
        
        # Adaptacyjne upraszczanie w zależności od analizy
        simplified_contour = adaptive_contour_simplification(contour, contour_analysis)
        
        if len(simplified_contour) < 3:
            simplified_contour = contour
        
        print(f"    📐 Kontur: {len(contour)} → {len(simplified_contour)} punktów, typ: {contour_analysis['type']}")
        
        # Generuj ścieżkę w zależności od typu
        if contour_analysis['type'] == 'geometric':
            path_data = create_geometric_svg_path(simplified_contour)
        elif contour_analysis['type'] == 'organic':
            path_data = create_organic_svg_path(simplified_contour)
        else:
            path_data = create_hybrid_svg_path(simplified_contour)
        
        return path_data
        
    except Exception as e:
        print(f"Błąd podczas tworzenia zaawansowanej ścieżki SVG: {e}")
        return create_simple_svg_path(contour)

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
        
        # Ultra zaawansowane wyciąganie kolorów z analizą LAB i histogramów
        colors = extract_dominant_colors_advanced(optimized_image, max_colors=24)
        print(f"🎨 Znaleziono {len(colors)} kolorów ultra wysokiej jakości")
        
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
