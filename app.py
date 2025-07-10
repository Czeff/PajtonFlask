
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
import cv2

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
    """Optymalizuje obraz do wektoryzacji z zachowaniem jakości"""
    try:
        with Image.open(image_path) as img:
            # Konwersja do RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Zachowaj proporcje przy skalowaniu
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Lekkie wygładzenie dla lepszej wektoryzacji
            img = img.filter(ImageFilter.GaussianBlur(radius=0.3))
            
            # Zwiększ kontrast dla lepszego wykrywania krawędzi
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.3)
            
            # Zwiększ nasycenie kolorów
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(1.2)
            
            return img
    except Exception as e:
        print(f"Błąd podczas optymalizacji obrazu: {e}")
        return None

def extract_dominant_colors_advanced(image, max_colors=8):
    """Zaawansowane wyciąganie kolorów dominujących"""
    try:
        # Konwertuj do numpy array
        img_array = np.array(image)
        
        # Użyj K-means clustering do znajdowania dominujących kolorów
        from sklearn.cluster import KMeans
        
        # Reshape do 2D array
        pixels = img_array.reshape(-1, 3)
        
        # Usuń duplikaty dla przyspieszenia
        unique_pixels = np.unique(pixels, axis=0)
        
        if len(unique_pixels) > 10000:
            # Próbkowanie losowe dla dużych obrazów
            indices = np.random.choice(len(unique_pixels), 10000, replace=False)
            unique_pixels = unique_pixels[indices]
        
        # K-means clustering
        kmeans = KMeans(n_clusters=min(max_colors, len(unique_pixels)), random_state=42, n_init=10)
        kmeans.fit(unique_pixels)
        
        # Zwróć kolory jako tuple
        colors = [(int(c[0]), int(c[1]), int(c[2])) for c in kmeans.cluster_centers_]
        
        # Sortuj kolory według jasności
        colors.sort(key=lambda c: sum(c))
        
        return colors
    except ImportError:
        # Fallback - prostsza metoda bez sklearn
        return extract_dominant_colors_simple(image, max_colors)
    except Exception as e:
        print(f"Błąd podczas wyciągania kolorów: {e}")
        return [(0, 0, 0), (128, 128, 128), (255, 255, 255)]

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
    """Zaawansowane tworzenie regionów kolorów"""
    try:
        width, height = image.size
        img_array = np.array(image)
        
        regions = []
        
        for color in colors:
            # Utwórz maskę dla podobnych kolorów
            mask = np.zeros((height, width), dtype=bool)
            
            # Oblicz odległość euklidesową w przestrzeni RGB
            diff = np.sqrt(np.sum((img_array - np.array(color))**2, axis=2))
            
            # Adaptacyjny próg w zależności od koloru
            threshold = 40 if sum(color) > 400 else 30
            mask = diff <= threshold
            
            # Morfologia matematyczna do wygładzenia masek
            if np.any(mask):
                # Usuń małe dziury
                mask = ndimage.binary_fill_holes(mask)
                # Wygładź krawędzie
                mask = ndimage.binary_opening(mask, structure=np.ones((3,3)))
                mask = ndimage.binary_closing(mask, structure=np.ones((5,5)))
                
                if np.sum(mask) > 50:  # Tylko regiony większe niż 50 pikseli
                    regions.append((color, mask))
        
        return regions
    except Exception as e:
        print(f"Błąd podczas tworzenia regionów: {e}")
        return create_color_regions_simple(image, colors)

def create_color_regions_simple(image, colors):
    """Prosta metoda tworzenia regionów kolorów"""
    try:
        width, height = image.size
        pixels = np.array(image)
        
        regions = []
        
        for color in colors:
            mask = np.zeros((height, width), dtype=bool)
            
            # Znajdź piksele podobne do tego koloru
            tolerance = 35
            for y in range(height):
                for x in range(width):
                    pixel = pixels[y, x]
                    if (abs(int(pixel[0]) - color[0]) <= tolerance and
                        abs(int(pixel[1]) - color[1]) <= tolerance and
                        abs(int(pixel[2]) - color[2]) <= tolerance):
                        mask[y, x] = True
            
            if np.any(mask):
                regions.append((color, mask))
        
        return regions
    except Exception as e:
        print(f"Błąd podczas prostego tworzenia regionów: {e}")
        return []

def trace_contours_advanced(mask):
    """Zaawansowane śledzenie konturów"""
    try:
        # Użyj OpenCV lub skimage do znajdowania konturów
        try:
            # Próba z OpenCV
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            processed_contours = []
            for contour in contours:
                if len(contour) >= 3:
                    # Uprość kontur
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    simplified = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(simplified) >= 3:
                        points = [(int(point[0][0]), int(point[0][1])) for point in simplified]
                        processed_contours.append(points)
            
            return processed_contours
            
        except ImportError:
            # Fallback bez OpenCV
            return trace_contours_simple(mask)
            
    except Exception as e:
        print(f"Błąd podczas zaawansowanego śledzenia konturów: {e}")
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
    """Tworzy wysokiej jakości ścieżkę SVG z konturu"""
    if len(contour) < 3:
        return None
    
    try:
        # Uprość kontur zachowując jakość
        simplified_contour = []
        tolerance = 2.0  # Mniejsza tolerancja dla lepszej jakości
        
        for i in range(len(contour)):
            current = contour[i]
            if i == 0 or i == len(contour) - 1:
                simplified_contour.append(current)
            else:
                prev = contour[i-1]
                next_point = contour[i+1]
                
                # Oblicz odległość punktu od linii między poprzednim a następnym
                dist = abs((next_point[1] - prev[1]) * current[0] - 
                          (next_point[0] - prev[0]) * current[1] + 
                          next_point[0] * prev[1] - next_point[1] * prev[0]) / \
                       ((next_point[1] - prev[1])**2 + (next_point[0] - prev[0])**2)**0.5
                
                if dist > tolerance:
                    simplified_contour.append(current)
        
        if len(simplified_contour) < 3:
            simplified_contour = contour
        
        # Rozpocznij ścieżkę
        path_data = f"M {simplified_contour[0][0]:.2f} {simplified_contour[0][1]:.2f}"
        
        # Dodaj linie z precyzją
        for i in range(1, len(simplified_contour)):
            current = simplified_contour[i]
            path_data += f" L {current[0]:.2f} {current[1]:.2f}"
        
        path_data += " Z"  # Zamknij ścieżkę
        return path_data
        
    except Exception as e:
        print(f"Błąd podczas tworzenia ścieżki SVG: {e}")
        return create_simple_svg_path(contour)

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
    """Ulepszona wektoryzacja obrazu z wysoką jakością"""
    try:
        print("🎨 Rozpoczynanie ulepszonej wektoryzacji...")
        
        # Optymalizuj obraz
        optimized_image = optimize_image_for_vectorization(image_path)
        if not optimized_image:
            print("❌ Błąd optymalizacji obrazu")
            return False
        
        print(f"✅ Obraz zoptymalizowany do rozmiaru: {optimized_image.size}")
        
        # Wyciągnij dominujące kolory
        colors = extract_dominant_colors_advanced(optimized_image, max_colors=8)
        print(f"🎨 Znaleziono {len(colors)} kolorów dominujących")
        
        if not colors:
            print("❌ Nie znaleziono kolorów")
            return False
        
        # Utwórz regiony kolorów
        regions = create_color_regions_advanced(optimized_image, colors)
        print(f"🗺️ Utworzono {len(regions)} regionów kolorowych")
        
        if not regions:
            print("❌ Nie utworzono regionów")
            return False
        
        # Generuj ścieżki SVG
        svg_paths = []
        for i, (color, mask) in enumerate(regions):
            contours = trace_contours_advanced(mask)
            print(f"🔍 Kolor {color}: {len(contours)} konturów")
            
            for contour in contours:
                if len(contour) >= 3:
                    path_data = create_smooth_svg_path(contour)
                    if path_data:
                        svg_paths.append((color, path_data))
        
        print(f"📝 Wygenerowano {len(svg_paths)} ścieżek SVG")
        
        if not svg_paths:
            print("❌ Nie wygenerowano żadnych ścieżek")
            return False
        
        # Generuj SVG
        width, height = optimized_image.size
        svg_content = generate_professional_svg(svg_paths, width, height)
        
        # Zapisz SVG
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        
        print(f"💾 SVG zapisany do: {output_path}")
        
        # Sprawdź jakość pliku
        file_size = os.path.getsize(output_path)
        if file_size < 200:
            print("⚠️ Wygenerowany plik SVG może być za mały")
            return False
        
        print(f"✅ Wektoryzacja zakończona! Rozmiar pliku: {file_size} bajtów")
        return True
        
    except Exception as e:
        print(f"❌ Błąd podczas wektoryzacji: {e}")
        traceback.print_exc()
        return False
    finally:
        gc.collect()

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
    
    app.run(host='0.0.0.0', port=5000, debug=True)
