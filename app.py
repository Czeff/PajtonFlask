
from flask import Flask, request, jsonify, render_template_string, send_file
from werkzeug.utils import secure_filename
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import os
import time
import math
import json
import io
import traceback
import gc
from collections import defaultdict

app = Flask(__name__)

# Konfiguracja
UPLOAD_FOLDER = 'uploads'
MAX_FILE_SIZE = 8 * 1024 * 1024  # Zmniejszono z 16MB do 8MB
MAX_IMAGE_SIZE = 600  # Zmniejszono z 800 do 600 pikseli
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'svg'}

# Upewnij się, że katalogi istnieją
for folder in ['raster', 'vector_auto', 'vector_manual', 'preview']:
    os.makedirs(os.path.join(UPLOAD_FOLDER, folder), exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def optimize_image_for_vectorization(image_path, max_size=MAX_IMAGE_SIZE):
    """Optymalizuje obraz do wektoryzacji z agresywną redukcją złożoności"""
    try:
        with Image.open(image_path) as img:
            # Konwersja do RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Zmniejsz rozmiar obrazu
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Zastosuj blur aby zmniejszyć detale
            img = img.filter(ImageFilter.GaussianBlur(radius=1))
            
            # Zwiększ kontrast dla lepszej segmentacji
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)
            
            # Ostrzeż obraz
            img = img.filter(ImageFilter.SHARPEN)
            
            return img
    except Exception as e:
        print(f"Błąd podczas optymalizacji obrazu: {e}")
        return None

def quantize_colors_aggressive(image, max_colors=4):
    """Agresywna kwantyzacja kolorów do maksymalnie 4 kolorów"""
    try:
        # Użyj PIL do kwantyzacji z methodą MAXCOVERAGE
        quantized = image.quantize(colors=max_colors, method=Image.Quantize.MAXCOVERAGE)
        
        # Konwertuj z powrotem do RGB
        return quantized.convert('RGB')
    except Exception as e:
        print(f"Błąd podczas kwantyzacji: {e}")
        return image

def simplified_edge_detection(image, threshold1=50, threshold2=150):
    """Uproszczona detekcja krawędzi z mniejszą złożonością"""
    try:
        # Konwertuj do skali szarości
        gray = image.convert('L')
        
        # Zastosuj filtry krawędzi
        edges = gray.filter(ImageFilter.FIND_EDGES)
        
        # Przekształć w binarny obraz
        edges = edges.point(lambda x: 255 if x > 30 else 0, mode='1')
        
        return edges.convert('L')
    except Exception as e:
        print(f"Błąd podczas detekcji krawędzi: {e}")
        return image.convert('L')

def create_simple_path_from_edges(edges_image, simplification_factor=5):
    """Tworzenie uproszczonych ścieżek SVG z obrazu krawędzi"""
    try:
        width, height = edges_image.size
        paths = []
        
        # Znajdź kontury przy pomocy prostego algorytmu
        contours = find_simple_contours(edges_image, simplification_factor)
        
        for contour in contours:
            if len(contour) > 3:  # Tylko kontury z więcej niż 3 punktami
                path_data = create_svg_path_from_contour(contour)
                if path_data:
                    paths.append(path_data)
        
        return paths
    except Exception as e:
        print(f"Błąd podczas tworzenia ścieżek: {e}")
        return []

def find_simple_contours(edges_image, simplification_factor=5):
    """Znajdowanie konturów z uproszczonym algorytmem"""
    width, height = edges_image.size
    contours = []
    visited = [[False for _ in range(width)] for _ in range(height)]
    
    # Konwertuj obraz do listy pikseli dla prostszej obróbki
    pixels = list(edges_image.getdata())
    
    # Skanuj obraz w większych krokach dla lepszej wydajności
    step = max(2, simplification_factor // 2)
    
    for y in range(0, height, step):
        for x in range(0, width, step):
            pixel_index = y * width + x
            if pixel_index < len(pixels) and pixels[pixel_index] > 128 and not visited[y][x]:
                # Znajdź kontur zaczynając od tego punktu
                contour = trace_contour_simple(pixels, visited, x, y, width, height, simplification_factor)
                if len(contour) > 5:  # Tylko znaczące kontury
                    contours.append(contour)
    
    return contours

def trace_contour_simple(pixels, visited, start_x, start_y, width, height, max_points=20):
    """Uproszczone śledzenie konturu"""
    contour = []
    
    # Kierunki: prawo, dół, lewo, góra
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    
    x, y = start_x, start_y
    direction = 0
    
    for _ in range(max_points):
        if 0 <= y < height and 0 <= x < width:
            pixel_index = y * width + x
            if pixel_index < len(pixels) and pixels[pixel_index] > 128 and not visited[y][x]:
                contour.append((x, y))
                visited[y][x] = True
                
                # Spróbuj znaleźć następny punkt
                found_next = False
                for i in range(4):
                    dx, dy = directions[(direction + i) % 4]
                    nx, ny = x + dx * 2, y + dy * 2  # Większe kroki
                    
                    if (0 <= ny < height and 0 <= nx < width):
                        next_pixel_index = ny * width + nx
                        if (next_pixel_index < len(pixels) and 
                            pixels[next_pixel_index] > 128 and not visited[ny][nx]):
                            x, y = nx, ny
                            direction = (direction + i) % 4
                            found_next = True
                            break
                
                if not found_next:
                    break
            else:
                break
        else:
            break
    
    return contour

def create_svg_path_from_contour(contour):
    """Tworzenie ścieżki SVG z konturu"""
    if len(contour) < 3:
        return None
    
    # Uproszczenie konturu - weź co n-ty punkt
    simplified = contour[::max(1, len(contour) // 10)]
    
    if len(simplified) < 3:
        simplified = contour
    
    path_data = f"M {simplified[0][0]} {simplified[0][1]}"
    
    # Dodaj linie do pozostałych punktów
    for point in simplified[1:]:
        path_data += f" L {point[0]} {point[1]}"
    
    path_data += " Z"  # Zamknij ścieżkę
    return path_data

def vectorize_image_optimized(image_path, output_path):
    """Zoptymalizowana wektoryzacja obrazu"""
    try:
        print("Rozpoczynanie optymalizowanej wektoryzacji...")
        
        # Optymalizuj obraz
        optimized_image = optimize_image_for_vectorization(image_path)
        if not optimized_image:
            return False
        
        print("Obraz zoptymalizowany")
        
        # Agresywna redukcja kolorów
        quantized_image = quantize_colors_aggressive(optimized_image, max_colors=3)
        print("Kolory zredukowane")
        
        # Detekcja krawędzi
        edges = simplified_edge_detection(quantized_image)
        print("Krawędzie wykryte")
        
        # Tworzenie ścieżek
        paths = create_simple_path_from_edges(edges, simplification_factor=8)
        print(f"Utworzono {len(paths)} ścieżek")
        
        # Uzyskaj kolory z skwantyzowanego obrazu
        colors = extract_dominant_colors(quantized_image, max_colors=3)
        
        # Generuj SVG
        width, height = optimized_image.size
        svg_content = generate_optimized_svg(paths, colors, width, height)
        
        # Zapisz SVG
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        
        print("SVG wygenerowany pomyślnie")
        return True
        
    except Exception as e:
        print(f"Błąd podczas wektoryzacji: {e}")
        traceback.print_exc()
        return False
    finally:
        # Wymuś garbage collection
        gc.collect()

def extract_dominant_colors(image, max_colors=3):
    """Wyciągnij dominujące kolory z obrazu"""
    try:
        # Zmniejsz obraz dla szybszej analizy
        small_image = image.copy()
        small_image.thumbnail((100, 100))
        
        # Uzyskaj palety kolorów
        palette_image = small_image.quantize(colors=max_colors)
        palette = palette_image.getpalette()
        
        colors = []
        for i in range(max_colors):
            r = palette[i * 3]
            g = palette[i * 3 + 1] 
            b = palette[i * 3 + 2]
            colors.append(f"rgb({r},{g},{b})")
        
        return colors
    except Exception as e:
        print(f"Błąd podczas wyciągania kolorów: {e}")
        return ["#000000", "#808080", "#ffffff"]

def generate_optimized_svg(paths, colors, width, height):
    """Generuj zoptymalizowany SVG z parametrami InkStitch"""
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" 
     xmlns="http://www.w3.org/2000/svg"
     xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
     xmlns:inkstitch="http://inkstitch.org/namespace">
  <defs>
    <style>
      .embroidery-path {{
        fill: none;
        stroke-width: 0.4;
        stroke-linejoin: round;
        stroke-linecap: round;
      }}
    </style>
  </defs>
  <g inkscape:label="Embroidery" inkscape:groupmode="layer">'''
    
    # Dodaj ścieżki z różnymi kolorami i parametrami haftu
    for i, path in enumerate(paths[:15]):  # Ogranicz do 15 ścieżek
        color = colors[i % len(colors)]
        svg_content += f'''
    <path d="{path}" 
          class="embroidery-path"
          style="stroke: {color};"
          inkstitch:stroke_method="running_stitch"
          inkstitch:running_stitch_length_mm="2.5"
          inkstitch:running_stitch_tolerance_mm="0.1"
          inkstitch:color="{color}" />'''
    
    svg_content += '''
  </g>
</svg>'''
    
    return svg_content

def create_preview_image(svg_path, preview_path, size=(400, 400)):
    """Tworzy podgląd PNG z pliku SVG"""
    try:
        # Dla uproszczenia, stwórz prosty podgląd tekstowy
        preview_img = Image.new('RGB', size, 'white')
        
        # Dodaj tekst informacyjny
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(preview_img)
        
        try:
            # Spróbuj użyć domyślnej czcionki
            font = ImageFont.load_default()
        except:
            font = None
        
        text = "Vector Preview\nEmbroidery Pattern\nGenerated"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (size[0] - text_width) // 2
        y = (size[1] - text_height) // 2
        
        draw.text((x, y), text, fill='black', font=font)
        
        preview_img.save(preview_path, 'PNG')
        return True
    except Exception as e:
        print(f"Błąd podczas tworzenia podglądu: {e}")
        return False

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Generator Wzorów Haftu - Zoptymalizowany</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .upload-area { border: 2px dashed #ccc; border-radius: 10px; padding: 40px; text-align: center; margin: 20px 0; }
        .upload-area:hover { border-color: #007bff; }
        .btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        .btn:hover { background: #0056b3; }
        .result { margin-top: 20px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
        .preview { max-width: 100%; height: auto; border: 1px solid #ddd; }
        .warning { color: #d9534f; font-weight: bold; }
        .info { color: #5bc0de; }
    </style>
</head>
<body>
    <h1>🧵 Generator Wzorów Haftu - Wersja Zoptymalizowana</h1>
    
    <div class="warning">
        ⚠️ UWAGA: Wersja zoptymalizowana pod względem wydajności
        <br>• Maksymalny rozmiar pliku: 8MB
        <br>• Obrazy są automatycznie zmniejszane do 600px
        <br>• Agresywna redukcja kolorów do maksymalnie 3-4 kolorów
        <br>• Uproszczone algorytmy wektoryzacji
    </div>
    
    <div class="upload-area" onclick="document.getElementById('file').click()">
        <p>Kliknij tutaj lub przeciągnij obraz do wektoryzacji</p>
        <p class="info">Obsługiwane formaty: PNG, JPG, JPEG, WebP, SVG</p>
        <input type="file" id="file" style="display: none" accept=".png,.jpg,.jpeg,.webp,.svg">
    </div>
    
    <button class="btn" onclick="uploadFile()">Zwektoryzuj Obraz</button>
    
    <div id="result" class="result" style="display: none;">
        <h3>Wynik wektoryzacji:</h3>
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
            document.getElementById('result-content').innerHTML = '<p>Przetwarzanie... To może potrwać kilka minut.</p>';
            
            fetch('/vectorize', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('result-content').innerHTML = 
                        '<p>✅ Wektoryzacja zakończona pomyślnie!</p>' +
                        '<img src="' + data.preview_url + '" class="preview" alt="Podgląd">' +
                        '<br><br>' +
                        '<a href="' + data.svg_url + '" download class="btn">Pobierz SVG</a>';
                } else {
                    document.getElementById('result-content').innerHTML = 
                        '<p>❌ Błąd: ' + data.error + '</p>';
                }
            })
            .catch(error => {
                document.getElementById('result-content').innerHTML = 
                    '<p>❌ Błąd połączenia: ' + error + '</p>';
            });
        }
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
        svg_filename = f"fx_{timestamp}.svg"
        svg_path = os.path.join(UPLOAD_FOLDER, 'vector_auto', svg_filename)
        preview_filename = f"{timestamp}_preview.png"
        preview_path = os.path.join(UPLOAD_FOLDER, 'preview', preview_filename)
        
        # Wektoryzacja
        success = vectorize_image_optimized(input_path, svg_path)
        
        if not success:
            return jsonify({'success': False, 'error': 'Nie można zwektoryzować obrazu'})
        
        # Tworzenie podglądu
        create_preview_image(svg_path, preview_path)
        
        # Wymuś czyszczenie pamięci
        gc.collect()
        
        return jsonify({
            'success': True,
            'svg_url': f'/download/vector_auto/{svg_filename}',
            'preview_url': f'/download/preview/{preview_filename}',
            'message': 'Wektoryzacja zakończona pomyślnie'
        })
        
    except Exception as e:
        print(f"Błąd podczas wektoryzacji: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': 'Błąd serwera podczas przetwarzania'})

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
    print("🧵 Generator Wzorów Haftu - Wersja Zoptymalizowana")
    print("⚡ Optymalizacje wydajności aktywne")
    print("🔧 Maksymalny rozmiar obrazu: 600px")
    print("🎨 Maksymalna liczba kolorów: 3-4")
    print("📡 Serwer uruchamiany na porcie 5000...")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
