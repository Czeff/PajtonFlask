
import os
import time
import re
import shutil
import xml.etree.ElementTree as ET
from flask import Flask, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont
import logging
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_UPLOAD = 'uploads'
RASTER_FOLDER = os.path.join(BASE_UPLOAD, 'raster')
VECTOR_AUTO = os.path.join(BASE_UPLOAD, 'vector_auto')
VECTOR_MANUAL = os.path.join(BASE_UPLOAD, 'vector_manual')
PREVIEW_FOLDER = os.path.join(BASE_UPLOAD, 'preview')

# Tworzenie katalogów
for d in (RASTER_FOLDER, VECTOR_AUTO, VECTOR_MANUAL, PREVIEW_FOLDER):
    os.makedirs(d, exist_ok=True)

HOOP_W_MM, HOOP_H_MM = 100, 100
DPI = 300
MAX_IMAGE_SIZE = 1024  # Zmniejszony rozmiar dla lepszej wydajności

# Registracja namespace'ów XML
ET.register_namespace('', "http://www.w3.org/2000/svg")
ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
ET.register_namespace('inkscape', 'http://www.inkscape.org/namespaces/inkscape')
ET.register_namespace('sodipodi', 'http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd')
INKSTITCH_NS = "http://inkstitch.org/namespace"

def optimize_image(image_path, max_size=MAX_IMAGE_SIZE):
    """Optymalizuje rozmiar obrazu przed przetwarzaniem"""
    try:
        with Image.open(image_path) as img:
            # Konwersja do RGB jeśli potrzebne
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'RGBA':
                    background.paste(img, mask=img.split()[-1])
                else:
                    background.paste(img)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Zmniejsz rozmiar jeśli za duży
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"Zmniejszono obraz do {new_size}")
            
            # Zapisz zoptymalizowany obraz
            img.save(image_path, 'JPEG', quality=85, optimize=True)
            return True
    except Exception as e:
        logger.error(f"Błąd optymalizacji obrazu: {e}")
        return False

def create_vector_svg_from_image(image_path, svg_path):
    """Tworzy SVG z lepszą analizą obrazu i grupowaniem kolorów"""
    try:
        with Image.open(image_path) as img:
            # Zwiększ rozdzielczość analizy
            original_size = img.size
            analysis_size = (min(200, original_size[0]), min(200, original_size[1]))
            img_analysis = img.resize(analysis_size, Image.Resampling.LANCZOS)
            
            width, height = img_analysis.size
            pixels = list(img_analysis.getdata())
            
            # Funkcja do grupowania podobnych kolorów
            def color_distance(c1, c2):
                if isinstance(c1, int):
                    c1 = (c1, c1, c1)
                if isinstance(c2, int):
                    c2 = (c2, c2, c2)
                return sum((a - b) ** 2 for a, b in zip(c1[:3], c2[:3])) ** 0.5
            
            # Znajdź dominujące kolory
            color_groups = {}
            tolerance = 30
            
            for pixel in pixels:
                if isinstance(pixel, int):
                    pixel = (pixel, pixel, pixel)
                
                # Sprawdź czy kolor pasuje do istniejącej grupy
                matched = False
                for group_color in color_groups:
                    if color_distance(pixel, group_color) < tolerance:
                        color_groups[group_color].append(pixel)
                        matched = True
                        break
                
                if not matched:
                    color_groups[pixel[:3]] = [pixel]
            
            # Wybierz najważniejsze kolory (maksymalnie 8)
            sorted_colors = sorted(color_groups.items(), key=lambda x: len(x[1]), reverse=True)[:8]
            
            # SVG header z większą dokładnością
            svg_width = 400
            svg_height = int(400 * height / width)
            
            svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" 
     width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">
  <title>Embroidery Pattern</title>
  <g id="embroidery-paths">'''
            
            # Twórz ścieżki dla każdego koloru
            scale_x = svg_width / width
            scale_y = svg_height / height
            
            for color_group, pixel_list in sorted_colors:
                if len(pixel_list) < 5:  # Ignoruj bardzo małe grupy
                    continue
                    
                rgb = f"rgb({color_group[0]},{color_group[1]},{color_group[2]})"
                
                # Znajdź wszystkie piksele tego koloru
                color_pixels = []
                for y in range(height):
                    for x in range(width):
                        pixel_idx = y * width + x
                        if pixel_idx < len(pixels):
                            pixel = pixels[pixel_idx]
                            if isinstance(pixel, int):
                                pixel = (pixel, pixel, pixel)
                            
                            if color_distance(pixel, color_group) < tolerance:
                                color_pixels.append((x, y))
                
                # Grupuj sąsiadujące piksele w prostokąty
                if color_pixels:
                    # Prosta segmentacja - twórz prostokąty dla grup pikseli
                    processed_pixels = set()
                    
                    for px, py in color_pixels:
                        if (px, py) in processed_pixels:
                            continue
                        
                        # Znajdź prostokąt zaczynający się od tego piksela
                        min_x, max_x = px, px
                        min_y, max_y = py, py
                        
                        # Rozszerz prostokąt w prawo
                        while max_x + 1 < width and (max_x + 1, py) in color_pixels:
                            max_x += 1
                        
                        # Rozszerz prostokąt w dół
                        while max_y + 1 < height:
                            can_extend = True
                            for x in range(min_x, max_x + 1):
                                if (x, max_y + 1) not in color_pixels:
                                    can_extend = False
                                    break
                            if can_extend:
                                max_y += 1
                            else:
                                break
                        
                        # Dodaj prostokąt do SVG
                        rect_x = min_x * scale_x
                        rect_y = min_y * scale_y
                        rect_w = (max_x - min_x + 1) * scale_x
                        rect_h = (max_y - min_y + 1) * scale_y
                        
                        # Dodaj jako ścieżkę
                        path_data = f"M {rect_x:.1f},{rect_y:.1f} L {rect_x + rect_w:.1f},{rect_y:.1f} L {rect_x + rect_w:.1f},{rect_y + rect_h:.1f} L {rect_x:.1f},{rect_y + rect_h:.1f} Z"
                        
                        svg_content += f'''
    <path d="{path_data}" fill="{rgb}" stroke="{rgb}" stroke-width="0.2"/>'''
                        
                        # Oznacz przetworzone piksele
                        for y in range(min_y, max_y + 1):
                            for x in range(min_x, max_x + 1):
                                processed_pixels.add((x, y))
            
            svg_content += '''
  </g>
</svg>'''
            
            os.makedirs(os.path.dirname(svg_path), exist_ok=True)
            with open(svg_path, 'w', encoding='utf-8') as f:
                f.write(svg_content)
                
            logger.info(f"Utworzono ulepszoną wektoryzację SVG: {svg_path}")
            return True
            
    except Exception as e:
        logger.error(f"Błąd tworzenia wektorowego SVG: {e}")
        return False

def trace_with_simple_vectorization(image_path, svg_path):
    """Prosta wektoryzacja obrazu bez zewnętrznych narzędzi"""
    return create_vector_svg_from_image(image_path, svg_path)

def scale_svg(svg_in, svg_out, max_w, max_h):
    """Skaluje SVG do określonych wymiarów"""
    try:
        with open(svg_in, 'r', encoding='utf-8') as f:
            txt = f.read()
        
        # Znajdź wymiary
        viewbox_match = re.search(r'viewBox="([\d.\s\-]+)"', txt)
        if viewbox_match:
            _, _, w, h = map(float, viewbox_match.group(1).split())
        else:
            w_match = re.search(r'width="([\d.]+)"', txt)
            h_match = re.search(r'height="([\d.]+)"', txt)
            w = float(w_match.group(1)) if w_match else 500
            h = float(h_match.group(1)) if h_match else 500
            txt = txt.replace("<svg ", f'<svg viewBox="0 0 {w} {h}" ', 1)
        
        # Oblicz skalowanie
        scale_x = max_w * DPI / 25.4 / w
        scale_y = max_h * DPI / 25.4 / h
        scale = min(scale_x, scale_y)
        
        new_w, new_h = w * scale, h * scale
        
        # Zastąp wymiary
        txt = re.sub(r'width="[^"]+"', f'width="{new_w}px"', txt)
        txt = re.sub(r'height="[^"]+"', f'height="{new_h}px"', txt)
        
        os.makedirs(os.path.dirname(svg_out), exist_ok=True)
        with open(svg_out, 'w', encoding='utf-8') as f:
            f.write(txt)
            
    except Exception as e:
        raise RuntimeError(f"Błąd skalowania SVG: {e}")

def ensure_svg_has_title(svg):
    """Dodaje tytuł do SVG jeśli go nie ma"""
    try:
        tree = ET.parse(svg)
        root = tree.getroot()
        ns = root.tag.split("}")[0].strip("{") if "}" in root.tag else ""
        prefix = {'svg': ns} if ns else {}
        
        if root.find('svg:title', prefix) is None:
            tag = f'{{{ns}}}title' if ns else 'title'
            title_elem = ET.Element(tag)
            title_elem.text = "Embroidery Pattern"
            root.insert(0, title_elem)
            tree.write(svg, encoding="utf-8", xml_declaration=True)
    except Exception as e:
        logger.warning(f"Nie można dodać tytułu do SVG: {e}")

def export_plain_svg(inp, out):
    """Eksportuje plain SVG"""
    os.makedirs(os.path.dirname(out), exist_ok=True)
    try:
        shutil.copy(inp, out)
        ensure_svg_has_title(out)
    except Exception as e:
        logger.warning(f"Nie można przetworzyć SVG: {e}")
        shutil.copy(inp, out)

def convert_to_paths(src_svg, out_svg):
    """Konwertuje obiekty SVG do ścieżek"""
    os.makedirs(os.path.dirname(out_svg), exist_ok=True)
    shutil.copy(src_svg, out_svg)
    
    if not os.path.exists(out_svg):
        raise FileNotFoundError(f"Nie udało się utworzyć pliku: {out_svg}")

def svg_has_paths(svg):
    """Sprawdza czy SVG zawiera ścieżki"""
    try:
        tree = ET.parse(svg)
        paths = tree.getroot().findall('.//{http://www.w3.org/2000/svg}path')
        rects = tree.getroot().findall('.//{http://www.w3.org/2000/svg}rect')
        circles = tree.getroot().findall('.//{http://www.w3.org/2000/svg}circle')
        ellipses = tree.getroot().findall('.//{http://www.w3.org/2000/svg}ellipse')
        return len(paths) > 0 or len(rects) > 0 or len(circles) > 0 or len(ellipses) > 0
    except:
        return False

def inject_inkstitch_params(svg_path):
    """Dodaje parametry InkStitch do SVG"""
    try:
        ET.register_namespace('inkstitch', INKSTITCH_NS)
        ET.register_namespace('inkscape', "http://www.inkscape.org/namespaces/inkscape")

        tree = ET.parse(svg_path)
        root = tree.getroot()

        # Sprawdź czy istnieje layer
        group_found = False
        for elem in root.findall('{http://www.w3.org/2000/svg}g'):
            if elem.attrib.get('{http://www.inkscape.org/namespaces/inkscape}groupmode') == 'layer':
                group_found = True
                break

        if not group_found:
            g = ET.Element('{http://www.w3.org/2000/svg}g', {
                '{http://www.inkscape.org/namespaces/inkscape}label': 'Embroidery Layer',
                '{http://www.inkscape.org/namespaces/inkscape}groupmode': 'layer',
                'id': 'embroidery-layer'
            })
            for elem in list(root):
                if elem.tag != f'{{{root.tag.split("}")[0].strip("{")}}}title':
                    g.append(elem)
                    root.remove(elem)
            root.append(g)

        # Dodaj parametry do ścieżek
        for path in root.findall('.//{http://www.w3.org/2000/svg}path'):
            if path.find(f'{{{INKSTITCH_NS}}}params') is None:
                param_elem = ET.Element(f'{{{INKSTITCH_NS}}}params')
                param_elem.text = '''{
  "stitch_type": "fill",
  "fill_angle": 45,
  "spacing": 0.4,
  "underlay": {"type": "none"},
  "pull_compensation": 0.0
}'''
                path.append(param_elem)

        tree.write(svg_path, encoding='utf-8', xml_declaration=True)
    except Exception as e:
        logger.warning(f"Nie można dodać parametrów InkStitch: {e}")

def create_preview_from_svg(svg_path, png_path):
    """Tworzy podgląd PNG z SVG z wizualną reprezentacją ścieżek"""
    try:
        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        
        # Odczytaj SVG
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # Pobierz wymiary
        width = 400
        height = 300
        
        viewbox = root.get('viewBox')
        if viewbox:
            _, _, vw, vh = map(float, viewbox.split())
            aspect = vw / vh
            if aspect > 1:
                height = int(width / aspect)
            else:
                width = int(height * aspect)
        
        # Utwórz obraz
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Spróbuj parsować i rysować rzeczywiste ścieżki z SVG
        paths = root.findall('.//{http://www.w3.org/2000/svg}path')
        if paths:
            for path_elem in paths:
                fill_color = path_elem.get('fill', 'black')
                
                # Parsuj kolor
                try:
                    if fill_color.startswith('rgb('):
                        # Wyciągnij wartości RGB
                        rgb_values = fill_color[4:-1].split(',')
                        color = tuple(int(v.strip()) for v in rgb_values)
                    elif fill_color.startswith('#'):
                        # Hex color
                        hex_color = fill_color[1:]
                        if len(hex_color) == 6:
                            color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                        else:
                            color = (100, 100, 100)
                    else:
                        color = (100, 100, 100)
                except:
                    color = (100, 100, 100)
                
                # Parsuj prostą ścieżkę (prostokąty)
                path_data = path_elem.get('d', '')
                if path_data.startswith('M ') and 'L ' in path_data and ' Z' in path_data:
                    try:
                        # Parsuj współrzędne prostokąta
                        parts = path_data.replace('M ', '').replace(' L ', ',').replace(' Z', '').split(',')
                        if len(parts) >= 8:
                            coords = [float(p.strip()) for p in parts[:8]]
                            
                            # Skaluj współrzędne do rozmiaru podglądu
                            scale_x = width / vw if viewbox else 1
                            scale_y = height / vh if viewbox else 1
                            
                            scaled_coords = []
                            for i in range(0, len(coords), 2):
                                x = coords[i] * scale_x
                                y = coords[i+1] * scale_y
                                scaled_coords.extend([x, y])
                            
                            # Rysuj prostokąt
                            if len(scaled_coords) >= 8:
                                x1, y1 = scaled_coords[0], scaled_coords[1]
                                x2, y2 = scaled_coords[4], scaled_coords[5]
                                
                                # Upewnij się, że współrzędne są w poprawnej kolejności
                                left = min(x1, x2)
                                top = min(y1, y2)
                                right = max(x1, x2)
                                bottom = max(y1, y2)
                                
                                # Rysuj wypełniony prostokąt
                                if right > left and bottom > top:
                                    draw.rectangle([left, top, right, bottom], fill=color, outline=color)
                    except:
                        pass
        
        # Jeśli nie udało się narysować ścieżek, użyj reprezentacji zastępczej
        if not paths or all(not path_elem.get('d') for path_elem in paths):
            try:
                font = ImageFont.load_default()
            except:
                font = None
                
            title = "Podgląd wzoru haftu"
            if font:
                bbox = draw.textbbox((0, 0), title, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                x = (width - text_width) // 2
                y = (height - text_height) // 2
                draw.text((x, y), title, fill='black', font=font)
        
        # Dodaj subtelną ramkę
        draw.rectangle([0, 0, width-1, height-1], outline='gray', width=1)
        
        img.save(png_path)
        logger.info(f"Utworzono podgląd: {png_path}")
        
    except Exception as e:
        logger.error(f"Błąd tworzenia podglądu: {e}")
        create_placeholder_image(png_path, "Błąd podglądu")

def create_placeholder_image(png_path, text="Preview"):
    """Tworzy placeholder image"""
    try:
        img = Image.new('RGB', (400, 300), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.load_default()
        except:
            font = None
            
        if font:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (400 - text_width) // 2
            y = (300 - text_height) // 2
            draw.text((x, y), text, fill='black', font=font)
        
        draw.rectangle([10, 10, 390, 290], outline='gray', width=2)
        
        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        img.save(png_path)
    except Exception as e:
        logger.error(f"Nie można utworzyć placeholder image: {e}")

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Generator Wzorów Haftu - Replit</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h2 { color: #333; text-align: center; }
            input[type="file"] { width: 100%; padding: 10px; margin: 10px 0; border: 2px dashed #ccc; border-radius: 5px; }
            input[type="submit"] { background-color: #4CAF50; color: white; padding: 15px 30px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; width: 100%; }
            input[type="submit"]:hover { background-color: #45a049; }
            .info { margin-top: 20px; padding: 15px; background-color: #e7f3ff; border-radius: 5px; }
            .status { margin-top: 10px; padding: 10px; background-color: #fff3cd; border-radius: 5px; font-size: 14px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🧵 Generator Wzorów Haftu - Replit</h2>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept=".png,.jpg,.jpeg,.webp,.svg" required>
                <input type="submit" value="Przetwórz na Wzór Haftu">
            </form>
            <div class="info">
                <h3>ℹ️ Informacje:</h3>
                <ul>
                    <li>Obsługiwane formaty: PNG, JPG, JPEG, WebP, SVG</li>
                    <li>Maksymalny rozmiar pliku: 16MB</li>
                    <li>Optymalizowane dla środowiska Replit</li>
                    <li>Najlepsze rezultaty dla prostych obrazów o wysokim kontraście</li>
                </ul>
            </div>
            <div class="status">
                <strong>Status środowiska:</strong> ✅ Zoptymalizowane dla Replit
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/upload', methods=["POST"])
def upload():
    try:
        f = request.files.get("file")
        if not f or f.filename == '':
            return jsonify({"error": "Nie przesłano pliku"}), 400

        filename = secure_filename(f.filename.lower())
        if not filename:
            return jsonify({"error": "Nieprawidłowa nazwa pliku"}), 400

        ext = os.path.splitext(filename)[1]
        if ext not in [".png", ".jpg", ".jpeg", ".webp", ".svg"]:
            return jsonify({"error": f"Nieobsługiwany format: {ext}"}), 400

        timestamp = str(int(time.time()))
        
        if ext in [".png", ".jpg", ".jpeg", ".webp"]:
            # Przetwarzanie obrazów rastrowych
            input_path = os.path.join(RASTER_FOLDER, f"{timestamp}_{filename}")
            f.save(input_path)
            
            # Optymalizuj obraz
            if not optimize_image(input_path):
                return jsonify({"error": "Nie można zoptymalizować obrazu"}), 500
            
            # Wektoryzacja
            traced_svg = os.path.join(VECTOR_AUTO, f"tr_{timestamp}.svg")
            if not trace_with_simple_vectorization(input_path, traced_svg):
                return jsonify({"error": "Nie można zwektoryzować obrazu"}), 500
            
            # Dalsze przetwarzanie
            fixed_svg = os.path.join(VECTOR_AUTO, f"fx_{timestamp}.svg")
            export_plain_svg(traced_svg, fixed_svg)
            
            scaled_svg = os.path.join(VECTOR_AUTO, f"sc_{timestamp}.svg")
            scale_svg(fixed_svg, scaled_svg, HOOP_W_MM, HOOP_H_MM)
            
            paths_svg = os.path.join(VECTOR_AUTO, f"path_{timestamp}.svg")
            convert_to_paths(scaled_svg, paths_svg)
            
        else:
            # Przetwarzanie plików SVG
            raw_svg = os.path.join(VECTOR_MANUAL, f"raw_{timestamp}.svg")
            f.save(raw_svg)
            
            fixed_svg = os.path.join(VECTOR_MANUAL, f"fx_{timestamp}.svg")
            export_plain_svg(raw_svg, fixed_svg)
            
            paths_svg = os.path.join(VECTOR_MANUAL, f"path_{timestamp}.svg")
            convert_to_paths(fixed_svg, paths_svg)

        # Sprawdź czy SVG ma ścieżki/kształty
        if not svg_has_paths(paths_svg):
            return jsonify({"error": "❌ Nie znaleziono elementów graficznych w pliku"}), 400

        # Dodaj parametry InkStitch
        inject_inkstitch_params(paths_svg)

        # Generuj podglądy
        preview_png = os.path.join(PREVIEW_FOLDER, f"{timestamp}_preview.png")
        create_preview_from_svg(paths_svg, preview_png)

        sim_svg = os.path.join(PREVIEW_FOLDER, f"{timestamp}_simulate.svg")
        sim_png = os.path.join(PREVIEW_FOLDER, f"{timestamp}_simulate.png")
        shutil.copy(paths_svg, sim_svg)
        create_preview_from_svg(sim_svg, sim_png)

        # Przygotuj ścieżki względne
        rel_preview = os.path.relpath(preview_png, BASE_UPLOAD).replace("\\", "/")
        rel_simulation = os.path.relpath(sim_png, BASE_UPLOAD).replace("\\", "/")

        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Wzór Haftu - Wynik</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h3 {{ color: #333; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; margin: 10px 0; }}
                .back-btn {{ display: inline-block; background-color: #008CBA; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin-top: 20px; }}
                .back-btn:hover {{ background-color: #007BB5; }}
                .success {{ color: #4CAF50; font-size: 24px; }}
                .info {{ background-color: #e7f3ff; padding: 15px; border-radius: 5px; margin: 15px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h3 class="success">✅ Wzór haftu został wygenerowany!</h3>
                
                <div class="info">
                    <strong>Środowisko:</strong> Replit (Zoptymalizowane)
                </div>
                
                <h3>🎯 Podgląd wzoru haftu:</h3>
                <img src="/uploads/{rel_preview}" alt="Podgląd wzoru haftu">
                
                <h3>🎮 Symulacja haftu:</h3>
                <img src="/uploads/{rel_simulation}" alt="Symulacja haftu">
                
                <a href="/" class="back-btn">← Powrót do generatora</a>
            </div>
        </body>
        </html>
        '''

    except Exception as e:
        logger.error(f"Błąd przetwarzania: {e}")
        return jsonify({"error": f"Błąd przetwarzania: {str(e)}"}), 500

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(BASE_UPLOAD, filename)

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "Plik jest za duży (max 16MB)"}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Nie znaleziono zasobu"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Błąd serwera"}), 500

if __name__ == "__main__":
    print("🎯 Aplikacja Generator Wzorów Haftu - Replit")
    print("📍 URL: http://0.0.0.0:5000")
    print("🔧 Status: Zoptymalizowane dla środowiska Replit")
    print("✅ Wektoryzacja: Własna implementacja")
    app.run(host='0.0.0.0', port=5000, debug=False)
