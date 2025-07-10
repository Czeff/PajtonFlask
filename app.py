

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
    """Tworzy SVG z prawdziwymi ścieżkami konturowymi zamiast brył"""
    try:
        with Image.open(image_path) as img:
            # Konwertuj do RGB jeśli potrzebne
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Zwiększ rozmiar analizy dla lepszych szczegółów
            original_size = img.size
            max_analysis_size = 400  # Zwiększony rozmiar dla lepszych konturów
            if max(original_size) > max_analysis_size:
                ratio = max_analysis_size / max(original_size)
                new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
                img_analysis = img.resize(new_size, Image.Resampling.LANCZOS)
            else:
                img_analysis = img.copy()
            
            width, height = img_analysis.size
            
            # Zastosuj filtry do detekcji krawędzi
            from PIL import ImageEnhance, ImageFilter
            
            # Zwiększ kontrast
            enhancer = ImageEnhance.Contrast(img_analysis)
            img_analysis = enhancer.enhance(1.5)
            
            # Zastosuj filtr do detekcji krawędzi
            edges = img_analysis.filter(ImageFilter.FIND_EDGES)
            
            # Konwertuj do skali szarości dla lepszej detekcji krawędzi
            gray = img_analysis.convert('L')
            
            # Utwórz mapę krawędzi
            edge_pixels = list(edges.getdata())
            gray_pixels = list(gray.getdata())
            
            # SVG wymiary - wyższa rozdzielczość
            svg_width = 800  # Zwiększone dla lepszych ścieżek
            svg_height = int(800 * height / width) if width > 0 else 600
            
            svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" 
     width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">
  <title>Vector Path Embroidery Pattern</title>
  <g id="embroidery-paths">'''
            
            # Skalowanie
            scale_x = svg_width / width if width > 0 else 1
            scale_y = svg_height / height if height > 0 else 1
            
            path_count = 0
            
            # Funkcja do znajdowania konturów
            def find_contours(threshold=50):
                contours = []
                visited = set()
                
                for y in range(1, height - 1):
                    for x in range(1, width - 1):
                        if (x, y) in visited:
                            continue
                        
                        pixel_idx = y * width + x
                        if pixel_idx < len(edge_pixels):
                            # Sprawdź czy to krawędź
                            edge_val = sum(edge_pixels[pixel_idx][:3]) / 3 if isinstance(edge_pixels[pixel_idx], tuple) else edge_pixels[pixel_idx]
                            
                            if edge_val > threshold:
                                # Rozpocznij śledzenie konturu
                                contour = trace_contour(x, y, edge_pixels, visited, threshold)
                                if len(contour) > 3:  # Minimum 3 punkty dla konturu
                                    contours.append(contour)
                
                return contours
            
            def trace_contour(start_x, start_y, edge_data, visited, threshold):
                contour = []
                current_x, current_y = start_x, start_y
                
                # Kierunki: prawo, dół, lewo, góra, i po przekątnej
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
                
                while len(contour) < 1000:  # Limit długości konturu
                    if (current_x, current_y) in visited:
                        break
                    
                    visited.add((current_x, current_y))
                    contour.append((current_x, current_y))
                    
                    # Znajdź następny punkt konturu
                    next_found = False
                    for dx, dy in directions:
                        next_x = current_x + dx
                        next_y = current_y + dy
                        
                        if (0 <= next_x < width and 0 <= next_y < height and 
                            (next_x, next_y) not in visited):
                            
                            pixel_idx = next_y * width + next_x
                            if pixel_idx < len(edge_data):
                                edge_val = sum(edge_data[pixel_idx][:3]) / 3 if isinstance(edge_data[pixel_idx], tuple) else edge_data[pixel_idx]
                                
                                if edge_val > threshold:
                                    current_x, current_y = next_x, next_y
                                    next_found = True
                                    break
                    
                    if not next_found:
                        break
                
                return contour
            
            # Znajdź kontury
            contours = find_contours(30)  # Próg dla detekcji krawędzi
            
            # Twórz ścieżki SVG z konturów
            for contour in contours:
                if len(contour) < 3:
                    continue
                
                # Uprość kontur (usuń zbędne punkty)
                simplified_contour = simplify_contour(contour)
                
                if len(simplified_contour) >= 3:
                    # Utwórz ścieżkę SVG
                    path_data = create_smooth_path(simplified_contour, scale_x, scale_y)
                    
                    # Określ kolor na podstawie oryginalnego obrazu
                    avg_x = sum(p[0] for p in simplified_contour) // len(simplified_contour)
                    avg_y = sum(p[1] for p in simplified_contour) // len(simplified_contour)
                    
                    if 0 <= avg_x < width and 0 <= avg_y < height:
                        pixel_idx = avg_y * width + avg_x
                        original_pixels = list(img_analysis.getdata())
                        if pixel_idx < len(original_pixels):
                            pixel = original_pixels[pixel_idx]
                            if isinstance(pixel, int):
                                pixel = (pixel, pixel, pixel)
                            
                            rgb = f"rgb({pixel[0]},{pixel[1]},{pixel[2]})"
                        else:
                            rgb = "rgb(0,0,0)"
                    else:
                        rgb = "rgb(0,0,0)"
                    
                    svg_content += f'''
    <path d="{path_data}" fill="none" stroke="{rgb}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>'''
                    path_count += 1
            
            # Jeśli nie znaleziono wystarczająco konturów, dodaj alternatywną metodę
            if path_count < 20:
                # Utwórz ścieżki na podstawie gradientu jasności
                for y in range(0, height, 8):
                    for x in range(0, width, 8):
                        # Sprawdź czy to interesujący obszar
                        if has_interesting_features(x, y, gray_pixels, width, height):
                            # Twórz lokalną ścieżkę
                            local_path = create_local_path(x, y, gray_pixels, width, height)
                            if local_path:
                                path_data = create_smooth_path(local_path, scale_x, scale_y)
                                
                                # Określ kolor
                                pixel_idx = y * width + x
                                original_pixels = list(img_analysis.getdata())
                                if pixel_idx < len(original_pixels):
                                    pixel = original_pixels[pixel_idx]
                                    if isinstance(pixel, int):
                                        pixel = (pixel, pixel, pixel)
                                    rgb = f"rgb({pixel[0]},{pixel[1]},{pixel[2]})"
                                else:
                                    rgb = "rgb(100,100,100)"
                                
                                svg_content += f'''
    <path d="{path_data}" fill="none" stroke="{rgb}" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/>'''
                                path_count += 1
            
            def simplify_contour(contour, tolerance=2):
                """Uprość kontur usuwając zbędne punkty"""
                if len(contour) <= 3:
                    return contour
                
                simplified = [contour[0]]
                
                for i in range(1, len(contour) - 1):
                    curr = contour[i]
                    prev = simplified[-1]
                    next_point = contour[i + 1]
                    
                    # Oblicz odległość od linii łączącej poprzedni i następny punkt
                    distance = point_to_line_distance(curr, prev, next_point)
                    
                    if distance > tolerance:
                        simplified.append(curr)
                
                simplified.append(contour[-1])
                return simplified
            
            def point_to_line_distance(point, line_start, line_end):
                """Oblicz odległość punktu od linii"""
                x0, y0 = point
                x1, y1 = line_start
                x2, y2 = line_end
                
                # Jeśli linia jest punktem
                if x1 == x2 and y1 == y2:
                    return ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5
                
                # Odległość punktu od linii
                A = y2 - y1
                B = x1 - x2
                C = x2 * y1 - x1 * y2
                
                return abs(A * x0 + B * y0 + C) / (A ** 2 + B ** 2) ** 0.5
            
            def create_smooth_path(contour, scale_x, scale_y):
                """Twórz gładką ścieżkę SVG z użyciem krzywych Béziera"""
                if len(contour) < 3:
                    return ""
                
                # Skaluj punkty
                scaled_points = [(x * scale_x, y * scale_y) for x, y in contour]
                
                # Rozpocznij ścieżkę
                path_data = f"M {scaled_points[0][0]:.2f},{scaled_points[0][1]:.2f}"
                
                # Twórz gładkie krzywe między punktami
                for i in range(1, len(scaled_points)):
                    curr = scaled_points[i]
                    prev = scaled_points[i - 1]
                    
                    # Użyj krzywych Béziera dla gładkości
                    if i < len(scaled_points) - 1:
                        next_point = scaled_points[i + 1]
                        
                        # Oblicz punkty kontrolne
                        control_factor = 0.3
                        cx1 = prev[0] + control_factor * (curr[0] - prev[0])
                        cy1 = prev[1] + control_factor * (curr[1] - prev[1])
                        cx2 = curr[0] - control_factor * (next_point[0] - curr[0])
                        cy2 = curr[1] - control_factor * (next_point[1] - curr[1])
                        
                        path_data += f" C {cx1:.2f},{cy1:.2f} {cx2:.2f},{cy2:.2f} {curr[0]:.2f},{curr[1]:.2f}"
                    else:
                        path_data += f" L {curr[0]:.2f},{curr[1]:.2f}"
                
                return path_data
            
            def has_interesting_features(x, y, gray_data, w, h):
                """Sprawdź czy obszar ma interesujące cechy"""
                if x + 8 >= w or y + 8 >= h:
                    return False
                
                # Sprawdź wariancję jasności w małym obszarze
                values = []
                for dy in range(8):
                    for dx in range(8):
                        if y + dy < h and x + dx < w:
                            pixel_idx = (y + dy) * w + (x + dx)
                            if pixel_idx < len(gray_data):
                                values.append(gray_data[pixel_idx])
                
                if len(values) < 4:
                    return False
                
                # Oblicz wariancję
                mean_val = sum(values) / len(values)
                variance = sum((v - mean_val) ** 2 for v in values) / len(values)
                
                return variance > 500  # Próg dla interesujących obszarów
            
            def create_local_path(x, y, gray_data, w, h):
                """Twórz lokalną ścieżkę na podstawie gradientu"""
                path_points = []
                
                # Znajdź punkty o wysokim gradiencie w okolicy
                for dy in range(-4, 5, 2):
                    for dx in range(-4, 5, 2):
                        px, py = x + dx, y + dy
                        if 0 <= px < w and 0 <= py < h:
                            if has_high_gradient(px, py, gray_data, w, h):
                                path_points.append((px, py))
                
                # Sortuj punkty by utworzyć ścieżkę
                if len(path_points) > 2:
                    path_points.sort(key=lambda p: (p[0], p[1]))
                    return path_points
                
                return None
            
            def has_high_gradient(x, y, gray_data, w, h):
                """Sprawdź czy punkt ma wysoki gradient"""
                if x <= 0 or x >= w-1 or y <= 0 or y >= h-1:
                    return False
                
                center_idx = y * w + x
                if center_idx >= len(gray_data):
                    return False
                
                center_val = gray_data[center_idx]
                
                # Sprawdź różnice z sąsiadami
                neighbors = [
                    (x-1, y), (x+1, y), (x, y-1), (x, y+1)
                ]
                
                max_diff = 0
                for nx, ny in neighbors:
                    if 0 <= nx < w and 0 <= ny < h:
                        neighbor_idx = ny * w + nx
                        if neighbor_idx < len(gray_data):
                            diff = abs(gray_data[neighbor_idx] - center_val)
                            max_diff = max(max_diff, diff)
                
                return max_diff > 30  # Próg dla wysokiego gradientu
            
            svg_content += '''
  </g>
</svg>'''
            
            os.makedirs(os.path.dirname(svg_path), exist_ok=True)
            with open(svg_path, 'w', encoding='utf-8') as f:
                f.write(svg_content)
                
            logger.info(f"Utworzono wektoryzację z {path_count} ścieżkami konturowymi: {svg_path}")
            return True
            
    except Exception as e:
        logger.error(f"Błąd tworzenia ścieżek SVG: {e}")
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
            w = float(w_match.group(1)) if w_match else 400
            h = float(h_match.group(1)) if h_match else 300
            txt = txt.replace("<svg ", f'<svg viewBox="0 0 {w} {h}" ', 1)
        
        # Oblicz skalowanie
        scale_x = max_w * DPI / 25.4 / w if w > 0 else 1
        scale_y = max_h * DPI / 25.4 / h if h > 0 else 1
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
    try:
        shutil.copy(src_svg, out_svg)
        
        # Sprawdź czy plik został utworzony i ma zawartość
        if not os.path.exists(out_svg):
            raise FileNotFoundError(f"Nie udało się utworzyć pliku: {out_svg}")
            
        # Sprawdź czy plik ma zawartość
        if os.path.getsize(out_svg) == 0:
            raise ValueError(f"Plik SVG jest pusty: {out_svg}")
            
    except Exception as e:
        logger.error(f"Błąd konwersji do ścieżek: {e}")
        raise

def svg_has_paths(svg):
    """Sprawdza czy SVG zawiera ścieżki z ulepszoną detekcją"""
    try:
        if not os.path.exists(svg):
            logger.error(f"Plik SVG nie istnieje: {svg}")
            return False
            
        with open(svg, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        if not content:
            logger.error(f"Plik SVG jest pusty: {svg}")
            return False
            
        try:
            tree = ET.parse(svg)
            root = tree.getroot()
            
            # Szukaj różnych elementów graficznych
            paths = root.findall('.//{http://www.w3.org/2000/svg}path')
            rects = root.findall('.//{http://www.w3.org/2000/svg}rect')
            circles = root.findall('.//{http://www.w3.org/2000/svg}circle')
            ellipses = root.findall('.//{http://www.w3.org/2000/svg}ellipse')
            polygons = root.findall('.//{http://www.w3.org/2000/svg}polygon')
            polylines = root.findall('.//{http://www.w3.org/2000/svg}polyline')
            
            total_elements = len(paths) + len(rects) + len(circles) + len(ellipses) + len(polygons) + len(polylines)
            
            logger.info(f"SVG zawiera {total_elements} elementów graficznych (paths: {len(paths)}, rects: {len(rects)}, circles: {len(circles)}, ellipses: {len(ellipses)}, polygons: {len(polygons)}, polylines: {len(polylines)})")
            
            return total_elements > 0
            
        except ET.ParseError as e:
            logger.error(f"Błąd parsowania XML: {e}")
            # Spróbuj prostej detekcji tekstowej
            return any(tag in content for tag in ['<path', '<rect', '<circle', '<ellipse', '<polygon', '<polyline'])
            
    except Exception as e:
        logger.error(f"Błąd sprawdzania ścieżek SVG: {e}")
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
    """Tworzy podgląd PNG z SVG z lepszym parsowaniem ścieżek"""
    try:
        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        
        if not os.path.exists(svg_path):
            create_placeholder_image(png_path, "Brak pliku SVG")
            return
        
        # Odczytaj SVG
        try:
            with open(svg_path, 'r', encoding='utf-8') as f:
                svg_content = f.read()
            
            tree = ET.parse(svg_path)
            root = tree.getroot()
        except Exception as e:
            logger.error(f"Błąd parsowania SVG: {e}")
            create_placeholder_image(png_path, "Błąd parsowania SVG")
            return
        
        # Pobierz wymiary SVG
        width = 400
        height = 300
        svg_width = svg_height = 400
        
        # Spróbuj wyciągnąć wymiary z atrybutów
        try:
            svg_width_attr = root.get('width', '400')
            svg_height_attr = root.get('height', '300')
            
            # Usuń jednostki jeśli są
            svg_width = float(svg_width_attr.replace('px', '').replace('pt', ''))
            svg_height = float(svg_height_attr.replace('px', '').replace('pt', ''))
        except:
            pass
        
        # Sprawdź viewBox
        viewbox = root.get('viewBox')
        if viewbox:
            try:
                _, _, vw, vh = map(float, viewbox.split())
                svg_width, svg_height = vw, vh
            except:
                pass
        
        # Oblicz proporcje
        if svg_width > 0 and svg_height > 0:
            aspect = svg_width / svg_height
            if aspect > 1:
                height = int(width / aspect)
            else:
                width = int(height * aspect)
        
        # Utwórz obraz
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Oblicz skalowanie
        scale_x = width / svg_width if svg_width > 0 else 1
        scale_y = height / svg_height if svg_height > 0 else 1
        
        elements_drawn = 0
        
        # Rysuj prostokąty
        rects = root.findall('.//{http://www.w3.org/2000/svg}rect')
        for rect in rects:
            try:
                x = float(rect.get('x', 0)) * scale_x
                y = float(rect.get('y', 0)) * scale_y
                w = float(rect.get('width', 10)) * scale_x
                h = float(rect.get('height', 10)) * scale_y
                
                fill = rect.get('fill', 'black')
                color = parse_color(fill)
                
                if w > 0 and h > 0:
                    draw.rectangle([x, y, x + w, y + h], fill=color, outline=color)
                    elements_drawn += 1
            except Exception as e:
                continue
        
        # Rysuj koła
        circles = root.findall('.//{http://www.w3.org/2000/svg}circle')
        for circle in circles:
            try:
                cx = float(circle.get('cx', 0)) * scale_x
                cy = float(circle.get('cy', 0)) * scale_y
                r = float(circle.get('r', 5)) * min(scale_x, scale_y)
                
                fill = circle.get('fill', 'black')
                color = parse_color(fill)
                
                if r > 0:
                    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline=color)
                    elements_drawn += 1
            except:
                continue
        
        # Rysuj ścieżki
        paths = root.findall('.//{http://www.w3.org/2000/svg}path')
        for path_elem in paths:
            try:
                fill_color = path_elem.get('fill', 'black')
                color = parse_color(fill_color)
                
                path_data = path_elem.get('d', '')
                if not path_data:
                    continue
                
                # Parsuj prostą ścieżkę prostokąta
                if 'M ' in path_data and 'L ' in path_data:
                    try:
                        # Usuń zbędne znaki i podziel na komponenty
                        path_clean = path_data.replace('M ', '').replace(' L ', ' ').replace(' Z', '')
                        coords_text = path_clean.replace(',', ' ').split()
                        
                        coords = []
                        for coord in coords_text:
                            try:
                                coords.append(float(coord))
                            except:
                                continue
                        
                        if len(coords) >= 8:  # Minimum 4 punkty (8 współrzędnych)
                            # Skaluj współrzędne
                            scaled_coords = []
                            for i in range(0, len(coords), 2):
                                if i + 1 < len(coords):
                                    x = coords[i] * scale_x
                                    y = coords[i + 1] * scale_y
                                    scaled_coords.extend([x, y])
                            
                            # Znajdź bounding box
                            if len(scaled_coords) >= 8:
                                x_coords = [scaled_coords[i] for i in range(0, len(scaled_coords), 2)]
                                y_coords = [scaled_coords[i] for i in range(1, len(scaled_coords), 2)]
                                
                                left = min(x_coords)
                                top = min(y_coords)
                                right = max(x_coords)
                                bottom = max(y_coords)
                                
                                # Rysuj jako prostokąt
                                if right > left and bottom > top and right - left > 0.5 and bottom - top > 0.5:
                                    draw.rectangle([left, top, right, bottom], fill=color, outline=color)
                                    elements_drawn += 1
                    except Exception as e:
                        # Fallback - rysuj małe punkty
                        try:
                            numbers = [float(x) for x in path_data.replace('M', '').replace('L', '').replace('Z', '').replace(',', ' ').split() if x.strip()]
                            if len(numbers) >= 2:
                                x = numbers[0] * scale_x
                                y = numbers[1] * scale_y
                                draw.rectangle([x-2, y-2, x+2, y+2], fill=color, outline=color)
                                elements_drawn += 1
                        except:
                            pass
            except:
                continue
        
        # Jeśli nie udało się narysować elementów, stwórz prostą wizualizację
        if elements_drawn == 0:
            # Spróbuj znaleźć jakiekolwiek liczby w SVG
            try:
                import re
                numbers = re.findall(r'\d+\.?\d*', svg_content)
                if len(numbers) > 20:  # Jeśli jest dużo liczb, to prawdopodobnie są jakieś dane
                    # Stwórz wzór na podstawie pierwszych liczb
                    for i in range(0, min(len(numbers), 40), 4):
                        try:
                            x = float(numbers[i]) * scale_x * 0.1
                            y = float(numbers[i+1]) * scale_y * 0.1
                            w = float(numbers[i+2]) * scale_x * 0.1
                            h = float(numbers[i+3]) * scale_y * 0.1
                            
                            if 0 < x < width and 0 < y < height and w > 0 and h > 0:
                                color = (100 + (i * 30) % 155, 100 + (i * 50) % 155, 150)
                                draw.rectangle([x, y, x + w, y + h], fill=color, outline=color)
                                elements_drawn += 1
                        except:
                            continue
            except:
                pass
        
        # Jeśli nadal nic nie ma, stwórz placeholder
        if elements_drawn == 0:
            try:
                font = ImageFont.load_default()
            except:
                font = None
                
            title = "Wzór haftu"
            if font:
                bbox = draw.textbbox((0, 0), title, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                x = (width - text_width) // 2
                y = (height - text_height) // 2 - 20
                draw.text((x, y), title, fill='black', font=font)
            
            # Dodaj dekoracyjny wzór
            for i in range(5):
                for j in range(5):
                    x = 50 + j * 60
                    y = 80 + i * 40
                    if x + 40 < width and y + 25 < height:
                        color = (80 + i*20, 80 + j*20, 120 + (i+j)*10)
                        draw.rectangle([x, y, x + 40, y + 25], fill=color, outline=(50, 50, 50))
        
        # Dodaj ramkę
        draw.rectangle([0, 0, width-1, height-1], outline='gray', width=1)
        
        img.save(png_path)
        logger.info(f"Utworzono podgląd z {elements_drawn} elementami: {png_path}")
        
    except Exception as e:
        logger.error(f"Błąd tworzenia podglądu: {e}")
        create_placeholder_image(png_path, "Błąd podglądu")

def parse_color(color_str):
    """Parsuje kolor SVG do tuple RGB"""
    try:
        if color_str.startswith('rgb('):
            # Wyciągnij wartości RGB
            rgb_values = color_str[4:-1].split(',')
            return tuple(int(v.strip()) for v in rgb_values)
        elif color_str.startswith('#'):
            # Hex color
            hex_color = color_str[1:]
            if len(hex_color) == 6:
                return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            elif len(hex_color) == 3:
                return tuple(int(hex_color[i]*2, 16) for i in range(3))
        elif color_str in {'black', 'none'}:
            return (0, 0, 0)
        elif color_str == 'white':
            return (255, 255, 255)
        else:
            return (100, 100, 100)
    except:
        return (100, 100, 100)

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
                <strong>Status środowiska:</strong> ✅ Zoptymalizowane dla Replit (v2.0)
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/upload', methods=["POST"])
def upload():
    try:
        # Sprawdź czy żądanie zawiera plik
        if 'file' not in request.files:
            logger.error("Brak pliku w żądaniu")
            return jsonify({"error": "Nie przesłano pliku"}), 400

        f = request.files['file']
        if not f or f.filename == '':
            logger.error("Pusty plik")
            return jsonify({"error": "Nie wybrano pliku"}), 400

        filename = secure_filename(f.filename.lower())
        if not filename:
            logger.error("Nieprawidłowa nazwa pliku")
            return jsonify({"error": "Nieprawidłowa nazwa pliku"}), 400

        ext = os.path.splitext(filename)[1]
        if ext not in [".png", ".jpg", ".jpeg", ".webp", ".svg"]:
            logger.error(f"Nieobsługiwany format: {ext}")
            return jsonify({"error": f"Nieobsługiwany format: {ext}"}), 400

        timestamp = str(int(time.time()))
        
        if ext in [".png", ".jpg", ".jpeg", ".webp"]:
            # Przetwarzanie obrazów rastrowych
            input_path = os.path.join(RASTER_FOLDER, f"{timestamp}_{filename}")
            f.save(input_path)
            
            logger.info(f"Zapisano plik rastrowy: {input_path}")
            
            # Optymalizuj obraz
            if not optimize_image(input_path):
                logger.error("Nie można zoptymalizować obrazu")
                return jsonify({"error": "Nie można zoptymalizować obrazu"}), 500
            
            # Wektoryzacja
            traced_svg = os.path.join(VECTOR_AUTO, f"tr_{timestamp}.svg")
            if not trace_with_simple_vectorization(input_path, traced_svg):
                logger.error("Nie można zwektoryzować obrazu")
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
            
            logger.info(f"Zapisano plik SVG: {raw_svg}")
            
            fixed_svg = os.path.join(VECTOR_MANUAL, f"fx_{timestamp}.svg")
            export_plain_svg(raw_svg, fixed_svg)
            
            paths_svg = os.path.join(VECTOR_MANUAL, f"path_{timestamp}.svg")
            convert_to_paths(fixed_svg, paths_svg)

        # Sprawdź czy SVG ma ścieżki/kształty
        if not svg_has_paths(paths_svg):
            logger.error(f"Brak elementów graficznych w pliku: {paths_svg}")
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

        logger.info(f"Przetwarzanie zakończone pomyślnie dla pliku: {filename}")

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
                    <strong>Plik:</strong> {filename}<br>
                    <strong>Środowisko:</strong> Replit (Zoptymalizowane v2.0)
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
    print("🎯 Aplikacja Generator Wzorów Haftu - Replit v2.0")
    print("📍 URL: http://0.0.0.0:5000")
    print("🔧 Status: Zoptymalizowane dla środowiska Replit")
    print("✅ Wektoryzacja: Własna implementacja z ulepszoną detekcją")
    app.run(host='0.0.0.0', port=5000, debug=False)

