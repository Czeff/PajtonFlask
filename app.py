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
MAX_IMAGE_SIZE = 800  # Zoptymalizowany rozmiar dla lepszej wydajności

# Registracja namespace'ów XML
ET.register_namespace('', "http://www.w3.org/2000/svg")
ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
ET.register_namespace('inkscape', 'http://www.inkscape.org/namespaces/inkscape')
ET.register_namespace('sodipodi', 'http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd')
INKSTITCH_NS = "http://inkstitch.org/namespace"

def optimize_image(image_path, max_size=MAX_IMAGE_SIZE):
    """Optymalizuje rozmiar obrazu przed przetwarzaniem zachowując jakość"""
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

            # Zwiększ ostrość przed zmniejszeniem
            from PIL import ImageEnhance
            sharpness = ImageEnhance.Sharpness(img)
            img = sharpness.enhance(1.2)

            # Zmniejsz rozmiar jeśli za duży
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"Zmniejszono obraz do {new_size}")

            # Zapisz zoptymalizowany obraz z wyższą jakością
            img.save(image_path, 'JPEG', quality=95, optimize=True)
            return True
    except Exception as e:
        logger.error(f"Błąd optymalizacji obrazu: {e}")
        return False

def create_vector_svg_from_image(image_path, svg_path):
    """Tworzy SVG z ulepszoną wektoryzacją zachowującą kolory i szczegóły"""
    try:
        with Image.open(image_path) as img:
            # Konwertuj do RGB jeśli potrzebne
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Zwiększ rozmiar analizy dla lepszych szczegółów
            original_size = img.size
            max_analysis_size = 600  # Zwiększony rozmiar dla lepszych konturów
            if max(original_size) > max_analysis_size:
                ratio = max_analysis_size / max(original_size)
                new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
                img_analysis = img.resize(new_size, Image.Resampling.LANCZOS)
            else:
                img_analysis = img.copy()

            width, height = img_analysis.size

            # Pobierz dane pikseli oryginału
            original_pixels = list(img_analysis.getdata())
            
            # Kwantyzacja kolorów - znajdź dominujące kolory
            from collections import Counter
            color_counts = Counter(original_pixels)
            
            # Wybierz najczęstsze kolory (maksymalnie 16)
            dominant_colors = [color for color, count in color_counts.most_common(16) if count > (width * height) // 200]
            
            # Jeśli za mało dominujących kolorów, użyj próbkowania
            if len(dominant_colors) < 8:
                sample_colors = []
                step = max(1, len(original_pixels) // 100)
                for i in range(0, len(original_pixels), step):
                    sample_colors.append(original_pixels[i])
                
                # Klasteryzacja kolorów
                unique_colors = list(set(sample_colors))
                if len(unique_colors) > 16:
                    # Prosta klasteryzacja kolorów
                    clustered_colors = cluster_colors(unique_colors, 12)
                    dominant_colors = clustered_colors
                else:
                    dominant_colors = unique_colors

            # Zwiększ kontrast i ostrość
            from PIL import ImageEnhance, ImageFilter
            
            # Zwiększ kontrast
            enhancer = ImageEnhance.Contrast(img_analysis)
            img_analysis = enhancer.enhance(1.3)
            
            # Zwiększ ostrość
            sharpness_enhancer = ImageEnhance.Sharpness(img_analysis)
            img_analysis = sharpness_enhancer.enhance(1.5)

            # Zastosuj filtry do detekcji krawędzi
            edges = img_analysis.filter(ImageFilter.FIND_EDGES)
            
            # Dodatkowe filtry dla lepszej detekcji
            contour_img = img_analysis.filter(ImageFilter.CONTOUR)
            
            # SVG wymiary - wyższa rozdzielczość
            svg_width = 1000  # Zwiększone dla lepszych ścieżek
            svg_height = int(1000 * height / width) if width > 0 else 800

            svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" 
     width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">
  <title>High-Quality Vector Embroidery Pattern</title>
  <defs>
    <filter id="embroideryTexture">
      <feGaussianBlur stdDeviation="0.5"/>
    </filter>
  </defs>
  <g id="embroidery-paths">'''

            # Skalowanie
            scale_x = svg_width / width if width > 0 else 1
            scale_y = svg_height / height if height > 0 else 1

            path_count = 0

            # Ulepszona funkcja do grupowania pikseli według kolorów
            def group_pixels_by_color(pixels, colors, tolerance=30):
                """Grupuj piksele według podobnych kolorów"""
                color_groups = {i: [] for i in range(len(colors))}
                
                for y in range(height):
                    for x in range(width):
                        pixel_idx = y * width + x
                        if pixel_idx < len(pixels):
                            pixel = pixels[pixel_idx]
                            
                            # Znajdź najbliższy kolor
                            min_dist = float('inf')
                            closest_color_idx = 0
                            
                            for i, color in enumerate(colors):
                                dist = color_distance(pixel, color)
                                if dist < min_dist:
                                    min_dist = dist
                                    closest_color_idx = i
                            
                            if min_dist <= tolerance:
                                color_groups[closest_color_idx].append((x, y))
                
                return color_groups

            def color_distance(c1, c2):
                """Oblicz odległość między kolorami"""
                return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 + (c1[2] - c2[2]) ** 2) ** 0.5

            def find_color_regions(pixel_positions):
                """Znajdź spójne regiony dla danego koloru"""
                if not pixel_positions:
                    return []
                
                regions = []
                visited = set()
                
                for pos in pixel_positions:
                    if pos in visited:
                        continue
                    
                    # Flood fill dla znalezienia spójnego regionu
                    region = []
                    stack = [pos]
                    
                    while stack:
                        x, y = stack.pop()
                        if (x, y) in visited:
                            continue
                        
                        visited.add((x, y))
                        region.append((x, y))
                        
                        # Sprawdź sąsiednie piksele
                        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                            nx, ny = x + dx, y + dy
                            if (nx, ny) in pixel_positions and (nx, ny) not in visited:
                                stack.append((nx, ny))
                    
                    if len(region) > 5:  # Minimum region size
                        regions.append(region)
                
                return regions

            def create_region_path(region, color):
                """Twórz ścieżkę SVG dla regionu"""
                if not region:
                    return ""
                
                # Znajdź kontur regionu
                contour_points = find_region_contour(region)
                
                if len(contour_points) < 3:
                    return ""
                
                # Uprość kontur
                simplified_contour = simplify_contour_advanced(contour_points)
                
                # Twórz gładką ścieżkę
                path_data = create_smooth_path_advanced(simplified_contour, scale_x, scale_y)
                
                return path_data

            def find_region_contour(region):
                """Znajdź kontur regionu"""
                region_set = set(region)
                contour = []
                
                for x, y in region:
                    is_edge = False
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if (nx, ny) not in region_set:
                            is_edge = True
                            break
                    
                    if is_edge:
                        contour.append((x, y))
                
                # Sortuj punkty konturu aby utworzyć spójną ścieżkę
                if contour:
                    contour = order_contour_points(contour)
                
                return contour

            def order_contour_points(points):
                """Uporządkuj punkty konturu w kolejności tworzenia ścieżki"""
                if len(points) <= 1:
                    return points
                
                ordered = [points[0]]
                remaining = points[1:]
                
                while remaining:
                    current = ordered[-1]
                    closest_dist = float('inf')
                    closest_idx = 0
                    
                    for i, point in enumerate(remaining):
                        dist = ((current[0] - point[0]) ** 2 + (current[1] - point[1]) ** 2) ** 0.5
                        if dist < closest_dist:
                            closest_dist = dist
                            closest_idx = i
                    
                    ordered.append(remaining.pop(closest_idx))
                
                return ordered

            def simplify_contour_advanced(contour, tolerance=1.5):
                """Zaawansowane uproszczenie konturu"""
                if len(contour) <= 3:
                    return contour
                
                # Algorytm Douglas-Peucker
                def douglas_peucker(points, epsilon):
                    if len(points) <= 2:
                        return points
                    
                    # Znajdź punkt z największą odległością od linii
                    max_dist = 0
                    max_index = 0
                    
                    for i in range(1, len(points) - 1):
                        dist = point_to_line_distance(points[i], points[0], points[-1])
                        if dist > max_dist:
                            max_dist = dist
                            max_index = i
                    
                    # Jeśli maksymalna odległość jest większa od epsilon, rekurencyjnie uprość
                    if max_dist > epsilon:
                        left = douglas_peucker(points[:max_index + 1], epsilon)
                        right = douglas_peucker(points[max_index:], epsilon)
                        return left[:-1] + right
                    else:
                        return [points[0], points[-1]]
                
                return douglas_peucker(contour, tolerance)

            def create_smooth_path_advanced(contour, scale_x, scale_y):
                """Twórz zaawansowaną gładką ścieżkę SVG"""
                if len(contour) < 3:
                    return ""
                
                # Skaluj punkty
                scaled_points = [(x * scale_x, y * scale_y) for x, y in contour]
                
                # Rozpocznij ścieżkę
                path_data = f"M {scaled_points[0][0]:.2f},{scaled_points[0][1]:.2f}"
                
                # Użyj krzywych Catmull-Rom dla gładkości
                for i in range(1, len(scaled_points)):
                    if i < len(scaled_points) - 1:
                        # Oblicz punkty kontrolne dla gładkiej krzywej
                        p0 = scaled_points[max(0, i-1)]
                        p1 = scaled_points[i]
                        p2 = scaled_points[min(len(scaled_points)-1, i+1)]
                        
                        # Punkty kontrolne Catmull-Rom
                        tension = 0.5
                        cp1x = p1[0] + (p2[0] - p0[0]) * tension / 6
                        cp1y = p1[1] + (p2[1] - p0[1]) * tension / 6
                        cp2x = p2[0] - (p2[0] - p1[0]) * tension / 6
                        cp2y = p2[1] - (p2[1] - p1[1]) * tension / 6
                        
                        path_data += f" C {cp1x:.2f},{cp1y:.2f} {cp2x:.2f},{cp2y:.2f} {p2[0]:.2f},{p2[1]:.2f}"
                    else:
                        path_data += f" L {scaled_points[i][0]:.2f},{scaled_points[i][1]:.2f}"
                
                path_data += " Z"
                return path_data

            # Funkcja do klastrowania kolorów
            def cluster_colors(colors, num_clusters):
                """Prosta klasteryzacja kolorów"""
                if len(colors) <= num_clusters:
                    return colors
                
                # K-means uproszczony
                import random
                centroids = random.sample(colors, num_clusters)
                
                for _ in range(10):  # 10 iteracji
                    clusters = [[] for _ in range(num_clusters)]
                    
                    for color in colors:
                        min_dist = float('inf')
                        closest_cluster = 0
                        
                        for i, centroid in enumerate(centroids):
                            dist = color_distance(color, centroid)
                            if dist < min_dist:
                                min_dist = dist
                                closest_cluster = i
                        
                        clusters[closest_cluster].append(color)
                    
                    # Aktualizuj centroidy
                    for i in range(num_clusters):
                        if clusters[i]:
                            avg_r = sum(c[0] for c in clusters[i]) // len(clusters[i])
                            avg_g = sum(c[1] for c in clusters[i]) // len(clusters[i])
                            avg_b = sum(c[2] for c in clusters[i]) // len(clusters[i])
                            centroids[i] = (avg_r, avg_g, avg_b)
                
                return centroids

            # Przetwórz każdy kolor
            color_groups = group_pixels_by_color(original_pixels, dominant_colors)
            
            for color_idx, color in enumerate(dominant_colors):
                pixel_positions = color_groups[color_idx]
                
                if not pixel_positions:
                    continue
                
                # Znajdź regiony dla tego koloru
                regions = find_color_regions(pixel_positions)
                
                for region in regions:
                    path_data = create_region_path(region, color)
                    
                    if path_data:
                        # Konwertuj kolor na format SVG
                        rgb_color = f"rgb({color[0]},{color[1]},{color[2]})"
                        
                        # Dodaj ścieżkę do SVG
                        svg_content += f'''
    <path d="{path_data}" fill="{rgb_color}" stroke="{rgb_color}" stroke-width="0.5" 
          opacity="0.95" filter="url(#embroideryTexture)"/>'''
                        path_count += 1

            # Funkcje pomocnicze
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
            svg_content += '''
  </g>
</svg>'''

            os.makedirs(os.path.dirname(svg_path), exist_ok=True)
            with open(svg_path, 'w', encoding='utf-8') as f:
                f.write(svg_content)

            logger.info(f"Utworzono wysokiej jakości wektoryzację z {path_count} regionami kolorów: {svg_path}")
            return True

    except Exception as e:
        logger.error(f"Błąd tworzenia wektoryzacji SVG: {e}")
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