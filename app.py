import os
import time
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from flask import Flask, request, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image, ImageFilter, ImageEnhance

app = Flask(__name__)

BASE_UPLOAD    = 'uploads'
RASTER_FOLDER  = os.path.join(BASE_UPLOAD, 'raster')
VECTOR_AUTO    = os.path.join(BASE_UPLOAD, 'vector_auto')
VECTOR_MANUAL  = os.path.join(BASE_UPLOAD, 'vector_manual')
PREVIEW_FOLDER = os.path.join(BASE_UPLOAD, 'preview')
for d in (RASTER_FOLDER, VECTOR_AUTO, VECTOR_MANUAL, PREVIEW_FOLDER):
    os.makedirs(d, exist_ok=True)

INKSCAPE_PATH = r"C:\Program Files\Inkscape\bin\inkscape.exe"
VTRACER_PATH  = r"C:\PajtonFlask\vtracer.exe"
INKSTITCH_CLI = r"C:\Users\wojci\AppData\Roaming\inkscape\extensions\inkstitch\inkstitch\bin\inkstitch.exe"

HOOP_W_MM, HOOP_H_MM = 100, 100
DPI = 300

ET.register_namespace('',    "http://www.w3.org/2000/svg")
ET.register_namespace('xlink','http://www.w3.org/1999/xlink')
ET.register_namespace('inkscape','http://www.inkscape.org/namespaces/inkscape')
ET.register_namespace('sodipodi','http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd')
INKSTITCH_NS = "http://inkstitch.org/namespace"

def trace_with_vtracer(inp, out_svg):
    os.makedirs(os.path.dirname(out_svg), exist_ok=True)
    r = subprocess.run([
        VTRACER_PATH,
        "--input",  os.path.abspath(inp),
        "--output", os.path.abspath(out_svg),
        "--mode",   "spline",
        "--filter_speckle", "4"
    ], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("vtracer:\n" + r.stderr)

def scale_svg(svg_in, svg_out, max_w, max_h):
    txt = open(svg_in, encoding="utf-8").read()
    m = re.search(r'viewBox="([\d.\s\-]+)"', txt)
    if m:
        _,_,w,h = map(float, m.group(1).split())
    else:
        wm = re.search(r'width="([\d.]+)"', txt)
        hm = re.search(r'height="([\d.]+)"', txt)
        w = float(wm.group(1)) if wm else 100
        h = float(hm.group(1)) if hm else 100
        txt = txt.replace("<svg ", f'<svg viewBox="0 0 {w} {h}" ', 1)
    sx = max_w * DPI/25.4 / w
    sy = max_h * DPI/25.4 / h
    sc = min(sx, sy)
    nw, nh = w*sc, h*sc
    txt = re.sub(r'width="[^"]+"',  f'width="{nw}px"',  txt)
    txt = re.sub(r'height="[^"]+"', f'height="{nh}px"', txt)
    os.makedirs(os.path.dirname(svg_out), exist_ok=True)
    open(svg_out, "w", encoding="utf-8").write(txt)

def ensure_svg_has_title(svg):
    tree = ET.parse(svg)
    root = tree.getroot()
    ns = root.tag.split("}")[0].strip("{") if "}" in root.tag else ""
    prefix = {'svg':ns} if ns else {}
    if root.find('svg:title', prefix) is None:
        tag = f'{{{ns}}}title' if ns else 'title'
        t = ET.Element(tag); t.text="Haft"
        root.insert(0, t)
        tree.write(svg, encoding="utf-8", xml_declaration=True)

def export_plain_svg(inp, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    subprocess.run([
        INKSCAPE_PATH,
        "--export-plain-svg",
        f"--export-filename={out}",
        inp
    ], check=True)
    ensure_svg_has_title(out)

def convert_to_paths(src_svg, out_svg):
    src = os.path.abspath(src_svg).replace("\\","/")
    out = os.path.abspath(out_svg).replace("\\","/")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    res = subprocess.run([
        INKSCAPE_PATH,
        "--actions=select-all;object-to-path",
        "--export-plain-svg",
        f"--export-filename={out}",
        src
    ], capture_output=True, text=True)
    if res.returncode:
        raise RuntimeError("Inkscape convert_to_paths failed:\n"+res.stderr)
    if not os.path.exists(out):
        raise FileNotFoundError(f"After convert_to_paths, missing:\n  {out}")

def svg_has_paths(svg):
    tree = ET.parse(svg)
    return bool(tree.getroot().findall('.//{http://www.w3.org/2000/svg}path'))

def inject_inkstitch_params(svg_path):
    ET.register_namespace('inkstitch', INKSTITCH_NS)
    ET.register_namespace('inkscape', "http://www.inkscape.org/namespaces/inkscape")

    tree = ET.parse(svg_path)
    root = tree.getroot()

    group_found = False
    for elem in root.findall('{http://www.w3.org/2000/svg}g'):
        if elem.attrib.get('{http://www.inkscape.org/namespaces/inkscape}groupmode') == 'layer':
            group_found = True
            break

    if not group_found:
        g = ET.Element('{http://www.w3.org/2000/svg}g', {
            '{http://www.inkscape.org/namespaces/inkscape}label': 'Layer 1',
            '{http://www.inkscape.org/namespaces/inkscape}groupmode': 'layer',
            'id': 'layer1'
        })
        for elem in list(root):
            g.append(elem)
            root.remove(elem)
        root.append(g)

    for path in root.findall('.//{http://www.w3.org/2000/svg}path'):
        has_params = path.find(f'{{{INKSTITCH_NS}}}params')
        if has_params is None:
            param_elem = ET.Element(f'{{{INKSTITCH_NS}}}params')
            param_elem.text = '''{
  "stitch_type": "fill",
  "fill_angle": 45,
  "spacing": 0.4,
  "underlay": {
    "type": "none"
  },
  "pull_compensation": 0.0
}'''
            path.append(param_elem)

    tree.write(svg_path, encoding='utf-8', xml_declaration=True)

def generate_stitch_plan_preview_png(svg_in, png_out):
    stitch_svg = svg_in.replace(".svg","_stitch_plan_preview.svg")
    os.makedirs(os.path.dirname(stitch_svg), exist_ok=True)
    cmd = [
        INKSTITCH_CLI,
        svg_in,
        "--extension=stitch_plan_preview",
        "--layer-visibility=visible"
    ]
    with open(stitch_svg, "wb") as out:
        r = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE)
    if r.returncode:
        raise RuntimeError("Ink/Stitch stitch_plan_preview:\n" + r.stderr.decode())
    subprocess.run([
        INKSCAPE_PATH,
        "--export-type=png",
        f"--export-filename={png_out}",
        "--export-dpi", str(DPI),
        stitch_svg
    ], check=True)
    if not os.path.exists(png_out):
        raise RuntimeError("PNG preview not generated")

def generate_simulation_svg(svg_in, sim_svg_out):
    cmd = [
        INKSTITCH_CLI,
        svg_in,
        "--extension=simulator"
    ]
    os.makedirs(os.path.dirname(sim_svg_out), exist_ok=True)
    with open(sim_svg_out, "wb") as out:
        r = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE)
    if r.returncode:
        raise RuntimeError("Ink/Stitch simulate:\n" + r.stderr.decode())
    if not os.path.exists(sim_svg_out):
        raise RuntimeError("Simulation SVG not generated")

def export_svg_to_png(svg_path, png_out):
    subprocess.run([
        INKSCAPE_PATH,
        "--export-type=png",
        f"--export-filename={png_out}",
        "--export-dpi", str(DPI),
        svg_path
    ], check=True)
    if not os.path.exists(png_out):
        raise RuntimeError("PNG not generated from SVG")

def enhance_simulation_image(input_png):
    img = Image.open(input_png)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.5)
    img.save(input_png)

@app.route('/')
def index():
    return '''<h2>Wgraj plik rastrowy lub wektorowy</h2>
<form action="/upload" method="post" enctype="multipart/form-data">
  <input type="file" name="file" required><br><br>
  <input type="submit" value="Wyślij">
</form>'''

@app.route('/upload', methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f:
        return "Nie przesłano pliku.", 400

    name = secure_filename(f.filename.lower())
    ts = str(int(time.time()))
    ext = os.path.splitext(name)[1]
    if ext not in [".png", ".jpg", ".jpeg", ".webp", ".svg"]:
        return f"Nieobsługiwany format: {ext}", 400

    in_dir, vec_dir = (
        (RASTER_FOLDER, VECTOR_AUTO)
        if ext in [".png", ".jpg", ".jpeg", ".webp"]
        else (VECTOR_MANUAL, VECTOR_MANUAL)
    )

    inp_path = os.path.join(in_dir, f"{ts}_{name}")
    f.save(inp_path)

    try:
        if ext in [".png", ".jpg", ".jpeg", ".webp"]:
            traced = os.path.join(vec_dir, f"tr_{ts}.svg")
            trace_with_vtracer(inp_path, traced)
            fixed = os.path.join(vec_dir, f"fx_{ts}.svg")
            export_plain_svg(traced, fixed)
            scaled = os.path.join(vec_dir, f"sc_{ts}.svg")
            scale_svg(fixed, scaled, HOOP_W_MM, HOOP_H_MM)
            paths = os.path.join(vec_dir, f"path_{ts}.svg")
            convert_to_paths(scaled, paths)
            inject_inkstitch_params(paths)
        else:
            raw = os.path.join(vec_dir, f"raw_{ts}.svg")
            shutil.copy(inp_path, raw)
            fixed = os.path.join(vec_dir, f"fx_{ts}.svg")
            export_plain_svg(raw, fixed)
            paths = os.path.join(vec_dir, f"path_{ts}.svg")
            convert_to_paths(fixed, paths)
            inject_inkstitch_params(paths)

        if not svg_has_paths(paths):
            snip = open(paths, encoding="utf-8").read(300)
            return f"❌ Brak ścieżek:<br><pre>{snip}</pre>", 400

        prev = os.path.join(PREVIEW_FOLDER, f"{ts}_preview.png")
        generate_stitch_plan_preview_png(paths, prev)

        sim_svg = os.path.join(PREVIEW_FOLDER, f"{ts}_simulate.svg")
        sim_png = os.path.join(PREVIEW_FOLDER, f"{ts}_simulate.png")
        generate_simulation_svg(paths, sim_svg)
        export_svg_to_png(sim_svg, sim_png)
        enhance_simulation_image(sim_png)

        rel_prev = os.path.relpath(prev, BASE_UPLOAD).replace("\\", "/")
        rel_sim = os.path.relpath(sim_png, BASE_UPLOAD).replace("\\", "/")

        return f'''
        <h3>✅ Podgląd haftu:</h3>
        <img src="/uploads/{rel_prev}" width="600"><br><br>
        <h3>🎮 Symulacja:</h3>
        <img src="/uploads/{rel_sim}" width="600"><br>
        <a href="/">Wróć</a>
        '''

    except Exception as e:
        return f'<pre style="color:red;">Błąd:\n{e}</pre>', 500

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(BASE_UPLOAD, filename)

if __name__ == "__main__":
    print("\u2705 Aplikacja: http://127.0.0.1:5000")
    app.run(debug=True)
