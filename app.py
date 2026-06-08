# ============================================================
#  ДЕМОНСТРАЦИОННО ПРИЛОЖЕНИЕ — ОБРАБОТКА НА ИЗОБРАЖЕНИЯ
#  Streamlit версия — работи в браузър
#  Три теми: Ротация | Grayscale | Edge Detection
# ============================================================

import streamlit as st
import numpy as np
from PIL import Image, ImageFilter
import cv2
import base64
import io

# ── СТРАНИЦА КОНФИГУРАЦИЯ ────────────────────────────────
st.set_page_config(
    page_title="Обработка на изображения",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS СТИЛОВЕ ──────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .stApp { background-color: #0d1117; }
    h1 { color: #4ecca3 !important; text-align: center; }
    h2 { color: #58a6ff !important; }
    h3 { color: #e3b341 !important; }
    .stButton > button {
        background-color: #1a5fa8;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 20px;
        font-size: 14px;
        width: 100%;
        transition: background 0.2s;
    }
    .stButton > button:hover { background-color: #2a7fd8; }
    .result-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
    }
    .info-box {
        background: rgba(88,166,255,0.1);
        border: 1px solid rgba(88,166,255,0.3);
        border-radius: 6px;
        padding: 12px;
        margin: 8px 0;
        color: #e6edf3;
        font-size: 14px;
    }
    .topic-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── ВГРАДЕНИ ИЗОБРАЖЕНИЯ ─────────────────────────────────
def make_planet(size=400):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cx, cy, r = size//2, size//2, size//2 - 10
    for y in range(size):
        for x in range(size):
            dx, dy = x - cx, y - cy
            d = np.sqrt(dx*dx + dy*dy)
            if d <= r:
                nx, ny = dx/r, dy/r
                nz = np.sqrt(max(0, 1 - nx*nx - ny*ny))
                lat = np.arcsin(nz)
                lon = np.arctan2(ny, nx)
                noise = (np.sin(lon*4 + lat*6)*0.5 +
                         np.cos(lat*8 - lon*3)*0.3 +
                         np.sin(lon*7 + 1.2)*0.2)
                light = max(0.2, nx*0.4 + nz*0.8)
                if noise > 0.15:
                    base = np.array([34, 120, 50])
                    if noise > 0.55: base = np.array([130, 110, 90])
                    if abs(lat) > 1.2: base = np.array([230, 240, 255])
                else:
                    depth = (noise + 1) / 2
                    base = np.array([20+int(depth*40), 60+int(depth*80), 180])
                img[y, x] = np.clip(base * light, 0, 255).astype(np.uint8)
            elif d <= r + 8:
                alpha = 1 - (d - r) / 8
                img[y, x] = (np.array([80, 140, 255]) * alpha).astype(np.uint8)
    bg = np.zeros((size, size, 3), dtype=np.uint8)
    rng = np.random.default_rng(42)
    sx = rng.integers(0, size, 200)
    sy = rng.integers(0, size, 200)
    for x, y in zip(sx, sy):
        if np.sqrt((x-cx)**2 + (y-cy)**2) > r + 12:
            b = rng.integers(150, 255)
            bg[y, x] = [b, b, b]
    mask = np.any(img > 0, axis=2)
    bg[mask] = img[mask]
    return Image.fromarray(bg)

def make_nature(size=400):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size//2):
        t = y / (size//2)
        img[y, :] = [int(100+t*135), int(160+t*75), int(220+t*35)]
    sx, sy, sr = size//4, size//5, 35
    for y in range(size):
        for x in range(size):
            if (x-sx)**2 + (y-sy)**2 <= sr**2:
                img[y, x] = [255, 240, 100]
            elif (x-sx)**2 + (y-sy)**2 <= (sr+6)**2:
                img[y, x] = [150, 130, 50]
    rng = np.random.default_rng(7)
    heights = np.zeros(size)
    heights[0] = size * 0.45
    for x in range(1, size):
        heights[x] = np.clip(heights[x-1] + rng.normal(0, 4), size*0.25, size*0.65)
    for x in range(size):
        h = int(heights[x])
        for y in range(h, size//2 + 20):
            t = (y - h) / max(1, size//2 + 20 - h)
            img[y, x] = [int(80+t*20), int(100+t*30), int(80+t*20)]
    for y in range(size//2 + 15, size):
        t = (y - size//2 - 15) / (size//2 - 15)
        img[y, :] = [int(30+t*20), int(120+t*40), int(30+t*10)]
    flowers = [(100,300,[255,80,80]),(180,320,[255,200,50]),
               (260,310,[200,80,255]),(340,295,[255,150,50])]
    for fx, fy, fc in flowers:
        for dy in range(-8, 9):
            for dx in range(-8, 9):
                if dx*dx+dy*dy <= 36 and 0<=fy+dy<size and 0<=fx+dx<size:
                    img[fy+dy, fx+dx] = fc
        for dy in range(8, 28):
            if fy+dy < size: img[fy+dy, fx] = [30, 150, 30]
    for cx2, cy2, cw, ch in [(200,60,60,20),(320,80,50,15),(80,90,45,18)]:
        for dy in range(-ch, ch+1):
            for dx in range(-cw, cw+1):
                if (dx/cw)**2+(dy/ch)**2<=1 and 0<=cy2+dy<size and 0<=cx2+dx<size:
                    img[cy2+dy, cx2+dx] = [240, 245, 255]
    return Image.fromarray(img.astype(np.uint8))

# ── ИНИЦИАЛИЗАЦИЯ ────────────────────────────────────────
@st.cache_resource
def load_builtin_images():
    return {
        "🌍  Планета": make_planet(400),
        "🌿  Природа": make_nature(400),
    }

builtin = load_builtin_images()

if 'original' not in st.session_state:
    st.session_state.original = builtin["🌍  Планета"]
if 'current' not in st.session_state:
    st.session_state.current  = builtin["🌍  Планета"].copy()
if 'total_rotation' not in st.session_state:
    st.session_state.total_rotation = 0
if 'history' not in st.session_state:
    st.session_state.history = []

def reset():
    st.session_state.current = st.session_state.original.copy()
    st.session_state.total_rotation = 0
    st.session_state.history = []

def add_history(action):
    st.session_state.history.append(action)

# ── ЗАГЛАВИЕ ─────────────────────────────────────────────
st.markdown('<h1>🖼️ Обработка на изображения</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#8b949e;font-size:15px;">Демонстрационно приложение · Три теми: Ротация | Grayscale | Edge Detection</p>', unsafe_allow_html=True)
st.markdown('---')

# ── SIDEBAR: КОНТРОЛИ ────────────────────────────────────
with st.sidebar:
    st.markdown('## 🗂️ Избери изображение')

    img_choice = st.radio('Вградени изображения:', list(builtin.keys()))
    if st.button('✅ Зареди избраното'):
        st.session_state.original = builtin[img_choice]
        st.session_state.current  = builtin[img_choice].copy()
        st.session_state.total_rotation = 0
        st.session_state.history = []
        st.rerun()

    uploaded = st.file_uploader('📂 или качи своя снимка:', type=['png','jpg','jpeg','bmp','webp'])
    if uploaded:
        img = Image.open(uploaded).convert('RGB')
        st.session_state.original = img
        st.session_state.current  = img.copy()
        st.session_state.total_rotation = 0
        st.session_state.history = []
        st.rerun()

    st.markdown('---')

    # ТЕМА 1
    st.markdown('## 🔄 Тема 1: Ротация')
    st.markdown('<div class="info-box">Завъртане на изображение на зададен ъгъл. Използва PIL rotate().</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button('↩ +90°'):
            st.session_state.current = st.session_state.current.rotate(-90, expand=True)
            st.session_state.total_rotation = (st.session_state.total_rotation + 90) % 360
            add_history('Ротация +90°')
            st.rerun()
        if st.button('↪ -90°'):
            st.session_state.current = st.session_state.current.rotate(90, expand=True)
            st.session_state.total_rotation = (st.session_state.total_rotation - 90) % 360
            add_history('Ротация -90°')
            st.rerun()
    with col2:
        if st.button('🔃 180°'):
            st.session_state.current = st.session_state.current.rotate(180, expand=True)
            st.session_state.total_rotation = (st.session_state.total_rotation + 180) % 360
            add_history('Ротация 180°')
            st.rerun()

    angle = st.slider('Произволен ъгъл:', 0, 360, 45)
    if st.button(f'✅ Завърти {angle}°'):
        st.session_state.current = st.session_state.current.rotate(-angle, expand=True)
        st.session_state.total_rotation = (st.session_state.total_rotation + angle) % 360
        add_history(f'Ротация {angle}°')
        st.rerun()

    st.markdown('---')

    # ТЕМА 2
    st.markdown('## ⬛ Тема 2: Grayscale')
    st.markdown('<div class="info-box">Конвертиране от цветно RGB в черно-бяло. Формула: Gray = 0.299R + 0.587G + 0.114B</div>', unsafe_allow_html=True)
    if st.button('🔲 Конвертирай в Ч/Б'):
        st.session_state.current = st.session_state.current.convert('L').convert('RGB')
        add_history('Grayscale')
        st.rerun()

    st.markdown('---')

    # ТЕМА 3
    st.markdown('## 🔍 Тема 3: Edge Detection')
    st.markdown('<div class="info-box">Засичане на ръбове с Canny алгоритъм или размазване с Gaussian Blur.</div>', unsafe_allow_html=True)

    t1 = st.slider('Canny праг 1:', 0, 255, 50)
    t2 = st.slider('Canny праг 2:', 0, 255, 150)
    if st.button('🔍 Canny Edge Detection'):
        img_np = np.array(st.session_state.current)
        gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges  = cv2.Canny(gray, t1, t2)
        st.session_state.current = Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))
        add_history(f'Canny ({t1}-{t2})')
        st.rerun()

    blur_r = st.slider('Blur радиус:', 1, 15, 3)
    if st.button('🌫️ Gaussian Blur'):
        st.session_state.current = st.session_state.current.filter(ImageFilter.GaussianBlur(radius=blur_r))
        add_history(f'Gaussian Blur (r={blur_r})')
        st.rerun()

    st.markdown('---')
    if st.button('↺ Нулирай всичко', type='secondary'):
        reset()
        st.rerun()

# ── ГЛАВНА ОБЛАСТ: ИЗОБРАЖЕНИЯ ───────────────────────────
col_orig, col_result = st.columns(2)

with col_orig:
    st.markdown('### 📷 Оригинал')
    st.image(st.session_state.original, use_container_width=True)
    w, h = st.session_state.original.size
    st.caption(f'Размер: {w} × {h} пиксела')

with col_result:
    st.markdown('### ✨ Резултат')
    st.image(st.session_state.current, use_container_width=True)
    w2, h2 = st.session_state.current.size
    st.caption(f'Размер: {w2} × {h2} пиксела  |  Общо завъртяне: {st.session_state.total_rotation}°')

# ── ИСТОРИЯ И ЗАПАЗВАНЕ ──────────────────────────────────
st.markdown('---')
col_hist, col_save = st.columns([2, 1])

with col_hist:
    st.markdown('### 📋 История на операциите')
    if st.session_state.history:
        for i, h in enumerate(st.session_state.history, 1):
            st.markdown(f'`{i}.` {h}')
    else:
        st.markdown('*Все още няма приложени операции.*')

with col_save:
    st.markdown('### 💾 Запази резултата')
    buf = io.BytesIO()
    st.session_state.current.save(buf, format='PNG')
    st.download_button(
        label='⬇️ Изтегли PNG',
        data=buf.getvalue(),
        file_name='rezultat.png',
        mime='image/png'
    )

# ── ОБЯСНЕНИЕ НА ТРИТЕ ТЕМИ ──────────────────────────────
st.markdown('---')
st.markdown('## 📚 Обяснение на трите теми')

tab1, tab2, tab3 = st.tabs(['🔄 Тема 1: Ротация', '⬛ Тема 2: Grayscale', '🔍 Тема 3: Edge Detection'])

with tab1:
    st.markdown('''
    ### Какво е ротация?
    Ротацията е геометрична трансформация която завърта изображението около централната му точка.

    **Как работи в Python (PIL/Pillow):**
    ```python
    from PIL import Image
    img = Image.open("снимка.jpg")
    rotated = img.rotate(90, expand=True)
    rotated.save("завъртяно.jpg")
    ```

    **Параметри:**
    - `angle` — ъгълът в градуси (положителен = обратно на часовниковата стрелка)
    - `expand=True` — изображението се разширява за да не се отреже при наклонени ъгли
    - `expand=False` — оригиналните размери се запазват (може да се отреже)
    ''')

with tab2:
    st.markdown('''
    ### Какво е Grayscale?
    Конвертиране от цветно изображение (RGB — 3 канала) в черно-бяло (L — 1 канал).

    **Формулата:**
    ```
    Gray = 0.299 × R  +  0.587 × G  +  0.114 × B
    ```
    Тежестите не са равни защото окото е най-чувствително към зеленото.

    **Как работи в Python:**
    ```python
    from PIL import Image
    img = Image.open("снимка.jpg")
    gray = img.convert("L")   # L = Luminance (яркост)
    gray.save("черно-бяло.jpg")
    ```
    ''')

with tab3:
    st.markdown('''
    ### Какво е Edge Detection?
    Засичане на ръбовете — местата в изображението където яркостта се променя рязко.

    **Canny алгоритъм (5 стъпки):**
    1. **Gaussian Blur** — леко размазване за намаляване на шума
    2. **Gradient** — изчислява промяната на яркостта (Sobel оператор)
    3. **Non-max suppression** — изтънява ръбовете до 1 пиксел
    4. **Double threshold** — класифицира: силен ръб, слаб ръб, не-ръб
    5. **Edge tracking** — запазва слабите само ако са свързани със силни

    **Как работи в Python:**
    ```python
    import cv2
    import numpy as np
    from PIL import Image

    img = np.array(Image.open("снимка.jpg"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)   # праг 1=50, праг 2=150
    ```
    ''')

st.markdown('---')
st.markdown('<p style="text-align:center;color:#8b949e;font-size:12px;">Демонстрационно приложение · Python · PIL/Pillow · OpenCV · Streamlit</p>', unsafe_allow_html=True)
