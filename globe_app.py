import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import requests
import io
import math
from scipy.stats import entropy as scipy_entropy

st.set_page_config(
    page_title="Анализ на Земята — Shannon Entropy",
    page_icon="🌍",
    layout="wide"
)

st.markdown("""
<style>
.stApp { background-color: #0d1117; color: #e6edf3; }
h1 { color: #4ecca3 !important; text-align: center; }
h2, h3 { color: #58a6ff !important; }
.metric-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    margin: 4px;
}
.metric-val { font-size: 28px; font-weight: bold; color: #4ecca3; }
.metric-lbl { font-size: 12px; color: #8b949e; margin-top: 4px; }
.info-box {
    background: rgba(88,166,255,0.08);
    border: 1px solid rgba(88,166,255,0.25);
    border-radius: 8px;
    padding: 14px;
    font-size: 14px;
    margin: 8px 0;
}
.formula-box {
    background: #161b22;
    border: 1px solid #e3b341;
    border-radius: 8px;
    padding: 14px;
    font-family: monospace;
    font-size: 15px;
    text-align: center;
    color: #e3b341;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ── SHANNON ЕНТРОПИЯ ─────────────────────────────────────
def shannon_entropy(image_array):
    """Изчислява Shannon ентропия на изображение"""
    if len(image_array.shape) == 3:
        gray = np.mean(image_array, axis=2).astype(np.uint8)
    else:
        gray = image_array
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    hist = hist[hist > 0]
    probs = hist / hist.sum()
    H = -np.sum(probs * np.log2(probs))
    return H

def entropy_map(image_array, window=16):
    """Създава карта на локалната ентропия"""
    if len(image_array.shape) == 3:
        gray = np.mean(image_array, axis=2).astype(np.uint8)
    else:
        gray = image_array
    h, w = gray.shape
    emap = np.zeros((h // window, w // window))
    for i in range(h // window):
        for j in range(w // window):
            patch = gray[i*window:(i+1)*window, j*window:(j+1)*window]
            emap[i, j] = shannon_entropy(patch)
    return emap

def entropy_interpretation(H):
    if H < 2:   return "Много ниска", "Почти еднороден регион (океан, пустиня)", "#58a6ff"
    elif H < 4: return "Ниска", "Слаба текстура (степи, тундра)", "#79c0ff"
    elif H < 5: return "Средна", "Умерена сложност (гори, поля)", "#e3b341"
    elif H < 6: return "Висока", "Богата текстура (планини, крайбрежие)", "#f78166"
    else:        return "Много висока", "Максимална сложност (градове, джунгла)", "#ff6e6e"

# ── ВГРАДЕНИ РЕГИОНИ ─────────────────────────────────────
def make_region_amazon(size=256):
    """Амазонска джунгла — тъмно зелена, висока ентропия"""
    rng = np.random.default_rng(1)
    base = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            n = (np.sin(x*0.3)*np.cos(y*0.2) + np.sin(x*0.7+y*0.5)*0.5 +
                 np.cos(x*0.15-y*0.4)*0.3)
            g = int(np.clip(80 + n*40 + rng.normal(0,8), 40, 140))
            r = int(np.clip(20 + n*15 + rng.normal(0,5), 10, 60))
            b = int(np.clip(15 + n*10 + rng.normal(0,4), 5, 40))
            base[y, x] = [r, g, b]
    return base

def make_region_sahara(size=256):
    """Сахара — пясъчни дюни, ниска ентропия"""
    rng = np.random.default_rng(2)
    base = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            n = np.sin(x*0.05)*np.cos(y*0.04) * 0.8
            r = int(np.clip(200 + n*30 + rng.normal(0,4), 160, 240))
            g = int(np.clip(170 + n*25 + rng.normal(0,4), 130, 210))
            b = int(np.clip(100 + n*15 + rng.normal(0,3), 70, 140))
            base[y, x] = [r, g, b]
    return base

def make_region_tokyo(size=256):
    """Токио — градска зона, много висока ентропия"""
    rng = np.random.default_rng(3)
    base = rng.integers(60, 180, (size, size, 3), dtype=np.uint8)
    # Улици
    for i in range(0, size, 20):
        base[i:i+3, :] = [40, 40, 40]
        base[:, i:i+3] = [40, 40, 40]
    # Сгради
    for _ in range(80):
        bx = rng.integers(0, size-15)
        by = rng.integers(0, size-15)
        bw = rng.integers(5, 18)
        bh = rng.integers(5, 18)
        c = rng.integers(80, 200, 3)
        base[by:by+bh, bx:bx+bw] = c
    return base

def make_region_ocean(size=256):
    """Тихи океан — почти еднороден, много ниска ентропия"""
    rng = np.random.default_rng(4)
    base = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            n = np.sin(x*0.02)*np.cos(y*0.02) * 0.3
            r = int(np.clip(10 + n*5  + rng.normal(0,2), 5, 25))
            g = int(np.clip(50 + n*15 + rng.normal(0,3), 30, 80))
            b = int(np.clip(160 + n*20 + rng.normal(0,4), 120, 200))
            base[y, x] = [r, g, b]
    return base

def make_region_alps(size=256):
    """Алпите — планини и снег, висока ентропия"""
    rng = np.random.default_rng(5)
    base = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            h = (np.sin(x*0.1)*np.cos(y*0.12) +
                 np.sin(x*0.25+y*0.2)*0.6 +
                 np.cos(x*0.05-y*0.08)*0.4)
            h = (h + 2) / 4
            if h > 0.7:
                c = [int(220+rng.normal(0,8))]*3
            elif h > 0.5:
                v = int(100+h*60)
                c = [v-20, v-10, v]
            elif h > 0.3:
                c = [int(60+h*40+rng.normal(0,6)),
                     int(90+h*50+rng.normal(0,6)),
                     int(40+h*30+rng.normal(0,4))]
            else:
                c = [int(30+h*30), int(80+h*40), int(20+h*20)]
            base[y, x] = np.clip(c, 0, 255)
    return base

REGIONS = {
    "🌿 Амазония (джунгла)":  make_region_amazon,
    "🏜️ Сахара (пустиня)":   make_region_sahara,
    "🏙️ Токио (град)":        make_region_tokyo,
    "🌊 Тихи океан (вода)":   make_region_ocean,
    "🏔️ Алпите (планини)":   make_region_alps,
}

@st.cache_resource
def get_regions():
    return {k: v() for k, v in REGIONS.items()}

# ── 3D ГЛОБУС ────────────────────────────────────────────
def make_globe(highlight_lat=None, highlight_lon=None):
    u = np.linspace(0, 2*np.pi, 180)
    v = np.linspace(0, np.pi, 90)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    rng = np.random.default_rng(42)
    noise = rng.random(z.shape) * 0.25
    color_val = z + noise

    colorscale = [
        [0.00, '#061428'], [0.28, '#0a2a5a'], [0.42, '#1a5a9a'],
        [0.48, '#c8b560'], [0.52, '#3d8a3d'], [0.65, '#2d6a2d'],
        [0.80, '#7a6a5a'], [1.00, '#ffffff'],
    ]

    traces = [go.Surface(
        x=x, y=y, z=z,
        surfacecolor=color_val,
        colorscale=colorscale,
        showscale=False,
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3, roughness=0.5),
        lightposition=dict(x=2, y=2, z=2),
        hoverinfo='skip', name='Земя',
        opacity=1.0,
    )]

    # Атмосфера
    r = 1.02
    traces.append(go.Surface(
        x=x*r, y=y*r, z=z*r,
        surfacecolor=np.ones_like(z),
        colorscale=[[0,'rgba(80,140,255,0.07)'],[1,'rgba(80,140,255,0.07)']],
        showscale=False, opacity=0.12, hoverinfo='skip', name='Атмосфера',
    ))

    # Звезди
    rng2 = np.random.default_rng(7)
    n = 400
    phi   = rng2.uniform(0, 2*np.pi, n)
    theta = rng2.uniform(0, np.pi, n)
    rs    = rng2.uniform(3, 5, n)
    traces.append(go.Scatter3d(
        x=rs*np.sin(theta)*np.cos(phi),
        y=rs*np.sin(theta)*np.sin(phi),
        z=rs*np.cos(theta),
        mode='markers',
        marker=dict(size=rng2.uniform(0.5,2.5,n), color='white', opacity=0.7),
        hoverinfo='skip', name='Звезди',
    ))

    # Екватор
    t = np.linspace(0, 2*np.pi, 200)
    traces.append(go.Scatter3d(
        x=np.cos(t), y=np.sin(t), z=np.zeros_like(t),
        mode='lines', line=dict(color='rgba(255,255,100,0.4)', width=2),
        hoverinfo='skip', name='Екватор',
    ))

    # Маркер на избрания регион
    if highlight_lat is not None and highlight_lon is not None:
        lat_r = math.radians(highlight_lat)
        lon_r = math.radians(highlight_lon)
        mx = math.cos(lat_r) * math.cos(lon_r)
        my = math.cos(lat_r) * math.sin(lon_r)
        mz = math.sin(lat_r)
        traces.append(go.Scatter3d(
            x=[mx*1.05], y=[my*1.05], z=[mz*1.05],
            mode='markers+text',
            marker=dict(size=10, color='#ff4444', symbol='circle',
                       line=dict(color='white', width=2)),
            text=['📍'], textposition='top center',
            hoverinfo='skip', name='Избран регион',
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        paper_bgcolor='#000010',
        scene=dict(
            bgcolor='#000010',
            xaxis=dict(visible=False, range=[-2,2]),
            yaxis=dict(visible=False, range=[-2,2]),
            zaxis=dict(visible=False, range=[-2,2]),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.8)),
        ),
        margin=dict(l=0,r=0,t=0,b=0),
        showlegend=False,
        height=500,
        annotations=[dict(
            text='🖱️ Влачи = въртене  |  Scroll = zoom  |  Двоен клик = reset',
            x=0.5, y=0.02, xref='paper', yref='paper',
            showarrow=False,
            font=dict(color='rgba(200,200,200,0.6)', size=11),
        )]
    )
    return fig

# ══════════════════════════════════════════════════════════
# ГЛАВНА СТРАНИЦА
# ══════════════════════════════════════════════════════════
st.markdown('<h1>🌍 Анализ на Земята — Shannon Ентропия</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#8b949e;font-size:14px;">Измерване на информационното богатство на различни региони чрез теорията на информацията</p>', unsafe_allow_html=True)

# Теоретично въведение
with st.expander("📚 Теория — Какво е Shannon Ентропия?", expanded=False):
    st.markdown("""
    ### Shannon Ентропия в изображенията

    Shannon ентропията измерва **количеството информация (неопределеност)** в даден сигнал.
    Въведена от Claude Shannon през 1948 г. като основа на теорията на информацията.
    """)
    st.markdown('<div class="formula-box">H = −Σ p(x) · log₂(p(x))</div>', unsafe_allow_html=True)
    st.markdown("""
    **Какво означава в контекста на изображения:**
    - `p(x)` = вероятността за поява на пиксел с яркост x (от хистограмата)
    - Висока ентропия → много различни яркости → богата текстура → **много информация**
    - Ниска ентропия → малко различни яркости → еднороден регион → **малко информация**

    **Диапазон:** от 0 (напълно еднороден) до 8 (максимален хаос за 8-битово изображение)

    | Регион | Очаквана ентропия |
    |--------|------------------|
    | Открит океан | 1-2 бита |
    | Пустиня | 2-3 бита |
    | Гора | 4-5 бита |
    | Планини | 5-6 бита |
    | Голям град | 6-8 бита |
    """)

st.markdown('---')

# ── ОСНОВЕН LAYOUT ───────────────────────────────────────
col_globe, col_analysis = st.columns([1.2, 1])

region_coords = {
    "🌿 Амазония (джунгла)":  (-3, -60),
    "🏜️ Сахара (пустиня)":   (23, 13),
    "🏙️ Токио (град)":        (35, 139),
    "🌊 Тихи океан (вода)":   (0, -150),
    "🏔️ Алпите (планини)":   (46, 10),
}

with col_globe:
    st.markdown('### 🌐 Интерактивен 3D Глобус')
    st.markdown('<div class="info-box">Въртете глобуса с мишката. Изберете регион от дясно за анализ.</div>', unsafe_allow_html=True)

    selected = st.selectbox('📍 Избери регион за анализ:', list(REGIONS.keys()))
    lat, lon = region_coords[selected]

    fig = make_globe(highlight_lat=lat, highlight_lon=lon)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f'📍 Координати: {lat}°{"N" if lat>=0 else "S"}, {lon}°{"E" if lon>=0 else "W"}')

with col_analysis:
    st.markdown('### 📊 Анализ на избрания регион')

    regions_data = get_regions()
    img_array = regions_data[selected]
    pil_img   = Image.fromarray(img_array)

    # Показваме изображението
    st.image(pil_img, caption=f'Сателитна текстура: {selected}', use_container_width=True)

    # Изчисляваме ентропията
    H = shannon_entropy(img_array)
    level, desc, color = entropy_interpretation(H)

    # Метрики
    st.markdown('#### 📈 Shannon Ентропия')
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-box"><div class="metric-val" style="color:{color}">{H:.3f}</div><div class="metric-lbl">бита</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-box"><div class="metric-val" style="font-size:16px;color:{color}">{level}</div><div class="metric-lbl">ниво</div></div>', unsafe_allow_html=True)
    with c3:
        pct = H / 8 * 100
        st.markdown(f'<div class="metric-box"><div class="metric-val">{pct:.0f}%</div><div class="metric-lbl">от максимум</div></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="info-box" style="border-color:{color}">💡 {desc}</div>', unsafe_allow_html=True)

    # Хистограма
    st.markdown('#### 📉 Хистограма на яркостта')
    gray = np.mean(img_array, axis=2).astype(np.uint8)
    hist, bins = np.histogram(gray.flatten(), bins=64, range=(0,256))

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Bar(
        x=bins[:-1], y=hist,
        marker_color='#58a6ff',
        marker_line_width=0,
        name='Яркост',
    ))
    fig_hist.add_annotation(
        text=f'H = {H:.3f} бита',
        x=0.98, y=0.95, xref='paper', yref='paper',
        showarrow=False, font=dict(color='#4ecca3', size=14),
        bgcolor='rgba(0,0,0,0.5)',
    )
    fig_hist.update_layout(
        paper_bgcolor='#161b22', plot_bgcolor='#0d1117',
        font=dict(color='#e6edf3', size=11),
        margin=dict(l=10,r=10,t=10,b=30),
        height=200,
        xaxis=dict(title='Яркост (0-255)', gridcolor='#30363d'),
        yaxis=dict(title='Брой пиксели', gridcolor='#30363d'),
        showlegend=False,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ── СРАВНЕНИЕ НА ВСИЧКИ РЕГИОНИ ──────────────────────────
st.markdown('---')
st.markdown('## 🔬 Сравнение на всички региони')

regions_data = get_regions()
names, entropies, colors_list = [], [], []
color_map = {'🌿': '#3fb950', '🏜': '#e3b341', '🏙': '#f78166', '🌊': '#58a6ff', '🏔': '#bc8cff'}

for name, arr in regions_data.items():
    H = shannon_entropy(arr)
    names.append(name)
    entropies.append(H)
    emoji = name[0]
    colors_list.append(color_map.get(emoji, '#58a6ff'))

# Bar chart
fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(
    x=names, y=entropies,
    marker_color=colors_list,
    text=[f'{h:.2f} бита' for h in entropies],
    textposition='outside',
    textfont=dict(color='#e6edf3', size=12),
))
fig_bar.add_hline(y=8, line_dash='dash', line_color='rgba(255,255,255,0.2)',
                   annotation_text='Максимум (8 бита)', annotation_font_color='#8b949e')
fig_bar.update_layout(
    paper_bgcolor='#161b22', plot_bgcolor='#0d1117',
    font=dict(color='#e6edf3'),
    margin=dict(l=10,r=10,t=20,b=10),
    height=320,
    xaxis=dict(gridcolor='#30363d'),
    yaxis=dict(title='Shannon ентропия (бита)', gridcolor='#30363d', range=[0,9]),
    showlegend=False,
)
st.plotly_chart(fig_bar, use_container_width=True)

# Таблица
st.markdown('### 📋 Детайлна таблица')
col_headers = st.columns([2.5, 1.5, 1.5, 3])
col_headers[0].markdown('**Регион**')
col_headers[1].markdown('**Ентропия**')
col_headers[2].markdown('**% от макс.**')
col_headers[3].markdown('**Интерпретация**')
st.markdown('---')

for name, H in zip(names, entropies):
    level, desc, color = entropy_interpretation(H)
    c1, c2, c3, c4 = st.columns([2.5, 1.5, 1.5, 3])
    c1.markdown(name)
    c2.markdown(f'<span style="color:{color};font-weight:bold">{H:.3f}</span>', unsafe_allow_html=True)
    c3.markdown(f'{H/8*100:.0f}%')
    c4.markdown(f'<span style="color:{color}">{level}</span> — {desc}', unsafe_allow_html=True)

# ── ЕНТРОПИЙНА КАРТА ─────────────────────────────────────
st.markdown('---')
st.markdown('## 🗺️ Ентропийна карта на избрания регион')
st.markdown('Визуализация на локалната ентропия — кои части на региона са най-информационно богати.')

img_array = regions_data[selected]
emap = entropy_map(img_array, window=16)

fig_emap = go.Figure(data=go.Heatmap(
    z=emap,
    colorscale='Plasma',
    colorbar=dict(title='бита', tickfont=dict(color='#e6edf3'), titlefont=dict(color='#e6edf3')),
))
fig_emap.update_layout(
    paper_bgcolor='#161b22', plot_bgcolor='#0d1117',
    font=dict(color='#e6edf3'),
    margin=dict(l=10,r=10,t=10,b=10),
    height=300,
    xaxis=dict(title='X блок', gridcolor='#30363d'),
    yaxis=dict(title='Y блок', gridcolor='#30363d'),
)
st.plotly_chart(fig_emap, use_container_width=True)
st.caption('🟣 Тъмно = ниска ентропия (еднородна зона)  |  🟡 Светло = висока ентропия (богата текстура)')

# ── ИЗВОД ────────────────────────────────────────────────
st.markdown('---')
st.markdown('## 🎯 Кибернетичен извод')
st.markdown(f"""
<div class="info-box">
<strong>Какво научихме?</strong><br><br>
Shannon ентропията е универсална метрика за <em>информационно богатство</em>.
Когато я прилагаме към сателитни изображения на Земята, тя ни позволява
<strong>математически да "видим"</strong> това, което окото само усеща:<br><br>
• Градовете са <strong>информационно богати</strong> — висока ентропия (много различни пиксели = сложна структура)<br>
• Океаните са <strong>информационно бедни</strong> — ниска ентропия (еднородност = предсказуемост)<br>
• Тази метрика може да се използва за <strong>автоматична класификация на земното покритие</strong>
без човешка намеса — само математика.<br><br>
<em>H = {shannon_entropy(regions_data[selected]):.3f} бита за {selected}</em>
</div>
""", unsafe_allow_html=True)

st.markdown('<p style="text-align:center;color:#8b949e;font-size:12px;margin-top:20px;">Shannon Ентропия · Теория на информацията · Python · Streamlit · Plotly</p>', unsafe_allow_html=True)
