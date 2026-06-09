import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io, math

st.set_page_config(
    page_title="Shannon Ентропия — Анализ на Информацията",
    page_icon="🌍", layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.stApp { background-color: #0d1117; color: #e6edf3; }
.stTabs [data-baseweb="tab-list"] {
    background-color: #161b22;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    color: #8b949e;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 15px;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background-color: #1a5fa8 !important;
    color: white !important;
}
h1 { color: #4ecca3 !important; text-align: center; font-size: 2.2rem !important; }
h2 { color: #58a6ff !important; }
h3 { color: #e3b341 !important; }
.metric-card {
    background: linear-gradient(135deg, #161b22, #1c2128);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 18px;
    text-align: center;
    margin: 6px 0;
}
.metric-val { font-size: 32px; font-weight: bold; }
.metric-lbl { font-size: 12px; color: #8b949e; margin-top: 4px; }
.info-card {
    background: rgba(88,166,255,0.08);
    border-left: 4px solid #58a6ff;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 14px;
    line-height: 1.7;
}
.formula-card {
    background: #161b22;
    border: 2px solid #e3b341;
    border-radius: 10px;
    padding: 16px;
    font-family: monospace;
    font-size: 18px;
    text-align: center;
    color: #e3b341;
    margin: 12px 0;
    letter-spacing: 1px;
}
.section-divider {
    height: 3px;
    background: linear-gradient(90deg, #1a5fa8, #4ecca3, #e3b341, #f78166);
    border-radius: 3px;
    margin: 20px 0;
}
.tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: bold;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# ГЕНЕРАТОРИ НА ИЗОБРАЖЕНИЯ
# ══════════════════════════════════════════════════════════
def shannon_entropy(arr):
    if len(arr.shape) == 3:
        gray = np.mean(arr, axis=2).astype(np.uint8)
    else:
        gray = arr
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0,256))
    hist = hist[hist > 0]
    p = hist / hist.sum()
    return float(-np.sum(p * np.log2(p)))

def entropy_map(arr, window=16):
    gray = np.mean(arr, axis=2).astype(np.uint8) if len(arr.shape)==3 else arr
    h, w = gray.shape
    emap = np.zeros((h//window, w//window))
    for i in range(h//window):
        for j in range(w//window):
            patch = gray[i*window:(i+1)*window, j*window:(j+1)*window]
            emap[i,j] = shannon_entropy(patch)
    return emap

def entropy_color(H):
    if H < 2:   return "#58a6ff", "Много ниска"
    elif H < 4: return "#79c0ff", "Ниска"
    elif H < 5: return "#e3b341", "Средна"
    elif H < 6: return "#f78166", "Висока"
    else:        return "#ff4444", "Много висока"

# ── РЕГИОНИ ──────────────────────────────────────────────
@st.cache_resource
def build_regions():
    def amazon(s=300):
        rng=np.random.default_rng(1); img=np.zeros((s,s,3),dtype=np.uint8)
        for y in range(s):
            for x in range(s):
                n=np.sin(x*.3)*np.cos(y*.2)+np.sin(x*.7+y*.5)*.5+np.cos(x*.15-y*.4)*.3
                g=int(np.clip(80+n*40+rng.normal(0,8),40,140))
                img[y,x]=[int(np.clip(20+n*15+rng.normal(0,5),10,60)),g,int(np.clip(15+n*10,5,40))]
        return img
    def sahara(s=300):
        rng=np.random.default_rng(2); img=np.zeros((s,s,3),dtype=np.uint8)
        for y in range(s):
            for x in range(s):
                n=np.sin(x*.05)*np.cos(y*.04)*.8
                img[y,x]=[int(np.clip(200+n*30+rng.normal(0,4),160,240)),
                           int(np.clip(170+n*25+rng.normal(0,4),130,210)),
                           int(np.clip(100+n*15+rng.normal(0,3),70,140))]
        return img
    def tokyo(s=300):
        rng=np.random.default_rng(3); img=rng.integers(60,180,(s,s,3),dtype=np.uint8)
        for i in range(0,s,20): img[i:i+3,:]=40; img[:,i:i+3]=40
        for _ in range(100):
            bx,by=rng.integers(0,s-15),rng.integers(0,s-15)
            bw,bh=rng.integers(5,18),rng.integers(5,18)
            img[by:by+bh,bx:bx+bw]=rng.integers(80,200,3)
        return img
    def ocean(s=300):
        rng=np.random.default_rng(4); img=np.zeros((s,s,3),dtype=np.uint8)
        for y in range(s):
            for x in range(s):
                n=np.sin(x*.02)*np.cos(y*.02)*.3
                img[y,x]=[int(np.clip(10+n*5+rng.normal(0,2),5,25)),
                           int(np.clip(50+n*15+rng.normal(0,3),30,80)),
                           int(np.clip(160+n*20+rng.normal(0,4),120,200))]
        return img
    def alps(s=300):
        rng=np.random.default_rng(5); img=np.zeros((s,s,3),dtype=np.uint8)
        for y in range(s):
            for x in range(s):
                h=(np.sin(x*.1)*np.cos(y*.12)+np.sin(x*.25+y*.2)*.6+np.cos(x*.05-y*.08)*.4+2)/4
                if h>.7: c=[int(np.clip(220+rng.normal(0,8),200,255))]*3
                elif h>.5: v=int(100+h*60); c=[v-20,v-10,v]
                elif h>.3: c=[int(60+h*40+rng.normal(0,6)),int(90+h*50+rng.normal(0,6)),int(40+h*30)]
                else: c=[int(30+h*30),int(80+h*40),int(20+h*20)]
                img[y,x]=np.clip(c,0,255)
        return img
    return {"🌿 Амазония":amazon(),"🏜️ Сахара":sahara(),
            "🏙️ Токио":tokyo(),"🌊 Тихи океан":ocean(),"🏔️ Алпите":alps()}

# ── ТЕКСТУРИ ─────────────────────────────────────────────
@st.cache_resource
def build_textures():
    def honeycomb(s=256):
        img=np.zeros((s,s,3),dtype=np.uint8)
        for y in range(s):
            for x in range(s):
                hx=x/20; hy=y/20
                v=np.sin(hx)+np.sin(hy/2+hx/2)*np.cos(hx/2-hy/2)
                c=int(np.clip((v+2)/4*255,0,255))
                img[y,x]=[int(c*.9),int(c*.7),20]
        return img
    def marble(s=256):
        rng=np.random.default_rng(10); img=np.zeros((s,s,3),dtype=np.uint8)
        noise=np.zeros((s,s))
        for freq in [1,2,4,8,16]:
            n=rng.random((s,s)); noise+=n/freq
        for y in range(s):
            for x in range(s):
                v=np.sin(x/20+noise[y,x]*5)
                c=int(np.clip((v+1)/2*255,0,255))
                img[y,x]=[c,int(c*.95),int(c*.9)]
        return img
    def wood(s=256):
        img=np.zeros((s,s,3),dtype=np.uint8)
        rng=np.random.default_rng(11)
        for y in range(s):
            for x in range(s):
                r=np.sqrt((x-s/2)**2+(y-s/2)**2)
                v=np.sin(r/4+rng.random()*.3)
                c=int(np.clip((v+1)/2*200+30,0,255))
                img[y,x]=[int(c*.8),int(c*.5),int(c*.2)]
        return img
    def crystal(s=256):
        img=np.zeros((s,s,3),dtype=np.uint8)
        for y in range(s):
            for x in range(s):
                v=(np.sin(x/8)*np.cos(y/8)+np.sin((x+y)/12)+np.cos((x-y)/10))/3
                r=int(np.clip(100+v*80,0,255))
                g=int(np.clip(150+v*60,0,255))
                b=int(np.clip(220+v*35,0,255))
                img[y,x]=[r,g,b]
        return img
    def sand(s=256):
        rng=np.random.default_rng(12); img=np.zeros((s,s,3),dtype=np.uint8)
        for y in range(s):
            for x in range(s):
                v=np.sin(x*.08+rng.random()*.1)*np.cos(y*.06)
                c=int(np.clip(180+v*40+rng.normal(0,5),140,230))
                img[y,x]=[c,int(c*.85),int(c*.6)]
        return img
    def lava(s=256):
        rng=np.random.default_rng(13); img=np.zeros((s,s,3),dtype=np.uint8)
        for y in range(s):
            for x in range(s):
                v=np.sin(x*.1)*np.cos(y*.08)+np.sin((x+y)*.05)*np.cos((x-y)*.07)
                v=(v+2)/4
                if v>.7: c=[255,int(200*v),0]
                elif v>.4: c=[int(200*v),int(50*v),0]
                else: c=[int(80*v),int(20*v),int(20*v)]
                img[y,x]=np.clip(c,0,255)
        return img
    return {"🐝 Пчелна пита":honeycomb(),"🪨 Мрамор":marble(),
            "🌲 Дърво":wood(),"💎 Кристал":crystal(),
            "🏖️ Пясък":sand(),"🌋 Лава":lava()}

# ── КАРТИНИ ──────────────────────────────────────────────
@st.cache_resource
def build_paintings():
    def starry_night(s=300):
        img=np.zeros((s,s,3),dtype=np.uint8)
        # Тъмно синьо небе с вихри
        for y in range(s):
            for x in range(s):
                swirl=np.sin(x*.15+y*.1)*np.cos(x*.08-y*.12)+np.sin((x+y)*.06)
                b=int(np.clip(100+swirl*60+y*.1,40,200))
                g=int(np.clip(50+swirl*30,20,120))
                r=int(np.clip(20+swirl*15,5,80))
                img[y,x]=[r,g,b]
        # Звезди
        rng=np.random.default_rng(20)
        for _ in range(25):
            sx,sy=rng.integers(10,s-10),rng.integers(10,s//2)
            for dy in range(-6,7):
                for dx in range(-6,7):
                    if dx*dx+dy*dy<=36:
                        t=1-(dx*dx+dy*dy)/36
                        ny2,nx2=sy+dy,sx+dx
                        if 0<=ny2<s and 0<=nx2<s:
                            img[ny2,nx2]=np.clip(img[ny2,nx2]+[int(200*t),int(200*t),int(100*t)],0,255)
        # Хълмове
        for x in range(s):
            h=int(s*.65+np.sin(x*.05)*20)
            for y in range(h,s):
                t=(y-h)/(s-h)
                img[y,x]=[int(20+t*30),int(40+t*60),int(20+t*30)]
        return img

    def monet_waterlilies(s=300):
        img=np.zeros((s,s,3),dtype=np.uint8)
        rng=np.random.default_rng(21)
        # Вода
        for y in range(s):
            for x in range(s):
                ripple=np.sin(x*.2+y*.15)*np.cos(x*.1-y*.2)
                r=int(np.clip(40+ripple*30+rng.normal(0,8),10,120))
                g=int(np.clip(80+ripple*40+rng.normal(0,8),40,160))
                b=int(np.clip(120+ripple*50+rng.normal(0,8),60,200))
                img[y,x]=[r,g,b]
        # Водни лилии
        lily_pos=[(80,120),(200,180),(150,80),(250,220),(60,200)]
        for lx,ly in lily_pos:
            for dy in range(-18,19):
                for dx in range(-25,26):
                    if (dx/25)**2+(dy/18)**2<=1:
                        t=1-(dx/25)**2-(dy/18)**2
                        ny2,nx2=ly+dy,lx+dx
                        if 0<=ny2<s and 0<=nx2<s:
                            gr=int(np.clip(30+t*80+rng.normal(0,10),0,255))
                            gg=int(np.clip(100+t*100+rng.normal(0,10),0,255))
                            gb=int(np.clip(30+t*40+rng.normal(0,5),0,255))
                            img[ny2,nx2]=[gr,gg,gb]
            # Цвят
            for dy in range(-8,9):
                for dx in range(-8,9):
                    if dx*dx+dy*dy<=50:
                        t=1-(dx*dx+dy*dy)/50
                        ny2,nx2=ly+dy,lx+dx
                        if 0<=ny2<s and 0<=nx2<s:
                            img[ny2,nx2]=np.clip([int(220*t+rng.normal(0,15)),int(100*t),int(100*t)],0,255)
        return img

    def picasso_cubist(s=300):
        img=np.zeros((s,s,3),dtype=np.uint8)
        rng=np.random.default_rng(22)
        colors_p=[[200,50,50],[50,100,200],[200,180,50],[50,180,100],[180,80,180],[240,140,50]]
        for _ in range(40):
            x1,y1=rng.integers(0,s),rng.integers(0,s)
            x2,y2=x1+rng.integers(30,100),y1+rng.integers(30,100)
            c=colors_p[rng.integers(0,len(colors_p))]
            for y in range(min(y1,y2),min(max(y1,y2),s)):
                for x in range(min(x1,x2),min(max(x1,x2),s)):
                    img[y,x]=np.clip([c[0]+rng.normal(0,15),c[1]+rng.normal(0,15),c[2]+rng.normal(0,15)],0,255).astype(np.uint8)
        # Линии
        for _ in range(20):
            x1,y1=rng.integers(0,s),rng.integers(0,s)
            x2,y2=rng.integers(0,s),rng.integers(0,s)
            steps=max(abs(x2-x1),abs(y2-y1))
            if steps>0:
                for t in range(steps):
                    px2=x1+int((x2-x1)*t/steps)
                    py2=y1+int((y2-y1)*t/steps)
                    for dy in range(-1,2):
                        for dx in range(-1,2):
                            if 0<=py2+dy<s and 0<=px2+dx<s:
                                img[py2+dy,px2+dx]=[20,20,20]
        return img

    def kandinsky(s=300):
        img=np.zeros((s,s,3),dtype=np.uint8)
        img[:]=50
        rng=np.random.default_rng(23)
        bright_colors=[[255,50,50],[50,200,255],[255,200,0],[100,255,100],[255,100,200],[200,100,255]]
        # Кръгове
        for _ in range(15):
            cx2,cy2=rng.integers(30,s-30),rng.integers(30,s-30)
            r=rng.integers(15,60)
            c=bright_colors[rng.integers(0,len(bright_colors))]
            for y in range(max(0,cy2-r),min(s,cy2+r)):
                for x in range(max(0,cx2-r),min(s,cx2+r)):
                    if (x-cx2)**2+(y-cy2)**2<=r**2:
                        img[y,x]=np.clip([c[0]+rng.normal(0,20),c[1]+rng.normal(0,20),c[2]+rng.normal(0,20)],0,255).astype(np.uint8)
        # Триъгълници и линии
        for _ in range(10):
            x1,y1=rng.integers(0,s),rng.integers(0,s)
            c=bright_colors[rng.integers(0,len(bright_colors))]
            for i in range(80):
                px2=int(x1+np.cos(i*.2)*i*.3)%s
                py2=int(y1+np.sin(i*.15)*i*.3)%s
                if 0<=py2<s and 0<=px2<s:
                    img[py2,px2]=c
        return img

    def rothko(s=300):
        img=np.zeros((s,s,3),dtype=np.uint8)
        rng=np.random.default_rng(24)
        zones=[
            ([180,40,40],[200,80,60],0,s//4),
            ([220,140,40],[240,160,60],s//4,s//2),
            ([40,80,160],[60,100,180],s//2,3*s//4),
            ([40,120,80],[60,140,100],3*s//4,s),
        ]
        for c1,c2,y1,y2 in zones:
            for y in range(y1,y2):
                t=(y-y1)/(y2-y1)
                for x in range(s):
                    r=int(c1[0]+(c2[0]-c1[0])*t+rng.normal(0,12))
                    g=int(c1[1]+(c2[1]-c1[1])*t+rng.normal(0,12))
                    b=int(c1[2]+(c2[2]-c1[2])*t+rng.normal(0,12))
                    img[y,x]=np.clip([r,g,b],0,255)
        return img

    def seurat_pointillism(s=300):
        img=np.full((s,s,3),240,dtype=np.uint8)
        rng=np.random.default_rng(25)
        palette=[[50,120,200],[200,100,50],[100,180,80],[220,180,60],[180,80,120],[80,160,160]]
        for _ in range(8000):
            x,y=rng.integers(0,s),rng.integers(0,s)
            c=palette[rng.integers(0,len(palette))]
            r2=rng.integers(2,5)
            for dy in range(-r2,r2+1):
                for dx in range(-r2,r2+1):
                    if dx*dx+dy*dy<=r2*r2 and 0<=y+dy<s and 0<=x+dx<s:
                        img[y+dy,x+dx]=c
        return img

    return {
        "🌙 Ван Гог — Звездна нощ":starry_night(),
        "🌸 Моне — Водни лилии":monet_waterlilies(),
        "🔷 Пикасо — Кубизъм":picasso_cubist(),
        "⭕ Кандински — Абстракция":kandinsky(),
        "🟥 Ротко — Цветни полета":rothko(),
        "🔵 Сьора — Пуантилизъм":seurat_pointillism(),
    }

regions   = build_regions()
textures  = build_textures()
paintings = build_paintings()

# ══════════════════════════════════════════════════════════
# ЗАГЛАВИЕ
# ══════════════════════════════════════════════════════════
st.markdown('<h1>🔬 Shannon Ентропия — Анализ на Информацията</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#8b949e;font-size:15px;margin-bottom:0">Измерване на информационното богатство на заобикалящия ни свят</p>', unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Формулата винаги видима
st.markdown('<div class="formula-card">H(X) = −Σ p(xᵢ) · log₂(p(xᵢ))</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#8b949e;font-size:12px">H = ентропия в битове &nbsp;|&nbsp; p(xᵢ) = вероятност за яркост i &nbsp;|&nbsp; Диапазон: 0 (еднородно) → 8 (максимален хаос)</p>', unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# ТАБОВЕ
# ══════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "🌍  Таб 1 — Земята от Космоса",
    "🔬  Таб 2 — Природни Текстури",
    "🎨  Таб 3 — Изкуство & Математика",
])

# ══════════════════════════════════════════════════════════
# ТАБ 1: ЗЕМЯТА
# ══════════════════════════════════════════════════════════
with tab1:
    st.markdown('## 🌍 Информационното богатство на Земята')
    st.markdown('<div class="info-card">Сателитните изображения на различни региони имат коренно различна Shannon ентропия. Математиката "вижда" разликата между хаотичния град и монотонния океан.</div>', unsafe_allow_html=True)

    region_coords = {
        "🌿 Амазония":  (-3,-60,"Тропическа джунгла","Смесица от зеленина, сенки, светлина"),
        "🏜️ Сахара":    (23,13,"Пясъчна пустиня","Еднородни пясъчни дюни"),
        "🏙️ Токио":     (35,139,"Мегаполис","Максимална градска сложност"),
        "🌊 Тихи океан":(0,-150,"Открит океан","Почти еднородна водна повърхност"),
        "🏔️ Алпите":    (46,10,"Планински масив","Сняг, скали и гори"),
    }

    col_g, col_a = st.columns([1.3, 1])

    with col_g:
        st.markdown('### 🌐 3D Интерактивен Глобус')
        selected_r = st.selectbox('📍 Избери регион:', list(regions.keys()), key='reg')
        lat, lon, rtype, rdesc = region_coords[selected_r]

        # Глобус
        u=np.linspace(0,2*np.pi,180); v=np.linspace(0,np.pi,90)
        x=np.outer(np.cos(u),np.sin(v)); y=np.outer(np.sin(u),np.sin(v))
        z=np.outer(np.ones_like(u),np.cos(v))
        rng0=np.random.default_rng(42); noise=rng0.random(z.shape)*.25
        colorscale_g=[[0,'#061428'],[0.28,'#0a2a5a'],[0.42,'#1a5a9a'],
                      [0.48,'#c8b560'],[0.52,'#3d8a3d'],[0.65,'#2d6a2d'],
                      [0.80,'#7a6a5a'],[1.0,'#ffffff']]
        traces=[go.Surface(x=x,y=y,z=z,surfacecolor=z+noise,colorscale=colorscale_g,
                           showscale=False,lighting=dict(ambient=0.5,diffuse=0.8,specular=0.3),
                           lightposition=dict(x=2,y=2,z=2),hoverinfo='skip')]
        r2=1.02
        traces.append(go.Surface(x=x*r2,y=y*r2,z=z*r2,
                                  surfacecolor=np.ones_like(z),
                                  colorscale=[[0,'rgba(80,140,255,0.06)'],[1,'rgba(80,140,255,0.06)']],
                                  showscale=False,opacity=0.1,hoverinfo='skip'))
        rng_s=np.random.default_rng(7); ns=400
        phi_s=rng_s.uniform(0,2*np.pi,ns); theta_s=rng_s.uniform(0,np.pi,ns)
        rs=rng_s.uniform(3,5,ns)
        traces.append(go.Scatter3d(x=rs*np.sin(theta_s)*np.cos(phi_s),
                                    y=rs*np.sin(theta_s)*np.sin(phi_s),
                                    z=rs*np.cos(theta_s),mode='markers',
                                    marker=dict(size=rng_s.uniform(.5,2.5,ns),color='white',opacity=.7),
                                    hoverinfo='skip'))
        t_eq=np.linspace(0,2*np.pi,200)
        traces.append(go.Scatter3d(x=np.cos(t_eq),y=np.sin(t_eq),z=np.zeros_like(t_eq),
                                    mode='lines',line=dict(color='rgba(255,255,100,0.35)',width=2),hoverinfo='skip'))
        lat_r=math.radians(lat); lon_r=math.radians(lon)
        mx=math.cos(lat_r)*math.cos(lon_r); my=math.cos(lat_r)*math.sin(lon_r); mz=math.sin(lat_r)
        traces.append(go.Scatter3d(x=[mx*1.06],y=[my*1.06],z=[mz*1.06],mode='markers+text',
                                    marker=dict(size=12,color='#ff4444',line=dict(color='white',width=2)),
                                    text=['📍'],textposition='top center',hoverinfo='skip'))
        fig_g=go.Figure(data=traces)
        fig_g.update_layout(paper_bgcolor='#000010',
            scene=dict(bgcolor='#000010',xaxis=dict(visible=False,range=[-2,2]),
                       yaxis=dict(visible=False,range=[-2,2]),zaxis=dict(visible=False,range=[-2,2]),
                       aspectmode='cube',camera=dict(eye=dict(x=1.6,y=1.6,z=0.8))),
            margin=dict(l=0,r=0,t=0,b=0),showlegend=False,height=440,
            annotations=[dict(text='🖱️ Влачи = въртене  |  Scroll = zoom',
                              x=.5,y=.02,xref='paper',yref='paper',showarrow=False,
                              font=dict(color='rgba(200,200,200,0.5)',size=10))])
        st.plotly_chart(fig_g, use_container_width=True)
        st.caption(f'📍 {rtype} | {lat}°{"N" if lat>=0 else "S"}, {lon}°{"E" if lon>=0 else "W"} | {rdesc}')

    with col_a:
        st.markdown('### 📊 Анализ на региона')
        arr = regions[selected_r]
        H = shannon_entropy(arr)
        color, level = entropy_color(H)

        st.image(Image.fromarray(arr), caption=f'Текстура: {selected_r}', use_container_width=True)

        c1,c2,c3 = st.columns(3)
        c1.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{color}">{H:.2f}</div><div class="metric-lbl">бита ентропия</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{color};font-size:20px">{level}</div><div class="metric-lbl">ниво</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-val">{H/8*100:.0f}%</div><div class="metric-lbl">от максимум</div></div>', unsafe_allow_html=True)

        # Хистограма
        gray=np.mean(arr,axis=2).astype(np.uint8)
        hist,bins=np.histogram(gray.flatten(),bins=64,range=(0,256))
        fig_h=go.Figure()
        fig_h.add_trace(go.Bar(x=bins[:-1],y=hist,marker_color=color,marker_line_width=0))
        fig_h.add_annotation(text=f'H = {H:.3f} бита',x=.98,y=.95,xref='paper',yref='paper',
                              showarrow=False,font=dict(color='#4ecca3',size=13),bgcolor='rgba(0,0,0,0.5)')
        fig_h.update_layout(paper_bgcolor='#161b22',plot_bgcolor='#0d1117',
                             font=dict(color='#e6edf3',size=10),
                             margin=dict(l=5,r=5,t=5,b=25),height=180,
                             xaxis=dict(title='Яркост',gridcolor='#30363d'),
                             yaxis=dict(gridcolor='#30363d'),showlegend=False)
        st.plotly_chart(fig_h, use_container_width=True)

        # Ентропийна карта
        emap=entropy_map(arr,window=15)
        fig_e=go.Figure(data=go.Heatmap(z=emap,colorscale='Plasma',
                                         colorbar=dict(title=dict(text='бита',font=dict(color='#e6edf3')),
                                                       tickfont=dict(color='#e6edf3'))))
        fig_e.update_layout(paper_bgcolor='#161b22',plot_bgcolor='#0d1117',
                             font=dict(color='#e6edf3'),margin=dict(l=5,r=5,t=5,b=5),height=160)
        st.plotly_chart(fig_e, use_container_width=True)
        st.caption('🗺️ Ентропийна карта — тъмно=ниска, светло=висока')

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('### 📈 Сравнение на всички региони')

    names=[]; vals=[]; clrs=[]
    cmap={'🌿':'#3fb950','🏜':'#e3b341','🏙':'#f78166','🌊':'#58a6ff','🏔':'#bc8cff'}
    for n,a in regions.items():
        names.append(n); vals.append(shannon_entropy(a))
        clrs.append(cmap.get(n[0],'#58a6ff'))

    fig_b=go.Figure()
    fig_b.add_trace(go.Bar(x=names,y=vals,marker_color=clrs,
                            text=[f'{v:.2f}' for v in vals],textposition='outside',
                            textfont=dict(color='#e6edf3',size=13)))
    fig_b.add_hline(y=8,line_dash='dash',line_color='rgba(255,255,255,0.2)',
                    annotation_text='Максимум 8 бита',annotation_font_color='#8b949e')
    fig_b.update_layout(paper_bgcolor='#161b22',plot_bgcolor='#0d1117',
                         font=dict(color='#e6edf3'),
                         margin=dict(l=10,r=10,t=20,b=10),height=300,
                         yaxis=dict(title='Shannon ентропия (бита)',gridcolor='#30363d',range=[0,9.5]),
                         xaxis=dict(gridcolor='#30363d'),showlegend=False)
    st.plotly_chart(fig_b, use_container_width=True)

# ══════════════════════════════════════════════════════════
# ТАБ 2: ТЕКСТУРИ
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown('## 🔬 Природни Текстури — Ентропия на Материята')
    st.markdown('<div class="info-card">Природата създава текстури с различна информационна сложност. От кристалния порядък до хаоса на лавата — Shannon ентропията измерва всичко.</div>', unsafe_allow_html=True)

    # Всички текстури в решетка
    st.markdown('### 🎯 Всички текстури наведнъж')
    tex_names = list(textures.keys())
    tex_arrays = list(textures.values())

    cols = st.columns(6)
    for i, (name, arr) in enumerate(textures.items()):
        H = shannon_entropy(arr)
        color, level = entropy_color(H)
        with cols[i]:
            st.image(Image.fromarray(arr), caption=name.split()[1], use_container_width=True)
            st.markdown(f'<div style="text-align:center;font-size:18px;font-weight:bold;color:{color}">{H:.2f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="text-align:center;font-size:10px;color:#8b949e">{level}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col_t1, col_t2 = st.columns([1, 1.4])

    with col_t1:
        st.markdown('### 🔍 Детайлен анализ')
        selected_t = st.selectbox('Избери текстура:', list(textures.keys()), key='tex')
        arr_t = textures[selected_t]
        H_t = shannon_entropy(arr_t)
        color_t, level_t = entropy_color(H_t)

        st.image(Image.fromarray(arr_t), use_container_width=True)

        st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{color_t}">{H_t:.4f} бита</div><div class="metric-lbl">{level_t} ентропия &nbsp;|&nbsp; {H_t/8*100:.1f}% от максимум</div></div>', unsafe_allow_html=True)

        # Ентропийна карта
        emap_t = entropy_map(arr_t, window=16)
        fig_et = go.Figure(data=go.Heatmap(z=emap_t, colorscale='Viridis',
                                            colorbar=dict(title=dict(text='бита',font=dict(color='#e6edf3')),
                                                          tickfont=dict(color='#e6edf3'))))
        fig_et.update_layout(paper_bgcolor='#161b22',plot_bgcolor='#0d1117',
                              font=dict(color='#e6edf3'),margin=dict(l=5,r=5,t=5,b=5),height=220)
        st.plotly_chart(fig_et, use_container_width=True)

    with col_t2:
        st.markdown('### 📊 RGB канали & хистограма')

        # RGB разбивка
        arr_t = textures[selected_t]
        channels = ['Червен', 'Зелен', 'Син']
        ch_colors = ['#ff6e6e','#3fb950','#58a6ff']
        fig_rgb = make_subplots(rows=1, cols=3, subplot_titles=channels)
        for i in range(3):
            h,b = np.histogram(arr_t[:,:,i].flatten(), bins=64, range=(0,256))
            fig_rgb.add_trace(go.Bar(x=b[:-1],y=h,marker_color=ch_colors[i],marker_line_width=0,
                                      showlegend=False), row=1, col=i+1)
        fig_rgb.update_layout(paper_bgcolor='#161b22',plot_bgcolor='#0d1117',
                               font=dict(color='#e6edf3',size=10),
                               margin=dict(l=5,r=5,t=30,b=5),height=200)
        fig_rgb.update_xaxes(gridcolor='#30363d'); fig_rgb.update_yaxes(gridcolor='#30363d')
        st.plotly_chart(fig_rgb, use_container_width=True)

        # Ентропии по канал
        st.markdown('#### Ентропия по RGB канал')
        ch_entropies = [shannon_entropy(arr_t[:,:,i]) for i in range(3)]
        fig_ch = go.Figure()
        fig_ch.add_trace(go.Bar(x=['Червен','Зелен','Син'], y=ch_entropies,
                                 marker_color=ch_colors, text=[f'{v:.2f}' for v in ch_entropies],
                                 textposition='outside', textfont=dict(color='#e6edf3')))
        fig_ch.update_layout(paper_bgcolor='#161b22',plot_bgcolor='#0d1117',
                              font=dict(color='#e6edf3'),margin=dict(l=5,r=5,t=10,b=5),height=220,
                              yaxis=dict(gridcolor='#30363d',range=[0,9]),
                              xaxis=dict(gridcolor='#30363d'),showlegend=False)
        st.plotly_chart(fig_ch, use_container_width=True)

        # Радар
        st.markdown('#### Профил на текстурата')
        all_H = {n: shannon_entropy(a) for n,a in textures.items()}
        categories = [n.split()[1] for n in all_H.keys()]
        values = list(all_H.values())
        values.append(values[0])
        categories.append(categories[0])
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(r=values, theta=categories,
                                         fill='toself', fillcolor='rgba(88,166,255,0.2)',
                                         line=dict(color='#58a6ff', width=2)))
        fig_r.update_layout(polar=dict(bgcolor='#161b22',
                                        radialaxis=dict(visible=True,range=[0,8],gridcolor='#30363d',color='#8b949e'),
                                        angularaxis=dict(gridcolor='#30363d',color='#e6edf3')),
                             paper_bgcolor='#161b22',font=dict(color='#e6edf3'),
                             margin=dict(l=30,r=30,t=20,b=20),height=260,showlegend=False)
        st.plotly_chart(fig_r, use_container_width=True)

# ══════════════════════════════════════════════════════════
# ТАБ 3: ИЗКУСТВО
# ══════════════════════════════════════════════════════════
with tab3:
    st.markdown('## 🎨 Изкуство & Математика — Ентропията на Творчеството')
    st.markdown('<div class="info-card">Различните художествени стилове имат различна информационна сложност. Кубизмът на Пикасо е математически по-сложен от плоските полета на Ротко. Shannon ентропията измерва "хаоса на творчеството".</div>', unsafe_allow_html=True)

    # Галерия
    st.markdown('### 🖼️ Галерия — Ентропия на изкуството')
    paint_cols = st.columns(6)
    paint_Hs = {}
    for i, (name, arr) in enumerate(paintings.items()):
        H = shannon_entropy(arr)
        paint_Hs[name] = H
        color, level = entropy_color(H)
        with paint_cols[i]:
            st.image(Image.fromarray(arr), use_container_width=True)
            artist = name.split('—')[0].strip()
            style  = name.split('—')[1].strip() if '—' in name else ''
            st.markdown(f'<div style="text-align:center;font-size:11px;color:#e6edf3;font-weight:bold">{artist}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="text-align:center;font-size:18px;font-weight:bold;color:{color}">{H:.2f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="text-align:center;font-size:9px;color:#8b949e">{level}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col_p1, col_p2 = st.columns([1, 1.4])

    with col_p1:
        st.markdown('### 🔍 Детайлен анализ')
        selected_p = st.selectbox('Избери картина:', list(paintings.keys()), key='paint')
        arr_p = paintings[selected_p]
        H_p = shannon_entropy(arr_p)
        color_p, level_p = entropy_color(H_p)

        st.image(Image.fromarray(arr_p), use_container_width=True)
        st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{color_p}">{H_p:.4f} бита</div><div class="metric-lbl">{level_p} ентропия &nbsp;|&nbsp; {H_p/8*100:.1f}% от максимум</div></div>', unsafe_allow_html=True)

        # Ентропийна карта
        emap_p = entropy_map(arr_p, window=15)
        fig_ep = go.Figure(data=go.Heatmap(z=emap_p, colorscale='Hot',
                                            colorbar=dict(title=dict(text='бита',font=dict(color='#e6edf3')),
                                                          tickfont=dict(color='#e6edf3'))))
        fig_ep.update_layout(paper_bgcolor='#161b22',plot_bgcolor='#0d1117',
                              font=dict(color='#e6edf3'),margin=dict(l=5,r=5,t=5,b=5),height=220)
        st.plotly_chart(fig_ep, use_container_width=True)
        st.caption('🔥 Ентропийна карта — кои части са най-сложни')

    with col_p2:
        st.markdown('### 📊 Сравнение на стиловете')

        # Bar chart
        p_names = list(paint_Hs.keys())
        p_vals  = list(paint_Hs.values())
        p_colors_list = ['#f78166','#3fb950','#e3b341','#bc8cff','#ff6e6e','#58a6ff']
        fig_pb = go.Figure()
        fig_pb.add_trace(go.Bar(
            x=[n.split('—')[0].strip() for n in p_names],
            y=p_vals, marker_color=p_colors_list,
            text=[f'{v:.2f}' for v in p_vals],
            textposition='outside', textfont=dict(color='#e6edf3',size=12)
        ))
        fig_pb.update_layout(paper_bgcolor='#161b22',plot_bgcolor='#0d1117',
                              font=dict(color='#e6edf3'),
                              margin=dict(l=5,r=5,t=10,b=5),height=260,
                              yaxis=dict(title='Shannon ентропия',gridcolor='#30363d',range=[0,9.5]),
                              xaxis=dict(gridcolor='#30363d'),showlegend=False)
        st.plotly_chart(fig_pb, use_container_width=True)

        # Стилове vs ентропия
        st.markdown('#### 💡 Какво казва математиката за стила')
        style_info = [
            ("🌙 Ван Гог", paint_Hs.get("🌙 Ван Гог — Звездна нощ",0),
             "Вихрите и звездите създават богата текстура. Висока ентропия = емоционален интензитет."),
            ("🌸 Моне", paint_Hs.get("🌸 Моне — Водни лилии",0),
             "Импресионизмът е мека светлина. Средна ентропия = баланс между ред и хаос."),
            ("🔷 Пикасо", paint_Hs.get("🔷 Пикасо — Кубизъм",0),
             "Кубизмът разбива формите. Висока ентропия = фрагментирана реалност."),
            ("⭕ Кандински", paint_Hs.get("⭕ Кандински — Абстракция",0),
             "Чиста абстракция = максимален хаос на цвета. Много висока ентропия."),
            ("🟥 Ротко", paint_Hs.get("🟥 Ротко — Цветни полета",0),
             "Плоски цветни полета = ниска ентропия. Простотата е математически измерима."),
            ("🔵 Сьора", paint_Hs.get("🔵 Сьора — Пуантилизъм",0),
             "Хиляди точки = висока ентропия на микро ниво, но структура на макро ниво."),
        ]
        for artist, H, desc in style_info:
            color, level = entropy_color(H)
            st.markdown(f'**{artist}** <span style="color:{color}">({H:.2f} бита — {level})</span><br><span style="font-size:12px;color:#8b949e">{desc}</span>', unsafe_allow_html=True)
            st.markdown('')

    # Scatter plot
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('### 🌐 Всичко заедно — Ентропия на света')

    all_names = (
        [f'🌍 {n.split()[1]}' for n in regions.keys()] +
        [f'🔬 {n.split()[1]}' for n in textures.keys()] +
        [f'🎨 {n.split("—")[0].strip().split()[1]}' for n in paintings.keys()]
    )
    all_vals = (
        [shannon_entropy(a) for a in regions.values()] +
        [shannon_entropy(a) for a in textures.values()] +
        [shannon_entropy(a) for a in paintings.values()]
    )
    all_cats = (['Земя']*5 + ['Текстура']*6 + ['Изкуство']*6)
    cat_colors = {'Земя':'#58a6ff','Текстура':'#3fb950','Изкуство':'#f78166'}

    fig_all = go.Figure()
    for cat in ['Земя','Текстура','Изкуство']:
        idxs = [i for i,c in enumerate(all_cats) if c==cat]
        fig_all.add_trace(go.Bar(
            x=[all_names[i] for i in idxs],
            y=[all_vals[i] for i in idxs],
            name=cat, marker_color=cat_colors[cat],
            text=[f'{all_vals[i]:.2f}' for i in idxs],
            textposition='outside', textfont=dict(color='#e6edf3',size=11)
        ))
    fig_all.add_hline(y=8, line_dash='dash', line_color='rgba(255,255,255,0.2)',
                       annotation_text='Теоретичен максимум 8 бита',
                       annotation_font_color='#8b949e')
    fig_all.update_layout(
        paper_bgcolor='#161b22', plot_bgcolor='#0d1117',
        font=dict(color='#e6edf3'), barmode='group',
        margin=dict(l=10,r=10,t=20,b=10), height=380,
        yaxis=dict(title='Shannon ентропия (бита)', gridcolor='#30363d', range=[0,9.8]),
        xaxis=dict(gridcolor='#30363d'),
        legend=dict(bgcolor='#161b22', bordercolor='#30363d', borderwidth=1),
    )
    st.plotly_chart(fig_all, use_container_width=True)

    st.markdown('<div class="info-card">💡 <b>Кибернетичен извод:</b> Shannon ентропията е универсална метрика. Тя измерва едно и също нещо в природата, изкуството и технологиите — количеството информация. Градовете, абстрактното изкуство и лавата са математически сродни: всички са "богати на информация". Океанът, Ротко и пясъкът споделят математическата простота.</div>', unsafe_allow_html=True)

st.markdown('<p style="text-align:center;color:#8b949e;font-size:12px;margin-top:20px">Shannon Ентропия · Теория на информацията · Python · Streamlit · Plotly · 2025</p>', unsafe_allow_html=True)
