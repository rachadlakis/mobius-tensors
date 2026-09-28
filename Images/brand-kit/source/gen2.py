import math, os, glob
from inf import mix, hexc, norm, cross, CYAN, VIOLET
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
BG, INK = "#050a14", "#0b1220"
OUT = "logos2"; os.makedirs(OUT, exist_ok=True)
TL = math.radians(42); W = 0.30; ZS = 0.24
def C(t): return [1.25*math.cos(t), 0.62*math.sin(2*t), ZS*math.sin(t)]
def V(t, v, twist):
    e=1e-4; a=C(t-e); b=C(t+e); T=norm([b[i]-a[i] for i in range(3)])
    Nn=norm(cross([0,0,1],T)); Z=cross(T,Nn)
    g = twist*math.pi/2*(1-math.cos(t))/2
    d=[math.cos(g)*Z[i]+math.sin(g)*Nn[i] for i in range(3)]
    c=C(t); x,y,z=[c[i]+v*d[i] for i in range(3)]
    return (x, -(-y*math.sin(TL)+z*math.cos(TL)), -(y*math.cos(TL)+z*math.sin(TL)))
L = norm([-0.35,-0.7,0.6])
def col(t, lit): g=1-abs(1-2*t); return hexc([v*(0.42+0.75*lit) for v in mix(CYAN,VIOLET,g)])
def quads(nu, nv, twist):
    qs=[]
    for i in range(nu):
        t0,t1=2*math.pi*i/nu,2*math.pi*(i+1)/nu
        for j in range(nv):
            v0,v1=-W+2*W*j/nv,-W+2*W*(j+1)/nv
            pts=[V(t0,v0,twist),V(t1,v0,twist),V(t1,v1,twist),V(t0,v1,twist)]
            e1=[pts[1][k]-pts[0][k] for k in range(3)]; e2=[pts[3][k]-pts[0][k] for k in range(3)]
            n=norm(cross(e1,e2)); lit=abs(sum(n[k]*L[k] for k in range(3)))
            qs.append((sum(p[2] for p in pts)/4, pts, (i+0.5)/nu, lit))
    qs.sort(key=lambda q:q[0]); return qs
_b=[V(2*math.pi*i/300,v,k) for i in range(300) for v in (-W,W) for k in (0,1)]
MX=(min(p[0] for p in _b)+max(p[0] for p in _b))/2; MY=(min(p[1] for p in _b)+max(p[1] for p in _b))/2
SP=max(max(p[0] for p in _b)-min(p[0] for p in _b),1.6*(max(p[1] for p in _b)-min(p[1] for p in _b)))
def pr(p,s,cx,cy): return (cx+(p[0]-MX)/SP*2*s, cy+(p[1]-MY)/SP*2*s)
def pts_s(pts,s,cx,cy): return " ".join("%.2f,%.2f"%pr(p,s,cx,cy) for p in pts)
def icon_ribbon(s,cx,cy,twist=0):
    return "".join('<polygon points="%s" fill="%s" stroke="%s" stroke-width="0.8"/>'%(pts_s(p,s,cx,cy),col(t,l),col(t,l)) for d,p,t,l in quads(420,1,twist))
def icon_lattice(s,cx,cy):
    return "".join('<polygon points="%s" fill="%s" stroke="%s" stroke-width="%.2f" stroke-linejoin="round"/>'%(pts_s(p,s,cx,cy),col(t,l),BG,s*0.022) for d,p,t,l in quads(64,3,0))
def icon_nodes(s,cx,cy):
    ps=[]
    for i in range(56):
        t=2*math.pi*i/56
        for j in range(3):
            v=-W+W*j; p=V(t,v,0); ps.append((p[2],p,i/56))
    ps.sort(key=lambda q:q[0]); zs=[q[0] for q in ps]; z0,z1=min(zs),max(zs); out=[]
    for d,p,t in ps:
        k=(d-z0)/(z1-z0); x,y=pr(p,s,cx,cy)
        out.append('<circle cx="%.2f" cy="%.2f" r="%.2f" fill="%s"/>'%(x,y,s*(0.028+0.03*k),col(t,0.35+0.65*k)))
    return "".join(out)
ICONS={"ribbon":icon_ribbon,"twist":lambda s,cx,cy:icon_ribbon(s,cx,cy,1),"lattice":icon_lattice,"nodes":icon_nodes}
FB=TTFont(glob.glob("*space-grotesk*/package/files/space-grotesk-latin-600-normal.woff2")[0])
FL=TTFont(glob.glob("*space-grotesk*/package/files/space-grotesk-latin-500-normal.woff2")[0])
def tp(font,txt,x,y,size,track=0):
    gs=font.getGlyphSet(); cm=font.getBestCmap(); sc=size/font["head"].unitsPerEm; d=[]; cur=x
    for ch in txt:
        g=cm[ord(ch)]; pen=SVGPathPen(gs); gs[g].draw(TransformPen(pen,(sc,0,0,-sc,cur,y))); d.append(pen.getCommands()); cur+=gs[g].width*sc+track*size
    return " ".join(d)
def lockup(n,bg,tc,sc_,f):
    icon=ICONS[n](112,130,120)
    svg=('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 700 240" width="700" height="240"><rect width="700" height="240" fill="%s"/>%s'
         '<path d="%s" fill="%s"/><path d="%s" fill="%s"/></svg>')%(bg,icon,tp(FB,"MOBIUS",268,128,74,0.04),tc,tp(FL,"TENSORS",271,176,34,0.42),sc_)
    open(f"{OUT}/{f}","w").write(svg)
def iconf(n,f,size=512):
    open(f"{OUT}/{f}","w").write('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d"><rect width="%d" height="%d" rx="%d" fill="%s"/>%s</svg>'%(size,size,size,size,size,size,size*0.22,BG,ICONS[n](size*0.37,size/2,size/2)))
for n in ICONS:
    lockup(n,BG,"#eaf4ff","#7ae1ff",f"mobius-{n}-lockup-dark.svg")
    lockup(n,"#ffffff",INK,"#5b4bd6",f"mobius-{n}-lockup-light.svg")
    iconf(n,f"mobius-{n}-icon.svg")
rows="".join(f'<img src="logos2/mobius-{n}-lockup-dark.svg"><img src="logos2/mobius-{n}-icon.svg" width=240>' for n in ICONS)
open("sheet2.html","w").write('<body style="margin:0;background:#1a1f2b;display:grid;grid-template-columns:700px 240px;gap:12px;padding:12px">'+rows+'<img src="logos2/mobius-ribbon-lockup-light.svg"><img src="logos2/mobius-lattice-lockup-light.svg" style="width:240px"></body>')
