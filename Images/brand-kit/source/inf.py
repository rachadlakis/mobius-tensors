import math
CYAN,VIOLET=(0x7a,0xe1,0xff),(0xa7,0x8b,0xfa)
def mix(a,b,t): return tuple(int(a[i]+(b[i]-a[i])*t) for i in range(3))
def hexc(c): return "#%02x%02x%02x"%tuple(max(0,min(255,int(v))) for v in c)
def norm(v): l=math.sqrt(sum(x*x for x in v)) or 1; return [x/l for x in v]
def cross(a,b): return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def make(mode, tilt, W=0.28, zsep=0.22, N=360):
    TL=math.radians(tilt)
    def C(t): return [1.25*math.cos(t), 0.62*math.sin(2*t), zsep*math.sin(t)]
    def ang(t):
        if mode=="half": return t/2
        if mode=="vert": return 0
        if mode=="flip": # twist concentrated near crossing t=pi/2
            return math.pi/2*(1-math.cos(t))/2* (1) 
        return 0
    def frame(t):
        e=1e-4; a=C(t-e); b=C(t+e); T=norm([b[i]-a[i] for i in range(3)])
        Nn=norm(cross([0,0,1],T)); Z=cross(T,Nn)
        g=ang(t); return [math.cos(g)*Z[i]+math.sin(g)*Nn[i] for i in range(3)]
    def rot(p):
        x,y,z=p; y2=y*math.cos(TL)-z*math.sin(TL); z2=y*math.sin(TL)+z*math.cos(TL)
        return (x,-z2, y2)  # screen x, screen y (z up -> negative), depth y2? 
    # view: camera looks along +y after tilt; use screen y = -z', depth = -y'
    def V(t,v):
        c=C(t); d=frame(t); p=[c[i]+v*d[i] for i in range(3)]
        x,y,z=p; y2=y*math.cos(TL)+z*math.sin(TL); z2=-y*math.sin(TL)+z*math.cos(TL)
        return (x,-z2,-y2)
    L=norm([-0.3,-0.6,0.75]); qs=[]
    for i in range(N):
        t0,t1=2*math.pi*i/N,2*math.pi*(i+1)/N
        pts=[V(t0,-W),V(t1,-W),V(t1,W),V(t0,W)]
        e1=[pts[1][k]-pts[0][k] for k in range(3)]; e2=[pts[3][k]-pts[0][k] for k in range(3)]
        n=norm(cross(e1,e2)); lit=abs(sum(n[k]*L[k] for k in range(3)))
        tt=i/N; g=1-abs(1-2*tt)
        c=[v*(0.6+0.5*lit) for v in mix(CYAN,VIOLET,g)]
        qs.append((sum(p[2] for p in pts)/4,pts,hexc(c)))
    qs.sort(key=lambda q:q[0])
    xs=[p[0] for q in qs for p in q[1]]; ys=[p[1] for q in qs for p in q[1]]
    mx,my=(min(xs)+max(xs))/2,(min(ys)+max(ys))/2; sp=max(max(xs)-min(xs),max(ys)-min(ys))
    return qs,mx,my,sp
def svg_icon(qs,mx,my,sp,s,cx,cy):
    out=[]
    for d,pts,c in qs:
        pp=" ".join("%.2f,%.2f"%(cx+(p[0]-mx)/sp*2*s, cy+(p[1]-my)/sp*2*s) for p in pts)
        out.append('<polygon points="%s" fill="%s" stroke="%s" stroke-width="0.7"/>'%(pp,c,c))
    return "".join(out)
if __name__=="__main__":
    cells=[]
    for mode in ("vert","half","flip"):
        for tilt in (20,45,70):
            q=make(mode,tilt); cells.append(f'<div style="color:#fff;font:12px sans-serif">{mode} {tilt}<svg viewBox="0 0 300 300" width=260 height=260 style="background:#050a14">{svg_icon(*q,120,150,150)}</svg></div>')
    open('try.html','w').write('<body style="margin:0;background:#222;display:grid;grid-template-columns:repeat(3,270px)">'+''.join(cells))
