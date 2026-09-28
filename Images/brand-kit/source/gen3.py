from gen2 import *
import gen2
def cells(nu,nv):
    qs=[]
    for i in range(nu):
        t0,t1=2*math.pi*i/nu,2*math.pi*(i+1)/nu
        for j in range(nv):
            v0,v1=-W+2*W*j/nv,-W+2*W*(j+1)/nv
            pts=[V(t0,v0,0),V(t1,v0,0),V(t1,v1,0),V(t0,v1,0)]
            e1=[pts[1][k]-pts[0][k] for k in range(3)]; e2=[pts[3][k]-pts[0][k] for k in range(3)]
            n=norm(cross(e1,e2)); lit=abs(sum(n[k]*L[k] for k in range(3)))
            qs.append((sum(p[2] for p in pts)/4,pts,(i+0.5)/nu,lit,i,j))
    qs.sort(key=lambda q:q[0]); return qs
def render(s,cx,cy,nv=3,lanes=True,every=0,dots=False,nu=360,lw=0.018,lc=None,lo=0.9):
    out=[]; lc=lc or BG; top=[]
    for d,p,t,l,i,j in cells(nu,nv):
        c=col(t,l); P=[pr(q,s,cx,cy) for q in p]
        out.append('<polygon points="%s" fill="%s" stroke="%s" stroke-width="0.8"/>'%(" ".join("%.2f,%.2f"%x for x in P),c,c))
        ln='<line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" stroke="'+lc+'" stroke-opacity="%.2f" stroke-width="%.2f" stroke-linecap="round"/>'
        if lanes and j<nv-1: out.append(ln%(P[3][0],P[3][1],P[2][0],P[2][1],lo,s*lw))
        if every and i%every==0: out.append(ln%(P[0][0],P[0][1],P[3][0],P[3][1],lo,s*lw))
        if dots and i%every==0 and j<nv-1:
            top.append((d,P[3][0],P[3][1]))
    # dots: draw after everything, but hide ones whose point is occluded (depth below a nearby front quad) -> approximate: only keep dots on front-ish surface
    for d,x,y in top:
        out.append('<circle cx="%.2f" cy="%.2f" r="%.2f" fill="#7ae1ff" fill-opacity="0.35"/><circle cx="%.2f" cy="%.2f" r="%.2f" fill="#ffffff"/>'%(x,y,s*0.05,x,y,s*0.026))
    return "".join(out)
V3={
 "mesh":  lambda s,cx,cy: render(s,cx,cy,3,True,15),
 "lanes": lambda s,cx,cy: render(s,cx,cy,3,True,0),
 "nodes": lambda s,cx,cy: render(s,cx,cy,3,True,20,dots=True,lo=0.55,lw=0.01),
}
gen2.ICONS.clear(); gen2.ICONS.update(V3); gen2.OUT="logos3"; os.makedirs("logos3",exist_ok=True)
for n in V3:
    lockup(n,BG,"#eaf4ff","#7ae1ff",f"mobius-{n}-lockup-dark.svg")
    lockup(n,"#ffffff",INK,"#5b4bd6",f"mobius-{n}-lockup-light.svg")
    iconf(n,f"mobius-{n}-icon.svg")
rows="".join(f'<img src="logos3/mobius-{n}-lockup-dark.svg"><img src="logos3/mobius-{n}-icon.svg" width=240>' for n in V3)
open("sheet3.html","w").write('<body style="margin:0;background:#1a1f2b;display:grid;grid-template-columns:700px 240px;gap:12px;padding:12px">'+rows+'</body>')
