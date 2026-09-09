import numpy as np, colorsys
from PIL import Image

def _hls(C):
    mx=C.max(-1); mn=C.min(-1); L=(mx+mn)/2; d=mx-mn
    S=np.zeros_like(L); m=d>1e-9
    S[m]=np.where(L[m]<0.5, d[m]/np.maximum(mx[m]+mn[m],1e-9), d[m]/np.maximum(2-mx[m]-mn[m],1e-9))
    r,g,b=C[...,0],C[...,1],C[...,2]
    H=np.zeros_like(L)
    rm=m&(mx==r); gm=m&(mx==g)&~rm; bm=m&~rm&~gm
    H[rm]=((g-b)[rm]/d[rm])%6; H[gm]=((b-r)[gm]/d[gm])+2; H[bm]=((r-g)[bm]/d[bm])+4
    return H/6, L, S

def dark_to_light(arr, ink_L=0.38, sat_gain=1.15, sat_thresh=0.12, surface=0.0):
    """Original was ink composited over a black surface: c = alpha*C.
    Recover alpha and the pure ink hue, map the ink to a dark equivalent,
    then re-composite over white. Antialiased pixels fade to white instead
    of turning into saturated fringe."""
    a = arr.astype(np.float64)/255.0
    alpha = a.max(-1)
    if surface > 0:  # map the original surface colour to true white
        alpha = np.clip((alpha-surface)/(1.0-surface), 0, 1)
    C = np.where(alpha[...,None] > 1e-6, a/np.maximum(alpha[...,None],1e-6), 0.0)
    H,L,S = _hls(C)
    chroma = S > sat_thresh
    Ln = np.where(chroma, ink_L, 0.0)               # colored ink -> mid-dark; white ink -> black
    Sn = np.where(chroma, np.clip(S*sat_gain,0,1), 0.0)
    ink = np.array([colorsys.hls_to_rgb(h,l,s) for h,l,s in
                    zip(H.ravel(), Ln.ravel(), Sn.ravel())]).reshape(a.shape)
    out = (1-alpha)[...,None]*1.0 + alpha[...,None]*ink
    return (np.clip(out,0,1)*255).astype(np.uint8)

if __name__ == "__main__":
    import sys
    im = Image.open(sys.argv[1]).convert("RGB")
    Image.fromarray(dark_to_light(np.array(im))).save(sys.argv[2])
    print("wrote", sys.argv[2])
