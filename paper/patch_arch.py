"""Patch generate_figures.py: replace fig_system_architecture with rich pipeline."""
import re, pathlib

src = pathlib.Path(__file__).parent / "generate_figures.py"
text = src.read_text(encoding="utf-8")

NEW_FN = r'''# ─── Figure 1: System Architecture ───────────────────────────────────────────
def fig_system_architecture():
    """Beautiful detailed Spectral Sentinel ML pipeline diagram."""
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.2)
    ax.axis('off')

    PAL = {'ch':'#C8E6FA','cb':'#FFD0B0','sk':'#EDE7F6','sp':'#FFF8E1',
           'fl':'#E8F5E9','bc':'#E3F2FD','md':'#FCE4EC','bg':'#FAFDF8'}
    EDG = {'ch':'#1565C0','cb':'#BF360C','sk':'#6A1B9A','sp':'#E65100',
           'fl':'#1B5E20','bc':'#0D47A1','md':'#880E4F','bg':'#8D7B2B'}

    def fbox(x,y,w,h,k,lw=1.3,alpha=0.88,rad=0.18,zo=3):
        ax.add_patch(FancyBboxPatch((x,y),w,h, boxstyle=f'round,pad={rad}',
            facecolor=PAL[k],edgecolor=EDG[k],linewidth=lw,zorder=zo,alpha=alpha))

    def arr(x1,y1,x2,y2,col='#555',lw=1.5,cs='arc3,rad=0',ms=12):
        ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
            arrowprops=dict(arrowstyle='-|>',color=col,lw=lw,
                mutation_scale=ms,connectionstyle=cs),zorder=7)

    def txt(x,y,s,fs=8.5,col='#222',bold=False,italic=False,
            ha='center',va='center',zo=8,**kw):
        ax.text(x,y,s,fontsize=fs,color=col,ha=ha,va=va,
            fontweight='bold' if bold else 'normal',
            fontstyle='italic' if italic else 'normal',zorder=zo,**kw)

    # ── CLIENTS ────────────────────────────────────────────────────────
    txt(0.85,6.0,'FL Clients',fs=9.5,bold=True,col='#333')
    for cy,cn,cs,byz in [(4.7,'Client 1','Honest',False),(3.6,'Client 2','Honest',False),
                          (2.5,'Client 3','Honest',False),(1.4,'Client 4','Byzantine',True)]:
        k='cb' if byz else 'ch'
        fbox(0.05,cy,1.6,0.78,k,lw=2.0 if byz else 1.3)
        txt(0.85,cy+0.54,cn,fs=9,bold=True,col=EDG[k])
        txt(0.85,cy+0.28,f'({cs})',fs=8,italic=True,col=EDG['cb'] if byz else '#555')
        grad=(r'$\tilde{g}_i$' if byz
              else r'$g_i\!\sim\!\mathcal{N}(\nabla F_i,\sigma^2 I)$')
        txt(0.85,cy+0.06,grad,fs=7,col='#666')
        arr(1.65,cy+0.39,2.05,cy+0.39,col=EDG['cb'] if byz else '#777',lw=1.2)
    txt(0.85,0.82,r'$n$ clients, $f<n/2$ Byzantine',fs=8.5,italic=True,col='#555')
    txt(0.85,0.5,r'Var: $\mathbb{E}[\xi_j^2]\!\leq\!\sigma^2$',fs=7.5,col='#777')

    # ── BIG AGGREGATOR BACKGROUND ──────────────────────────────────────
    fbox(1.95,0.12,7.35,5.75,'bg',lw=2.2,alpha=0.45,rad=0.3,zo=2)
    txt(5.62,6.0,'Spectral Sentinel \u2014 Byzantine-Robust Aggregator',
        fs=10.5,bold=True,col='#5D4037')
    for sx,slbl,sc in [(2.95,'\u2460  FD Sketch','#6A1B9A'),
                        (4.55,'\u2461  Eigenspectrum','#E65100'),
                        (6.25,'\u2462  KS Test','#C62828'),
                        (7.95,'\u2463  Filter+Agg','#1B5E20')]:
        txt(sx,5.72,slbl,fs=8.5,bold=True,col=sc)

    # ── STAGE A: Frequent Directions ───────────────────────────────────
    fbox(2.05,0.25,1.8,5.2,'sk',lw=1.5,alpha=0.8,rad=0.2)
    txt(2.95,5.12,r'$G \in \mathbb{R}^{n \times d}$',fs=9,bold=True,col=EDG['sk'])
    for r in range(5):
        for c_ in range(3):
            ax.add_patch(plt.Rectangle((2.12+c_*0.47,4.35-r*0.49),0.40,0.40,
                facecolor='#EF9A9A' if r==4 else '#9FA8DA',
                edgecolor='white',lw=0.7,alpha=0.85,zorder=4))
    ax.text(2.0,2.48,'Byzantine\nrow',fontsize=6.5,ha='right',va='center',
            color=EDG['cb'],style='italic',zorder=5)
    arr(2.02,2.52,2.12,2.52,col=EDG['cb'],lw=0.8)
    txt(2.95,2.95,'Frequent Directions',fs=9,bold=True,col=EDG['sk'])
    txt(2.95,2.67,r'$\tilde{G}=\mathrm{Sketch}(G,k)$',fs=8.5)
    txt(2.95,2.42,r'$\tilde{G}\in\mathbb{R}^{k\times d},\;k\!\ll\!n$',fs=8)
    for i,(lbl,pct,fc) in enumerate([(r'Full $O(d^2)$\u201428GB',1.0,'#EF9A9A'),
                                       (r'Sketch $O(k^2)$\u20140.9GB',0.032,'#A5D6A7')]):
        yb=1.85-i*0.48
        ax.add_patch(plt.Rectangle((2.12,yb),pct*1.52,0.33,
            facecolor=fc,edgecolor='gray',lw=0.5,alpha=0.85,zorder=4))
        txt(2.88,yb+0.165,lbl,fs=7,col='#222',zo=5)
    arr(2.88,1.28,2.88,1.08,col='#2E7D32',lw=1.3)
    txt(3.35,1.18,r'$31\!\times$\u2193 mem',fs=8,bold=True,col='#2E7D32')
    txt(2.95,0.72,
        r'$\hat{C}=\frac{1}{n}\tilde{G}\tilde{G}^T\!\in\!\mathbb{R}^{n\times n}$',fs=9)
    txt(2.95,0.42,'Gram matrix',fs=8,italic=True,col='#777')
    arr(3.85,2.8,4.05,2.8,col='#555',lw=1.8)

    # ── STAGE B: Eigenspectrum ─────────────────────────────────────────
    fbox(4.05,0.25,1.8,5.2,'sp',lw=1.5,alpha=0.8,rad=0.2)
    txt(4.95,5.12,'Eigendecomposition',fs=9,bold=True,col=EDG['sp'])
    txt(4.95,4.85,r'$\lambda_1\geq\lambda_2\geq\ldots\geq\lambda_n$',fs=8.5)
    np.random.seed(2026)
    n_s,d_s=20,300
    Gs=np.random.randn(n_s,d_s)*0.25; Gs[-4:]=np.random.randn(4,d_s)*1.8
    ev_s=np.linalg.eigvalsh((Gs@Gs.T)/d_s)
    lp_s=np.mean(ev_s[:n_s-4])*(1+np.sqrt(n_s/d_s))**2
    bins_s=np.linspace(ev_s.min(),ev_s.max()*1.05,15)
    hist_v,bin_e=np.histogram(ev_s,bins=bins_s)
    px_min,px_max,py_bot,py_top=4.12,5.78,2.8,4.55
    ev_range=bins_s[-1]-bins_s[0]; bw=(px_max-px_min)/len(hist_v)
    for i,(bst,cnt) in enumerate(zip(bin_e[:-1],hist_v)):
        bx=px_min+(bst-bins_s[0])/ev_range*(px_max-px_min)
        fc=C['red'] if bst>lp_s*0.9 else C['blue']
        ax.add_patch(plt.Rectangle((bx,py_bot),bw*0.82,
            cnt/hist_v.max()*(py_top-py_bot)*0.88,
            facecolor=fc,edgecolor='white',lw=0.4,alpha=0.75,zorder=4))
    lp_px=px_min+(lp_s-bins_s[0])/ev_range*(px_max-px_min)
    ax.plot([lp_px,lp_px],[py_bot,py_top],color=C['orange'],lw=2.0,ls='--',zorder=5)
    txt(lp_px+0.05,py_top-0.12,r'$\lambda_+$',fs=8.5,col=C['orange'],ha='left')
    txt(px_min+0.1,py_bot-0.12,r'$\lambda$',fs=7.5,col='#666',ha='left')
    ax.legend(handles=[
        mpatches.Patch(facecolor=C['blue'],alpha=0.7,label='Honest'),
        mpatches.Patch(facecolor=C['red'],alpha=0.7,label='Byzantine spike')],
        loc='upper right',fontsize=6.5,
        bbox_to_anchor=(5.82,4.62),bbox_transform=ax.transData,framealpha=0.9)
    txt(4.95,2.52,
        r'$\hat{\sigma}^2,\hat{\gamma}$ from honest bulk',fs=8,italic=True,col='#777')
    txt(4.95,2.22,r'$\lambda_+=\hat{\sigma}^2(1\!+\!\sqrt{\hat{\gamma}})^2$',fs=9)
    txt(4.95,1.88,'MP law upper edge',fs=8,italic=True,col='#777')
    txt(4.95,0.72,r'$n/d\!\to\!\gamma$, MP holds',fs=8,col='#555')
    txt(4.95,0.44,'for honest (Non-IID OK)',fs=7.5,italic=True,col='#777')
    arr(5.85,2.8,6.05,2.8,col='#555',lw=1.8)

    # ── STAGE C: KS Test ──────────────────────────────────────────────
    fbox(6.05,0.25,1.7,5.2,'sp',lw=1.5,alpha=0.8,rad=0.2)
    txt(6.9,5.12,'KS Detection',fs=9,bold=True,col='#C62828')
    txt(6.9,4.82,'KS goodness-of-fit:',fs=8.5,bold=True,col='#C62828')
    txt(6.9,4.55,r'$D_{KS}=\sup_\lambda|F_n-F_{MP}|$',fs=7.5)
    txt(6.9,4.28,r'Reject if $D_{KS}>\tau_{KS}$',fs=8,italic=True,col='#555')
    txt(6.9,3.95,'Tail anomaly test:',fs=8.5,bold=True,col='#C62828')
    txt(6.9,3.68,
        r'$\mathcal{A}=\{i:\lambda_i>\lambda_++\tau\hat{\sigma}^2\}$',fs=7.5)
    txt(6.9,3.41,r'Detect if $|\mathcal{A}|>0$',fs=8,italic=True,col='#555')
    fbox(6.12,2.70,1.55,0.52,'fl',lw=1.0,alpha=0.85,rad=0.1)
    txt(6.9,2.97,r'$P[\mathrm{detect}]\!\geq\!1-e^{-k/\log^2 k}$',
        fs=7.8,col=EDG['fl'])
    txt(6.9,2.74,'1-round completeness',fs=7.5,italic=True,col=EDG['fl'])
    fbox(6.12,2.05,1.55,0.55,'cb',lw=1.2,alpha=0.85,rad=0.1)
    txt(6.9,2.35,r'Phase trans.: $\sigma^2 f^2=0.25$',
        fs=8,bold=True,col='#BF360C')
    txt(6.9,2.12,r'$<0.25$: detectable; $\geq0.25$: impossible',
        fs=7.5,col='#C62828')
    txt(6.9,1.72,'Sketching error:',fs=8,bold=True,col='#555')
    txt(6.9,1.48,
        r'$|\lambda_i(C)-\lambda_i(\tilde{C})|\leq O(k^{-1/2})$',fs=7.5)
    txt(6.9,0.72,r'$k\geq\Omega(\log d)$ sufficient',fs=8,col='#555')
    txt(6.9,0.44,'for detection guarantee',fs=7.5,italic=True,col='#777')
    arr(7.75,2.8,7.95,2.8,col='#555',lw=1.8)

    # ── STAGE D: Filter & Aggregate ────────────────────────────────────
    fbox(7.95,0.25,1.7,5.2,'fl',lw=1.5,alpha=0.8,rad=0.2)
    txt(8.8,5.12,'Filter & Aggregate',fs=9,bold=True,col=EDG['fl'])
    txt(8.8,4.82,
        r'$\hat{\mathcal{B}}=\{i:g_i\notin\mathrm{MP\ bulk}\}$',fs=8.5)
    txt(8.8,4.52,r'$\mathcal{H}=[n]\setminus\hat{\mathcal{B}}$',fs=8.5)
    for ci in range(4):
        mark='✗' if ci==3 else '✓'
        col_=EDG['cb'] if ci==3 else EDG['ch']
        bg_='#FFCDD2' if ci==3 else '#C8E6C9'
        yp=4.18-ci*0.48
        ax.add_patch(plt.Rectangle((8.08,yp),1.42,0.36,
            facecolor=bg_,edgecolor=col_,lw=0.7,alpha=0.85,zorder=4))
        ax.text(8.35,yp+0.18,mark,fontsize=10,ha='center',va='center',
            color=col_,fontweight='bold',zorder=5)
        ax.text(8.82,yp+0.18,
            f'C{ci+1} {"excl." if ci==3 else "incl."}',
            fontsize=8,ha='center',va='center',color='#333',zorder=5)
    txt(8.8,2.22,'Honest-only mean:',fs=8.5,bold=True,col=EDG['fl'])
    txt(8.8,1.92,
        r'$\hat{g}=\frac{1}{|\mathcal{H}|}\sum_{i\in\mathcal{H}}g_i$',fs=10)
    txt(8.8,1.55,'Convergence:',fs=8.5,bold=True,col=EDG['fl'])
    txt(8.8,1.28,
        r'$\min_t\mathbb{E}[\|\nabla F\|^2]\leq O\!\left(\frac{\sigma f}{\sqrt{T}}+\frac{f^2}{T}\right)$',
        fs=7.8)
    txt(8.8,0.92,'Minimax optimal',fs=8,italic=True,col='#2E7D32')
    txt(8.8,0.64,
        r'$T^*=O(\sigma f/\varepsilon^2+f^2/\varepsilon)$',fs=7.8)

    # ── BLOCKCHAIN ─────────────────────────────────────────────────────
    arr(9.65,2.75,9.82,2.75,col=C['blue'],lw=2.0)
    txt(9.73,3.0,r'$\mathrm{hash}(w^t)$',fs=8,col=C['blue'],bold=True)
    fbox(9.82,0.4,1.55,5.5,'bc',lw=1.8,alpha=0.85,rad=0.2)
    txt(10.6,5.75,'Blockchain',fs=9.5,bold=True,col=EDG['bc'])
    txt(10.6,5.48,'Self-Stab. Shared Memory',fs=8,italic=True,col='#1565C0')
    for bi,(by,blbl,bfc) in enumerate([
            (4.85,r'Round $t{-}1$: $h_{t-1}$','#BBDEFB'),
            (4.22,r'Round $t$: $h_t$','#90CAF9'),
            (3.59,r'Round $t{+}1$: pending','#64B5F6')]):
        ax.add_patch(FancyBboxPatch((9.92,by),1.32,0.45,
            boxstyle='round,pad=0.05',facecolor=bfc,
            edgecolor=EDG['bc'],linewidth=0.9,alpha=0.9,zorder=4))
        txt(10.58,by+0.225,blbl,fs=7.5,col='#0D47A1',zo=5)
        if bi<2: arr(10.58,by,10.58,by-0.1,col=EDG['bc'],lw=0.8)
    txt(10.6,3.12,'BFT Consensus',fs=9,bold=True,col=EDG['bc'])
    txt(10.6,2.88,r'$f<n/2$ Byzantine',fs=8)
    for i,line in enumerate(['✦ Safety','✦ Liveness','✦ Closure (Self-Stab.)']):
        txt(10.6,2.60-i*0.22,line,fs=8,col='#0D47A1')
    txt(10.6,1.82,'Write-once immutability:',fs=8,bold=True,col='#555')
    txt(10.6,1.6,'No Byzantine rollback.',fs=7.5,italic=True,col='#777')
    txt(10.6,0.95,'Polygon/Hardhat testnet',fs=8,col='#555')
    txt(10.6,0.68,'Gas: 0.15 MATIC/round',fs=7.5,col='#777')

    # ── GLOBAL MODEL ───────────────────────────────────────────────────
    arr(11.37,2.75,11.52,2.75,col=EDG['md'],lw=2.0)
    txt(11.44,3.0,r'$w^{t+1}$',fs=8,col=EDG['md'],bold=True)
    fbox(11.52,1.0,1.4,4.2,'md',lw=1.8,alpha=0.88,rad=0.2)
    txt(12.22,5.0,'Global\nModel',fs=9.5,bold=True,col=EDG['md'])
    txt(12.22,4.5,r'$w^{t+1}$',fs=12,bold=True,col='#922B21')
    txt(12.22,4.12,r'$w^{t+1}=w^t-\eta\hat{g}$',fs=8.5)
    txt(12.22,3.78,r'$\eta=O(1/\sqrt{T})$',fs=8,italic=True,col='#777')
    txt(12.22,3.42,'Self-Stabilization:',fs=8.5,bold=True,col=EDG['md'])
    txt(12.22,3.16,r'Any $w^0\in\mathbb{R}^d$',fs=8)
    txt(12.22,2.9,'converges in $T^*$ rounds',fs=8)
    txt(12.22,2.55,'Phase transition:',fs=8.5,bold=True,col='#C62828')
    txt(12.22,2.3,r'$\sigma^2 f^2\geq0.25$',fs=8)
    txt(12.22,2.06,'impossible (FLP analogy)',fs=7.5,italic=True,col='#C62828')
    txt(12.22,1.55,'Experiment:',fs=8,bold=True,col='#555')
    txt(12.22,1.28,'78.4% vs 63.4%',fs=8,col='#2E7D32')

    # ── SELF-STABILIZING LOOP ──────────────────────────────────────────
    ax.annotate('',xy=(5.62,6.08),xytext=(12.22,5.22),
        arrowprops=dict(arrowstyle='-|>',color=C['purple'],lw=2.2,
            mutation_scale=14,connectionstyle='arc3,rad=-0.28'),zorder=9)
    ax.add_patch(FancyBboxPatch((3.8,5.9),5.65,0.24,boxstyle='round,pad=0.05',
        facecolor='white',edgecolor=C['purple'],linewidth=1.0,alpha=0.92,zorder=8))
    txt(6.62,6.02,
        r'Self-Stabilizing Loop \u2014 $T^*=O(\sigma f/\varepsilon^2+f^2/\varepsilon)$'
        r' from any initial $w^0$',
        fs=8.5,bold=True,col=C['purple'])

    fig.tight_layout(pad=0.3)
    path = OUTDIR / 'fig_architecture.pdf'
    fig.savefig(path)
    path_png = OUTDIR / 'fig_architecture.png'
    fig.savefig(path_png)
    plt.close()
    print(f"  Saved: {path}")

'''

# Replace between the two sentinel comments
pattern = (r'# \u2500+ Figure 1: System Architecture \u2500+\r?\n'
           r'def fig_system_architecture\(\):.*?'
           r'(?=\r?\n# \u2500+ Figure 2:)')
replacement = NEW_FN + '\n'
new_text, n = re.subn(pattern, replacement, text, flags=re.DOTALL)
if n == 0:
    raise RuntimeError("Pattern not found — check sentinel comments.")
src.write_text(new_text, encoding="utf-8")
print(f"Patched {n} occurrence(s) in {src}")
