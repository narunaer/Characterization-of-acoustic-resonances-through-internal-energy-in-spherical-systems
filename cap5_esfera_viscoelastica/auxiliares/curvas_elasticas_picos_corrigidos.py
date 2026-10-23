from google.colab import drive
drive.mount('/content/drive')
import sys,os,subprocess,importlib.util
if importlib.util.find_spec('scipy') is None:
    subprocess.check_call([sys.executable,'-m','pip','install','-q','scipy'])
def _garantir_latex_cm_super():
    def _tem_cm_super():
        try:
            subprocess.run(['kpsewhich','type1ec.sty'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False
    if _tem_cm_super(): return
    try:
        subprocess.run(['apt-get','update','-qq'],check=False)
        subprocess.run(['apt-get','install','-y','-qq','cm-super','texlive-latex-extra','dvipng'],check=False)
    except Exception:
        pass
_garantir_latex_cm_super()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,NullFormatter
from scipy.special import spherical_jn,spherical_yn,spherical_in
from scipy.signal import find_peaks
CHECKPOINT_PATH='/content/drive/MyDrive/dataset_elastico/dataset8.npz'
OUTPUT_DIR=os.path.join(os.path.dirname(CHECKPOINT_PATH),'curvas_individuais_parametros')
os.makedirs(OUTPUT_DIR,exist_ok=True)
CM_DEFAULT=1500.0
RHOM_DEFAULT=1000.0
LMAX=12
X_MIN=0.0001
X_MAX=1.0
N_MALHA_GLOBAL=3000
A_FIXED_CL_RATIO=1.0
A_FIXED_ETA=2.0
A_RHO_RATIOS=[0.1,1.0,10.0]
B_FIXED_RHO_RATIO=1.0
B_FIXED_ETA=2.0
B_CL_RATIOS=[0.1,1.0,10.0]
C_FIXED_CL_RATIO=1.0
C_FIXED_RHO_RATIO=1.0
C_ETAS=[1.16,2.0,5.0]
COLORS=['#0f223d','#810000','black']
LINESTYLES=['-','--','-.']
LINEWIDTH=3.0
ALPHAS=[1.0,1.0,1.0]
FIGSIZE=(6.4,4.5)
OUTPUT_DPI=1200
PEAK_PROMINENCE_DECADES=0.10
PEAK_MARKER_SIZE=25
PANEL_CONFIG={
'a':{'xlim':(X_MIN,X_MAX),'xticks':np.array([0.25,0.50,0.75,1.00]),'xticklabels':[r'$0{,}25$',r'$0{,}50$',r'$0{,}75$',r'$1{,}00$'],'ylim':(1e-10,1e8),'yticks':np.array([1e-10,1e-4,1e2,1e8]),'yticklabels':[r'$10^{-10}$',r'$10^{-4}$',r'$10^{2}$',r'$10^{8}$'],'xlabel':r'$x_M=k_Ma$','ylabel':'Energia interna normalizada','legend_loc':'lower right','legend_bbox':None,'box_pos':(0.035,0.055)},
'b':{'xlim':(X_MIN,X_MAX),'xticks':np.array([0.25,0.50,0.75,1.00]),'xticklabels':[r'$0{,}25$',r'$0{,}50$',r'$0{,}75$',r'$1{,}00$'],'ylim':(1e-10,1e8),'yticks':np.array([1e-10,1e-4,1e2,1e8]),'yticklabels':[r'$10^{-10}$',r'$10^{-4}$',r'$10^{2}$',r'$10^{8}$'],'xlabel':r'$x_M=k_Ma$','ylabel':'Energia interna normalizada','legend_loc':'lower right','legend_bbox':None,'box_pos':(0.035,0.055)},
'c':{'xlim':(X_MIN,X_MAX),'xticks':np.array([0.25,0.50,0.75,1.00]),'xticklabels':[r'$0{,}25$',r'$0{,}50$',r'$0{,}75$',r'$1{,}00$'],'ylim':(1e-10,1e8),'yticks':np.array([1e-10,1e-4,1e2,1e8]),'yticklabels':[r'$10^{-10}$',r'$10^{-4}$',r'$10^{2}$',r'$10^{8}$'],'xlabel':r'$x_M=k_Ma$','ylabel':'Energia interna normalizada','legend_loc':'lower right','legend_bbox':None,'box_pos':(0.035,0.055)}
}
plt.rcParams.update({'text.usetex':True,'font.family':'serif','font.serif':['Computer Modern Roman'],'font.size':16,'axes.labelsize':20,'xtick.labelsize':16,'ytick.labelsize':16,'legend.fontsize':16,'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white','savefig.edgecolor':'white'})
def malha_global_naouniforme(x_min,x_max,x_split=0.1,n_baixo=1500,n_alto=1500):
    x_split=float(np.clip(x_split,x_min,x_max)); n_baixo=max(int(n_baixo),2); n_alto=max(int(n_alto),2)
    baixo=np.geomspace(x_min,x_split,n_baixo,endpoint=False); alto=np.linspace(x_split,x_max,n_alto)
    return np.unique(np.concatenate([baixo,alto]))
def _J(n,z):
    if n>=0: return spherical_jn(n,z)
    m=-n-1; return (-1)**(m+1)*spherical_yn(m,z)
def _dJ(n,z):
    if n>=0: return spherical_jn(n,z,derivative=True)
    m=-n-1; return (-1)**(m+1)*spherical_yn(m,z,derivative=True)
def _H(n,z):
    if n>=0: return spherical_jn(n,z)+1j*spherical_yn(n,z)
    m=-n-1; return 1j*(-1)**m*(spherical_jn(m,z)+1j*spherical_yn(m,z))
def _dH(n,z):
    if n>=0: return spherical_jn(n,z,derivative=True)+1j*spherical_yn(n,z,derivative=True)
    m=-n-1; return 1j*(-1)**m*(spherical_jn(m,z,derivative=True)+1j*spherical_yn(m,z,derivative=True))
def _ddJ(n,z):
    z=np.asarray(z,dtype=complex); pequeno=np.isclose(z,0.0); z_seguro=np.where(pequeno,1.0,z)
    out=-(2.0/z_seguro)*_dJ(n,z_seguro)+(n*(n+1)/z_seguro**2-1.0)*_J(n,z_seguro)
    if np.any(pequeno):
        limite=-1.0/3.0 if n==0 else 2.0/15.0 if n==2 else 0.0
        out=np.where(pequeno,limite+0j,out)
    return out
def _I(n,z):
    if n>=0: return spherical_in(n,z)
    if n==-1: return np.cosh(z)/z
    raise NotImplementedError
def _dI(n,z): return spherical_in(n,z,derivative=True)
def _coefs_vec(xM,l_max,cL,cT,cM,rhoS,rhoM):
    xM=np.asarray(xM,dtype=float); xL_real=xM*(cM/np.abs(cL)); eta=cL/cT; phi=np.angle(1.0/cL); zL=xL_real*np.exp(1j*phi); zT=eta*zL; zM=xM.astype(complex)
    mu=rhoS*cT**2; lamb=rhoS*cL**2-2*mu; n=xM.size; a_l=np.zeros((l_max+1,n),dtype=complex); b_l=np.zeros((l_max+1,n),dtype=complex)
    for l in range(l_max+1):
        den_M=rhoM*cM**2*zM**2*_H(l,zM)
        A_l=(lamb*zL**2*_J(l,zL)-2*mu*zL**2*_ddJ(l,zL))/den_M
        B_l=-2*mu*l*(l+1)*(zT*_dJ(l,zT)-_J(l,zT))/den_M
        C_l=-_J(l,zM)/_H(l,zM); den_M_tilde=zM*_dH(l,zM); A_l_tilde=zL*_dJ(l,zL)/den_M_tilde; B_l_tilde=l*(l+1)*_J(l,zT)/den_M_tilde; C_l_tilde=-_dJ(l,zM)/_dH(l,zM)
        num_R=2*(_J(l,zL)-zL*_dJ(l,zL)); den_R=zT**2*_ddJ(l,zT)+(l*(l+1)-2)*_J(l,zT); den_R_safe=np.where(den_R!=0,den_R,1.0); R_l=np.where(den_R!=0,num_R/den_R_safe,0j)
        num_a=C_l_tilde-C_l; den_a=A_l-A_l_tilde-(B_l_tilde-B_l)*R_l; den_a_safe=np.where(den_a!=0,den_a,1.0)
        a_l[l]=np.where(den_a!=0,num_a/den_a_safe,0j); b_l[l]=R_l*a_l[l]
    return a_l,b_l
def _G_vec(z,l,tol=1e-6):
    z=np.asarray(z,dtype=complex); x=np.abs(z); out=np.zeros_like(z,dtype=complex); real_branch=np.abs(z.imag)<tol*x; imag_branch=(~real_branch)&(np.abs(z.real)<tol*x); general=~(real_branch|imag_branch)
    if np.any(real_branch):
        xr=x[real_branch]; out[real_branch]=xr*_J(l,xr)*_dJ(l,xr)+xr**2/2*(_J(l,xr)**2-_J(l-1,xr)*_J(l+1,xr))
    if np.any(imag_branch):
        xi=x[imag_branch]; out[imag_branch]=xi*_I(l,xi)*_dI(l,xi)-xi**2/2*(_I(l,xi)**2-_I(l-1,xi)*_I(l+1,xi))
    if np.any(general):
        zg=z[general]; xg=x[general]; zc=np.conj(zg); den=zg**2-zc**2; out[general]=xg**2/den*(zg*_J(l,zg)*_dJ(l,zc)-zc*_dJ(l,zg)*_J(l,zc))
    return out
def W_clapeyron_vec(xM,l_max,cL,cT,cM,rhoS,rhoM,tol=1e-6):
    xM=np.asarray(xM,dtype=float); a_l,b_l=_coefs_vec(xM=xM,l_max=l_max,cL=cL,cT=cT,cM=cM,rhoS=rhoS,rhoM=rhoM)
    zL=cM/cL*xM; xL=np.abs(zL); eta=cL/cT; zT=eta*zL; zTc=np.conj(zT); mu=rhoS*cT**2; lamb=rhoS*cL**2-2*mu; dois_T=np.zeros_like(xM,dtype=complex); IS_sum=np.zeros_like(xM,dtype=complex)
    for l in range(l_max+1):
        a=a_l[l]; b=b_l[l]; L=l*(l+1); termo_a=xL*_G_vec(zL,l,tol); termo_b=xL*(_J(l,zT)*_J(l,zTc)+_G_vec(zT,l,tol)); termo_cruzado=xL*_J(l,zL)*_J(l,zTc)
        dois_T+=2*(2*l+1)*(np.abs(a)**2*termo_a+np.abs(b)**2*L*termo_b+2*L*np.real(a*np.conj(b)*termo_cruzado))
        fator1=np.conj(a)*_dJ(l,np.conj(zL))+np.conj(b)*(L/np.conj(zL))*_J(l,zTc)
        fator2=a*zL**2*(lamb*_J(l,zL)-2*mu*_ddJ(l,zL))-2*mu*L*b*(zT*_dJ(l,zT)-_J(l,zT))
        IS_sum+=(2*l+1)*fator1*fator2
    IS_termo=-np.real(IS_sum/(rhoS*np.abs(cL)*np.conj(cL))); return np.real(dois_T)+IS_termo
def load_dataset_elastico(checkpoint_path):
    if not os.path.exists(checkpoint_path): raise FileNotFoundError(f'Checkpoint nao encontrado:\n{checkpoint_path}')
    with np.load(checkpoint_path,allow_pickle=True) as data:
        files=list(data.files); required={'_indice','alpha','beta','N_res'}
        if not required.issubset(files): raise KeyError('O dataset nao possui as colunas _indice, alpha, beta e N_res.')
        idx=np.asarray(data['_indice']); alpha=np.asarray(data['alpha'],dtype=float); beta=np.asarray(data['beta'],dtype=float); n_res=np.asarray(data['N_res'],dtype=float)
        if 'eta' in files: eta=np.asarray(data['eta'],dtype=float)
        elif 'cL0' in files and 'cT0' in files: eta=np.asarray(data['cL0'],dtype=float)/np.asarray(data['cT0'],dtype=float)
        else: raise KeyError('Nao encontrei eta nem cL0/cT0.')
        xM_pico=np.asarray(data['xM_pico'],dtype=float) if 'xM_pico' in files else np.full(len(idx),np.nan)
        cL0_array=np.asarray(data['cL0'],dtype=float) if 'cL0' in files else None; cT0_array=np.asarray(data['cT0'],dtype=float) if 'cT0' in files else None; rhoS_array=np.asarray(data['rhoS'],dtype=float) if 'rhoS' in files else None; cM_array=np.asarray(data['cM'],dtype=float) if 'cM' in files else None; rhoM_array=np.asarray(data['rhoM'],dtype=float) if 'rhoM' in files else None
    rows=[]
    for curve_id in np.unique(idx):
        mask=idx==curve_id; posicoes=np.flatnonzero(mask)
        if len(posicoes)==0: continue
        p0=posicoes[0]; alpha0=float(alpha[p0]); beta0=float(beta[p0]); eta0=float(eta[p0])
        if not (np.isfinite(alpha0) and alpha0>0 and np.isfinite(beta0) and beta0>0 and np.isfinite(eta0) and 1<=eta0<=10): continue
        rho_ratio=1.0/alpha0; cL_ratio=1.0/beta0; cM=float(cM_array[p0]) if cM_array is not None else CM_DEFAULT; rhoM=float(rhoM_array[p0]) if rhoM_array is not None else RHOM_DEFAULT; cL=float(cL0_array[p0]) if cL0_array is not None else cL_ratio*cM; cT=float(cT0_array[p0]) if cT0_array is not None else cL/eta0; rhoS=float(rhoS_array[p0]) if rhoS_array is not None else rho_ratio*rhoM
        peaks=xM_pico[mask]; peaks=peaks[np.isfinite(peaks)&(peaks>0)]; peaks=np.unique(peaks); peaks.sort(); nres_mask=n_res[mask]; nres_mask=nres_mask[np.isfinite(nres_mask)]; n_res0=int(np.max(nres_mask)) if len(nres_mask) else len(peaks)
        rows.append({'curve_id':curve_id,'alpha':alpha0,'beta':beta0,'cL_ratio':cL_ratio,'rho_ratio':rho_ratio,'eta':eta0,'cL':cL,'cT':cT,'rhoS':rhoS,'cM':cM,'rhoM':rhoM,'N_res':n_res0,'peaks':peaks})
    df=pd.DataFrame(rows)
    if len(df)==0: raise RuntimeError('Nenhuma curva valida foi encontrada.')
    print('\n'+'='*90+'\nDATASET ELASTICO\n'+'='*90)
    print('Curvas validas:',len(df)); print('cL/cM:',df['cL_ratio'].min(),'a',df['cL_ratio'].max()); print('rhoS/rhoM:',df['rho_ratio'].min(),'a',df['rho_ratio'].max()); print('eta:',df['eta'].min(),'a',df['eta'].max()); print('N_res:',df['N_res'].min(),'a',df['N_res'].max()); print('='*90+'\n')
    return df
def find_nearest_case(df,target_cL_ratio,target_rho_ratio,target_eta,weight_cL=1.0,weight_rho=1.0,weight_eta=1.0,excluded_ids=None):
    work=df.copy()
    if excluded_ids is not None: work=work.loc[~work['curve_id'].isin(excluded_ids)].copy()
    if len(work)==0: raise RuntimeError('Nao restaram curvas candidatas.')
    d_cL=np.log10(work['cL_ratio'].to_numpy(dtype=float))-np.log10(target_cL_ratio); d_rho=np.log10(work['rho_ratio'].to_numpy(dtype=float))-np.log10(target_rho_ratio); d_eta=np.log10(work['eta'].to_numpy(dtype=float))-np.log10(target_eta)
    distance=weight_cL*d_cL**2+weight_rho*d_rho**2+weight_eta*d_eta**2
    return work.iloc[int(np.argmin(distance))]
def select_cases(df):
    panel_a=[]; used=set()
    for rho_target in A_RHO_RATIOS:
        row=find_nearest_case(df,A_FIXED_CL_RATIO,rho_target,A_FIXED_ETA,30.0,1.0,30.0,used); panel_a.append(row); used.add(row['curve_id'])
    panel_b=[]; used=set()
    for cL_target in B_CL_RATIOS:
        row=find_nearest_case(df,cL_target,B_FIXED_RHO_RATIO,B_FIXED_ETA,1.0,30.0,30.0,used); panel_b.append(row); used.add(row['curve_id'])
    panel_c=[]; used=set()
    for eta_target in C_ETAS:
        row=find_nearest_case(df,C_FIXED_CL_RATIO,C_FIXED_RHO_RATIO,eta_target,30.0,30.0,1.0,used); panel_c.append(row); used.add(row['curve_id'])
    return panel_a,panel_b,panel_c
def calculate_panel_curves(selected_rows,x_mesh):
    curves=[]
    for row in selected_rows:
        W=W_clapeyron_vec(x_mesh,LMAX,float(row['cL']),float(row['cT']),float(row['cM']),float(row['rhoS']),float(row['rhoM']))
        W_plot=np.asarray(W,dtype=float); W_plot[~np.isfinite(W_plot)]=np.nan; W_plot[W_plot<=0]=np.nan; curves.append(W_plot)
    return curves
def fmt_latex(value,sig=2):
    value=float(value)
    if np.isclose(value,1.0): return '1'
    if abs(value)>=100 or (0<abs(value)<0.01):
        exponent=int(np.floor(np.log10(abs(value)))); mantissa=value/10.0**exponent; mantissa_txt=f'{mantissa:.{max(sig-1,0)}f}'.rstrip('0').rstrip('.').replace('.','{,}')
        return mantissa_txt+rf'\times10^{{{exponent}}}'
    return f'{value:.{sig}g}'.replace('.','{,}')
def plot_curve_peaks(ax,x_mesh,W,color,panel_key):
    cfg=PANEL_CONFIG[panel_key]; valid=np.isfinite(W)&(W>0)&(x_mesh>=cfg['xlim'][0])&(x_mesh<=cfg['xlim'][1])
    if np.sum(valid)<3: return 0
    xv=x_mesh[valid]; yv=W[valid]; logW=np.log10(yv); idx,_=find_peaks(logW,prominence=PEAK_PROMINENCE_DECADES)
    if len(idx)==0: return 0
    ax.scatter(xv[idx],yv[idx],s=PEAK_MARKER_SIZE,color=color,edgecolors='white',linewidths=0.45,zorder=5)
    return len(idx)
def style_axis(ax,panel_key):
    cfg=PANEL_CONFIG[panel_key]; ax.set_yscale('log'); ax.set_xlim(*cfg['xlim']); ax.set_ylim(*cfg['ylim']); ax.xaxis.set_major_locator(FixedLocator(cfg['xticks'])); ax.set_xticklabels(cfg['xticklabels']); ax.yaxis.set_major_locator(FixedLocator(cfg['yticks'])); ax.set_yticklabels(cfg['yticklabels']); ax.minorticks_off(); ax.yaxis.set_minor_formatter(NullFormatter()); ax.set_xlabel(cfg['xlabel'],fontsize=20,labelpad=8); ax.set_ylabel(cfg['ylabel'],fontsize=19,labelpad=9); ax.tick_params(direction='in',length=5,width=1.2,labelsize=16,pad=6)
    for spine in ax.spines.values(): spine.set_linewidth(1.8); spine.set_edgecolor('black')
    ax.grid(False)
def style_legend(ax,panel_key):
    cfg=PANEL_CONFIG[panel_key]; kwargs={'loc':cfg['legend_loc'],'fontsize':16,'frameon':True,'handlelength':2.5,'borderpad':0.5,'labelspacing':0.4,'handletextpad':0.65}
    if cfg['legend_bbox'] is not None: kwargs['bbox_to_anchor']=cfg['legend_bbox']
    legend=ax.legend(**kwargs); legend.get_frame().set(edgecolor='black',facecolor='white',linewidth=1.8,alpha=1.0); return legend
def fixed_parameter_box(ax,text,panel_key):
    cfg=PANEL_CONFIG[panel_key]; ax.text(cfg['box_pos'][0],cfg['box_pos'][1],text,transform=ax.transAxes,ha='left',va='bottom',fontsize=15,linespacing=1.12,color='black',bbox={'boxstyle':'square,pad=0.4','facecolor':'white','edgecolor':'black','linewidth':2.0,'alpha':1.0},zorder=20)
def salvar_figura_individual(panel_key,rows,curves,x_mesh,nome,tipo):
    fig,ax=plt.subplots(figsize=FIGSIZE,facecolor='white')
    for i,(row,W) in enumerate(zip(rows,curves)):
        if tipo=='densidade': label=r'$\rho_S/\rho_M='+fmt_latex(row['rho_ratio'],2)+'$'
        elif tipo=='velocidade': label=r'$c_L/c_M='+fmt_latex(row['cL_ratio'],2)+'$'
        else: label=r'$\eta='+fmt_latex(row['eta'],3)+'$'
        ax.plot(x_mesh,W,color=COLORS[i],linestyle=LINESTYLES[i],linewidth=LINEWIDTH,alpha=ALPHAS[i],label=label)
        n_detectados=plot_curve_peaks(ax,x_mesh,W,COLORS[i],panel_key)
        print(f'{nome} | curva {i+1}: {n_detectados} máximos locais marcados | checkpoint: {len(row["peaks"])}')
    style_axis(ax,panel_key); style_legend(ax,panel_key)
    if tipo=='densidade':
        cl_fixed=np.exp(np.mean(np.log([float(row['cL_ratio']) for row in rows]))); eta_fixed=np.exp(np.mean(np.log([float(row['eta']) for row in rows]))); texto=r'$c_L/c_M\approx'+fmt_latex(cl_fixed,3)+'$'+'\n'+r'$\eta\approx'+fmt_latex(eta_fixed,3)+'$'
    elif tipo=='velocidade':
        rho_fixed=np.exp(np.mean(np.log([float(row['rho_ratio']) for row in rows]))); eta_fixed=np.exp(np.mean(np.log([float(row['eta']) for row in rows]))); texto=r'$\rho_S/\rho_M\approx'+fmt_latex(rho_fixed,3)+'$'+'\n'+r'$\eta\approx'+fmt_latex(eta_fixed,3)+'$'
    else:
        cl_fixed=np.exp(np.mean(np.log([float(row['cL_ratio']) for row in rows]))); rho_fixed=np.exp(np.mean(np.log([float(row['rho_ratio']) for row in rows]))); texto=r'$c_L/c_M\approx'+fmt_latex(cl_fixed,3)+'$'+'\n'+r'$\rho_S/\rho_M\approx'+fmt_latex(rho_fixed,3)+'$'
    fixed_parameter_box(ax,texto,panel_key); fig.subplots_adjust(left=0.17,right=0.97,bottom=0.18,top=0.96); base=os.path.join(OUTPUT_DIR,nome)
    fig.savefig(base+'.png',dpi=OUTPUT_DPI,bbox_inches=None,pad_inches=0,facecolor='white',edgecolor='white'); fig.savefig(base+'.pdf',bbox_inches=None,pad_inches=0,facecolor='white',edgecolor='white'); fig.savefig(base+'.eps',format='eps',bbox_inches=None,pad_inches=0,facecolor='white',edgecolor='white'); plt.show(); plt.close(fig)
    print(base+'.png'); print(base+'.pdf'); print(base+'.eps'); return base
def plot_curvas_individuais(panel_a,panel_b,panel_c):
    x_mesh=malha_global_naouniforme(X_MIN,X_MAX,n_baixo=N_MALHA_GLOBAL//2,n_alto=N_MALHA_GLOBAL-N_MALHA_GLOBAL//2)
    print('Calculando variação de densidade...'); curves_a=calculate_panel_curves(panel_a,x_mesh)
    print('Calculando variação de velocidade longitudinal...'); curves_b=calculate_panel_curves(panel_b,x_mesh)
    print('Calculando variação de eta...'); curves_c=calculate_panel_curves(panel_c,x_mesh)
    print('\n'+'='*90+'\nSALVANDO FIGURAS INDIVIDUAIS\n'+'='*90)
    bases=[]; bases.append(salvar_figura_individual('a',panel_a,curves_a,x_mesh,'WT_variacao_densidade','densidade')); bases.append(salvar_figura_individual('b',panel_b,curves_b,x_mesh,'WT_variacao_velocidade_longitudinal','velocidade')); bases.append(salvar_figura_individual('c',panel_c,curves_c,x_mesh,'WT_variacao_eta','eta')); print('='*90); return bases
def print_selected_cases(panel_a,panel_b,panel_c):
    print('\n'+'='*110+'\nCASOS REAIS SELECIONADOS NO DATASET\n'+'='*110+'\n')
    print('VARIACAO DE DENSIDADE\nObjetivo: cL/cM ~= 1, eta ~= 2\n'+'-'*110)
    for row in panel_a: print(f"rhoS/rhoM={row['rho_ratio']:.6g} | cL/cM={row['cL_ratio']:.6g} | eta={row['eta']:.6g} | N_res={int(row['N_res'])}")
    print('\nVARIACAO DE VELOCIDADE LONGITUDINAL\nObjetivo: rhoS/rhoM ~= 1, eta ~= 2\n'+'-'*110)
    for row in panel_b: print(f"cL/cM={row['cL_ratio']:.6g} | rhoS/rhoM={row['rho_ratio']:.6g} | eta={row['eta']:.6g} | N_res={int(row['N_res'])}")
    print('\nVARIACAO DE ETA\nObjetivo: cL/cM ~= 1, rhoS/rhoM ~= 1\n'+'-'*110)
    for row in panel_c: print(f"eta={row['eta']:.6g} | cL/cM={row['cL_ratio']:.6g} | rhoS/rhoM={row['rho_ratio']:.6g} | N_res={int(row['N_res'])}")
    print('='*110+'\n')
print('='*90+'\nCURVAS PARAMETRICAS - ESFERA ELASTICA\n'+'='*90); print('Checkpoint:',CHECKPOINT_PATH)
df=load_dataset_elastico(CHECKPOINT_PATH); panel_a,panel_b,panel_c=select_cases(df); print_selected_cases(panel_a,panel_b,panel_c); output_bases=plot_curvas_individuais(panel_a,panel_b,panel_c)
