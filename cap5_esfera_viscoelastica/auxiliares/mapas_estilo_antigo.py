# -*- coding: utf-8 -*-
"""
MAPAS DE RESSONANCIA (ESTILO ANTIGO) - ELASTICO, KELVIN-VOIGT E MAXWELL

Elastico: mapa em eta = 1,16 + fronteiras de eta = 2, 5 e 10 (eta exato, recalculado celula a celula).
Kelvin-Voigt e Maxwell: tres mapas por faixa de eta, lidos do banco do gerador (sem recalculo).

Contagem por celula do elastico (sem Newton no plano complexo):
  1. malha de 5000 pontos em 1e-3 <= x_M <= 1, energia vetorizada, find_peaks com proeminencia 0,1
     em log10 W_T (descartando W_T > 1e30, que e estouro numerico) -> decide se existe ressonancia;
  2. raizes reais da equacao caracteristica de baixa frequencia (Eq. da relacao de recorrencia),
     l = 0..10, no mesmo intervalo -> conta as ressonancias estreitas que a malha nao resolve;
  N = max(N_malha, N_raizes).
"""
import os, warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.special import spherical_jn, spherical_yn
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

warnings.filterwarnings("ignore")

# ---------------- pastas: dados e resultados ----------------
try:
    from google.colab import drive
    NO_COLAB = True
except ImportError:
    NO_COLAB = False
if NO_COLAB:
    drive.mount("/content/drive")
    PASTA = "/content/drive/MyDrive/visco_final"
    SAIDA = os.path.join(PASTA, "resultados")
else:
    PASTA = os.path.abspath(os.environ.get("VISCO_DADOS", os.path.join(os.path.expanduser("~"), "Downloads", "visco_final")))
    SAIDA = os.path.abspath(os.environ.get("VISCO_SAIDA", "resultados"))
FIG_DIR = os.path.join(SAIDA, "figuras")
CACHE_DIR = os.path.join(SAIDA, "cache_eta_exato")
for p in (FIG_DIR, CACHE_DIR):
    os.makedirs(p, exist_ok=True)

# ---------------- parâmetros ----------------
CM, RHOM = 1500.0, 1000.0
L_MAX = 10
N_MALHA, X_MIN, X_MAX = 5000, 1e-3, 1.0
PROMINENCIA = 0.1                 # em log10 W_T, como no gerador
LIMIAR_ESTOURO = 1e30             # W_T acima disso é estouro numérico
PASSOS_POR_PI = 16                # malha das raízes: pontos por pi de fase (1 + eta) z_L
FASE_MINIMA = 0.1                 # abaixo disso a contagem vem só das raízes
N_PROCESSOS = 1 if os.name == "nt" else (os.cpu_count() or 1)   # no Windows, processos paralelos não funcionam dentro do notebook
ETAS_ELASTICO = (1.16, 2.0, 5.0, 10.0)
GERAR_INGLES = True
MOSTRAR_FIGURAS = True
PREENCHER_VAZIOS_BANCO = True

LARGURA_BIN = 0.25
LIMITES_MAPA = {"elastico": ((-3.0, 3.5), (-3.0, 3.0)), "voigt": ((-3.0, 3.5), (-3.0, 3.0)),
                "maxwell": ((-3.0, 3.5), (-4.0, 3.0))}
ETA_FAIXAS = [(2 / np.sqrt(3), 2.0), (2.0, 5.0), (5.0, np.inf)]
ROT_FAIXA = [r"$\eta\in[2/\sqrt{3};\,2)$", r"$\eta\in[2;\,5)$", r"$\eta\in[5;\,\infty)$"]
CELL_LINECOLOR, CELL_LINEWIDTH = "white", 0.42
BOX_COLOR = "#600B00"
FRONTEIRAS_ESTILOS = {2.0: dict(color="#8b3f2b", linestyle=(0, (5, 3)), linewidth=1.5),
                      5.0: dict(color="#3b82c4", linestyle=(0, (3, 2)), linewidth=1.5),
                      10.0: dict(color="#444444", linestyle=(0, (6, 2, 1.2, 2)), linewidth=1.7)}
ESTILO_FRONTEIRA_BASE = dict(color="black", linestyle="-", linewidth=1.2)
GUIAS = dict(color="0.85", linestyle=":", linewidth=1.0, zorder=3)
RESONANCE_CMAP = LinearSegmentedColormap.from_list("ressonancias", ["white", "#D9E1EA", "#526F92", "black"], N=256)
RESONANCE_CMAP.set_bad("#D9D9D9")


# =========================================================
# ESFERA ELÁSTICA: COEFICIENTES E ENERGIA (VETORIZADO EM x_M)
# =========================================================
def jn(l, z):
    return spherical_jn(l, z)


def djn(l, z):
    return spherical_jn(l, z, derivative=True)


def ddjn(l, z):
    # z^2 j'' = (l^2 - l - z^2) j_l + 2 z j_{l+1}: sem cancelamento para z pequeno
    return ((l * l - l - z * z) * jn(l, z) + 2 * z * jn(l + 1, z)) / (z * z)


def hn(l, x):
    return spherical_jn(l, x) + 1j * spherical_yn(l, x)


def dhn(l, x):
    return spherical_jn(l, x, derivative=True) + 1j * spherical_yn(l, x, derivative=True)


def resolver(M, rhs):
    # resolve em bloco; se algum ponto for singular, resolve ponto a ponto e deixa NaN nele
    try:
        return np.linalg.solve(M, rhs[..., None])[..., 0]
    except np.linalg.LinAlgError:
        out = np.full(rhs.shape, np.nan, dtype=complex)
        for i in range(rhs.shape[0]):
            try:
                out[i] = np.linalg.solve(M[i], rhs[i])
            except np.linalg.LinAlgError:
                pass
        return out


def energia_l(x, l, cL, cT, rhoS):
    # termo l de 2 f0 <W>/(Z_S |A|^2) da esfera elástica (teorema de Clapeyron, relações de Lommel)
    zL, zT = x * CM / cL, x * CM / cT
    mu, rc = rhoS * cT ** 2, RHOM * CM ** 2
    lam = rhoS * cL ** 2 - 2 * mu
    L, f = l * (l + 1), 2 * l + 1
    jL, djL, ddjL = jn(l, zL), djn(l, zL), ddjn(l, zL)
    h, dh = hn(l, x), dhn(l, x)
    PA = zL ** 2 * (lam * jL - 2 * mu * ddjL)
    if l == 0:
        M = np.stack([np.stack([zL * djL, -x * dh], -1), np.stack([PA / rc + 0j, -x ** 2 * h], -1)], -2)
        a = resolver(M, np.stack([x * djn(0, x), x ** 2 * jn(0, x)], -1).astype(complex))[:, 0]
        b = np.zeros_like(a)
    else:
        jT, djT, ddjT = jn(l, zT), djn(l, zT), ddjn(l, zT)
        PB = -2 * mu * L * (zT * djT - jT)
        z0 = np.zeros_like(x, dtype=complex)
        M = np.stack([np.stack([zL * djL + 0j, L * jT + 0j, -x * dh], -1),
                      np.stack([PA / rc + 0j, PB / rc + 0j, -x ** 2 * h], -1),
                      np.stack([2 * (zL * djL - jL) + 0j, zT ** 2 * ddjT + (L - 2) * jT + 0j, z0], -1)], -2)
        a, b = resolver(M, np.stack([x * djn(l, x), x ** 2 * jn(l, x), z0], -1).astype(complex))[:, :2].T
    jm = jn(l - 1, zL) if l > 0 else np.cos(zL) / zL
    GL = zL * jL * djL + 0.5 * zL ** 2 * (jL ** 2 - jm * jn(l + 1, zL))           # Lommel, argumento real
    dois_T = 2 * f * np.abs(a) ** 2 * zL * GL
    sup = np.conj(a) * djL * a * PA
    if l > 0:
        jm = jn(l - 1, zT)
        GT = zT * jT * djT + 0.5 * zT ** 2 * (jT ** 2 - jm * jn(l + 1, zT))
        dois_T = dois_T + 2 * f * (np.abs(b) ** 2 * L * zL * (jT ** 2 + GT) + 2 * L * np.real(a * np.conj(b)) * zL * jL * jT)
        sup = (np.conj(a) * djL + np.conj(b) * L / zL * jT) * (a * PA + b * PB)
    return np.real(dois_T) - np.real(f * sup) / (rhoS * cL ** 2)


def energia_elastica(x, cL, cT, rhoS):
    with np.errstate(all="ignore"):
        return sum(energia_l(x, l, cL, cT, rhoS) for l in range(L_MAX + 1))


# =========================================================
# RAÍZES REAIS DA EQUAÇÃO CARACTERÍSTICA (BAIXA FREQUÊNCIA)
# =========================================================
def termos_caracteristica(l, z, rho, eta, J, JT):
    # (l+1) E_l + rho_M c_L^2 z_L^2 G_l = 0 dividida por rho_S c_L^2 (l >= 1); l = 0 sem o fator T_0
    eps, mu = 1.0 / eta ** 2, z ** 2 / rho
    zT = eta * z
    jl, pl = J[l], J[l + 1]
    AL = (z ** 2 - 2 * eps * (l * l - l)) * jl - 4 * eps * z * pl
    dL = l * jl - z * pl
    if l == 0:
        return (AL, mu * dL)
    jt, pt = JT[l], JT[l + 1]
    T = (2 * l * l - 2 - zT ** 2) * jt + 2 * zT * pt
    BT, BL = (l - 1) * jt - zT * pt, (l - 1) * jl - z * pl
    Lf = l * (l + 1)
    return ((l + 1) * AL * T, (l + 1) * 4 * eps * Lf * BT * BL, mu * dL * T, -2 * Lf * mu * jt * BL)


def F_caracteristica(l, z, rho, eta):
    o = np.arange(l + 2)[:, None]
    z = np.atleast_1d(z)
    return sum(termos_caracteristica(l, z, rho, eta, spherical_jn(o, z[None]), spherical_jn(o, eta * z[None])))


def n_raizes(rho, eta, z_min, z_max, it=50):
    # número de raízes reais em [z_min, z_max] somando l = 0..L_MAX (troca de sinal + bissecção)
    n = max(int(PASSOS_POR_PI * (1 + eta) * z_max / np.pi), 200)
    z = np.unique(np.concatenate([np.geomspace(z_min, min(1.0, z_max), 200), np.linspace(z_min, z_max, n)]))
    o = np.arange(L_MAX + 2)[:, None]
    J, JT = spherical_jn(o, z[None]), spherical_jn(o, eta * z[None])
    total = 0
    for l in range(L_MAX + 1):
        t = termos_caracteristica(l, z, rho, eta, J, JT)
        f, escala = sum(t), sum(np.abs(u) for u in t)
        bom = np.abs(f) > 1e-12 * escala
        total += int(np.count_nonzero(bom[:-1] & bom[1:] & (np.sign(f[:-1]) * np.sign(f[1:]) < 0)))
    return total


def contar_celula(args):
    cx, ry, eta = args
    cL, rhoS = CM * cx, RHOM * ry
    cT = cL / eta
    x = np.linspace(X_MIN, X_MAX, N_MALHA)
    W = energia_elastica(x, cL, cT, rhoS)
    ok = np.isfinite(W) & (W > 0) & (W < LIMIAR_ESTOURO)
    n_malha = len(find_peaks(np.log10(W[ok]), prominence=PROMINENCIA)[0]) if ok.sum() > 10 else 0
    n_r = n_raizes(ry, eta, X_MIN / cx, X_MAX / cx)
    # fase total X_MAX (c_M/c_L)(1 + eta) < 0,1: sólido rígido, a malha só registra estouro numérico (z_L ~ 1e-6),
    # e a única ressonância possível (tipo Minnaert, via módulo volumétrico) já está nas raízes de l = 0
    N = n_r if X_MAX * (1 + eta) / cx < FASE_MINIMA else max(n_malha, n_r)
    return dict(c_razao=cx, rho_ratio=ry, eta=eta, N=N, N_malha=n_malha, N_raizes=n_r,
                frac_malha_ok=float(ok.mean()))


# =========================================================
# BANCO DO GERADOR (KELVIN-VOIGT E MAXWELL)
# =========================================================
def ler_banco(modelo):
    # só o hipercubo_<modelo>.npz: uma linha por pico (curvas sem ressonância têm uma linha com pico NaN),
    # com os parâmetros da curva repetidos em cada linha
    arq = Path(PASTA) / f"viscoelastico_lmax10_{modelo}" / f"hipercubo_{modelo}.npz"
    with np.load(arq, allow_pickle=True) as d:
        pk = pd.DataFrame({k: d[k] for k in ("_indice", "xM_pico", "altura", "origem", "alpha", "beta", "eta", "N_res", "status")})
    pk["origem"], pk["status"] = pk["origem"].astype(str), pk["status"].astype(str)
    r = pk.groupby("_indice").first().reset_index()
    r = r.loc[r["status"] == "ok"].copy()
    v = pk.loc[np.isfinite(pk["xM_pico"]) & (pk["xM_pico"] >= 1e-4) & (pk["xM_pico"] <= 1.0)]
    # contagem feita só pela malha base (sem polos) e posição da primeira ressonância de cada curva
    r["so_malha"] = r["_indice"].map(v.groupby("_indice")["origem"].agg(lambda o: (o == "malha_base").all())).fillna(False)
    r["x1"] = r["_indice"].map(v.groupby("_indice")["xM_pico"].min())
    # curvas com máximos espúrios (estouro numérico em sólidos muito rígidos) saem do mapa
    estouro = set(v.loc[(v["origem"] == "malha_base") & (v["altura"] > LIMIAR_ESTOURO), "_indice"])
    rig = r["so_malha"] & (r["N_res"] > 0) & (r["x1"] < 1e-3) & (1 / r["beta"] > 100)
    if modelo != "maxwell":
        # fase interna total (c_M/c_L)(1 + eta) < 0,1 com contagem pela malha: só estouro numérico
        # (no Maxwell, c_T efetivo tende a zero em baixa frequência e esse critério não vale)
        rig |= r["so_malha"] & (r["N_res"] > 0) & (X_MAX * r["beta"] * (1 + r["eta"]) < FASE_MINIMA)
    fora = estouro | set(r.loc[rig, "_indice"])
    r = r.loc[~r["_indice"].isin(fora)]
    print(f"[{modelo}] {len(r)} curvas ({len(fora)} com máximos espúrios excluídas)")
    return pd.DataFrame({"c_razao": 1 / r["beta"], "rho_ratio": 1 / r["alpha"], "eta": r["eta"], "N": r["N_res"]})


# =========================================================
# MAPAS (ESTÉTICA ANTIGA)
# =========================================================
class Grade:
    def __init__(self, log_x, log_y):
        self.LOG_X, self.LOG_Y = log_x, log_y
        self.NX = int(round((log_x[1] - log_x[0]) / LARGURA_BIN))
        self.NY = int(round((log_y[1] - log_y[0]) / LARGURA_BIN))

    def celula_x(self, v):
        return (np.log10(v) - self.LOG_X[0]) / LARGURA_BIN - 0.5

    def celula_y(self, v):
        return (np.log10(v) - self.LOG_Y[0]) / LARGURA_BIN - 0.5

    def centros(self):
        return [(i, j, 10 ** (self.LOG_X[0] + (i + .5) * LARGURA_BIN), 10 ** (self.LOG_Y[0] + (j + .5) * LARGURA_BIN))
                for i in range(self.NX) for j in range(self.NY)]


def calcular_eta_elastico(eta, gr):
    destino = Path(CACHE_DIR) / f"elastico_eta_{f'{eta:g}'.replace('.', 'p')}.csv"
    if destino.is_file():
        print(f"[Elástico eta={eta:g}] lido do cache {destino.name}")
        return pd.read_csv(destino)
    cel = gr.centros()
    print(f"[Elástico eta={eta:g}] calculando {len(cel)} células em {N_PROCESSOS} processos")
    with ProcessPoolExecutor(N_PROCESSOS) as ex:
        res = list(ex.map(contar_celula, [(cx, ry, eta) for _, _, cx, ry in cel], chunksize=8))
    df = pd.DataFrame(res)
    df.insert(0, "iy", [j for _, j, _, _ in cel])
    df.insert(0, "ix", [i for i, _, _, _ in cel])
    df.to_csv(destino, index=False)
    print(f"   N máximo {int(df['N'].max())}; células ressonantes {np.mean(df['N'] > 0):.1%}; "
          f"células em que as raízes acrescentam ressonâncias à malha: {np.mean(df['N_raizes'] > df['N_malha']):.1%}")
    return df


def mapa_binado(df, gr, preencher=False):
    ex, ey = np.linspace(*gr.LOG_X, gr.NX + 1), np.linspace(*gr.LOG_Y, gr.NY + 1)
    cont = np.zeros((gr.NX, gr.NY), dtype=int)
    mapa = np.full((gr.NX, gr.NY), np.nan)
    lx, ly, n = np.log10(df["c_razao"].to_numpy(float)), np.log10(df["rho_ratio"].to_numpy(float)), df["N"].to_numpy(float)
    dentro = np.isfinite(lx) & np.isfinite(ly) & np.isfinite(n) & (lx >= ex[0]) & (lx <= ex[-1]) & (ly >= ey[0]) & (ly <= ey[-1])
    lx, ly, n = lx[dentro], ly[dentro], n[dentro]
    ix = np.clip(np.searchsorted(ex, lx, side="right") - 1, 0, gr.NX - 1)
    iy = np.clip(np.searchsorted(ey, ly, side="right") - 1, 0, gr.NY - 1)
    np.add.at(cont, (ix, iy), 1)
    m = np.full((gr.NX, gr.NY), -np.inf)
    np.maximum.at(m, (ix, iy), n)
    mapa[cont > 0] = m[cont > 0]
    if preencher and np.any(cont == 0):
        cx, cy = np.meshgrid(0.5 * (ex[:-1] + ex[1:]), 0.5 * (ey[:-1] + ey[1:]), indexing="ij")
        _, ids = cKDTree(np.column_stack((lx, ly))).query(np.column_stack((cx[cont == 0], cy[cont == 0])))
        mapa[cont == 0] = n[ids]
    return mapa


def segmentos_fronteira(mask):
    seg, (nx, ny) = [], mask.shape
    for i in range(nx):
        for j in range(ny):
            if not mask[i, j]:
                continue
            xl, xr, yb, yt = i - .5, i + .5, j - .5, j + .5
            if i == 0 or not mask[i - 1, j]: seg.append(((xl, yb), (xl, yt)))
            if i == nx - 1 or not mask[i + 1, j]: seg.append(((xr, yb), (xr, yt)))
            if j == 0 or not mask[i, j - 1]: seg.append(((xl, yb), (xr, yb)))
            if j == ny - 1 or not mask[i, j + 1]: seg.append(((xl, yt), (xr, yt)))
    return seg


def texto_decimal(t, pt):
    v = f"{t:.4f}".rstrip("0").rstrip(".") if t < 1 else f"{t:g}"
    return v.replace(".", ",") if pt else v


def desenhar_mapa(ax, mapa, gr, idioma, escala_max):
    pt = idioma == "pt"
    norm = Normalize(vmin=0.0, vmax=np.log10(1.0 + max(escala_max, 1)))
    for i in range(gr.NX):
        for j in range(gr.NY):
            v = mapa[i, j]
            cor = RESONANCE_CMAP(norm(np.log10(1 + v))) if np.isfinite(v) else "#D9D9D9"
            ax.add_patch(Rectangle((i - .5, j - .5), 1, 1, facecolor=cor, edgecolor=CELL_LINECOLOR,
                                   linewidth=CELL_LINEWIDTH, zorder=1))
    tx = 10.0 ** np.arange(np.ceil(gr.LOG_X[0]), gr.LOG_X[1] + 1e-9)
    ty = 10.0 ** np.arange(np.ceil(gr.LOG_Y[0]), gr.LOG_Y[1] + 1e-9)
    ax.set_xticks(gr.celula_x(tx)); ax.set_xticklabels([texto_decimal(t, pt) for t in tx])
    ax.set_yticks(gr.celula_y(ty)); ax.set_yticklabels([texto_decimal(t, pt) for t in ty])
    ax.set_xlabel(r"$c_L/c_M$", fontsize=23, labelpad=9)
    ax.set_ylabel(r"$\rho_S/\rho_M$", fontsize=23, labelpad=10)
    ax.set_xlim(-.5, gr.NX - .5); ax.set_ylim(-.5, gr.NY - .5)
    ax.set_aspect("equal", adjustable="box")
    return norm


def barra_cores(fig, cax, norm, escala_max, idioma):
    nmax = max(int(np.ceil(escala_max)), 1)
    valores = np.unique(np.array([0] + [10 ** k for k in range(1, 6) if 10 ** k < .8 * nmax] + [nmax], dtype=int))
    sm = cm.ScalarMappable(norm=norm, cmap=RESONANCE_CMAP)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, ticks=np.log10(1 + valores))
    cb.ax.set_yticklabels([str(int(v)) for v in valores])
    cb.set_label("Ressonâncias" if idioma == "pt" else "Resonances", fontsize=20, labelpad=18, rotation=270)
    return cb


def estilizar_eixos(ax, cb):
    for eixo in (ax, cb.ax):
        eixo.tick_params(axis="both", labelsize=15, width=1.25, length=4.5, direction="out")
        for borda in eixo.spines.values():
            borda.set_visible(True); borda.set_edgecolor(BOX_COLOR); borda.set_linewidth(1.8)


def criar_figura(gr):
    altura = .70
    largura = altura * gr.NX / gr.NY
    fig = plt.figure(figsize=(10.5, 10.5), facecolor="white")
    return fig, fig.add_axes([.12, .13, largura, altura]), fig.add_axes([.12 + largura + .02, .13, .03, altura])


def salvar(fig, nome, mostrar):
    for ext in ("png", "pdf", "eps"):
        fig.savefig(os.path.join(FIG_DIR, f"{nome}.{ext}"), dpi=600, bbox_inches="tight")
    if mostrar:
        plt.show()
    plt.close(fig)


def caixa(ax, x, texto, ha):
    ax.text(x, .965, texto, transform=ax.transAxes, fontsize=22, color="white", va="top", ha=ha,
            bbox=dict(facecolor=BOX_COLOR, edgecolor="none", boxstyle="square,pad=0.14"), zorder=111)


def mapa_elastico(dados, idioma, escala_max, mostrar):
    gr = Grade(*LIMITES_MAPA["elastico"])
    mats = {eta: mapa_binado(df, gr) for eta, df in dados.items()}
    with plt.rc_context({"font.family": "sans-serif"}):
        fig, ax, cax = criar_figura(gr)
        norm = desenhar_mapa(ax, mats[1.16], gr, idioma, escala_max)
        ax.axvline(gr.celula_x(1.0), **GUIAS)
        ax.axhline(gr.celula_y(1.0), **GUIAS)
        for eta in ETAS_ELASTICO:
            estilo = ESTILO_FRONTEIRA_BASE if eta == 1.16 else FRONTEIRAS_ESTILOS[eta]
            for a, b in segmentos_fronteira(mats[eta] > 0):
                ax.plot([a[0], b[0]], [a[1], b[1]], solid_capstyle="butt", zorder=15 if eta == 1.16 else 16, **estilo)
        rot = "Fronteira de referência\n" if idioma == "pt" else "Reference frontier\n"
        handles = [Line2D([0], [0], label=rot + rf"$\eta={eta:g}$", **FRONTEIRAS_ESTILOS[eta]) for eta in (2.0, 5.0, 10.0)]
        lg = ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(.985, .83), fontsize=16, frameon=True,
                       framealpha=1, facecolor="white", edgecolor="none", fancybox=False, borderpad=.22,
                       handlelength=2.3, handletextpad=.8, labelspacing=.55)
        lg.set_zorder(100)
        caixa(ax, .40, r"$\eta=1{,}16$" if idioma == "pt" else r"$\eta=1.16$", "left")
        caixa(ax, .955, r"$N_{\mathrm{res}}$", "right")
        estilizar_eixos(ax, barra_cores(fig, cax, norm, escala_max, idioma))
        salvar(fig, f"mapa_fase_elastico_{idioma}", mostrar)


def mapa_separado(df, modelo, faixa, idioma, escala_max, mostrar):
    gr = Grade(*LIMITES_MAPA[modelo])
    mat = mapa_binado(df, gr, preencher=PREENCHER_VAZIOS_BANCO)
    with plt.rc_context({"font.family": "sans-serif"}):
        fig, ax, cax = criar_figura(gr)
        norm = desenhar_mapa(ax, mat, gr, idioma, escala_max)
        caixa(ax, .955, ROT_FAIXA[faixa] + r"   $N_{\mathrm{res}}$", "right")
        estilizar_eixos(ax, barra_cores(fig, cax, norm, escala_max, idioma))
        salvar(fig, f"mapa_fase_{modelo}_eta{faixa + 1}_{idioma}", mostrar)


def executar():
    gr = Grade(*LIMITES_MAPA["elastico"])
    elasticos = {eta: calcular_eta_elastico(eta, gr) for eta in ETAS_ELASTICO}
    series = {}
    for modelo in ("voigt", "maxwell"):
        banco = ler_banco(modelo)
        for k, (lo, hi) in enumerate(ETA_FAIXAS):
            series[(modelo, k)] = banco.loc[(banco["eta"] >= lo * (1 - 1e-9)) & (banco["eta"] < hi)]
    # escala comum aos mapas coloridos (eta = 1,16 no elástico e as faixas dos viscoelásticos);
    # eta = 2, 5 e 10 entram só como fronteira
    nmax = max(float(df["N"].max()) for df in [elasticos[1.16]] + list(series.values()))
    print("Escala de cores compartilhada, N_res máximo:", int(nmax))
    for idioma in (["pt", "en"] if GERAR_INGLES else ["pt"]):
        mostrar = MOSTRAR_FIGURAS and idioma == "pt"
        mapa_elastico(elasticos, idioma, nmax, mostrar)
        for (modelo, k), df in series.items():
            mapa_separado(df, modelo, k, idioma, nmax, mostrar)
    print("Figuras em:", FIG_DIR)


if __name__ == "__main__":
    executar()
