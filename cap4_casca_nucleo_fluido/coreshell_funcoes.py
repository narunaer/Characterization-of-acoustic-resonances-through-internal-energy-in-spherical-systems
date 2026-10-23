import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import newton, minimize_scalar
from scipy.signal import find_peaks
from scipy.special import spherical_jn, spherical_yn
import matplotlib.pyplot as plt

# ============================================================
# CASO ESPECIFICO
# ============================================================
ALPHA = 1.0e-3
BETA = 1.0e-3
DELTA = 1.5
LMAX = 10
X_MIN = 1.0e-4
X_MAX = 1.0

RHO_M = 1000.0
C_M = 1480.0
RHO_A = RHO_B = RHO_M
C_A = ALPHA * C_M
C_B = BETA * C_M
MATERIALS = (RHO_M, RHO_A, RHO_B, C_M, C_A, C_B)

# D_l regularizado: F_l(z)=z^4 D_l(z)
Y_BOTTOM = -1.0e-2
Y_TOP = 1.0e-3
ROOT_IMAG_MAX = 1.0e-8

# Escala radial aproximada usada apenas para distribuir sementes
PHI = C_M / C_A + DELTA * C_M / C_B
DX_EST = np.pi / PHI

# Busca local
SEEDS_PER_SPACING = 2.5
SEED_IMAGS = (-1.0e-8, -1.0e-5)
MAXITER = 100
ROOT_RESIDUAL_MAX = 1.0e-6
ROOT_DEDUP_TOL = 1.0e-8

# Principio do argumento
ARG_LEVELS = (2048, 4096, 8192)
ARG_INTEGER_TOL = 0.05
ARG_PHASE_JUMP_MAX = np.pi / 2

# Energia
ENERGY_RATIO_MIN = 10.0**0.75
ENERGY_MERGE_TOL = 1.0e-9

# Malhas usadas apenas para verificar a convergencia da contagem direta de picos
MESH_N_EACH = (750, 1500, 3000, 6000, 12000, 60000, 100000)
MESH_PROMINENCE = 0.75

OUT_DIR = (
    "/content/drive/MyDrive/validacao_velocity_001_001_delta15_simplificada"
    if os.path.isdir("/content/drive/MyDrive")
    else "./validacao_velocity_001_001_delta15_simplificada"
)
os.makedirs(OUT_DIR, exist_ok=True)
warnings.filterwarnings("ignore")


# ============================================================
# FUNCOES ESFERICAS
# ============================================================
def J(n, x):
    if n >= 0:
        return spherical_jn(n, x)
    m = -n - 1
    return (-1)**(m + 1) * spherical_yn(m, x)

def Y(n, x):
    if n >= 0:
        return spherical_yn(n, x)
    m = -n - 1
    return (-1)**m * spherical_jn(m, x)

def dJ(n, x):
    if n >= 0:
        return spherical_jn(n, x, derivative=True)
    m = -n - 1
    return (-1)**(m + 1) * spherical_yn(m, x, derivative=True)

def dY(n, x):
    if n >= 0:
        return spherical_yn(n, x, derivative=True)
    m = -n - 1
    return (-1)**m * spherical_jn(m, x, derivative=True)

def H(n, x):
    if n >= 0:
        return spherical_jn(n, x) + 1j * spherical_yn(n, x)
    m = -n - 1
    return 1j * (-1)**m * (spherical_jn(m, x) + 1j * spherical_yn(m, x))

def dH(n, x):
    if n >= 0:
        return spherical_jn(n, x, derivative=True) + 1j * spherical_yn(n, x, derivative=True)
    m = -n - 1
    return 1j * (-1)**m * (spherical_jn(m, x, derivative=True) + 1j * spherical_yn(m, x, derivative=True))

def Fjj(l, xB, yB):
    def F(x):
        return x**3 * (J(l, x)**2 - J(l - 1, x) * J(l + 1, x))
    return F(yB) - F(xB)

def Fyy(l, xB, yB):
    def F(x):
        return x**3 * (Y(l, x)**2 - Y(l - 1, x) * Y(l + 1, x))
    return F(yB) - F(xB)

def Fjy(l, xB, yB):
    def F(x):
        return 0.5 * x**3 * (2 * J(l, x) * Y(l, x) - J(l + 1, x) * Y(l - 1, x) - J(l - 1, x) * Y(l + 1, x))
    return F(yB) - F(xB)

def Bjj(l, xB, yB):
    def B(x):
        return x**2 * J(l, x) * dJ(l, x)
    return B(yB) - B(xB)

def Byy(l, xB, yB):
    def B(x):
        return x**2 * Y(l, x) * dY(l, x)
    return B(yB) - B(xB)

def Bjy(l, xB, yB):
    def B(x):
        return x**2 * J(l, x) * dY(l, x)
    return B(yB) - B(xB)

def FA(l, xA):
    return 0.5 * xA**3 * (J(l, xA)**2 - J(l - 1, xA) * J(l + 1, xA))

def BA(l, xA):
    return xA**2 * J(l, xA) * dJ(l, xA)


# ============================================================
# D_l E COEFICIENTES
# ============================================================
def D_l_coreshell(l, xA, xB, yM, yB, rho0, rho1, rho2, c0, c1, c2):
    rho_MB = rho0 / rho2
    rho_AB = rho1 / rho2
    k_MB = c2 / c0
    k_AB = c2 / c1

    A = -J(l, yB)
    B = -Y(l, yB)
    F = rho_MB * H(l, yM)
    G = -dJ(l, yB)
    HH = -dY(l, yB)
    JJ = k_MB * dH(l, yM)
    L = -rho_AB * J(l, xA)
    M = J(l, xB)
    P = Y(l, xB)
    Q = -k_AB * dJ(l, xA)
    R = dJ(l, xB)
    S = dY(l, xB)

    return (L * R - M * Q) * (F * HH - B * JJ) + (L * S - P * Q) * (A * JJ - F * G)

def coeffs(l, xA, xB, yM, yB, rho0, rho1, rho2, c0, c1, c2):
    rho_MB = rho0 / rho2
    rho_AB = rho1 / rho2
    k_MB = c2 / c0
    k_AB = c2 / c1

    A = -J(l, yB)
    B = -Y(l, yB)
    F = rho_MB * H(l, yM)
    G = -dJ(l, yB)
    HH = -dY(l, yB)
    JJ = k_MB * dH(l, yM)
    L = -rho_AB * J(l, xA)
    M = J(l, xB)
    P = Y(l, xB)
    Q = -k_AB * dJ(l, xA)
    R = dJ(l, xB)
    S = dY(l, xB)

    D = (L * R - M * Q) * (F * HH - B * JJ) + (L * S - P * Q) * (A * JJ - F * G)
    X = 1j * rho_MB * k_MB / yM**2
    b_l = (M * S - P * R) * X / D
    c_l = (P * Q - L * S) * X / D
    d_l = (L * R - M * Q) * X / D
    return b_l, c_l, d_l

def D_regularizado(l, z):
    rho0, rho1, rho2, c0, c1, c2 = MATERIALS
    z = np.asarray(z, dtype=complex)
    xA = (c0 / c1) * z
    xB = (c0 / c2) * z
    yM = (1.0 + DELTA) * z
    yB = (1.0 + DELTA) * xB
    return z**4 * D_l_coreshell(l, xA, xB, yM, yB, rho0, rho1, rho2, c0, c1, c2)


# ============================================================
# ENERGIA TOTAL W_T
# ============================================================
def energia_total(x_vals):
    rho0, rho1, rho2, c0, c1, c2 = MATERIALS
    x_vals = np.asarray(x_vals, dtype=float)

    xA = (c0 / c1) * x_vals
    xB = (c0 / c2) * x_vals
    yM = (1.0 + DELTA) * x_vals
    yB = (1.0 + DELTA) * xB

    WT = np.zeros_like(x_vals, dtype=float)

    with np.errstate(all="ignore"):
        for l in range(LMAX + 1):
            bl, cl, dl = coeffs(l, xA, xB, yM, yB, rho0, rho1, rho2, c0, c1, c2)
            fator = 2 * l + 1

            bl2 = np.abs(bl)**2
            fa = FA(l, xA)
            ba = BA(l, xA)
            w_core = np.real(fator * bl2 * (2.0 * fa + ba))

            cl2 = np.abs(cl)**2
            dl2 = np.abs(dl)**2
            re_cd = np.real(np.conj(cl) * dl)
            potencial = 0.5 * cl2 * Fjj(l, xB, yB) + 0.5 * dl2 * Fyy(l, xB, yB) + re_cd * Fjy(l, xB, yB)
            contorno = cl2 * Bjj(l, xB, yB) + dl2 * Byy(l, xB, yB) + 2.0 * re_cd * Bjy(l, xB, yB)
            w_shell = np.real(fator * (2.0 * potencial + contorno))

            WT += w_core + w_shell

    return np.where(np.isfinite(WT) & (WT > 0.0), WT, 0.0)


# ============================================================
# UTILIDADES DE RAIZ
# ============================================================
def dentro(z, tol=1e-10):
    z = complex(z)
    return (
        np.isfinite(z.real)
        and np.isfinite(z.imag)
        and X_MIN - tol <= z.real <= X_MAX + tol
        and Y_BOTTOM - tol <= z.imag <= ROOT_IMAG_MAX + tol
    )

def residuo_raiz(l, z):
    F = lambda zz: D_regularizado(l, zz)
    z = complex(z)
    valor = F(z)
    if not np.isfinite(valor):
        return np.inf

    h = 1e-7 * max(1.0, abs(z))
    vizinhos = np.array([z + h, z - h, z + 1j*h, z - 1j*h], dtype=complex)
    valores = np.asarray(F(vizinhos), dtype=complex)
    validos = np.isfinite(valores)

    if not np.any(validos):
        return np.inf

    escala = max(float(np.median(np.abs(valores[validos]))), np.finfo(float).tiny)
    return float(abs(valor) / escala)

def remover_duplicatas(raizes, tol=ROOT_DEDUP_TOL):
    raizes = sorted((complex(z) for z in raizes if dentro(z)), key=lambda z: (z.real, z.imag))
    unicas = []

    for z in raizes:
        if not unicas or min(abs(z - u) for u in unicas[-20:]) > tol:
            unicas.append(z)

    return unicas

def sementes_x():
    passo = DX_EST / SEEDS_PER_SPACING
    return np.arange(X_MIN + 0.5 * passo, X_MAX, passo), passo


# ============================================================
# NEWTON, SECANTE E MULLER
# ============================================================
def raizes_newton(l):
    F = lambda z: D_regularizado(l, z)
    xs, passo = sementes_x()
    raizes = []

    def dF(z):
        h = 2e-7 * max(1.0, abs(z))
        return (F(z + h) - F(z - h)) / (2.0 * h)

    for yi in SEED_IMAGS:
        for xr in xs:
            try:
                raiz = complex(newton(F, complex(xr, yi), fprime=dF, tol=1e-13, maxiter=MAXITER))
            except Exception:
                continue

            if dentro(raiz) and residuo_raiz(l, raiz) <= ROOT_RESIDUAL_MAX:
                raizes.append(raiz)

    return remover_duplicatas(raizes)

def raizes_secante(l):
    F = lambda z: D_regularizado(l, z)
    xs, passo = sementes_x()
    raizes = []

    for yi in SEED_IMAGS:
        for xr in xs:
            z0 = complex(xr - 0.08 * passo, yi)
            z1 = complex(xr + 0.08 * passo, 1.2 * yi)

            try:
                raiz = complex(newton(F, z0, x1=z1, tol=1e-13, maxiter=MAXITER))
            except Exception:
                continue

            if dentro(raiz) and residuo_raiz(l, raiz) <= ROOT_RESIDUAL_MAX:
                raizes.append(raiz)

    return remover_duplicatas(raizes)

def muller(F, z0, z1, z2, maxiter=MAXITER):
    z0, z1, z2 = complex(z0), complex(z1), complex(z2)
    f0, f1, f2 = F(z0), F(z1), F(z2)

    for _ in range(maxiter):
        h1 = z1 - z0
        h2 = z2 - z1

        if h1 == 0 or h2 == 0 or h1 + h2 == 0:
            return None

        d1 = (f1 - f0) / h1
        d2 = (f2 - f1) / h2
        d = (d2 - d1) / (h2 + h1)
        b = d2 + h2 * d
        disc = np.sqrt(b*b - 4.0*f2*d + 0j)
        e1, e2 = b + disc, b - disc
        den = e1 if abs(e1) > abs(e2) else e2

        if den == 0 or not np.isfinite(den):
            return None

        h = -2.0 * f2 / den
        z3 = z2 + h

        if not (np.isfinite(z3.real) and np.isfinite(z3.imag)):
            return None

        if abs(h) <= 1e-13 * max(1.0, abs(z3)):
            return z3

        z0, z1, z2 = z1, z2, z3
        f0, f1, f2 = f1, f2, F(z2)

    return None

def raizes_muller(l):
    F = lambda z: D_regularizado(l, z)
    xs, passo = sementes_x()
    raizes = []

    for yi in SEED_IMAGS:
        for xr in xs:
            zc = complex(xr, yi)
            raiz = muller(F, zc - 0.10 * passo, zc, zc + 0.10 * passo)

            if raiz is not None and dentro(raiz) and residuo_raiz(l, raiz) <= ROOT_RESIDUAL_MAX:
                raizes.append(raiz)

    return remover_duplicatas(raizes)


# ============================================================
# PRINCIPIO DO ARGUMENTO
# ============================================================
def contorno(xa, xb, ya, yb, n):
    baixo = np.linspace(xa, xb, n, endpoint=False) + 1j * ya
    direita = xb + 1j * np.linspace(ya, yb, n, endpoint=False)
    cima = np.linspace(xb, xa, n, endpoint=False) + 1j * yb
    esquerda = xa + 1j * np.linspace(yb, ya, n, endpoint=False)
    return np.concatenate([baixo, direita, cima, esquerda])

def winding(F, n):
    z = contorno(X_MIN, X_MAX, Y_BOTTOM, Y_TOP, n)
    valores = np.asarray(F(z), dtype=complex)

    if not np.all(np.isfinite(valores)) or np.any(np.abs(valores) == 0.0):
        return None, None

    fase = valores / np.abs(valores)
    razao = np.roll(fase, -1) / fase

    if not np.all(np.isfinite(razao)):
        return None, None

    incrementos = np.angle(razao)
    return float(np.sum(incrementos) / (2.0 * np.pi)), float(np.max(np.abs(incrementos)))

def contagem_argumento(l):
    F = lambda z: D_regularizado(l, z)
    anterior = None

    for n in ARG_LEVELS:
        valor, salto = winding(F, n)

        if valor is None:
            anterior = None
            continue

        contagem = int(round(valor))
        valido = contagem >= 0 and abs(valor - contagem) <= ARG_INTEGER_TOL and salto <= ARG_PHASE_JUMP_MAX

        if not valido:
            anterior = None
            continue

        if anterior == contagem:
            return contagem, True

        anterior = contagem

    return None, False

def raizes_argumento(l):
    """O princípio do argumento fornece a contagem; Müller localiza as raízes."""
    esperado, ok = contagem_argumento(l)

    if not ok or esperado is None or esperado <= 0:
        return [], esperado, False

    F = lambda z: D_regularizado(l, z)
    n_sementes = max(int(np.ceil(3.2 * esperado)), esperado + 8)
    passo = (X_MAX - X_MIN) / n_sementes
    raizes = []

    for fase in (0.25, 0.75):
        xs = X_MIN + (np.arange(n_sementes) + fase) * passo
        xs = xs[(xs > X_MIN) & (xs < X_MAX)]

        for yi in (-1e-7, -1e-5):
            for xr in xs:
                zc = complex(float(xr), yi)
                raiz = muller(F, zc - 0.12 * passo, zc, zc + 0.12 * passo, maxiter=80)

                if raiz is not None and dentro(raiz) and residuo_raiz(l, raiz) <= ROOT_RESIDUAL_MAX:
                    raizes.append(raiz)

        raizes = remover_duplicatas(raizes)

        if len(raizes) == esperado:
            break

    return raizes, esperado, len(raizes) == esperado


# ============================================================
# TESTE DOS POLOS NA ENERGIA TOTAL
# ============================================================
def janelas_entre_polos(raizes_por_l):
    """Cada polo so pode procurar um pico entre os pontos medios dos polos vizinhos."""
    polos = []
    for l, raizes in raizes_por_l.items():
        for i, z in enumerate(raizes):
            polos.append((float(z.real), int(l), int(i), complex(z)))

    polos.sort(key=lambda item: item[0])
    if not polos:
        return {}

    grupos = []
    atual = [polos[0]]

    for item in polos[1:]:
        if abs(item[0] - atual[-1][0]) <= ENERGY_MERGE_TOL:
            atual.append(item)
        else:
            grupos.append(atual)
            atual = [item]
    grupos.append(atual)

    centros = np.array([np.mean([p[0] for p in grupo]) for grupo in grupos], dtype=float)
    janelas = {}

    for g, grupo in enumerate(grupos):
        lo = X_MIN if g == 0 else 0.5 * (centros[g - 1] + centros[g])
        hi = X_MAX if g == len(grupos) - 1 else 0.5 * (centros[g] + centros[g + 1])
        for _, l, i, _ in grupo:
            janelas[(l, i)] = (float(lo), float(hi))

    return janelas


def validar_polo_na_energia(z, lo, hi):
    """Procura um maximo de W_T somente dentro da celula real atribuida ao polo."""
    x0 = float(np.real(z))
    gamma = 2.0 * abs(float(np.imag(z)))

    if not (hi > lo and lo <= x0 <= hi):
        return None

    linear = np.linspace(lo, hi, 121)
    escalas = np.array([0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    perto = np.concatenate([x0 - gamma * escalas, [x0], x0 + gamma * escalas]) if gamma > 0 else np.array([x0])
    x = np.unique(np.concatenate([linear, perto[(perto >= lo) & (perto <= hi)]]))
    w = energia_total(x)

    validos = np.isfinite(w) & (w > 0.0)
    if np.count_nonzero(validos) < 3:
        return None

    indices = find_peaks(np.where(validos, w, -np.inf))[0]
    indices = indices[(indices > 0) & (indices < len(x) - 1)]
    if len(indices) == 0:
        return None

    i = int(indices[np.argmin(np.abs(x[indices] - x0))])
    a, b = float(x[i - 1]), float(x[i + 1])

    def objetivo(xx):
        valor = energia_total(np.array([xx], dtype=float))[0]
        return -np.log(max(valor, np.finfo(float).tiny)) if np.isfinite(valor) and valor > 0 else 1e300

    try:
        resultado = minimize_scalar(objetivo, bounds=(a, b), method="bounded", options={"xatol": 1e-13, "maxiter": 200})
        x_pico = float(resultado.x) if resultado.success else float(x[i])
    except Exception:
        x_pico = float(x[i])

    w_pico = float(energia_total(np.array([x_pico], dtype=float))[0])
    if not (np.isfinite(w_pico) and w_pico > 0.0 and lo <= x_pico <= hi):
        return None

    eps = max(0.05 * gamma, (hi - lo) * 1e-7, np.finfo(float).eps * max(1.0, abs(x_pico)) * 64.0)
    xl = max(lo, x_pico - eps)
    xr = min(hi, x_pico + eps)
    wl, wr = energia_total(np.array([xl, xr], dtype=float))

    maximo_local = np.isfinite(wl) and np.isfinite(wr) and w_pico > wl and w_pico > wr
    externo = validos & ((x <= lo + 0.2 * (hi - lo)) | (x >= hi - 0.2 * (hi - lo)))
    fundo = float(np.median(w[externo])) if np.any(externo) else float(np.median(w[validos]))
    fundo = max(fundo, np.finfo(float).tiny)
    razao = float(w_pico / fundo)

    return {
        "x_pico": x_pico,
        "WT_pico": w_pico,
        "WT_fundo": fundo,
        "energy_ratio": razao,
        "maximo_local": bool(maximo_local),
        "visivel": bool(maximo_local and razao >= ENERGY_RATIO_MIN),
        "janela_lo": float(lo),
        "janela_hi": float(hi),
    }


def validar_metodo(nome, raizes_por_l):
    """Valida todos os polos do metodo usando uma particao unica do eixo real."""
    janelas = janelas_entre_polos(raizes_por_l)
    linhas = []

    for l in range(LMAX + 1):
        for i, z in enumerate(raizes_por_l.get(l, [])):
            lo, hi = janelas[(l, i)]
            energia = validar_polo_na_energia(z, lo, hi)
            linha = {
                "metodo": nome, "l": l, "pole_real": float(z.real), "pole_imag": float(z.imag),
                "residuo": residuo_raiz(l, z), "janela_lo": lo, "janela_hi": hi,
                "maximo_WT": False, "visivel": False, "x_pico_WT": np.nan, "energy_ratio": np.nan,
            }

            if energia is not None:
                linha["maximo_WT"] = energia["maximo_local"]
                linha["visivel"] = energia["visivel"]
                linha["x_pico_WT"] = energia["x_pico"]
                linha["energy_ratio"] = energia["energy_ratio"]

            linhas.append(linha)

    df = pd.DataFrame(linhas)
    if len(df) == 0:
        return df, 0, 0, 0

    picos = np.sort(df.loc[df["visivel"], "x_pico_WT"].dropna().to_numpy(dtype=float))
    unicos = []
    for x in picos:
        if not unicos or abs(x - unicos[-1]) > ENERGY_MERGE_TOL:
            unicos.append(float(x))

    return df, int(df["maximo_WT"].sum()), int(df["visivel"].sum()), len(unicos)


# ============================================================
# CONVERGENCIA DA CONTAGEM DIRETA NA MALHA
# ============================================================
def malha_direta(n_each):
    log = np.geomspace(X_MIN, min(0.1, X_MAX), int(n_each))
    linear = np.linspace(min(0.1, X_MAX), X_MAX, int(n_each))
    return np.unique(np.concatenate([log, linear]))


def contar_picos_na_malha(x):
    w = energia_total(x)
    validos = np.isfinite(w) & (w > 0.0)
    if np.count_nonzero(validos) < 3:
        return 0

    piso = max(float(np.min(w[validos])) * 1e-14, np.finfo(float).tiny)
    log_w = np.log10(np.where(validos, np.maximum(w, piso), piso))
    picos, _ = find_peaks(log_w, prominence=MESH_PROMINENCE)
    return int(len(picos))


def grafico_convergencia_malha():
    linhas = []

    print("\nCONVERGENCIA DA CONTAGEM DIRETA EM W_T")
    for n_each in MESH_N_EACH:
        x = malha_direta(n_each)
        n_picos = contar_picos_na_malha(x)
        linhas.append({"N_grid": len(x), "N_picos_WT": n_picos})
        print(f"N_grid={len(x):6d} -> {n_picos:5d} picos", flush=True)

    df = pd.DataFrame(linhas)
    df.to_csv(os.path.join(OUT_DIR, "convergencia_malha.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(df["N_grid"], df["N_picos_WT"], marker="o")
    ax.set_xlabel("Número de pontos da malha")
    ax.set_ylabel("Número de picos em $W_T$")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "convergencia_malha.png"), dpi=300, bbox_inches="tight")
    plt.show()
    return df
