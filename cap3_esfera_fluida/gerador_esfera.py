#!/usr/bin/env python3
# Gerador de ressonancias da esfera fluida simples (esfera S de raio a num meio fluido M), versao 3.
# Mesma metodologia do gerador core-shell fluido (versao 8).
#
# Parametros de cada curva: m = c_M/c_S (contraste de velocidade; x_S = m x_M) e rho_ratio = rho_S/rho_M.
# Coeficiente interno (potencial de velocidade, pressao e velocidade normal continuas em r = a):
#     b_l = i (rho_M/rho_S) / (x_M^2 D_l),   D_l = j_l(x_S) h_l'(x_M) - (m/rho_ratio) j_l'(x_S) h_l(x_M).
# Energia interna (potencial + cinetica), normalizada como no codigo antigo:
#     W_l = (2l+1) |b_l|^2 [ 2 F_l + B_l ],  F_l = (1/2) x_S^3 (j_l^2 - j_{l-1} j_{l+1}),  B_l = x_S^2 j_l j_l'  (em x_S),
#     W_T = soma de W_l para l = 0..LMAX. Uma regiao so: nao ha soma de regioes nem peso de impedancia.
# CORRECAO em relacao ao codigo antigo: j_{-1}(x) = cos(x)/x = -y_0(x). O codigo antigo usava +y_0(x), o que deixava
# F_0 (termo l = 0 de WP e WK) errado. Aqui o sinal e' o correto (o mesmo do core-shell), conferido contra a integral
# numerica de x^2 j_0^2 (validacao 1).
#
# Como a contagem funciona:
#   Toda curva comeca na malha base de 5000 pontos em x_M: 1000 log em [x_min, 0.1], 3000 lineares em [0.1, 0.9] e
#   1000 lineares em [0.9, 1], mais uma extensao de min(Delta_W, 0.05) depois de x_max para enxergar pico na borda.
#   Espacamento medio entre ressonancias de um mesmo l: Delta_W = pi/Phi, Phi = c_M/c_S = m (no codigo, 'esp').
#
#   m <= 1 (curva normal: som dentro da esfera nao e' mais lento que fora): as ressonancias sao poucas e largas e a
#   malha base resolve. N_peaks = maximos de W_T com proeminencia >= 0.1 em log10. A busca de polos tambem roda (e' barata
#   aqui), mas so serve de metadado: associa cada pico ao l e ao polo (D0_confirmado, primary_l) e mostra o que a malha
#   base deveria ver (N_polos_visiveis, N_previsto_malha). Desligue com --sem-polos-normais.
#
#   m > 1 (curva de risco: som lento dentro, picos estreitos e amontoados):
#     1. zeros de z^2 D_l no plano complexo, l = 0..10 (Newton a partir dos minimos de |D_l| no eixo real, auditoria pelo
#        principio do argumento, busca completa por subdivisao se nao fechar). z^2 tira o polo duplo de D_l em z = 0;
#     2. polos com gamma/x < 1e-10 tem o gamma refeito so com valores no eixo real (D_l = D_j + i D_y);
#     3. peso de cada polo: W_l(x) ~ S/((x - x_k)^2 + gamma^2), entao FWHM = 2 gamma, Q = x_k/(2 gamma), area = pi S/gamma;
#     4. malha refinada pelo passo que cada pico precisa, dx_nec = 2 sqrt(S/(k fundo) - gamma^2), k = 10^0.1 - 1;
#     5. contagem final pelos polos (Q >= 2 e fracao de area >= 1e-3) quando a busca fechou em todos os l; senao, a malha
#        refinada.
#
# No Colab: cole o arquivo inteiro numa celula e rode. A rodada usa os valores de COLAB (logo abaixo das constantes),
# monta o Drive e salva em COLAB['saida']. Se a sessao cair, rodar a celula de novo continua do checkpoint.
#
# Na linha de comando:
#   python -u gerador_esfera.py --workers 8 --saida /content/drive/MyDrive/resultados_esfera
#   python -u gerador_esfera.py --max-curvas 20               (teste rapido)
#   python -u gerador_esfera.py --indices 120 4500 --workers 2

import argparse, json, multiprocessing, os, signal, sys, time, uuid, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import scipy.linalg as la
from scipy.optimize import minimize_scalar, newton
from scipy.signal import find_peaks
from scipy.special import spherical_jn, spherical_yn

warnings.filterwarnings("ignore")

LMAX = 10
X_MIN, X_MAX = 1e-4, 1.0
IMAG_MIN, IMAG_MAX = -0.5, 1e-3   # topo do retangulo acima do eixo: polo com gamma ~ 1e-30 nao fica colado na borda do contorno
RHO_EXTRA = np.linspace(1.0e-4, 9.0e-4, 10)
P_POLO = 2             # D_l ~ z^-2 perto de z = 0 em todo l; a busca usa z^2 D_l

# malha e contagem
N_LOG, N_MEIO, N_FIM = 1000, 3000, 1000
PROM_WT = 0.1          # proeminencia minima (log10) dos picos de W_T
PROM_WL = 0.75         # proeminencia minima (log10) dos picos de W_l quando zeros e malha nao batem
Q_MIN = 2.0
FRAC_AREA = 1e-3       # polo conta se a area dele for >= 0.1% da energia integrada na janela de largura pi/Phi
Q_MIN_POLO = 2.0       # e se Q = x/(2 gamma) >= 2 (abaixo disso nao e' pico: a lorentziana nem vale)
DX_MIN = 1e-9          # abaixo disso o refino nao vale a pena (precisao do double)
N_MAX_PONTOS = 2_000_000

# busca de polos
NIVEIS_CONTORNO = (64, 128, 256)
NIVEIS_AUDITORIA = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
TOL_INTEIRO = 0.05
PROF_MAX = 40
BEYN_MAX, BEYN_FOLGA, BEYN_N = 8, 4, 500
RESIDUO_MAX = 1e-7
GAMMA_RAD = 1e-10      # gamma/x abaixo disso o gamma do Newton em complexo e' ruido; refaz pelo eixo real (polo_estreito)
TOL_DUPLICATA = 1e-8
VERSAO = 3

# valores usados quando o arquivo roda dentro de um notebook (Colab/Jupyter), onde nao ha linha de comando
COLAB = dict(workers=os.cpu_count() or 1, saida="/content/drive/MyDrive/resultados_esfera", prazo_polos=120.0, limite_curva=600.0,
             salvar_a_cada=25, max_curvas=None, indices=None, refazer_falhas=False, sem_polos_normais=False)


# ---------------------------------------------------------------------------
# Funcoes esfericas (ordem negativa: j_{-n-1} = (-1)^{n+1} y_n, y_{-n-1} = (-1)^n j_n). Aceitam argumento complexo.
# ---------------------------------------------------------------------------
def J(n, x):
    if n >= 0: return spherical_jn(n, x)
    m = -n - 1; return (-1) ** (m + 1) * spherical_yn(m, x)

def Y(n, x):
    if n >= 0: return spherical_yn(n, x)
    m = -n - 1; return (-1) ** m * spherical_jn(m, x)

def dJ(n, x):
    if n >= 0: return spherical_jn(n, x, derivative=True)
    m = -n - 1; return (-1) ** (m + 1) * spherical_yn(m, x, derivative=True)

def dY(n, x):
    if n >= 0: return spherical_yn(n, x, derivative=True)
    m = -n - 1; return (-1) ** m * spherical_jn(m, x, derivative=True)

def H(n, x): return J(n, x) + 1j * Y(n, x)
def dH(n, x): return dJ(n, x) + 1j * dY(n, x)

# primitivas da energia no interior (0 ate x_S): integral de t^2 j_l(t)^2 dt = F_l, e o termo de borda B_l
def FS(l, xS): return 0.5 * xS**3 * (J(l, xS) ** 2 - J(l - 1, xS) * J(l + 1, xS))
def BS(l, xS): return xS**2 * J(l, xS) * dJ(l, xS)


# ---------------------------------------------------------------------------
# Denominador, coeficiente e energia
# ---------------------------------------------------------------------------
def D_l_sphere(l, xM, m, rho_ratio):
    """D_l = j_l(m x) h_l'(x) - (m/rho_ratio) j_l'(m x) h_l(x). Mesmo nome e conta do codigo antigo, agora aceita x complexo."""
    xS = m * xM
    return J(l, xS) * dH(l, xM) - (m / rho_ratio) * dJ(l, xS) * H(l, xM)


def coef_b(l, x, mat):
    """b_l = i (rho_M/rho_S) / (x^2 D_l): sai das condicoes de contorno em r = a
        rho_S b j_l(x_S) = rho_M [ j_l(x) + c_l h_l(x) ]    (pressao p = i omega rho phi continua)
        m b j_l'(x_S)    = j_l'(x) + c_l h_l'(x)            (velocidade normal continua; k_S/k_M = m)
    eliminando c_l e usando o wronskiano j_l h_l' - j_l' h_l = i/x^2."""
    m, rho = mat
    return 1j / (rho * x**2 * D_l_sphere(l, x, m, rho))


def D_l(l, z, mat):
    """z^2 D_l: tem os mesmos zeros, mas sem o polo duplo em z = 0 colado em x_min
    (j_l(mz) ~ z^l, h_l(z) ~ z^-(l+1), h_l'(z) ~ z^-(l+2), entao D_l ~ z^-2 em todo l)."""
    z = np.asarray(z, dtype=complex)
    with np.errstate(all="ignore"):
        return z**P_POLO * D_l_sphere(l, z, *mat)


def D_partes(l, x, mat):
    """No eixo real so h_l(x) = j_l(x) + i y_l(x) e' complexo e D_l e' linear nele. Entao D_l = D_j + i D_y, com
        D_j = j_l(x_S) j_l'(x) - (m/rho) j_l'(x_S) j_l(x),   D_y = j_l(x_S) y_l'(x) - (m/rho) j_l'(x_S) y_l(x),
    as duas reais e calculadas separadas, cada uma com a precisao do double."""
    x = np.asarray(x, dtype=float); m, rho = mat; xS = m * x
    with np.errstate(all="ignore"):
        jS, djS = J(l, xS), (m / rho) * dJ(l, xS)
        return jS * dJ(l, x) - djS * J(l, x), jS * dY(l, x) - djS * Y(l, x)


def energia_l(l, x, mat):
    """W_l = (2l+1) |b_l|^2 [2 F_l(x_S) + B_l(x_S)] de um l so."""
    x = np.asarray(x, dtype=float); xS = mat[0] * x
    with np.errstate(all="ignore"):
        w = np.real((2 * l + 1) * np.abs(coef_b(l, x, mat)) ** 2 * (2.0 * FS(l, xS) + BS(l, xS)))
    return np.where(np.isfinite(w), w, np.nan)


def sphere_energy_parts_vs_xM(xM_vals, m, rho_ratio, lmax=LMAX, bloco=400_000):
    """Mesma assinatura e saida do codigo antigo: (W_T, W_P, W_K), com
        W_P = soma (2l+1) |b_l|^2 F_l,   W_K = soma (2l+1) |b_l|^2 (F_l + B_l),   W_T = W_P + W_K.
    Cada Bessel e' calculada uma vez so (ordens -1 a lmax+1 em x_S e em x) e as derivadas saem da recorrencia
    f_l' = (l/x) f_l - f_{l+1}. (A forma f_l' = f_{l-1} - (l+1)/x f_l e' identica, mas em l = 0 e x pequeno vira
    cos(x)/x - sin(x)/x^2, diferenca de dois numeros ~ 1/x que perde ~1e-7 de precisao; a forma de cima nao cancela.)
    Unica diferenca da conta antiga: o sinal correto de j_{-1}."""
    x = np.asarray(xM_vals, dtype=float)
    if len(x) > bloco:
        p = [sphere_energy_parts_vs_xM(x[i:i + bloco], m, rho_ratio, lmax) for i in range(0, len(x), bloco)]
        return tuple(np.concatenate([q[k] for q in p]) for k in range(3))
    if np.any(x <= 0): raise ValueError("All x_M values must be positive.")
    if not np.isfinite(m) or m <= 0 or not np.isfinite(rho_ratio) or rho_ratio <= 0: raise ValueError("m e rho_ratio devem ser positivos e finitos.")
    xS, mt = m * x, m / rho_ratio; WP, WK = np.zeros_like(x), np.zeros_like(x)
    with np.errstate(all="ignore"):
        JS = {n: J(n, xS) for n in range(-1, lmax + 2)}; JM = {n: J(n, x) for n in range(lmax + 2)}; YM = {n: Y(n, x) for n in range(lmax + 2)}
        for l in range(lmax + 1):
            d = lambda F, z: l / z * F[l] - F[l + 1]
            jS, djS, h, dh = JS[l], d(JS, xS), JM[l] + 1j * YM[l], d(JM, x) + 1j * d(YM, x)
            peso = (2 * l + 1) * np.abs(1j / (rho_ratio * x**2 * (jS * dh - mt * djS * h))) ** 2
            F = 0.5 * xS**3 * (jS**2 - JS[l - 1] * JS[l + 1]); B = xS**2 * jS * djS
            WP += np.nan_to_num(peso * F); WK += np.nan_to_num(peso * (F + B))
    WP = np.where(np.isfinite(WP) & (WP >= 0), WP, 0.0); WK = np.where(np.isfinite(WK) & (WK >= 0), WK, 0.0)
    WT = WP + WK
    return np.where(np.isfinite(WT) & (WT > 0), WT, 0.0), WP, WK


def sphere_total_energy_vs_xM(xM_vals, m, rho_ratio, lmax=LMAX): return sphere_energy_parts_vs_xM(xM_vals, m, rho_ratio, lmax)[0]
def energia_total(x, mat, lmax=LMAX): return sphere_energy_parts_vs_xM(x, mat[0], mat[1], lmax)[0]


def polo_estreito(l, z, mat):
    """Polo colado no eixo real (gamma/x < GAMMA_RAD): o Newton em complexo devolve gamma ~ 1e-20 de ruido, porque
    Im de j_l(x + i eps) se perde no double. Aqui o Newton usa so valores no eixo real:
        D_l(x0 - i gamma) ~ D_l(x0) - i gamma D_l'(x0) = 0  ->  z0 = x0 - D_l(x0)/D_l'(x0),  D_l = D_j + i D_y,
    itera x0 <- x0 - Re(D_l/D_l') ate parar e fica gamma = Im(D_l(x0)/D_l'(x0))."""
    x = z.real
    if -z.imag > GAMMA_RAD * x: return z
    Dc = lambda t: (lambda p: p[0] + 1j * p[1])(D_partes(l, t, mat))
    q = np.nan
    for _ in range(50):
        h = 1e-7 * x; q = complex(Dc(x) / ((Dc(x + h) - Dc(x - h)) / (2 * h))); x -= q.real
        if abs(q.real) < 1e-15 * x: break
    return complex(x, -q.imag) if np.isfinite(q) and q.imag > 0 else z


def peso_polo(l, z, mat):
    """Perto do polo z_k = x_k - i gamma: b_l ~ R/(x - z_k), com R = i (rho_M/rho_S) / (z_k^2 D_l'(z_k)). Entao
        W_l(x) ~ S / ((x - x_k)^2 + gamma^2),   S = (2l+1) |R|^2 [2 F_l(m x_k) + B_l(m x_k)].
    Altura do pico = S/gamma^2, largura a meia altura = 2 gamma e area = pi S/gamma. Nada disso depende de malha."""
    z = complex(z); h = 1e-7 * abs(z); m, rho = mat
    with np.errstate(all="ignore"):
        dD = (D_l_sphere(l, z + h, m, rho) - D_l_sphere(l, z - h, m, rho)) / (2 * h)
        R = 1j / (rho * z**2 * dD); xS = m * z.real
        S = (2 * l + 1) * abs(R) ** 2 * (2 * FS(l, xS) + BS(l, xS))
    return float(np.real(S))


# ---------------------------------------------------------------------------
# Grade (m, rho_ratio): a mesma do codigo antigo
# ---------------------------------------------------------------------------
def _legacy_values():
    p = [np.linspace(a, b, 10) for a, b in ((0.001, 0.009), (0.01, 0.09), (0.1, 0.9), (1.0, 9.0), (10.0, 90.0), (100.0, 900.0))]
    return np.unique(np.concatenate(p + [[1000.0, 2000.0, 3000.0]]))


def grade():
    """m_grid, rho_grid achatados com indexing='ij' (curva i = indice im * N_rho + ir), como no codigo antigo."""
    m_vals = np.sort(_legacy_values()); rho_vals = np.sort(np.unique(np.concatenate([_legacy_values(), RHO_EXTRA])))
    M, R = np.meshgrid(m_vals, rho_vals, indexing="ij")
    return M.ravel(), R.ravel()


# ---------------------------------------------------------------------------
# Malha e picos
# ---------------------------------------------------------------------------
def malha_base(mat, x_min=X_MIN, x_max=X_MAX):
    esp = np.pi / mat[0]                              # Delta_W = pi/Phi, Phi = m: distancia media entre modos de um mesmo l
    ext = min(esp, 0.05 * x_max); a, b = min(0.1, x_max), min(0.9, x_max)
    x = np.unique(np.concatenate([np.geomspace(x_min, a, N_LOG), np.linspace(a, b, N_MEIO), np.linspace(b, x_max + ext, N_FIM)]))
    return x, esp


def maximos(energia, x, esp, prom=None, x_min=X_MIN, x_max=X_MAX, w=None):
    """Indices e posicoes dos maximos de energia(x) (proeminencia em log10, se prom for dado). Pico perto de x_max tem a
    posicao refinada antes de decidir se fica dentro do intervalo."""
    w = energia(x) if w is None else w
    if prom is None:
        idx = find_peaks(np.where(np.isfinite(w), w, -np.inf))[0]
    else:
        ok = np.isfinite(w) & (w > 0)
        if not ok.any(): return np.array([], int), np.array([])
        piso = max(w[ok].min() * 1e-14, np.finfo(float).tiny)
        idx = find_peaks(np.log10(np.where(ok, np.maximum(w, piso), piso)), prominence=prom)[0]
    fica, xs = [], []
    for i in idx:
        xp = float(x[i])
        if xp > x_max - 2 * min(esp, 0.05 * x_max) and 0 < i < len(x) - 1:
            alvo = lambda t: -np.log(max(float(energia(np.array([t]))[0]), np.finfo(float).tiny))
            xp = float(minimize_scalar(alvo, bounds=(x[i - 1], x[i + 1]), method="bounded", options={"xatol": 1e-12}).x)
        if x_min <= xp <= x_max: fica.append(i); xs.append(xp)
    return np.array(fica, int), np.array(xs)


def refina_posicoes(energia, x, xs, it=60):
    """Secao aurea em todos os picos de uma vez, cada um no seu intervalo [x_{i-1}, x_{i+1}]."""
    if len(xs) == 0: return xs
    i = np.clip(np.searchsorted(x, xs), 1, len(x) - 1)
    a, b = x[i - 1], x[np.minimum(i + 1, len(x) - 1)]; g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a); fc, fd = energia(c), energia(d)
    for _ in range(it):
        mk = np.nan_to_num(fc, nan=-np.inf) > np.nan_to_num(fd, nan=-np.inf)
        b, a = np.where(mk, d, b), np.where(mk, a, c)
        c, d = b - g * (b - a), a + g * (b - a); fc, fd = energia(c), energia(d)
    return 0.5 * (a + b)


def medir_picos(WT, x, xs, esp, escala=None, n_scan=80, it=50):
    """Mede cada pico: topo refinado (secao aurea), fundo = mediana de W_T perto das duas bordas da janela do pico (metade do
    caminho ate os picos vizinhos, no maximo esp/2), e FWHM procurando o cruzamento da meia altura
    fundo + (topo - fundo)/2 de cada lado (varredura em log + bissecao). Q = x/FWHM. Vetorizado em todos os picos."""
    n = len(xs)
    if n == 0: return []
    xs0 = np.asarray(xs, dtype=float); w0 = WT(xs0); xs = refina_posicoes(WT, x, xs0); wp = WT(xs)
    xs, wp = np.where(wp >= w0, xs, xs0), np.maximum(wp, w0)     # se o refino escorregar pra fora de um pico muito fino, fica o ponto da malha
    viz_e = np.r_[X_MIN, 0.5 * (xs[1:] + xs[:-1])]; viz_d = np.r_[0.5 * (xs[1:] + xs[:-1]), X_MAX + min(esp, 0.05)]
    lo, hi = np.maximum(viz_e, xs - 0.5 * esp), np.minimum(viz_d, xs + 0.5 * esp)
    fr = np.array([0.02, 0.06, 0.12, 0.2])
    pts = np.concatenate([lo[:, None] + fr * (xs - lo)[:, None], hi[:, None] - fr * (hi - xs)[:, None]], axis=1)
    fundo = np.median(WT(pts.ravel()).reshape(n, -1), axis=1)
    nivel = fundo + 0.5 * (wp - fundo)
    escala = np.full(n, np.nan) if escala is None else np.asarray(escala, dtype=float)
    lados = []
    for lado, dmax in ((-1, xs - lo), (+1, hi - xs)):
        base = np.where(np.isfinite(escala) & (escala > 0), escala, dmax / 100)
        dmin = np.maximum(32 * np.finfo(float).eps * np.maximum(1, xs), np.minimum(1e-3 * base, 1e-3 * dmax))
        D = np.exp(np.linspace(np.log(dmin), np.log(np.maximum(dmax, 2 * dmin)), n_scan, axis=1))
        abaixo = (WT((xs[:, None] + lado * D).ravel()).reshape(n, -1) < nivel[:, None])
        tem = abaixo.any(axis=1); k = np.argmax(abaixo, axis=1)
        a = np.where(k > 0, D[np.arange(n), np.maximum(k - 1, 0)], 0.0); b = D[np.arange(n), k]
        for _ in range(it):                     # bissecao entre o ultimo ponto acima e o primeiro abaixo do nivel
            mm = 0.5 * (a + b); acima = WT(xs + lado * mm) >= nivel; a, b = np.where(acima, mm, a), np.where(acima, b, mm)
        lados.append(np.where(tem, 0.5 * (a + b), np.nan))
    fwhm = lados[0] + lados[1]; ok = (wp > fundo) & (fundo > 0)
    saida = []
    for i in range(n):
        razao = wp[i] / fundo[i] if fundo[i] > 0 else np.inf
        saida.append({"xM_peak": float(xs[i]), "WT_peak": float(wp[i]), "WT_background": float(fundo[i]), "energy_ratio": float(razao),
                      "energy_log10_prominence": float(np.log10(razao)) if razao > 0 else np.nan,
                      "FWHM": float(fwhm[i]) if ok[i] and np.isfinite(fwhm[i]) else np.nan,
                      "Q": float(xs[i] / fwhm[i]) if ok[i] and np.isfinite(fwhm[i]) and fwhm[i] > 0 else np.nan,
                      "area_energy": np.nan, "frac_area_local": np.nan})
    return saida


def casar(xs, zeros, esp):
    """Casa maximos da malha com zeros, 1 a 1, do par mais proximo pro mais distante (tolerancia max(2 gamma, esp/4))."""
    xs = np.asarray(xs, dtype=float)
    if not len(xs) or not len(zeros): return {}
    o = np.argsort(xs); xo = xs[o]; pares = []
    for j, z in enumerate(zeros):
        tol = max(2 * abs(z.imag), 0.25 * esp); a, b = np.searchsorted(xo, z.real - tol), np.searchsorted(xo, z.real + tol, side="right")
        pares += [(abs(xo[k] - z.real), int(o[k]), j) for k in range(a, b)]
    usado_x, usado_z, casados = set(), set(), {}
    for _, i, j in sorted(pares):
        if i not in usado_x and j not in usado_z: usado_x.add(i); usado_z.add(j); casados[j] = i
    return casados


def modos_por_l(mat, zeros_por_l, x, esp, lmax=LMAX):
    """Posicoes dos modos de cada l, decididas pela comparacao zeros x maximos de W_l na malha base. So e' usado como alvo
    da malha refinada nos l em que a busca de polos nao fechou."""
    saida = []
    for l in range(lmax + 1):
        zeros = zeros_por_l.get(l); E = lambda v, l=l: energia_l(l, v, mat)
        _, xs = maximos(E, x, esp)
        r = {"l": l, "N_zeros": None if zeros is None else len(zeros), "N_max_malha": len(xs)}
        if zeros is not None:
            cas = casar(xs, zeros, esp)
            if len(xs) == len(zeros) == len(cas):
                r.update(regra="bate", x=[z.real for z in zeros]); saida.append(r); continue
            bons = [j for j, z in enumerate(zeros) if z.real / (2 * abs(z.imag)) >= Q_MIN]
            sobra_x = [v for i, v in enumerate(xs) if i not in {cas[j] for j in cas if j not in bons}]
            z2 = [zeros[j] for j in bons]
            if len(sobra_x) == len(z2) == len(casar(sobra_x, z2, esp)):
                r.update(regra="bate_Q2", x=[z.real for z in z2]); saida.append(r); continue
        _, xp = maximos(E, x, esp, prom=PROM_WL)
        r.update(regra="proeminencia" if zeros is not None else "sem_polos", x=list(refina_posicoes(E, x, xp)))
        saida.append(r)
    return saida


def separacao_laurent(polos, mat, n=400):
    """Perto de dois polos vizinhos, 1/D_l ~ A1/(x - z1) + A2/(x - z2), com A = 1/D_l'(z). Procura no eixo real onde esse
    modelo de dois polos tem os seus maximos e devolve a distancia entre eles (nan quando so sobra um maximo: os dois
    picos se fundem e nenhuma malha separa). Polos do mesmo l somam na amplitude; de l diferentes somam na energia."""
    if len(polos) < 2: return np.array([]), np.array([])
    pol = sorted(polos, key=lambda p: p["pole_real"])
    z = np.array([complex(p["pole_real"], p["pole_imag"]) for p in pol]); l = np.array([p["l"] for p in pol])
    A, h = np.empty(len(z), complex), np.empty(len(z))
    for ll in np.unique(l):
        k = l == ll; zz = z[k]; e = 1e-7 * np.maximum(np.abs(zz), 1e-4)
        A[k] = 2 * e / (D_l(ll, zz + e, mat) - D_l(ll, zz - e, mat))
        gam = np.abs(zz.imag); s_ = np.maximum(gam, 1e-6 * zz.real)          # altura do pico tirada da cauda lorentziana
        h[k] = energia_l(ll, zz.real + s_, mat) * (s_**2 + gam**2) / gam**2
    z1, z2 = z[:-1], z[1:]; x1, x2 = z1.real, z2.real
    G1, G2 = np.abs(z1.imag)[:, None], np.abs(z2.imag)[:, None]
    g1, g2 = np.maximum(G1[:, 0], 1e3 * np.finfo(float).eps * x1), np.maximum(G2[:, 0], 1e3 * np.finfo(float).eps * x2)
    u = np.geomspace(1e-3, 30, n // 4)
    X = np.sort(np.concatenate([x1[:, None] - g1[:, None] * u[::-1], x1[:, None] + g1[:, None] * u, np.linspace(x1, x2, n // 4 + 2, axis=1)[:, 1:-1],
                                x2[:, None] - g2[:, None] * u[::-1], x2[:, None] + g2[:, None] * u], axis=1), axis=1)
    mesmo = (l[:-1] == l[1:])[:, None]
    with np.errstate(all="ignore"):
        W = np.where(mesmo, np.abs(A[:-1, None] / (X - z1[:, None]) + A[1:, None] / (X - z2[:, None])) ** 2,
                     h[:-1, None] * G1**2 / ((X - x1[:, None]) ** 2 + G1**2) + h[1:, None] * G2**2 / ((X - x2[:, None]) ** 2 + G2**2))
    mx = (W[:, 1:-1] > W[:, :-2]) & (W[:, 1:-1] >= W[:, 2:]) & np.isfinite(W[:, 1:-1])
    sep = np.full(len(z1), np.nan)
    for i in np.flatnonzero(mx.sum(axis=1) >= 2):
        p = X[i, 1:-1][mx[i]]; sep[i] = p.max() - p.min()
    return 0.5 * (x1 + x2), sep


# ---------------------------------------------------------------------------
# Busca de polos (principio do argumento + Beyn + Newton)
# ---------------------------------------------------------------------------
def contorno(xa, xb, ya, yb, n):
    t = np.linspace(0, 1, max(int(n), 4), endpoint=False)
    return np.concatenate([xa + (xb - xa) * t + 1j * ya, xb + 1j * (ya + (yb - ya) * t), xb - (xb - xa) * t + 1j * yb, xa + 1j * (yb - (yb - ya) * t)])


def voltas(F, caixa, n):
    """Numero de voltas de F em torno da origem ao longo da borda da caixa: (1/2 pi) soma de arg(F_{k+1}/F_k)."""
    v = np.asarray(F(contorno(*caixa, n)), dtype=complex)
    if not np.all(np.isfinite(v)) or np.any(v == 0): return None
    with np.errstate(all="ignore"): r = np.roll(v, -1) / v
    return float(np.sum(np.angle(r)) / (2 * np.pi)) if np.all(np.isfinite(r)) else None


def conta_zeros(F, caixa, niveis=NIVEIS_CONTORNO):
    """Numero de zeros na caixa; so aceita quando duas resolucoes seguidas dao o mesmo inteiro."""
    antes = None
    for n in niveis:
        w = voltas(F, caixa, n); k = None if w is None else int(round(w))
        if k is None or k < 0 or abs(w - k) > TOL_INTEIRO: antes = None; continue
        if k == antes: return k
        antes = k
    return None


def auditoria(F, caixa, achados):
    """Recontagem do contorno inteiro com resolucao crescente; passa se bater duas vezes seguidas com o que foi achado."""
    seguidas, ultimo = 0, None
    for n in NIVEIS_AUDITORIA:
        w = voltas(F, caixa, n); k = None if w is None else int(round(w))
        if k is None or k < 0 or abs(w - k) > TOL_INTEIRO: seguidas = 0; continue
        ultimo = k; seguidas = seguidas + 1 if k == achados else 0
        if seguidas >= 2: return k, True
    return ultimo, False


def dentro(z, xa, xb, ya, yb, tol=1e-10): return xa - tol <= z.real <= xb + tol and ya - tol <= z.imag <= yb + tol


def residuo(F, z):
    """|F(z)| relativo a mediana de |F| em quatro vizinhos a distancia 1e-7 |z|."""
    v = F(z); h = 1e-7 * max(1.0, abs(z))
    viz = np.asarray(F(np.array([z + h, z - h, z + 1j * h, z - 1j * h])), dtype=complex); viz = viz[np.isfinite(viz)]
    return abs(v) / max(np.median(np.abs(viz)), np.finfo(float).tiny) if np.isfinite(v) and len(viz) else np.inf


def polir(F, z0, caixa):
    try: z = complex(newton(F, z0, tol=1e-13, maxiter=150))
    except Exception: return None
    return z if dentro(z, *caixa) and residuo(F, z) <= RESIDUO_MAX else None


def beyn(F, caixa, r):
    """Momentos da integral de contorno -> autovalores = zeros. So usado pra caixas com 2 a 8 zeros."""
    L = r + BEYN_FOLGA; xa, xb, ya, yb = caixa; c = 0.5 * (xa + xb) + 0.5j * (ya + yb)
    z = contorno(*caixa, BEYN_N); fz = np.asarray(F(z), dtype=complex)
    if not np.all(np.isfinite(fz)) or np.any(fz == 0): return None
    dlog = (F(z + 1e-6) - F(z - 1e-6)) / 2e-6 / fz; dz = np.roll(z, -1) - z; w = z - c
    mo = [np.sum(0.5 * ((w**k) * dlog + np.roll((w**k) * dlog, -1)) * dz) / (2j * np.pi) for k in range(2 * L)]
    H0 = np.array([[mo[i + j] for j in range(L)] for i in range(L)]); H1 = np.array([[mo[i + j + 1] for j in range(L)] for i in range(L)])
    try:
        U, S, Vh = la.svd(H0)
        if np.any(S[:r] <= 0): return None
        return la.eigvals(U[:, :r].conj().T @ H1 @ Vh[:r].conj().T @ np.diag(1 / S[:r])) + c
    except Exception:
        return None


def sem_duplicatas(zs):
    out = []
    for z in sorted(zs, key=lambda z: (z.real, z.imag)):
        if not out or min(abs(z - u) for u in out[-8:]) > TOL_DUPLICATA: out.append(z)
    return out


def zeros_na_caixa(F, caixa, prazo, prof=0):
    """Busca completa: conta os zeros da caixa; 1 zero -> Newton do centro; 2 a 8 -> Beyn + Newton; senao divide em 4."""
    if time.time() >= prazo: return []
    k = conta_zeros(F, caixa); xa, xb, ya, yb = caixa
    if k == 0: return []
    if k == 1:
        z = polir(F, 0.5 * (xa + xb) + 0.5j * (ya + yb), caixa)
        if z is not None: return [z]
    elif k is not None and k <= BEYN_MAX:
        cand = beyn(F, caixa, k)
        if cand is not None and len(cand) == k and all(dentro(c, *caixa) for c in cand):
            pol = [polir(F, c, caixa) for c in cand]
            if all(p is not None for p in pol) and len(sem_duplicatas(pol)) == k: return sem_duplicatas(pol)
    if prof >= PROF_MAX: return []
    xm, ym = 0.5 * (xa + xb), 0.5 * (ya + yb); achados = []
    for sub in ((xa, xm, ya, ym), (xm, xb, ya, ym), (xa, xm, ym, yb), (xm, xb, ym, yb)):
        achados += zeros_na_caixa(F, sub, prazo, prof + 1)
    return sem_duplicatas(achados)


def no_eixo(z):
    """Zero com 0 <= Im z < GAMMA_RAD |z|: nao existe polo no semiplano de cima (sistema passivo), entao e' um polo colado
    no eixo cujo gamma o Newton nao resolve (o sinal de Im e' ruido). Fica com Im negativo minimo e o polo_estreito refaz."""
    return complex(z.real, -1e-300) if 0 <= z.imag < GAMMA_RAD * abs(z) else z


def zeros_rapido(l, F, esp, caixa, prazo, mat):
    """Primeira tentativa, barata: sementes nos minimos de |x^2 D_l| no eixo real, numa malha de passo ~ esp/200 (no minimo
    2e5 pontos), avaliada com Bessel de argumento real (|D_l| = sqrt(D_j^2 + D_y^2)), e Newton em complexo a partir de cada
    uma com Im inicial = -1e-12, -1e-9, -1e-6, -1e-3."""
    xr = np.linspace(X_MIN, X_MAX, int(min(2e6, max(2e5, 200 / esp))))
    Dj, Dy = D_partes(l, xr, mat); a = xr**P_POLO * np.hypot(Dj, Dy); a = np.where(np.isfinite(a), a, np.inf)
    # minimo de verdade: abaixo dos dois vizinhos por mais que o arredondamento (|D_l| constante, como em m = rho = 1,
    # gera minimos de ruido em todo ponto e cada um viraria um Newton perdido)
    c = a[1:-1] * (1 + 1e-13); sem = xr[np.flatnonzero((c < a[:-2]) & (c <= a[2:])) + 1]; zs = []
    for s0 in sem:
        if time.time() >= prazo: return None
        for y0 in (-1e-12, -1e-9, -1e-6, -1e-3):
            z = polir(F, complex(s0, y0), caixa)
            if z is not None: zs.append(z); break
    return zs


def buscar_polos(mat, prazo, x, esp, lmax=LMAX):
    """Zeros de z^2 D_l para cada l no retangulo [x_min, x_max] x [y_min, +1e-3].
    0. fundo do retangulo: Im do argumento da Bessel interna e' m Im(z); comeca com |Im| <= 150, ou seja
       y_min = -min(0.5, 150/max(m, 1)), e sobe pela metade enquanto D_l zerar ou estourar no contorno;
    1. Newton a partir dos minimos de |D_l| no eixo real (rapido, acha quase tudo);
    2. auditoria pelo principio do argumento no retangulo inteiro;
    3. se a contagem nao fechar, busca completa: faixas verticais com borda nos maximos de |D_l| no eixo, cada faixa
       subdividida ate ter 1 zero (Newton) ou ate 8 (Beyn). Junta com o passo 1 e audita de novo.
    No fim, todo polo com gamma/x < 1e-10 tem o gamma refeito pelo eixo real (polo_estreito)."""
    y0 = -min(-IMAG_MIN, 150.0 / max(mat[0], 1.0))
    polos, zeros_por_l, info = [], {}, {"imag_min": {}}
    xr = x[x <= X_MAX]
    for l in range(lmax + 1):
        if time.time() >= prazo: zeros_por_l[l] = None; info[l] = "sem_tempo"; continue
        F = lambda z, l=l: D_l(l, z, mat)
        ymin = y0
        while ymin < -1e-4:
            v = F(contorno(X_MIN, X_MAX, ymin, IMAG_MAX, 4096))
            if np.all(np.isfinite(v)) and np.all(v != 0): break
            ymin *= 0.5
        caixa = (X_MIN, X_MAX, ymin, IMAG_MAX); info["imag_min"][l] = ymin
        filtra = lambda zs: [z for z in sem_duplicatas([no_eixo(z) for z in zs]) if ymin <= z.imag < 0 and X_MIN <= z.real <= X_MAX]
        zs = zeros_rapido(l, F, esp, caixa, prazo, mat)
        if zs is None: zeros_por_l[l] = None; info[l] = "sem_tempo"; continue
        zs = filtra(zs); esperado, ok = auditoria(F, caixa, len(zs)); via = "rapida"
        if not ok:
            a = np.abs(F(xr.astype(complex))); topo = np.flatnonzero((a[1:-1] > a[:-2]) & (a[1:-1] > a[2:])) + 1
            bordas = np.unique(np.r_[X_MIN, xr[topo], X_MAX]); mais = []
            for xa, xb in zip(bordas[:-1], bordas[1:]):
                mais += zeros_na_caixa(F, (xa, xb, ymin, IMAG_MAX), prazo)
                if time.time() >= prazo: break
            if time.time() >= prazo: zeros_por_l[l] = None; info[l] = "sem_tempo"; continue
            zs = filtra(zs + mais); esperado, ok = auditoria(F, caixa, len(zs)); via = "completa"
        zs = [polo_estreito(l, z, mat) for z in zs]
        info[l] = "ok" if ok else f"auditoria_falhou ({esperado} esperados, {len(zs)} achados)"
        zeros_por_l[l] = zs if ok else None       # se a contagem nao fechou, esse l fica com a malha
        polos += [{"l": l, "pole_real": z.real, "pole_imag": z.imag, "Gamma_pole": 2 * abs(z.imag), "Q_pole": z.real / (2 * abs(z.imag)),
                   "argument_count_l": esperado, "argument_audit_ok": ok, "busca": via, "energy_peak_found": False, "xM_energy_peak": np.nan} for z in zs]
    return polos, zeros_por_l, info


# ---------------------------------------------------------------------------
# Uma curva
# ---------------------------------------------------------------------------
SEM_POLO = dict(primary_l=None, primary_pole_real=np.nan, primary_pole_imag=np.nan, primary_Gamma_pole=np.nan, primary_Q_pole=np.nan,
                associated_l="", N_associated_Dzeros=0, D0_confirmado=False, xM_D0=np.nan)

def associa_polos(res, polos, esp):
    """Pendura em cada pico os polos que caem dentro dele (o mais proximo vira o 'primary'). Compatibilidade com o formato
    antigo: D0_confirmado = algum polo de l = 0 associado ao pico; xM_D0 = Re desse polo (antes era o minimo de |D_0|^2)."""
    if not polos:
        for r in res: r.update(SEM_POLO)
        return
    P = np.array([p["pole_real"] for p in polos])
    for r in res:
        tol = max(r["FWHM"] if np.isfinite(r["FWHM"]) else 0, 0.25 * esp)
        perto = [polos[j] for j in np.flatnonzero(np.abs(P - r["xM_peak"]) <= tol)]
        if not perto: r.update(SEM_POLO); continue
        p = min(perto, key=lambda q: abs(q["pole_real"] - r["xM_peak"])); l0 = [q for q in perto if q["l"] == 0]
        r.update(primary_l=p["l"], primary_pole_real=p["pole_real"], primary_pole_imag=p["pole_imag"], primary_Gamma_pole=p["Gamma_pole"], primary_Q_pole=p["Q_pole"],
                 associated_l=",".join(str(l) for l in sorted({q["l"] for q in perto})), N_associated_Dzeros=len(perto), D0_confirmado=bool(l0),
                 xM_D0=min(l0, key=lambda q: abs(q["pole_real"] - r["xM_peak"]))["pole_real"] if l0 else np.nan)
        for q in perto: q["energy_peak_found"] = True; q["xM_energy_peak"] = r["xM_peak"]


def medir_polos(polos, mat, x, w, esp):
    """Para cada polo: peso S, area pi S/gamma, altura S/gamma^2, fundo de W_T na janela [x_k - esp/2, x_k + esp/2]
    (mediana da malha base) e a fracao da energia integrada da janela que vem dele:
        fracao = area_k / (soma das areas dos polos na janela, com arctan truncado + fundo * largura da janela).
    A area de um polo dentro de [lo, hi] e' (area/pi) [arctan((hi - x_k)/gamma) - arctan((lo - x_k)/gamma)]."""
    if not polos: return
    for p in polos:
        z = complex(p["pole_real"], p["pole_imag"]); g = -z.imag; S = peso_polo(p["l"], z, mat)
        p.update(S_pole=S, area_pole=np.pi * S / g, WT_peak_pole=S / g**2)
    ordem = np.argsort([p["pole_real"] for p in polos]); X = np.array([polos[i]["pole_real"] for i in ordem])
    G = np.array([-polos[i]["pole_imag"] for i in ordem]); A = np.maximum(np.array([polos[i]["area_pole"] for i in ordem]), 0)
    for p in polos:
        xk = p["pole_real"]; lo, hi = max(xk - 0.5 * esp, X_MIN), min(xk + 0.5 * esp, X_MAX)
        k = (x >= lo) & (x <= hi) & np.isfinite(w); fundo = float(np.median(w[k])) if k.sum() >= 3 else float(np.interp(xk, x, w))
        i0, i1 = np.searchsorted(X, xk - 2 * esp), np.searchsorted(X, xk + 2 * esp)
        Ij = np.sum(A[i0:i1] * (np.arctan((hi - X[i0:i1]) / G[i0:i1]) - np.arctan((lo - X[i0:i1]) / G[i0:i1])) / np.pi)
        I_jan = Ij + max(fundo, 0) * (hi - lo)
        p.update(WT_background_pole=fundo, frac_area_local=float(p["area_pole"] / I_jan) if I_jan > 0 else np.nan,
                 prom_log10_pole=float(np.log10(1 + p["WT_peak_pole"] / fundo)) if fundo > 0 and p["WT_peak_pole"] > 0 else np.nan,
                 borda_x_min=bool(xk - 2 * (-p["pole_imag"]) < X_MIN))


def ressonancia_de_polo(p):
    """Ressonancia lida direto do polo: posicao x_k, FWHM = 2 gamma, Q = x_k/(2 gamma), altura e area analiticas."""
    x, g, fundo = p["pole_real"], -p["pole_imag"], p["WT_background_pole"]; topo = p["WT_peak_pole"] + fundo
    return {"xM_peak": x, "WT_peak": topo, "WT_background": fundo, "energy_ratio": topo / fundo if fundo > 0 else np.inf,
            "energy_log10_prominence": p["prom_log10_pole"], "FWHM": 2 * g, "Q": x / (2 * g), "origem": "polo",
            "area_energy": p["area_pole"], "frac_area_local": p["frac_area_local"], "primary_l": p["l"], "primary_pole_real": x,
            "primary_pole_imag": p["pole_imag"], "primary_Gamma_pole": 2 * g, "primary_Q_pole": x / (2 * g), "associated_l": str(p["l"]),
            "N_associated_Dzeros": 1, "D0_confirmado": p["l"] == 0, "xM_D0": x if p["l"] == 0 else np.nan, "borda_x_min": p["borda_x_min"]}


K_VIS = 10**PROM_WT - 1     # pico visivel: W_T pelo menos 10^PROM_WT acima do fundo

def passo_necessario(p):
    """Maior passo de malha uniforme que garante ver o pico desse polo. No pior caso o ponto mais perto do topo fica a dx/2
    dele, e ainda assim W_T tem que ficar 10^PROM_WT acima do fundo:
        S / (gamma^2 + (dx/2)^2) >= k fundo   ->   dx_nec = 2 sqrt(S/(k fundo) - gamma^2),   k = 10^PROM_WT - 1."""
    S, fu, g = p.get("S_pole", np.nan), p.get("WT_background_pole", np.nan), abs(p["pole_imag"])
    if not (np.isfinite(S) and np.isfinite(fu) and S > 0 and fu > 0): return np.nan
    return float(2 * np.sqrt(max(S / (K_VIS * fu) - g * g, 0.0)))


def malha_dx(x, alvos, esp, laurent=None, x_max=X_MAX):
    """Malha base + um trecho uniforme em volta de cada grupo de picos (vizinhos a menos de esp/2). O passo do trecho e' o
    do pico mais exigente do grupo (dx_nec) ou, se a expansao de Laurent disser que dois picos do grupo estao mais perto
    que isso, metade da separacao deles. Os pontos comecam na borda do trecho, nao em cima do pico. Se nao couber em
    N_MAX_PONTOS, sobe um piso de passo (bissecao em log) ate caber; os picos com dx_nec abaixo do piso ficam sem resolver.
    alvos: lista de (posicao, dx_alvo, objeto ou None). Devolve a malha e o piso, e grava 'dx_malha_usado' em cada objeto."""
    A = sorted([a for a in alvos if np.isfinite(a[1]) and a[1] > 0], key=lambda a: a[0])
    if not A: return x, np.nan
    P = np.array([a[0] for a in A]); Dn = np.array([a[1] for a in A]); ext = min(esp, 0.05 * x_max); seg = []
    for gi in np.split(np.arange(len(P)), np.flatnonzero(np.diff(P) > 0.5 * esp) + 1):
        lo, hi = max(P[gi[0]] - 0.25 * esp, X_MIN), min(P[gi[-1]] + 0.25 * esp, x_max + ext); d = float(Dn[gi].min())
        if laurent is not None and len(laurent[0]):
            k = (laurent[0] >= lo) & (laurent[0] <= hi) & np.isfinite(laurent[1]) & (laurent[1] > 0)
            if k.any(): d = min(d, 0.5 * float(laurent[1][k].min()))
        seg.append((lo, hi, d, gi))
    livre = max(N_MAX_PONTOS - len(x), 1000)
    custo = lambda piso: sum((hi - lo) / max(d, piso) for lo, hi, d, _ in seg)
    piso = DX_MIN
    if custo(piso) > livre:
        a, b = DX_MIN, max(hi - lo for lo, hi, _, _ in seg)
        for _ in range(60):
            mm = np.sqrt(a * b); a, b = (mm, b) if custo(mm) > livre else (a, mm)
        piso = b
    extras = []
    for lo, hi, d, gi in seg:
        h = max(d, piso); extras.append(lo + h * np.arange(int((hi - lo) / h) + 1))
        for i in gi:
            if A[i][2] is not None: A[i][2]["dx_malha_usado"] = h
    return np.unique(np.concatenate([x] + extras)), float(piso)


def conta_polos(cand, frac_area):
    """Polos que contam como ressonancia: Q >= 2 e fracao de area >= frac_area. Polos sobrepostos (|x1 - x2| < gamma1 + gamma2)
    contam como modos separados em N_res_polos_area e como um pico so em N_res_polos_picos."""
    conta = [p for p in sorted(cand, key=lambda q: q["pole_real"]) if np.isfinite(p.get("frac_area_local", np.nan)) and p["frac_area_local"] >= frac_area and p["Q_pole"] >= Q_MIN_POLO]
    ids = {id(p) for p in conta}
    for p in cand: p["contado_area"] = id(p) in ids; p["sobreposto"] = False
    Xc, Gc = np.array([p["pole_real"] for p in conta]), np.array([-p["pole_imag"] for p in conta])
    junta = np.diff(Xc) < Gc[1:] + Gc[:-1] if len(conta) > 1 else np.array([], bool)
    for i, p in enumerate(conta): p["sobreposto"] = bool((i > 0 and junta[i - 1]) or (i < len(junta) and junta[i]))
    return conta, len(conta), int(len(conta) - junta.sum())


def processa_curva(m, rho, prazo_polos, polos_normais=True, forcar_polos=False, frac_area=FRAC_AREA):
    t0 = time.time(); mat = (float(m), float(rho))
    x, esp = malha_base(mat); WT = lambda v: energia_total(v, mat)
    risco = mat[0] > 1 or forcar_polos          # forcar_polos: so pra teste/calibracao em curvas normais

    # malha base: sempre roda, e nas curvas normais ja e' a resposta
    w = WT(x); _, xs_base = maximos(WT, x, esp, prom=PROM_WT, w=w)
    saida = {"risco": bool(risco), "esp": esp, "candidatos": [], "busca_polos": "nao_feita", "N_res_malha_base": len(xs_base),
             "N_res_malha_refinada": None, "N_modos": None, "pontos_malha": len(x), "dx_min": None}

    # polos: obrigatorios nas curvas de risco, metadado nas normais
    zeros_por_l = {l: None for l in range(LMAX + 1)}
    if risco or polos_normais:
        polos, zeros_por_l, info = buscar_polos(mat, time.time() + prazo_polos, x, esp)
        imag_min = min(info.pop("imag_min").values(), default=np.nan)
        saida.update(candidatos=polos, auditoria=info, imag_min_busca=imag_min, busca_polos="incompleta" if "sem_tempo" in info.values() else "completa")
        medir_polos(polos, mat, x, w, esp)
        for p in polos: p["dx_nec"] = passo_necessario(p)
    cand = saida["candidatos"]; bons = [p for p in cand if p["argument_audit_ok"]]
    vis = [p for p in bons if p["Q_pole"] >= Q_MIN_POLO and np.isfinite(p.get("prom_log10_pole", np.nan)) and p["prom_log10_pole"] >= PROM_WT]
    dn = np.array([p["dx_nec"] for p in vis if np.isfinite(p["dx_nec"])])
    completa = saida["busca_polos"] == "completa" and all(v == "ok" for v in saida.get("auditoria", {}).values())
    saida.update(N_polos_visiveis=len(vis) if cand or saida["busca_polos"] == "completa" else None, dx_necessario=float(dn.min()) if len(dn) else None,
                 dx_90=float(np.percentile(dn, 10)) if len(dn) else None, N_polos_estreitos=sum(-p["pole_imag"] < GAMMA_RAD * p["pole_real"] for p in cand),
                 N_polos_borda_x_min=sum(p.get("borda_x_min", False) for p in cand))
    if completa:
        conta, n_area, n_picos = conta_polos(cand, frac_area)
        saida.update(N_res_polos_area=n_area, N_res_polos_picos=n_picos, area_polos_total=float(sum(p["area_pole"] for p in cand)))

    if not risco:
        dxl = np.interp(xs_base, x[1:], np.diff(x)) if len(xs_base) else None
        res = medir_picos(WT, x, xs_base, esp, escala=dxl)
        for r in res: r["origem"] = "malha_base"
        associa_polos(res, cand, esp)
        for p in vis: p["dx_malha_usado"] = float(np.interp(p["pole_real"], x[1:], np.diff(x)))      # passo local da malha base
        if saida["N_polos_visiveis"] is not None:
            saida["N_previsto_malha"] = int(sum(p["dx_malha_usado"] <= p["dx_nec"] for p in vis if np.isfinite(p["dx_nec"])))
        saida.update(modo="malha_base", ressonancias=res, ressonancias_malha=res, runtime_s=time.time() - t0)
        return saida

    # curva de risco: modos por l, separacao de Laurent, malha refinada
    modos = modos_por_l(mat, zeros_por_l, x, esp)
    laurent = separacao_laurent(bons, mat) if len(bons) >= 2 else None
    if laurent is not None:
        d_polos = np.diff(np.sort([p["pole_real"] for p in bons]))
        saida.update(N_pares_fundidos_laurent=int(np.sum(np.isnan(laurent[1]))), dx_min_polos=float(0.5 * d_polos.min()),
                     dx_min_laurent=float(0.5 * np.nanmin(laurent[1])) if np.isfinite(laurent[1]).any() else None)
    alvos = [(p["pole_real"], p["dx_nec"], p) for p in vis]
    for mo in modos:                                   # l sem polos (auditoria falhou): modos da malha, passo = metade da distancia
        if mo.get("N_zeros") is None or mo["regra"] in ("proeminencia", "sem_polos"):
            P = np.sort(np.asarray(mo["x"], float)); viz = np.minimum(np.r_[np.inf, np.diff(P)], np.r_[np.diff(P), np.inf]) if len(P) else []
            alvos += [(v, 0.5 * min(d, 0.25 * esp), None) for v, d in zip(P, viz)]
    xr, piso = malha_dx(x, alvos, esp, laurent); coube = not (np.isfinite(piso) and piso > DX_MIN); dx_min = float(np.min(np.diff(xr)))
    saida.update(dx_piso_malha=piso if np.isfinite(piso) else None,
                 N_previsto_malha=int(sum(p.get("dx_malha_usado", np.inf) <= p["dx_nec"] for p in vis if np.isfinite(p["dx_nec"]))))
    wr = WT(xr); _, xs = maximos(WT, xr, esp, prom=PROM_WT if coube else None, w=wr)

    # escala de largura pra medir cada pico: Gamma do polo mais proximo, se tiver; senao o passo local da malha
    escala = np.interp(xs, xr[1:], np.diff(xr)) if len(xs) else np.array([])
    if cand and len(xs):
        o = np.argsort([p["pole_real"] for p in cand]); P = np.array([cand[i]["pole_real"] for i in o]); G = np.array([cand[i]["Gamma_pole"] for i in o])
        k = np.clip(np.searchsorted(P, xs), 0, len(P) - 1); k0 = np.maximum(k - 1, 0)
        j = np.where(np.abs(P[k0] - xs) < np.abs(P[k] - xs), k0, k)          # polo mais proximo de cada pico
        perto = np.abs(P[j] - xs) <= 0.25 * esp; escala = np.where(perto, G[j], escala)
    res = medir_picos(WT, xr, xs, esp, escala=escala)
    for r in res: r["origem"] = "malha_refinada" if coube else "malha_limite"
    associa_polos(res, cand, esp)
    modo = "malha_refinada" if coube else "malha_limite"; saida["ressonancias_malha"] = list(res)

    # contagem final pelos polos (nao depende de malha), quando a busca fechou em todos os l
    if completa:
        for p in conta: p["energy_peak_found"] = True; p["xM_energy_peak"] = p["pole_real"]
        res, modo = [ressonancia_de_polo(p) for p in conta], "polos_area"
    saida.update(modo=modo, ressonancias=res, N_res_malha_refinada=len(xs), N_modos=sum(len(mo["x"]) for mo in modos),
                 regras_l={mo["l"]: mo["regra"] for mo in modos}, pontos_malha=len(xr), dx_min=dx_min, runtime_s=time.time() - t0)
    return saida


class Estourou(Exception): pass

def _alarme(*_): raise Estourou()


def trabalho(args):
    """Roda uma curva dentro de um limite duro de tempo (SIGALRM). Nunca derruba o processo."""
    chave, m, rho, prazo_polos, limite, frac, polos_normais = args
    signal.signal(signal.SIGALRM, _alarme); signal.setitimer(signal.ITIMER_REAL, limite)
    try:
        r = processa_curva(m, rho, prazo_polos, polos_normais=polos_normais, frac_area=frac); r["status"] = "ok"
    except Estourou:
        r = {"status": "timeout"}
    except Exception as e:
        r = {"status": "erro", "erro": f"{type(e).__name__}: {e}"}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    r.update(m=m, rho_ratio=rho)
    return chave, r


# ---------------------------------------------------------------------------
# Checkpoint e CSV
# ---------------------------------------------------------------------------
NOME = f"esfera_fluida_lmax{LMAX}"

def caminhos(raiz):
    pasta = os.path.join(os.path.abspath(raiz), NOME); os.makedirs(pasta, exist_ok=True)
    arq = lambda pre, ext="csv": os.path.join(pasta, f"{pre}_{NOME}.{ext}")
    return {"ckpt": arq("checkpoint", "npz"), "resumo": arq("resumo"), "res": arq("ressonancias"), "cand": arq("candidatos"),
            "res_malha": arq("ressonancias_malha"), "param": arq("parametros", "json"), "leiame": os.path.join(pasta, "LEIAME.txt")}


CAMPOS_ANTIGOS = ("xM_peak", "FWHM", "Q", "WT_peak", "origem", "D0_confirmado", "xM_D0")

def picos_formato_antigo(c):
    """Lista de picos de uma curva no formato antigo (picos_por_curva), para quem le o checkpoint do jeito antigo."""
    return [{**{k: r.get(k) for k in CAMPOS_ANTIGOS}, "modo_deteccao": c.get("modo", ""), "lmax_usado": LMAX} for r in c.get("ressonancias", [])]


def salva(cam, curvas, m_grid, rho_grid):
    """Checkpoint novo ('curves', tudo) mais as chaves do formato antigo (picos_por_curva, done_flat, m_grid, rho_grid),
    pra que os scripts antigos de leitura e revisao continuem funcionando."""
    done = np.zeros(len(m_grid), bool)
    for k, c in curvas.items(): done[int(k)] = c["status"] == "ok"
    antigo = {k: {"m": c["m"], "rho_ratio": c["rho_ratio"], "picos": picos_formato_antigo(c)} for k, c in curvas.items() if c["status"] == "ok"}
    tmp = cam["ckpt"] + f".tmp_{uuid.uuid4().hex}.npz"
    np.savez_compressed(tmp, versao=VERSAO, lmax=LMAX, curves=np.array(curvas, dtype=object), m_grid=m_grid, rho_grid=rho_grid,
                        checkpoint_version=np.array(VERSAO), picos_por_curva=np.array(antigo, dtype=object), done_flat=done,
                        pipeline_kind=np.array("polos_v3", dtype=object))
    os.replace(tmp, cam["ckpt"])


def carrega(cam):
    if not os.path.exists(cam["ckpt"]): return {}
    with np.load(cam["ckpt"], allow_pickle=True) as d:
        if "versao" not in d.files or int(d["versao"]) != VERSAO:
            print("Checkpoint de outra versao, comecando do zero."); return {}
        return d["curves"].item()


LEIAME = """Gerador da esfera fluida simples, versao {versao} (metodologia do gerador core-shell fluido, versao 8). Arquivos:

checkpoint_*.npz          tudo o que foi calculado, curva a curva (dict 'curves', chave = indice da curva na grade);
                          tambem traz as chaves do formato antigo (picos_por_curva, done_flat, m_grid, rho_grid)
parametros_*.json         constantes e opcoes de linha de comando usadas nesta rodada
resumo_*.csv              uma linha por curva (substitui o antigo *_summary.csv; as colunas antigas continuam)
ressonancias_*.csv        uma linha por ressonancia da contagem final (substitui o antigo *_peaks.csv; colunas antigas continuam)
ressonancias_malha_*.csv  uma linha por pico achado na malha (base nas curvas normais, refinada nas de risco)
candidatos_*.csv          uma linha por polo (zero de D_l) achado na busca

Grade: m = c_M/c_S e rho_ratio = rho_S/rho_M, os mesmos valores do codigo antigo ({nm} x {nr} = {nt} curvas).
'global' = indice da curva + 1, como antes. Nao ha mais regimes velocity/density: a grade (m, rho_ratio) ja e' o espaco
de parametros inteiro.

Fisica: b_l = i (rho_M/rho_S)/(x^2 D_l), D_l = j_l(m x) h_l'(x) - (m/rho_ratio) j_l'(m x) h_l(x);
W_T = soma (2l+1) |b_l|^2 [2 F_l + B_l], com F_l = (1/2) x_S^3 (j_l^2 - j_l-1 j_l+1) e B_l = x_S^2 j_l j_l' em x_S = m x.
Uma regiao so: nao ha normalizacao por regiao (W_T = W~_S, como no codigo antigo).
CORRECAO: o codigo antigo usava j_-1(x) = +y_0(x); o correto e' j_-1(x) = cos(x)/x = -y_0(x). Isso mudava F_0, ou seja a
energia do termo l = 0 (altura e fundo dos picos de monopolo). Aqui o sinal e' o correto.
Espacamento medio entre ressonancias de um mesmo l (tese): Delta_W = pi/Phi, Phi = c_M/c_S = m; no codigo e' 'esp'.

Como cada curva e' contada
  m <= 1 (normal): maximos de W_T na malha base (5000 pontos) com proeminencia >= 0,1 em log10. A busca de polos tambem
    roda e so preenche os metadados (primary_l, D0_confirmado, candidatos, N_polos_visiveis, N_previsto_malha,
    N_res_polos_area), sem mudar a contagem.
  m > 1 (risco):
    1. zeros de z^2 D_l para l = 0..10 (Newton + principio do argumento; busca completa se nao fechar); polos com
       gamma/x < 1e-10 tem gamma refeito pelo eixo real (D_l = D_j + i D_y);
    2. malha refinada: cada grupo de picos ganha passo dx_nec = 2 sqrt(S/(k fundo) - gamma^2), k = 10^0,1 - 1, ou metade
       da separacao de Laurent se for menor; teto de {nmax} pontos (acima disso sobe um piso de passo);
    3. se a busca fechou em todos os l: ressonancias = polos com Q >= {qmin} e area >= {frac} da energia da janela pi/Phi.
       Senao: picos da malha refinada.

resumo
  global, m, rho_ratio    curva (como antes)
  N_peaks                 contagem final (a que vai pro mapa; mesmo significado de antes, agora pelo metodo novo)
  x_first, Q_max          posicao da primeira ressonancia e maior Q da contagem final (como antes)
  modo_contagem           malha_base | malha_refinada | malha_limite | polos_area
  curva_risco             m > 1
  N_res_malha_base        maximos de W_T na malha base
  N_res_malha_refinada    maximos de W_T na malha refinada (passo dx_nec; so curvas de risco)
  N_res_polos_area        polos contados (um por modo); N_res_polos_picos: idem juntando polos sobrepostos
  N_polos_visiveis        polos com Q >= 2 e proeminencia analitica >= 0,1 (o que uma malha infinitamente fina veria)
  N_previsto_malha        quantos desses o passo usado deveria ver (passo <= dx_nec; nas curvas normais, passo local da malha base)
  frac_malha_refinada     N_res_malha_refinada / N_polos_visiveis
  dx_necessario, dx_90    passo para ver todos / 90% dos polos visiveis
  dx_piso_malha           piso de passo usado na malha refinada (> 1e-9 quando nao coube)
  N_Dzeros_total          polos achados; N_polos_estreitos: polos com gamma/x < 1e-10
  N_polos_borda_x_min     polos com x_k - 2 gamma < x_min (meio pico fica abaixo de x_min: a malha nao enxerga, o polo sim)
  busca_polos, audit_all_ok  se a busca de polos fechou em todos os l
  imag_min_busca          fundo do retangulo da busca; runtime_s, status, error

ressonancias e ressonancias_malha
  peak_number, xM_peak, FWHM, Q, WT_peak  como antes. Origem 'polo': FWHM = 2 gamma, Q = x/(2 gamma), WT_peak = S/gamma^2 + fundo
  origem                  malha_base | malha_refinada | malha_limite | polo (antes: malha_global, zoom_D0, derivada_periodo)
  D0_confirmado, xM_D0    pico associado a um polo de l = 0 / Re desse polo (antes: minimo de |D_0|^2 confirmado)
  modo_deteccao           agora traz o modo de contagem da curva (os modos velocity/density nao existem mais)
  WT_background, energy_ratio, energy_log10_prominence   fundo, razao pico/fundo e log10 dela
  area_energy             pi S/gamma (so origem 'polo'); frac_area_local: area / energia integrada na janela pi/Phi
  primary_l, primary_pole_real, primary_pole_imag, primary_Gamma_pole, primary_Q_pole, associated_l, N_associated_Dzeros

candidatos
  l, pole_real, pole_imag (z = x - i gamma), Gamma_pole = 2 gamma, Q_pole = x/(2 gamma)
  argument_count_l, argument_audit_ok, busca (rapida | completa)
  S_pole                  peso: W_l(x) ~ S/((x - x_k)^2 + gamma^2) perto do polo
  area_pole, WT_peak_pole, WT_background_pole, frac_area_local, prom_log10_pole
  contado_area, sobreposto, dx_nec, dx_malha_usado, borda_x_min
"""


def exporta(cam, curvas, m_grid, rho_grid, args=None):
    resumo, res, cand, res_malha = [], [], [], []
    for chave in sorted(curvas, key=int):
        c = curvas[chave]; i = int(chave); ok = c["status"] == "ok"
        base = {"global": i + 1, "m": float(m_grid[i]), "rho_ratio": float(rho_grid[i])}
        rr, cc = c.get("ressonancias", []), c.get("candidatos", [])
        Q = [r["Q"] for r in rr if np.isfinite(r["Q"])]; razoes = [r["energy_ratio"] for r in rr if np.isfinite(r["energy_ratio"])]
        aud = c.get("auditoria", {}); nv = c.get("N_polos_visiveis")
        resumo.append({**base, "N_peaks": len(rr) if ok else np.nan, "x_first": rr[0]["xM_peak"] if rr else np.nan, "Q_max": max(Q) if Q else np.nan,
                       "status": c["status"], "lmax": LMAX, "curva_risco": c.get("risco"), "modo_contagem": c.get("modo"), "busca_polos": c.get("busca_polos"),
                       "audit_all_ok": all(v == "ok" for v in aud.values()) if aud else np.nan,
                       "N_res_malha_base": c.get("N_res_malha_base"), "N_res_malha_refinada": c.get("N_res_malha_refinada"),
                       "N_res_polos_area": c.get("N_res_polos_area"), "N_res_polos_picos": c.get("N_res_polos_picos"),
                       "N_polos_visiveis": nv, "N_previsto_malha": c.get("N_previsto_malha"),
                       "frac_malha_refinada": c["N_res_malha_refinada"] / nv if nv and c.get("N_res_malha_refinada") is not None else None,
                       "dx_necessario": c.get("dx_necessario"), "dx_90": c.get("dx_90"), "dx_piso_malha": c.get("dx_piso_malha"),
                       "N_Dzeros_total": len(cc) if ok else np.nan, "N_Dzeros_com_maximo_energia": sum(p["energy_peak_found"] for p in cc) if ok else np.nan,
                       "N_l_res": len({r["primary_l"] for r in rr if r.get("primary_l") is not None}) if ok else np.nan,
                       "energy_ratio_max": max(razoes) if razoes else np.nan, "area_polos_total": c.get("area_polos_total"),
                       "N_polos_estreitos": c.get("N_polos_estreitos"), "N_polos_borda_x_min": c.get("N_polos_borda_x_min"),
                       "N_modos_por_l": c.get("N_modos"), "pontos_malha": c.get("pontos_malha"), "dx_min_refino": c.get("dx_min"),
                       "dx_min_polos": c.get("dx_min_polos"), "dx_min_laurent": c.get("dx_min_laurent"), "N_pares_fundidos_laurent": c.get("N_pares_fundidos_laurent"),
                       "imag_min_busca": c.get("imag_min_busca"), "runtime_s": c.get("runtime_s"), "error": c.get("erro", "")})
        extra = {"modo_deteccao": c.get("modo", ""), "lmax_usado": LMAX}
        res += [{**base, "peak_number": k + 1, **{f: r.get(f) for f in CAMPOS_ANTIGOS}, **extra, **{f: v for f, v in r.items() if f not in CAMPOS_ANTIGOS}} for k, r in enumerate(rr)]
        res_malha += [{**base, "peak_number": k + 1, **{f: r.get(f) for f in CAMPOS_ANTIGOS}, **extra, **{f: v for f, v in r.items() if f not in CAMPOS_ANTIGOS}} for k, r in enumerate(c.get("ressonancias_malha", []))]
        cand += [{**base, "candidate_num": k + 1, **p} for k, p in enumerate(cc)]
    for nome, linhas in (("resumo", resumo), ("res", res), ("res_malha", res_malha), ("cand", cand)): pd.DataFrame(linhas).to_csv(cam[nome], index=False)
    with open(cam["leiame"], "w", encoding="utf-8") as fh:
        fh.write(LEIAME.format(versao=VERSAO, nmax=N_MAX_PONTOS, qmin=Q_MIN_POLO, frac=getattr(args, "frac_area", FRAC_AREA),
                               nm=len(np.unique(m_grid)), nr=len(np.unique(rho_grid)), nt=len(m_grid)))
    const = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str, tuple, np.ndarray)) and k != "LEIAME"}
    with open(cam["param"], "w", encoding="utf-8") as fh:
        json.dump({"versao": VERSAO, "argumentos": vars(args) if args is not None else None, "constantes": const,
                   "curvas_ok": sum(c["status"] == "ok" for c in curvas.values()), "curvas_total": len(m_grid)}, fh, indent=1, default=str)
    print(f"CSVs salvos em {os.path.dirname(cam['resumo'])}")


# ---------------------------------------------------------------------------
# Disparo
# ---------------------------------------------------------------------------
def roda(args):
    m_grid, rho_grid = grade(); cam = caminhos(args.saida); curvas = carrega(cam)
    idx = args.indices if args.indices else range(len(m_grid))
    pend = [(str(i), float(m_grid[i]), float(rho_grid[i]), args.prazo_polos, args.limite_curva, args.frac_area, not args.sem_polos_normais)
            for i in idx if str(i) not in curvas or (args.refazer_falhas and curvas[str(i)]["status"] != "ok")]
    pend.sort(key=lambda t: -t[1])               # m grande primeiro: as curvas mais demoradas nao ficam pro fim
    if args.max_curvas: pend = pend[: args.max_curvas]
    prontas = sum(c["status"] == "ok" for c in curvas.values())
    print(f"{prontas} de {len(m_grid)} curvas ja feitas, {len(pend)} pra rodar agora com {args.workers} workers")
    t0, feitas = time.time(), 0
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("fork")) as ex:
            for fut in as_completed([ex.submit(trabalho, t) for t in pend]):
                chave, r = fut.result(); curvas[chave] = r; feitas += 1
                resto = (time.time() - t0) / feitas * (len(pend) - feitas) / 3600
                print(f"[{feitas}/{len(pend)}] {chave} m={r['m']:.4g} rho={r['rho_ratio']:.4g} {r['status']} {r.get('modo', '')} "
                      f"N={len(r.get('ressonancias', [])) if r['status'] == 'ok' else 'NA'} (base {r.get('N_res_malha_base', '-')}) polos={len(r.get('candidatos', []))} "
                      f"t={r.get('runtime_s', 0):.1f}s falta~{resto:.1f}h", flush=True)
                if feitas % args.salvar_a_cada == 0: salva(cam, curvas, m_grid, rho_grid)
    except KeyboardInterrupt:
        print("Interrompido, salvando o que ja foi feito...")
    salva(cam, curvas, m_grid, rho_grid); exporta(cam, curvas, m_grid, rho_grid, args)


def em_notebook(): return "ipykernel" in sys.modules


def monta_drive(pasta):
    """No Colab, monta o Google Drive se a pasta de saida estiver nele."""
    if not pasta.startswith("/content/drive"): return
    try:
        from google.colab import drive
        if not os.path.ismount("/content/drive"): drive.mount("/content/drive")
    except ImportError:
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ressonancias da esfera fluida simples: malha base nas curvas normais, polos nas de risco (m > 1).")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--saida", default="./resultados_esfera")
    ap.add_argument("--prazo-polos", type=float, default=120.0, help="tempo maximo (s) da busca de polos por curva, somando todos os l")
    ap.add_argument("--limite-curva", type=float, default=600.0, help="limite duro (s) por curva")
    ap.add_argument("--salvar-a-cada", type=int, default=25)
    ap.add_argument("--frac-area", type=float, default=FRAC_AREA, help="fracao minima da energia da janela pra um polo contar")
    ap.add_argument("--sem-polos-normais", action="store_true", help="nao busca polos nas curvas normais (m <= 1)")
    ap.add_argument("--max-curvas", type=int, default=None, help="roda so as primeiras N pendentes (teste)")
    ap.add_argument("--indices", type=int, nargs="+", default=None, help="roda so essas curvas (indice na grade, global - 1)")
    ap.add_argument("--refazer-falhas", action="store_true", help="roda de novo as curvas com timeout ou erro")
    if em_notebook() and argv is None:          # no notebook o kernel passa '-f <arquivo>': ignora e usa COLAB
        ap.set_defaults(**COLAB); argv = []
    args = ap.parse_args(argv)
    monta_drive(args.saida); roda(args)


if __name__ == "__main__":
    main()
