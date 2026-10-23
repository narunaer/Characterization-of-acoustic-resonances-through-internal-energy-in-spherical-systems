#!/usr/bin/env python3
# Gerador de ressonancias da esfera solida imersa em agua: elastica (dataset8), Kelvin-Voigt e Maxwell.
# Mesma metodologia do gerador_coreshell.py versao 8; fisica do espalhamento igual a do elastico.py/voigt.py/maxwell.py
# (coeficientes a_l, b_l, denominador e energia de Clapeyron), com a perda escrita como na secao
# "Esfera Viscoelastica" da tese. Os tres modelos usam o mesmo codigo: o elastico e' o caso c_L, c_T reais e
# constantes (limite tau -> 0 do Voigt), entao malha, busca de polos, modelo analitico dos polos, recorrencia e
# contagem sao identicos nos tres.
#
#   Elastico:     c_L, c_T reais (colunas cL, cT do dataset8 antigo, repetidas em cL0, cT0); tau_adim = 0.
# Modelo de perda (texto da tese): so o cisalhamento dissipa, o modulo volumetrico K e' real.
#   Kelvin-Voigt: tan d_T(x) = tau x,  tan d_L(x) = 4/(3 eta0^2) tau x,  c_j(x) = c_j0 sqrt(1 - i tan d_j(x)).
#                 c_L0, c_T0 sao as velocidades do limite x -> 0 (modulo de armazenamento constante).
#   Maxwell:      tan d_T(x) = 1/(tau x),  c_T(x) = c_Tinf / sqrt(1 + i tan d_T),
#                 c_L(x) = c_Linf sqrt(1 - 4/(3 eta_inf^2) [1 - 1/(1 + i tan d_T)]).
#                 c_Linf, c_Tinf sao as velocidades do limite de alta frequencia.
#   tau e' o tempo de relaxacao adimensional tau~ = tau_T c_M / a da tese. As colunas cL0, cT0 e eta do dataset
#   guardam sempre as velocidades de referencia de cada modelo (c_j0 no Voigt, c_jinf no Maxwell), como nos mapas.
#
# Como a contagem funciona:
#
#   Toda curva comeca na malha base de 5000 pontos em x_M: 1000 log em [x_min, 0.1], 3000 lineares em
#   [0.1, 0.9] e 1000 lineares em [0.9, 1], mais min(Delta_W, 0.05) depois de x_max para ver pico na borda.
#
#   Espacamento entre ressonancias de um mesmo l (tese, Eq. deltaW_elastica): Delta_W = pi/Phi, Phi = fase acumulada
#   por unidade de x na onda mais lenta que propaga. Num solido dispersivo Phi depende de x: Phi(x) = max(d Re z_L/dx,
#   d Re z_T/dx) entre as ondas com Q da onda >= 1,5, z_j = x c_M/c_j(x). Sem perda, Phi = c_M/c_T. No codigo, 'esp'.
#
#   Curva normal (versao 2, secao 'Relacoes de recorrencia'): a condicao de ressonancia da tese,
#   f_l = kappa_l E_l + (rho_M/rho_S) z_L^2 G_l = 0 (kappa_l = -x h'/h), com as grandezas complexas do material e da
#   radiacao, nao tem zero com Q >= 1 em x <= X_PREV_MAX = 1,1. Os zeros saem de Newton complexo a partir dos minimos
#   de |f_l/f_l'| no eixo real (~1 s por curva) e sao os proprios polos de a_l (f_l = D_l/h_l a menos de fatores sem
#   zero). A resposta e' a malha base. Sem perda e sem radiacao f_l e' real e os zeros sao as raizes reais da tese.
#   O criterio da versao 1 (alguma onda que propaga e' mais lenta que o meio) fica so como coluna (risco_fase_v1): ele
#   marcava risco em solido com c_T < c_M cujo primeiro modo cai em x > 1 e perdia ressonancias que nao dependem de
#   onda lenta: o modo quase estatico de esfera leve ou com K pequeno (tipo Minnaert, x^2 ~ 3K/(rho_M c_M^2)) e o de
#   dilatacao uniforme da esfera Voigt muito viscosa (omega^2 ~ 15K/(rho_S a^2)), com Q de dezenas a centenas.
#
#   Curva de risco (as outras):
#     1. zeros de z^p D_l (D_l = denominador sem divisoes, ver campos) para l = 0..10: Newton a partir dos minimos
#        de |D_l| no eixo real e das raizes previstas pela recorrencia, auditoria pelo principio do argumento e busca
#        completa se nao fechar. Para l = 0 o fator espurio T_0(z_T) da tese sai de D_0 (b_0 = 0);
#     2. peso S de cada polo pelo residuo: W_l(x) ~ S/((x - x_k)^2 + gamma^2), area pi S/gamma;
#     3. proeminencia analitica: maximos do modelo fundo + soma das lorentzianas de todos os polos (sem malha). Num
#        solido os modos se sobrepoem muito mais que no core-shell, e a proeminencia de cada polo isolado sobre o
#        fundo da janela previa o dobro dos picos que a malha converge a ver; o modelo com todos os polos acerta;
#     4. malha refinada pelo passo que cada pico precisa (dx_nec), maximos de W_T nela: N_res_malha_refinada;
#     5. se a busca fechou em todos os l, contagem final pelos polos com Q >= 2 que sao visiveis no modelo analitico
#        (proeminencia >= 0,1) ou tem fracao de area >= FRAC_AREA = 3e-4 (no core-shell era so a area, 1e-3; ver FRAC_AREA).
#
#   Diferencas em relacao ao core-shell, por causa do solido com perda:
#     - as funcoes de Bessel internas entram escaladas por e^{-|Im z|} (j_l cresce como e^{|Im z|}/|z|). A escala
#       cancela exatamente em a_l j_l, |a_l|^2 G, etc., e e' um fator real positivo em D_l, entao nao muda nem os
#       zeros nem o principio do argumento. Com isso nada estoura, mesmo com |Im z_T| ~ 1e3 (o codigo antigo
#       caia em NaN a partir de |Im z| ~ 350 e jogava a curva para a malha grosseira);
#     - a regiao de busca e' o retangulo [x_min, x_max] x [-, 1e-3] com o fundo cortado na reta y = -x/3 (Q = 1,5).
#       Os polos que contam tem Q >= 2, gamma <= x/4, e ficam todos dentro com folga; o corte so tira os zeros super
#       amortecidos, que no solido com perda podem ser centenas e faziam a busca completa estourar o prazo. Com a
#       escala nao ha mais motivo para encolher a regiao como 150/k_max (isso esconderia polos com Q >= 2); o fundo
#       so sobe pela metade se D_l zerar ou estourar no contorno;
#     - no principio do argumento os valores de D_l no contorno sao multiplicados por e^{i theta}, theta = -soma_j
#       sgn(Im z_j) Re z_j (so onde |Im z_j| > 1): com perda alta j(z_j) ~ e^{-i z_j} e o argumento de D_l giraria como
#       Re z_j (~1e4 no Maxwell com tau pequeno) sem haver zero. theta e' continua e univoca no contorno, entao nao muda
#       o numero de voltas (ver D_contorno); sem isso essas curvas nao fechavam a auditoria nem com 32768 pontos;
#     - o peso S pelo residuo so vale se a onda interna quase nao muda de amplitude entre z_k e x_k; polo largo num
#       solido lento (gamma maior que ~1/Phi) nao forma lorentziana propria e fica sem S (ver peso_polo);
#     - se o numero estimado de zeros passar de N_ZEROS_MAX (solido com c_T de poucos m/s), a busca e' pulada e a
#       contagem fica com a malha refinada (busca_polos = 'pulada_densidade').
#
# Exemplos:
#   python -u gerador_viscoelastico.py --modelo voigt --n-curvas 8000 --workers 25
#   python -u gerador_viscoelastico.py --modelo elastico voigt maxwell --n-curvas 8000 --workers 25
#   python -u gerador_viscoelastico.py --modelo maxwell --n-curvas 40 --max-curvas 40   (teste rapido)
#   python -u gerador_viscoelastico.py --modelo voigt --recontar --frac-area 1e-3       (so refaz a contagem final)

import argparse, json, os, signal, time, uuid, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import scipy.linalg as la
from scipy.optimize import minimize_scalar, newton
from scipy.signal import find_peaks
from scipy.special import jv, yv, jve

warnings.filterwarnings("ignore")

VERSAO = 2                 # versao deste gerador (metodologia do core-shell v8 + criterio de risco pela recorrencia)
LMAX = 10
X_MIN, X_MAX = 1e-4, 1.0
IMAG_MAX = 1e-3          # topo da regiao de busca, acima do eixo para nao passar colado em polos estreitos
CM_PADRAO, RHOM_PADRAO = 1500.0, 1000.0
LIMITE_FISICO = np.sqrt(3) / 2   # c_T/c_L <= isso
XM_REF = 0.5               # so para preencher as colunas antigas tanL_ref/tanT_ref (tan d de cada canal em x = 0.5)
CT_MIN = 1e-3
FAIXAS = {"elastico": {"alpha": (1e-3, 1e3), "beta": (1 / 3000, 1000.0)},
          "voigt": {"alpha": (1e-3, 1e3), "beta": (1 / 3000, 1000.0), "tau": (1e-4, 1e3)},
          "maxwell": {"alpha": (1e-3, 1e4), "beta": (1 / 3000, 1000.0), "tau": (1e-4, 1e3)}}
# tau = tau~ da tese e' variavel nova, log-uniforme em [1e-4, 1e3] nos dois modelos (versao 2). As faixas nao se
# correspondem: Voigt com tau pequeno e' quase elastico, Maxwell com tau pequeno e' quase fluido (tan d_T = 1/(tau x)).

# malha e contagem
N_LOG, N_MEIO, N_FIM = 1000, 3000, 1000
PROM_WT = 0.1
PROM_WL = 0.75
Q_MIN = 2.0
FRAC_AREA = 3e-4           # contagem final (versao 2): polo com Q >= 2 conta se for visivel no modelo analitico (proeminencia
                           # >= 0,1) OU se tiver fracao de area >= 3e-4 (energia de modo escondido sob vizinho). Com tau em
                           # [1e-4, 1e3] aparecem modos acusticos internos de Q ~ 1e5 a 1e7 (Maxwell com tau pequeno: esfera
                           # quase fluida, sem perda, radiacao evanescente em l alto), bem visiveis e com area ~1e-8 da janela;
                           # so a area cortaria 11% dos visiveis (validacao 5). 3e-4 e' o valor calibrado na versao 1
Q_MIN_POLO = 2.0
DX_MIN = 1e-9
N_MAX_PONTOS = 2_000_000
BLOCO = 100_000            # pontos por chamada vetorizada (memoria)

# busca de polos
NIVEIS_CONTORNO = (64, 128, 256, 512, 1024, 2048)
ANG_MAX = 1.0              # salto maximo de argumento (rad) entre pontos vizinhos do contorno para a contagem valer
NIVEIS_AUDITORIA = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
TOL_INTEIRO = 0.05
PROF_MAX = 40
BEYN_MAX, BEYN_FOLGA, BEYN_N = 8, 4, 500
RESIDUO_MAX = 1e-7
GAMMA_RAD = 1e-10
DELTA_ESCALA_MAX = 1.0     # validade da lorentziana do residuo (ver peso_polo)
TOL_DUPLICATA = 1e-8
Q_BORDA = 1.5              # fundo da regiao de busca: y = -x/(2 Q_BORDA) (polos com Q < 1,5 ficam de fora; os que contam tem Q >= 2)
Q_BORDA_EXTRA = (1.75, 1.9)  # fundos alternativos da auditoria quando um zero fica colado no fundo
N_ZEROS_MAX = 5000         # acima disso (estimativa) a busca de polos e' pulada
N_RAPIDA_MIN, N_RAPIDA_MAX, PASSOS_POR_ESP = 200_000, 2_000_000, 200
PROM_SEMENTE = 1e-3        # minimo de |D_l| no eixo vira semente se for um vale de pelo menos 0,1% (1e-3 em log10);
                           # tira as oscilacoes de arredondamento perto de x_min, que so gastavam Newton

# precisao numerica
TOL_TAYLOR = 1e-7          # |Im z| < 1e-7 min(1,|z|): Bessel por Taylor em volta do eixo real
TOL_G_REAL = 1e-6          # |Im z| < 1e-6 min(1,|z|): ramo real da integral de Lommel G
NORMALIZACAO = "ZM"


# ---------------------------------------------------------------------------
# Material: velocidades complexas do texto da tese
# ---------------------------------------------------------------------------
def velocidades(z, cur):
    """c_L(z), c_T(z) para x real ou complexo (a busca de polos usa o prolongamento analitico).
    Voigt:   c_T = c_T0 sqrt(1 - i tau z),  c_L = c_L0 sqrt(1 - i (4/(3 eta0^2)) tau z).
    Maxwell: com q = 1 - 1/(1 + i tan d_T) = i/(tau z + i),  c_T = c_Tinf sqrt(1 - q),  c_L = c_Linf sqrt(1 - 4 q/(3 eta_inf^2)).
    Para x > 0 (e todo o retangulo de busca) os argumentos das raizes nunca cruzam o semieixo real negativo,
    entao c_L(z), c_T(z) sao analiticas e o principio do argumento vale."""
    z = np.asarray(z, dtype=complex); e2 = (cur["cL"] / cur["cT"]) ** 2; tau = cur["tau"]
    with np.errstate(all="ignore"):
        if cur["modelo"] == "elastico":   # sem perda: c_L, c_T reais e constantes (limite tau -> 0 do Voigt)
            return cur["cL"] * np.ones_like(z), cur["cT"] * np.ones_like(z)
        if cur["modelo"] == "voigt":
            return cur["cL"] * np.sqrt(1 - 1j * 4 / (3 * e2) * tau * z), cur["cT"] * np.sqrt(1 - 1j * tau * z)
        q = 1j / (tau * z + 1j) if np.isfinite(tau) else np.zeros_like(z)
        return cur["cL"] * np.sqrt(1 - 4 / (3 * e2) * q), cur["cT"] * np.sqrt(1 - q)


def argumentos(z, cur):
    cL, cT = velocidades(z, cur); z = np.asarray(z, dtype=complex)
    return z * CM_PADRAO / cL, z * CM_PADRAO / cT, cL, cT


def tan_perdas(x, cur):
    """tan d_T e tan d_L efetivos no eixo real: tan d = -Im(c^2)/Re(c^2) (M* = rho c^2 = M'(1 - i tan d))."""
    cL, cT = velocidades(x, cur)
    cL, cT = np.ravel(cL)[0], np.ravel(cT)[0]
    return float(-np.imag(cL**2) / np.real(cL**2)), float(-np.imag(cT**2) / np.real(cT**2))


def escala_s(z, cur, l=1):
    """s(z) = |Im z_L| + |Im z_T| (l >= 1) ou |Im z_L| (l = 0, D_0 nao tem z_T): fator da escala real e^{-s} de D_l."""
    zL, zT, _, _ = argumentos(z, cur)
    return np.abs(zL.imag) + (np.abs(zT.imag) if l > 0 else 0.0)


# ---------------------------------------------------------------------------
# Funcoes esfericas
# ---------------------------------------------------------------------------
def _j_topo(nmax, z, qr):
    """j_n(z) e^{-|Im z|} para n = nmax e nmax+1. Quase real: Taylor j(x + ie) = j + ie j' - e^2 j''/2 com Bessel real
    (o erro e' O(e^3) e a parte imaginaria, que carrega a perda, nao se perde no double). Senao: j_n = sqrt(pi/(2z))
    J_{n+1/2}, com jve = J e^{-|Im z|}. Todas as ordens numa chamada so de jv/jve (bem mais rapido que spherical_jn)."""
    v = np.empty((2,) + z.shape, dtype=complex); n = np.array([nmax, nmax + 1])[:, None]
    if qr.any():
        x, e = z.real[qr], z.imag[qr]; J = np.sqrt(np.pi / (2 * x)) * jv(np.arange(nmax - 1, nmax + 2)[:, None] + 0.5, x[None, :])
        f, fp = J[1:], J[:-1] - (n + 1) / x * J[1:]; fpp = -(2 / x) * fp + (n * (n + 1) / x**2 - 1) * f
        v[:, qr] = (f + 1j * e * fp - 0.5 * e * e * fpp) * np.exp(-np.abs(e))
    if not qr.all():
        zz = z[~qr]; v[:, ~qr] = np.sqrt(np.pi / (2 * zz)) * jve(n + 0.5, zz[None, :])
    return v


def tab_j(z, nmax):
    """Tabela j_n(z) e^{-|Im z|}, n = -1..nmax (linha n+1), e s = |Im z|. Cada funcao e' calculada uma vez por ordem e
    por argumento; as derivadas saem depois de f_l' = f_{l-1} - (l+1)/z f_l.
    |z| > 2(nmax+1) (fora do eixo): recorrencia ascendente j_{n+1} = (2n+1)/z j_n - j_{n-1} a partir de
        j_{-1} = cos z/z e j_0 = sin z/z, com sin e cos ja multiplicados por e^{-|Im z|} (estavel para n < |z|);
    resto: as duas ordens do topo saem de _j_topo e as outras da recorrencia descendente (estavel para j_l)."""
    z = np.asarray(z, dtype=complex); s = np.abs(z.imag); out = np.empty((nmax + 2,) + z.shape, dtype=complex)
    qr = s < TOL_TAYLOR * np.minimum(1.0, np.abs(z)); big = ~qr & (np.abs(z) > 2 * (nmax + 1)); rest = ~big
    with np.errstate(all="ignore"):
        if rest.any():
            zr = z[rest]; b, a = _j_topo(nmax, zr, qr[rest]); out[nmax + 1, rest] = b
            for n in range(nmax, -1, -1):
                c = (2 * n + 1) / zr * b - a; out[n, rest] = c; a, b = b, c
        if big.any():
            zb = z[big]; sb = s[big]; e1, e2 = np.exp(1j * zb - sb), np.exp(-1j * zb - sb)
            a, b = (e1 + e2) / 2 / zb, (e1 - e2) / 2j / zb; out[0, big], out[1, big] = a, b
            for n in range(0, nmax):
                c = (2 * n + 1) / zb * b - a; out[n + 2, big] = c; a, b = b, c
    return out, s


def tab_h(z, nmax):
    """h_n = j_n + i y_n do meio externo (argumento x_M), n = -1..nmax (linha n+1).
    Eixo real: j_n e y_n separados, j_n = sqrt(pi/(2x)) J_{n+1/2}, y_n = sqrt(pi/(2x)) Y_{n+1/2}, os dois reais; assim
    Re h e Im h guardam cada um a precisao do double (e' o que permite o polo_estreito). Com n = -1 isso da
    j_{-1} = cos x/x = -y_0 e y_{-1} = sin x/x = j_0, como nas funcoes J, Y dos codigos antigos.
    Plano complexo (contornos e Newton): recorrencia ascendente h_{n+1} = (2n+1)/z h_n - h_{n-1}, estavel para h,
    a partir de h_0 = -i e^{iz}/z e h_{-1} = i h_0 = e^{iz}/z."""
    z = np.asarray(z); H = np.empty((nmax + 2,) + z.shape, dtype=complex)
    with np.errstate(all="ignore"):
        if not np.iscomplexobj(z):
            nu = np.arange(-1, nmax + 1)[:, None] + 0.5; c = np.sqrt(np.pi / (2 * z))[None, :]
            return c * jv(nu, z[None, :]) + 1j * (c * yv(nu, z[None, :]))
        e = np.exp(1j * z) / z; H[0], H[1] = e, -1j * e
        for n in range(0, nmax): H[n + 2] = (2 * n + 1) / z * H[n + 1] - H[n]
    return H


def G_esc(z, l, J, s):
    """Peca de Lommel G(z,l) do codigo antigo, ja multiplicada por e^{-2|Im z|} (a mesma escala de |a_l|^2 G).
    Ramo complexo: G = |z|^2/(z^2 - z*^2) [z j_l(z) j_l'(z*) - z* j_l'(z) j_l(z*)], com j(z*) = j(z)*.
    Ramo real (|Im z| < 1e-6 min(1,|z|)): G = x j j' + x^2/2 (j^2 - j_{l-1} j_{l+1}); o erro e' O((Im z)^2).
    O ramo imaginario do codigo antigo (Re z ~ 0) nao ocorre: Re z_L, Re z_T > 0 em todo o eixo real dos dois modelos."""
    x = np.abs(z); jl, jm, jp = J[l + 1], J[l], J[l + 2]
    with np.errstate(all="ignore"):
        dj = jm - (l + 1) / z * jl; zc = np.conj(z)
        Gc = (x**2 / (z**2 - zc**2)) * (z * jl * np.conj(dj) - zc * dj * np.conj(jl))
        es = np.exp(s); r0, rm, rp = np.real(jl * es), np.real(jm * es), np.real(jp * es); xr = z.real
        Gr = (xr * r0 * (rm - (l + 1) / xr * r0) + xr**2 / 2 * (r0**2 - rm * rp)) * np.exp(-2 * s)
    return np.where(np.abs(z.imag) < TOL_G_REAL * np.minimum(1.0, x), Gr, Gc)


# ---------------------------------------------------------------------------
# Coeficientes, denominador e energia
# ---------------------------------------------------------------------------
def campos(z, cur, ls=None, energia=False, lmax=LMAX):
    """Para cada l: D_l (sem divisoes), numeradores de a_l e b_l e, no eixo real, a forma da energia.

    Condicoes de contorno em r = a (tese, Eq. do sistema matricial), por l:
        z_L j'(z_L) a + L j(z_T) b - x h'(x) s = x j'(x)                                   (velocidade normal)
        z_L^2 [lam j(z_L) - 2 mu j''(z_L)] a - 2 mu L [z_T j'(z_T) - j(z_T)] b - rho_M c_M^2 x^2 h(x) s = rho_M c_M^2 x^2 j(x)
        2 [z_L j'(z_L) - j(z_L)] a + [z_T^2 j''(z_T) + (L - 2) j(z_T)] b = 0                 (cisalhamento nulo)
    com L = l(l+1). Os coeficientes do codigo antigo, a_l = (C~ - C)/[(A - A~) - (B~ - B) R], b_l = R a_l, multiplicados
    em cima e embaixo por den_M den_M~ den_R = (rho_M c_M^2 x^2 h)(x h')(den_R) e usando o wronskiano
    C~ - C = i/(x^2 h h'), ficam sem nenhuma divisao:
        a_l = i rho_M c_M^2 x den_R / D_l,   b_l = i rho_M c_M^2 x num_R / D_l,
        D_l = den_R P_1 - num_R P_2,
        P_1 = z_L^2 [lam j(z_L) - 2 mu j''(z_L)] x h' - z_L j'(z_L) rho_M c_M^2 x^2 h,
        P_2 = L j(z_T) rho_M c_M^2 x^2 h + 2 mu L [z_T j'(z_T) - j(z_T)] x h',
        num_R = 2 [j(z_L) - z_L j'(z_L)],  den_R = z_T^2 j''(z_T) + (L - 2) j(z_T).
    D_l e' inteira no retangulo de busca (o D antigo tinha polos nos zeros de h, h' e den_R), entao o principio do
    argumento conta exatamente os zeros. Tudo que depende de z_L sai multiplicado por e^{-|Im z_L|} e o que depende de
    z_T por e^{-|Im z_T|}: aqui 'D' = D_l e^{-s}, 'Na' = num(a_l) e^{-|Im z_T|}, 'Nb' = num(b_l) e^{-|Im z_L|}, e
    a = Na/D, b = Nb/D sao a_l e^{|Im z_L|}, b_l e^{|Im z_T|}.

    As combinacoes com j' e j'' sao reescritas so com j_l e j_{l+1} (z j' = l j - z j_{l+1}, z^2 j'' pela equacao de
    Bessel), o que e' identico analiticamente mas evita o cancelamento em z pequeno (por exemplo, para l = 1,
    z_T^2 j'' + (L-2) j = z_T^2 j'' inteiro e' da ordem de z^3 enquanto cada termo da equacao de Bessel e' O(z)):
        num_R = 2 [(1 - l) j(z_L) + z_L j_{l+1}(z_L)],  den_R = (2l^2 - 2 - z_T^2) j(z_T) + 2 z_T j_{l+1}(z_T),
        z_L^2 j''(z_L) = (l^2 - l - z_L^2) j(z_L) + 2 z_L j_{l+1}(z_L),  z_T j'(z_T) - j(z_T) = (l - 1) j(z_T) - z_T j_{l+1}(z_T).
    Energia (energia=True, x real): a mesma energia de Clapeyron do voigt.py, 2<T> + Re{I_S}/4, escrita como forma
    hermitiana nos coeficientes, W_l = Re[c^H M c], c = (a_l, b_l):
        M11 = 2f |z_L| G(z_L) - f k u1 P_A,   M12 = 2fL conj(C) - f k u1 P_B,
        M21 = 2fL C - f k u2 P_A,             M22 = 2fL |z_L| (|j(z_T)|^2 + G(z_T)) - f k u2 P_B,
    f = 2l+1, C = |z_L| j(z_L) conj(j(z_T)), u1 = conj(j'(z_L)), u2 = L conj(j(z_T))/conj(z_L),
    P_A = z_L^2 [lam j(z_L) - 2 mu j''(z_L)], P_B = -2 mu L [z_T j'(z_T) - j(z_T)], k = 1/(rho_S |c_L| conj(c_L)).
    Com as funcoes escaladas M sai multiplicada pelas mesmas escalas de |a|^2, a b*, |b|^2 e W_l nao muda.
    A forma M tambem da o peso S de cada polo (peso_polo)."""
    real = not np.iscomplexobj(z); z = np.asarray(z, dtype=float if real else complex); zc_ = z.astype(complex)
    zL, zT, cL, cT = argumentos(zc_, cur); rhoS = cur["rhoS"]
    mu = rhoS * cT**2; lam = rhoS * cL**2 - 2 * mu; rcm = RHOM_PADRAO * CM_PADRAO**2
    JL, sL = tab_j(zL, lmax + 1); JT, sT = tab_j(zT, lmax + 1); HM = tab_h(z, lmax)
    res = {"sL": sL, "sT": sT, "cL": cL, "cT": cT}
    with np.errstate(all="ignore"):
        for l in (range(lmax + 1) if ls is None else ls):
            L, f, i = l * (l + 1), 2 * l + 1, l + 1
            jL, jT, pL, pT = JL[i], JT[i], JL[i + 1], JT[i + 1]; djL = (l / zL) * jL - pL
            h = HM[i]; dh = HM[i - 1] - (l + 1) / z * HM[i]
            PA = lam * zL**2 * jL - 2 * mu * ((l * l - l - zL**2) * jL + 2 * zL * pL); PB = -2 * mu * L * ((l - 1) * jT - zT * pT)
            nR = 2 * ((1 - l) * jL + zL * pL); dR = (2 * l * l - 2 - zT**2) * jT + 2 * zT * pT
            P1 = PA * z * dh - (l * jL - zL * pL) * rcm * z**2 * h
            P2 = L * jT * rcm * z**2 * h - PB * z * dh
            if l == 0:   # P_2 = 0 e den_R = T_0(z_T) e' fator comum espurio (secao 'Relacoes de recorrencia'): sai
                r = {"D": P1, "Na": 1j * rcm * z * np.ones_like(P1), "Nb": np.zeros_like(P1)}
            else:
                r = {"D": dR * P1 - nR * P2, "Na": 1j * rcm * z * dR, "Nb": 1j * rcm * z * nR}
            if energia:
                a, b = r["Na"] / r["D"], r["Nb"] / r["D"]; xL = np.abs(zL); k = 1 / (rhoS * np.abs(cL) * np.conj(cL))
                GL, GT = G_esc(zL, l, JL, sL), G_esc(zT, l, JT, sT); C = xL * jL * np.conj(jT)
                u1, u2 = np.conj(djL), L / np.conj(zL) * np.conj(jT)
                M = (2 * f * xL * GL - f * k * u1 * PA, 2 * f * L * np.conj(C) - f * k * u1 * PB,
                     2 * f * L * C - f * k * u2 * PA, 2 * f * L * xL * (np.abs(jT) ** 2 + GT) - f * k * u2 * PB)
                r.update(a=a, b=b, M=M, W=np.real(np.conj(a) * (M[0] * a + M[1] * b) + np.conj(b) * (M[2] * a + M[3] * b)))
            res[l] = r
    return res


def peso_Z(cL, cur):
    """Normalizacao da energia. A energia do codigo antigo e' W~_S = 2 f0 W_S/(Z_S |A|^2), Z_S = rho_S |c_L(x)| (tese).
    Na referencia do meio: 2 f0 W_S/(Z_M |A|^2) = (Z_S/Z_M) W~_S. Como so ha uma regiao, o fator e' liso em x
    e nao mexe em posicao, Q, fracao de area nem dx_nec; muda so a escala absoluta (altura, S, area).
    '--normalizacao separada' devolve W~_S, como nas versoes antigas."""
    if NORMALIZACAO == "separada": return np.ones(np.shape(cL))
    return cur["rhoS"] * np.abs(cL) / (RHOM_PADRAO * CM_PADRAO)


def energia(x, cur, por_l=False, ls=None):
    """W_T = soma de W_l (l = 0..10) no eixo real, com o peso de peso_Z. Vetorizado em blocos de BLOCO pontos."""
    x = np.atleast_1d(np.asarray(x, dtype=float)); ls = list(range(LMAX + 1)) if ls is None else list(ls)
    if len(x) > BLOCO:
        partes = [energia(x[i:i + BLOCO], cur, True, ls) for i in range(0, len(x), BLOCO)]
        WT, Wl = np.concatenate([p[0] for p in partes]), np.concatenate([p[1] for p in partes], axis=1)
        return (WT, Wl) if por_l else WT
    r = campos(x, cur, ls=ls, energia=True, lmax=max(ls)); w = peso_Z(r["cL"], cur)
    Wl = np.array([np.where(np.isfinite(r[l]["W"]), r[l]["W"], 0.0) * w for l in ls])
    WT = Wl.sum(axis=0); WT = np.where(np.isfinite(WT) & (WT > 0), WT, 0.0)
    return (WT, Wl) if por_l else WT


def energia_l(l, x, cur): return energia(x, cur, True, [l])[1][0]


def D_todos(z, cur, ls, p):
    """z^p_l D_l e^{-s} para varios l de uma vez (as tabelas de Bessel sao comuns). O fator real positivo e^{-s} nao
    muda zeros nem argumento; z^p_l tira o comportamento de potencia de D_l perto de z = 0."""
    z = np.asarray(z); out = np.empty((len(ls), len(z)), dtype=complex)
    for i0 in range(0, len(z), BLOCO):
        zz = z[i0:i0 + BLOCO]; r = campos(zz, cur, ls=ls, lmax=max(ls)); zc = zz.astype(complex)
        for k, l in enumerate(ls): out[k, i0:i0 + BLOCO] = zc ** p[l] * r[l]["D"]
    return out


def D_contorno(z, cur, ls, p):
    """z^p D_l e^{-s} e^{i theta(z)} para o principio do argumento. Onda interna com |Im z_j| grande (perda alta) faz
    j(z_j) ~ e^{-i sgn(Im z_j) z_j}/(2iz): o argumento de D_l gira como Re z_j ao longo do contorno sem haver zero (no
    Maxwell com tau pequeno, Re z_T ~ 1e4 em x = 1) e o contorno precisaria de dezenas de milhares de pontos por lado.
    A fase theta = - soma_j sgn(Im z_j) Re z_j chi(|Im z_j|), chi = 0 ate |Im z| = 1 e 1 a partir de 3 (rampa
    linear), desfaz essa rotacao. theta e' continua e univoca no contorno, entao a integral de d theta numa volta
    fechada e' zero e o numero de voltas (de zeros) nao muda; o fator tem modulo 1 e nao muda zero nenhum."""
    z = np.asarray(z); v = D_todos(z, cur, ls, p); zL, zT, _, _ = argumentos(z, cur)
    chi = lambda w: np.clip((np.abs(w.imag) - 1) / 2, 0, 1) * np.sign(w.imag) * w.real
    tL, tT = chi(zL), chi(zT)
    for k, l in enumerate(ls): v[k] = v[k] * np.exp(1j * (tL + (tT if l > 0 else 0)))
    return v


def F_analitica(l, cur, p, z_ref):
    """F(z) = z^p D_l(z) e^{-s(z_ref)}: analitica (o fator e' constante) e com modulo O(1) perto de z_ref.
    E' a funcao usada no Newton, no Beyn e nas derivadas (residuos)."""
    s_ref = float(escala_s(np.array([z_ref]), cur, l)[0])
    def F(z):
        zz = np.atleast_1d(z); r = campos(zz, cur, ls=[l], lmax=l)
        v = zz.astype(complex) ** p * r[l]["D"] * np.exp(r["sL"] + (r["sT"] if l > 0 else 0) - s_ref)
        return v if np.ndim(z) else complex(v[0])
    return F


def expoentes_p(cur, ls=range(LMAX + 1)):
    """p_l = -(inclinacao de log|D_l| em log x perto de x_min), arredondada: z^p D_l fica O(1) em z -> 0.
    No core-shell isso era o fator z^4 fixo; aqui a potencia depende de l e do material, entao e' medida."""
    x = np.array([X_MIN, 4 * X_MIN]); d = np.abs(D_todos(x, cur, list(ls), {l: 0 for l in ls}))
    with np.errstate(all="ignore"): incl = np.log(d[:, 1] / d[:, 0]) / np.log(4.0)
    return {l: int(-np.round(v)) if np.isfinite(v) else 0 for l, v in zip(ls, incl)}


# ---------------------------------------------------------------------------
# Fase acumulada, espacamento e risco
# ---------------------------------------------------------------------------
def fase(cur, x=None):
    """Phi(x) = max(d Re z_L/dx, d Re z_T/dx) entre as ondas que propagam (Q da onda >= Q_BORDA): fase por unidade de x
    da onda mais lenta que ainda forma ressonancia. Delta_W(x) = pi/Phi(x) (Eq. deltaW_elastica: pi c_T/c_M sem perda).
    Onda super amortecida nao forma modo (o Q do polo nao passa do Q da onda) e nao encolhe a janela; onde nenhuma
    propaga fica a longitudinal."""
    x = np.unique(np.concatenate([np.geomspace(X_MIN, 0.1, 400), np.linspace(0.1, X_MAX + 0.05, 1200)])) if x is None else x
    zl, zt, QL, QT, _ = ondas(x, cur); dL, dT = np.gradient(zl, x), np.gradient(zt, x)
    Phi = np.where(QT >= Q_BORDA, np.maximum(dL, dT), dL)
    return x, np.maximum(Phi, 1e-6)


def espacamento(cur):
    xg, Phi = fase(cur); espg = np.pi / Phi
    return lambda v: np.interp(v, xg, espg)


def risco_fase(cur, x=None):
    """Criterio da versao 1 (guardado como coluna risco_fase_v1 para comparacao): risco quando alguma onda interna que propaga (Q da onda = Re z/(2|Im z|) >= 2) e' mais lenta que o meio:
    Re z_j(x)/x > 1. Sem perda isso e' simplesmente c_T < c_M (Voigt: c_T0 < c_M). No Maxwell a onda transversal
    fica lenta em x pequeno, mas la tan d_T = 1/(tau x) e' enorme e ela nao propaga, entao essa regiao nao conta."""
    x = np.geomspace(X_MIN, X_MAX, 2000) if x is None else x; zL, zT, _, _ = argumentos(x, cur)
    for z in (zL, zT):
        with np.errstate(all="ignore"): q = z.real / (2 * np.abs(z.imag))
        if np.any((z.real / x > 1) & (q >= Q_MIN)): return True
    return False


def estimativa_zeros(cur):
    """Numero esperado de zeros de D_l em [x_min, x_max], somando l = 0..10: cada familia que propaga (L e T, Q da onda
    >= Q_BORDA) da ~ Delta(Re z)/pi zeros (Eq. contagem_elastica da tese: N_l = x/pi (c_M/c_L + c_M/c_T))."""
    xg = np.unique(np.concatenate([np.geomspace(X_MIN, 0.1, 400), np.linspace(0.1, X_MAX, 1200)])); th = fase_propagante(xg, cur)
    return float((LMAX + 1) * th[-1] / np.pi)


# ---------------------------------------------------------------------------
# Relacoes de recorrencia (secao da tese): previsao das ressonancias a partir do eixo real
# ---------------------------------------------------------------------------
# Condicao de ressonancia (Eq. det_elastico), dividida por M = rho_S c_L^2:
#     f_l(x) = kappa_l(x) E_l + mu_x G_l = 0,   kappa_l = -x h'(x)/h(x),   mu_x = rho_M c_M^2 x^2/M = (rho_M/rho_S) z_L^2,
#     E_l = z_L^2 [(1 - 2 eps) j(z_L) - 2 eps j''(z_L)] T_l(z_T) + 4 eps L [z_T j'(z_T) - j(z_T)][z_L j'(z_L) - j(z_L)],
#     G_l = z_L j'(z_L) T_l(z_T) - 2 L j(z_T) [z_L j'(z_L) - j(z_L)],   T_l = z_T^2 j''(z_T) + (L - 2) j(z_T),   eps = 1/eta^2 = c_T^2/c_L^2.
# f_l e' D_l/h_l a menos de fatores sem zero: tem exatamente os mesmos zeros. A tese troca kappa por l + 1 (x << 1)
# e, no solido elastico, a equacao fica real e as ressonancias sao as raizes reais (Eq. robin_elastica).
#
# Modo 'complexa' (padrao): f_l com tudo complexo (kappa com a radiacao, z_j e eps do material com perda), avaliada
# no eixo real; perto de um polo z_k = x_k - i gamma o passo de Newton w(x) = f/f' = x - z_k + O((x - z_k)^2), entao
# |w| tem minimo ~gamma em x ~ x_k (liso, mesmo para gamma/x ~ 1e-15). Cada minimo de |w| com |w| <= x, e cada raiz
# da versao real (abaixo), e' semente de Newton complexo em f_l (40 iteracoes, vetorizado), e os zeros convergidos com
# Q >= Q_PREV_MIN sao a previsao
# (posicao e Q). Sem perda e sem radiacao f e w sao reais, gamma = 0 e os zeros sao as raizes reais da tese.
# Por que nao a equacao real da tese com Re z_j no lugar de z_j: no Voigt com tau x grande o cisalhamento e' quase so
# viscoso (|mu''| >> mu', eps -> 3/4) e aparece um modo quase estatico de dilatacao uniforme (u ~ r, sem cisalhamento,
# omega^2 = 15 K/(rho_S a^2) mais o carregamento do fluido) com Q de dezenas a centenas, que nenhuma versao real da
# equacao ve; a 'complexa' ve (validacao 6). As versoes reais ficam como '--recorrencia real' (kappa = -Re(x h'/h),
# Re z_j, razoes dos modulos de armazenamento, so ondas que propagam) e '--recorrencia texto' (kappa = l + 1).
# l = 0: sem o fator espurio T_0 (Eq. reduzida da tese).
# As previsoes com Q >= Q_PREV_MIN em x <= X_PREV_MAX decidem se a curva e' de risco e viram sementes do Newton.
X_PREV_MAX = 1.1           # margem acima de x_max: cobre a extensao da malha (<= 0,05) com folga (ver validacao 6)
Q_PREV_MIN = 1.0           # previsao de primeira ordem: margem abaixo do Q = 2 dos polos que contam
PREV_PASSOS_POR_PI = 16    # pontos da malha por pi de fase acumulada (Re z_L + Re z_T)
PREV_MAX_RAIZES = 3000     # fase acumulada / pi acima disso: so a contagem pela fase (solido muito lento)
RECORRENCIA = "complexa"   # 'complexa' | 'real' (kappa = -Re(x h'/h)) | 'texto' (kappa = l + 1)
ZETA_QS = np.pi / 2        # versoes reais: onda com fase <= pi/2 entra sempre (quase estatica)
TOL_CANCELAMENTO = 1e-12   # |f| < 1e-12 x (soma dos modulos das parcelas): sinal indefinido (arredondamento), nao conta


def _f_complexa(x, cur, ls):
    """f_l(x) e^{-s} e a soma dos modulos das parcelas, x real (array), todos os l de ls com as mesmas tabelas.
    s = |Im z_L| + |Im z_T| (l >= 1) ou |Im z_L| (l = 0): cada parcela tem uma Bessel de cada onda (as de campos)."""
    zL, zT, cL, cT = argumentos(x, cur); JL, sL = tab_j(zL, LMAX + 1); JT, sT = tab_j(zT, LMAX + 1); H = tab_h(x, LMAX)
    eps = cT**2 / cL**2; mu = (RHOM_PADRAO / cur["rhoS"]) * (CM_PADRAO * x) ** 2 / cL**2; out = {}
    with np.errstate(all="ignore"):
        for l in ls:
            L = l * (l + 1); jl, pl = JL[l + 1], JL[l + 2]; k = -x * (H[l] - (l + 1) / x * H[l + 1]) / H[l + 1]
            AL = (zL**2 - 2 * eps * (l * l - l)) * jl - 4 * eps * zL * pl; dL = l * jl - zL * pl
            if l == 0: t = (k * AL, mu * dL)
            else:
                jt, pt = JT[l + 1], JT[l + 2]; T = (2 * l * l - 2 - zT**2) * jt + 2 * zT * pt; BT, BL = (l - 1) * jt - zT * pt, (l - 1) * jl - zL * pl
                t = (k * AL * T, k * 4 * eps * L * BT * BL, mu * dL * T, -mu * 2 * L * jt * BL)
            out[l] = (sum(t), sum(np.abs(u) for u in t), sL + (sT if l > 0 else 0))
    return out


def passo_newton(x, cur, ls):
    """w_l(x) = f_l/f_l' (diferenca central com h = 1e-6 x; a escala e^{-s} e' desfeita na razao) e |f| relativo
    ao cancelamento. Devolve {l: (w, bom)}."""
    x = np.asarray(x); x = x.astype(complex) if np.iscomplexobj(x) else x.astype(float); h = 1e-6 * np.abs(x); F0, Fp, Fm = (_f_complexa(v, cur, ls) for v in (x, x + h, x - h)); out = {}
    with np.errstate(all="ignore"):
        for l in ls:
            f0, e0, s0 = F0[l]; fp, _, sp = Fp[l]; fm, _, sm = Fm[l]
            d = (fp * np.exp(s0 - sp) - fm * np.exp(s0 - sm)) / (2 * h); out[l] = (f0 / d, np.abs(f0) > TOL_CANCELAMENTO * e0)
    return out


def _malha_prev(cur, x_max, so_propagantes):
    xg = np.unique(np.concatenate([np.geomspace(X_MIN, 0.1, 2000), np.linspace(0.1, x_max, 4000)]))
    if so_propagantes: th = fase_propagante(xg, cur)
    else:
        zl, zt = ondas(xg, cur)[:2]; th = np.r_[0.0, np.cumsum(np.abs(np.diff(zl)) + np.abs(np.diff(zt)))]
    n = int(PREV_PASSOS_POR_PI * th[-1] / np.pi)
    x = np.unique(np.concatenate([xg, np.interp(np.linspace(0, th[-1], max(n, 2)), th, xg)])) if 1 < n <= PREV_PASSOS_POR_PI * PREV_MAX_RAIZES else xg
    return x, float(th[-1] / np.pi)


def raizes_previstas(cur, ls=range(LMAX + 1), x_max=X_PREV_MAX, modo=None, ramos=False):
    """Ressonancias previstas pela condicao de recorrencia para cada l em [x_min, x_max] (ver bloco acima).
    Devolve ({l: posicoes}, info). info: 'N_fase' (fase acumulada/pi em x_max), 'aproximado' (fase/pi > PREV_MAX_RAIZES:
    sem posicoes, so a contagem), 'Q' ({l: Q de primeira ordem}, modo complexa) e, com ramos=True nas versoes reais,
    'ramos' ({l: (raizes do ramo longitudinal, raizes de T_l)}) para a validacao 6."""
    modo = modo or RECORRENCIA; ls = list(ls)
    x, nf = _malha_prev(cur, x_max, modo != "complexa"); info = {"N_fase": nf, "aproximado": nf > PREV_MAX_RAIZES}
    if info["aproximado"]: return {l: None for l in ls}, info
    out, br, Qs = {}, {}, {}
    with np.errstate(all="ignore"):
        if modo == "complexa":
            reais = raizes_previstas(cur, ls, x_max, modo="real")[0]; W = passo_newton(x, cur, ls)
            for l in ls:
                w, bom = W[l]; a = np.where(bom & np.isfinite(w), np.abs(w), np.inf)
                i = np.flatnonzero((a[1:-1] <= a[:-2]) & (a[1:-1] < a[2:]) & (a[1:-1] <= x[1:-1])) + 1   # minimos de |f/f'|
                z = x[i] - w[i]
                if l in reais and len(reais[l]):   # + raizes da versao real: com as duas familias intercaladas e densas,
                    wr0 = passo_newton(reais[l], cur, [l])[l][0]; z = np.r_[z, reais[l] - wr0]   # dois polos podem dividir um minimo
                for _ in range(40):                                                                      # Newton complexo
                    if not len(z): break
                    dz = passo_newton(z, cur, [l])[l][0]; z = np.where(np.isfinite(dz), z - dz, np.nan)
                dz = passo_newton(z, cur, [l])[l][0] if len(z) else z
                z = np.where(np.abs(z.imag) < GAMMA_RAD * np.abs(z.real), z.real - 1e-300j, z)   # polo colado no eixo (sinal de Im e' ruido)
                ok = np.isfinite(z) & (np.abs(dz) <= 1e-9 * np.abs(z)) & (z.imag < 0) & (z.real >= X_MIN) & (z.real <= x_max)
                z = np.array(sem_duplicatas(list(z[ok])), complex); z = z[z.real / (2 * np.abs(z.imag)) >= Q_PREV_MIN] if len(z) else z
                out[l], Qs[l] = z.real, z.real / (2 * np.abs(z.imag))
            info["Q"] = Qs; return out, info
        for l in ls:
            f, lo, T, reg, esc = condicao_recorrencia(l, x, cur, modo); bom = np.abs(f) > TOL_CANCELAMENTO * esc
            F = lambda v, l=l: condicao_recorrencia(l, v, cur, modo)[0]
            # so troca de sinal dentro do mesmo regime (na fronteira a funcao muda de forma) e fora do cancelamento
            mesmo = (reg[:-1] == reg[1:]) & (reg[1:] > 0) & bom[:-1] & bom[1:]; out[l] = _raizes(F, x, f, mesmo)
            if ramos:
                FL = lambda v, l=l: condicao_recorrencia(l, v, cur, modo)[1]; FT = lambda v, l=l: condicao_recorrencia(l, v, cur, modo)[2]
                br[l] = (_raizes(FL, x, lo, (reg[:-1] > 0) & (reg[1:] > 0)),
                         _raizes(FT, x, T, (reg[:-1] == 2) & (reg[1:] == 2)) if l > 0 else np.array([]))
    if ramos: info["ramos"] = br
    return out, info


def curva_de_risco(cur, prev=None):
    """Criterio da versao 2 (relacoes de recorrencia): a curva e' de risco se a condicao de recorrencia preve pelo
    menos uma ressonancia (Q >= Q_PREV_MIN no modo complexa) em x <= X_PREV_MAX. Devolve (risco, N_previsto, prev, info)."""
    prev, info = raizes_previstas(cur) if prev is None else prev
    N = int(round(info["N_fase"] * (LMAX + 1))) if info["aproximado"] else int(sum(len(v) for v in prev.values()))
    return N >= 1, N, prev, info


def ondas(x, cur, carga=False):
    """No eixo real: zeta_L, zeta_T (Re z_j), Q_L, Q_T (Q de cada onda) e eps = Re(c_T^2)/Re(c_L^2) = mu'/M' (razao dos
    modulos de armazenamento; no Voigt e' constante, 1/eta0^2). carga=True devolve tambem
    mu_x = rho_M c_M^2 x^2/(rho_S Re c_L^2), o fator do carregamento do fluido (= (rho_M/rho_S) z_L^2 sem perda)."""
    x = np.asarray(x, dtype=float); zL, zT, cL, cT = argumentos(x, cur)
    with np.errstate(all="ignore"): QL, QT = zL.real / (2 * np.abs(zL.imag)), zT.real / (2 * np.abs(zT.imag))
    r = (zL.real, zT.real, QL, QT, np.real(cT**2) / np.real(cL**2))
    return r + ((RHOM_PADRAO / cur["rhoS"]) * (CM_PADRAO * x) ** 2 / np.real(cL**2),) if carga else r


def fase_propagante(x, cur):
    """theta(x) = integral de (d zeta_L/dx [Q_L >= Q_BORDA] + d zeta_T/dx [Q_T >= Q_BORDA]): fase acumulada so das
    ondas que propagam. theta/pi conta as ressonancias de um l (Eq. contagem_elastica)."""
    zl, zt, QL, QT, _ = ondas(x, cur)
    d = np.maximum(np.diff(zl), 0) * (QL[1:] >= Q_BORDA) + np.maximum(np.diff(zt), 0) * (QT[1:] >= Q_BORDA)
    return np.r_[0.0, np.cumsum(d)]


def kappa(l, x, modo=None):
    """-Re(x h'_l(x)/h_l(x)) (modo 'exata') ou l + 1 (modo 'texto', limite x << 1 da tese)."""
    if (modo or RECORRENCIA) == "texto": return np.full(np.shape(x), l + 1.0)
    H = tab_h(np.asarray(x, dtype=float), l); h = H[l + 1]; dh = H[l] - (l + 1) / x * H[l + 1]
    return -np.real(x * dh / h)


def condicao_recorrencia(l, x, cur, modo=None):
    """Devolve (completa, longitudinal, transversal, regime, escala) no eixo real: o lado esquerdo da condicao inteira
    (multiplicada por zeta_L^2 no ramo longitudinal), T_l(zeta_T), o regime (2: as duas ondas propagam, 1: so a
    longitudinal, 0: nenhuma) e a soma dos modulos das parcelas (para separar zero de verdade de cancelamento).
    Como em campos, tudo so com j_l e j_{l+1} (z j' = l j - z j_{l+1}, z^2 j'' pela equacao de Bessel):
        zeta_L^2 [(1 - 2 eps) j - 2 eps j''] = (zeta_L^2 - 2 eps (l^2 - l)) j(zeta_L) - 4 eps zeta_L j_{l+1}(zeta_L),
        T_l = (2l^2 - 2 - zeta_T^2) j(zeta_T) + 2 zeta_T j_{l+1}(zeta_T),   z j' - j = (l - 1) j - z j_{l+1}."""
    from scipy.special import spherical_jn as sj
    x = np.asarray(x, dtype=float); zl, zt, QL, QT, eps, mu_x = ondas(x, cur, carga=True); k = kappa(l, x, modo); L = l * (l + 1)
    jl, pl = sj(l, zl), sj(l + 1, zl); AL = (zl**2 - 2 * eps * (l * l - l)) * jl - 4 * eps * zl * pl; dL = l * jl - zl * pl
    longit = k * AL + mu_x * dL; esc_l = np.abs(k * AL) + np.abs(mu_x * dL)
    pL, pT = (QL >= Q_BORDA) | (zl <= ZETA_QS), (QT >= Q_BORDA) | (zt <= ZETA_QS)
    if l == 0: return longit, longit, np.full_like(x, np.nan), np.where(pL, 1, 0), esc_l
    jt, pt = sj(l, zt), sj(l + 1, zt); T = (2 * l * l - 2 - zt**2) * jt + 2 * zt * pt; BT, BL = (l - 1) * jt - zt * pt, (l - 1) * jl - zl * pl
    e1, e2, g1, g2 = k * AL * T, k * 4 * eps * L * BT * BL, mu_x * dL * T, -mu_x * 2 * L * jt * BL
    regime = np.where(pL, np.where(pT, 2, 1), 0)
    return np.where(regime == 2, e1 + e2 + g1 + g2, longit), longit, T, regime, np.where(regime == 2, np.abs(e1) + np.abs(e2) + np.abs(g1) + np.abs(g2), esc_l)


def _raizes(F, x, f, par, it=60):
    """Raizes por troca de sinal de f na malha x (so nos intervalos [x_i, x_i+1] com par[i]), por bissecao vetorizada."""
    i = np.flatnonzero(par & (np.sign(f[:-1]) * np.sign(f[1:]) < 0))
    a, b, fa = x[i], x[i + 1], f[i]
    for _ in range(it):
        m = 0.5 * (a + b); fm = F(m); s = np.sign(fm) == np.sign(fa); a, fa, b = np.where(s, m, a), np.where(s, fm, fa), np.where(s, b, m)
    return 0.5 * (a + b)


# ---------------------------------------------------------------------------
# Malha e picos (iguais ao core-shell, com Delta_W dependente de x)
# ---------------------------------------------------------------------------
def malha_base(cur, x_min=X_MIN, x_max=X_MAX):
    esp_f = espacamento(cur); ext = min(float(esp_f(x_max)), 0.05 * x_max)
    a, b = min(0.1, x_max), min(0.9, x_max)
    x = np.unique(np.concatenate([np.geomspace(x_min, a, N_LOG), np.linspace(a, b, N_MEIO), np.linspace(b, x_max + ext, N_FIM)]))
    return x, esp_f


def maximos(energia_f, x, esp_f, prom=None, x_min=X_MIN, x_max=X_MAX, w=None):
    """Indices e posicoes dos maximos de energia_f(x). Pico perto de x_max tem a posicao refinada antes de decidir se fica."""
    w = energia_f(x) if w is None else w
    if prom is None:
        idx = find_peaks(np.where(np.isfinite(w), w, -np.inf))[0]
    else:
        ok = np.isfinite(w) & (w > 0)
        if not ok.any(): return np.array([], int), np.array([])
        piso = max(w[ok].min() * 1e-14, np.finfo(float).tiny)
        idx = find_peaks(np.log10(np.where(ok, np.maximum(w, piso), piso)), prominence=prom)[0]
    fica, xs = [], []; e_max = min(float(esp_f(x_max)), 0.05 * x_max)
    for i in idx:
        xp = float(x[i])
        if xp > x_max - 2 * e_max and 0 < i < len(x) - 1:
            alvo = lambda t: -np.log(max(float(energia_f(np.array([t]))[0]), np.finfo(float).tiny))
            xp = float(minimize_scalar(alvo, bounds=(x[i - 1], x[i + 1]), method="bounded", options={"xatol": 1e-12}).x)
        if x_min <= xp <= x_max: fica.append(i); xs.append(xp)
    return np.array(fica, int), np.array(xs)


def refina_posicoes(energia_f, x, xs, it=60):
    """Secao aurea em todos os picos de uma vez, cada um no seu intervalo [x_{i-1}, x_{i+1}]."""
    if len(xs) == 0: return xs
    i = np.clip(np.searchsorted(x, xs), 1, len(x) - 1)
    a, b = x[i - 1], x[np.minimum(i + 1, len(x) - 1)]; g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a); fc, fd = energia_f(c), energia_f(d)
    for _ in range(it):
        m = np.nan_to_num(fc, nan=-np.inf) > np.nan_to_num(fd, nan=-np.inf)
        b, a = np.where(m, d, b), np.where(m, a, c)
        c, d = b - g * (b - a), a + g * (b - a); fc, fd = energia_f(c), energia_f(d)
    return 0.5 * (a + b)


def medir_picos(WT, x, xs, esp_f, escala=None, n_scan=80, it=50):
    """Topo refinado, fundo (mediana de W_T perto das bordas da janela do pico, que vai ate a metade do caminho para os
    vizinhos e no maximo Delta_W/2 de cada lado) e FWHM procurando o cruzamento da meia altura dos dois lados."""
    n = len(xs)
    if n == 0: return []
    xs0 = np.asarray(xs, dtype=float); w0 = WT(xs0); xs = refina_posicoes(WT, x, xs0); wp = WT(xs)
    xs, wp = np.where(wp >= w0, xs, xs0), np.maximum(wp, w0)
    esp = esp_f(xs); viz_e = np.r_[X_MIN, 0.5 * (xs[1:] + xs[:-1])]; viz_d = np.r_[0.5 * (xs[1:] + xs[:-1]), X_MAX + min(float(esp_f(X_MAX)), 0.05)]
    lo, hi = np.maximum(viz_e, xs - 0.5 * esp), np.minimum(viz_d, xs + 0.5 * esp)
    fr = np.array([0.02, 0.06, 0.12, 0.2])
    pts = np.concatenate([lo[:, None] + fr * (xs - lo)[:, None], hi[:, None] - fr * (hi - xs)[:, None]], axis=1)
    fundo = np.median(WT(pts.ravel()).reshape(n, -1), axis=1); nivel = fundo + 0.5 * (wp - fundo)
    escala = np.full(n, np.nan) if escala is None else np.asarray(escala, dtype=float); lados = []
    for lado, dmax in ((-1, xs - lo), (+1, hi - xs)):
        base = np.where(np.isfinite(escala) & (escala > 0), escala, dmax / 100)
        dmin = np.maximum(32 * np.finfo(float).eps * np.maximum(1, xs), np.minimum(1e-3 * base, 1e-3 * dmax))
        D = np.exp(np.linspace(np.log(dmin), np.log(np.maximum(dmax, 2 * dmin)), n_scan, axis=1))
        abaixo = (WT((xs[:, None] + lado * D).ravel()).reshape(n, -1) < nivel[:, None])
        tem = abaixo.any(axis=1); k = np.argmax(abaixo, axis=1)
        a = np.where(k > 0, D[np.arange(n), np.maximum(k - 1, 0)], 0.0); b = D[np.arange(n), k]
        for _ in range(it):
            m = 0.5 * (a + b); acima = WT(xs + lado * m) >= nivel; a, b = np.where(acima, m, a), np.where(acima, b, m)
        lados.append(np.where(tem, 0.5 * (a + b), np.nan))
    fwhm = lados[0] + lados[1]; ok = (wp > fundo) & (fundo > 0); saida = []
    for i in range(n):
        razao = wp[i] / fundo[i] if fundo[i] > 0 else np.inf
        bom = ok[i] and np.isfinite(fwhm[i]) and fwhm[i] > 0
        saida.append({"xM_pico": float(xs[i]), "altura": float(wp[i]), "fundo": float(fundo[i]), "razao_pico_fundo": float(razao),
                      "proeminencia_log10": float(np.log10(razao)) if razao > 0 else np.nan,
                      "largura": float(fwhm[i]) if bom else np.nan, "Q": float(xs[i] / fwhm[i]) if bom else np.nan,
                      "area": np.nan, "frac_area_local": np.nan})
    return saida


def casar(xs, zeros, esp_f):
    """Casa maximos da malha com zeros, 1 a 1, do par mais proximo para o mais distante."""
    pares = sorted((abs(x - z.real), i, j) for i, x in enumerate(xs) for j, z in enumerate(zeros) if abs(x - z.real) <= max(2 * abs(z.imag), 0.25 * float(esp_f(z.real))))
    usado_x, usado_z, casados = set(), set(), {}
    for _, i, j in pares:
        if i not in usado_x and j not in usado_z: usado_x.add(i); usado_z.add(j); casados[j] = i
    return casados


def modos_por_l(cur, zeros_por_l, x, Wl, esp_f, lmax=LMAX):
    """Posicoes dos modos de cada l, pela comparacao zeros x maximos de W_l na malha base (igual ao core-shell).
    W_l abaixo de 1e-12 do maximo de W_T vira esse piso: la o W_l e' ruido de arredondamento e nao aparece na curva."""
    saida = []; Wl = np.maximum(Wl, 1e-12 * Wl.sum(axis=0).max())
    for l in range(lmax + 1):
        zeros = zeros_por_l.get(l); E = lambda v, l=l: energia_l(l, v, cur)
        _, xs = maximos(E, x, esp_f, w=Wl[l])
        r = {"l": l, "N_zeros": None if zeros is None else len(zeros), "N_max_malha": len(xs)}
        if zeros is not None:
            cas = casar(xs, zeros, esp_f)
            if len(xs) == len(zeros) == len(cas):
                r.update(regra="bate", x=[z.real for z in zeros]); saida.append(r); continue
            bons = [j for j, z in enumerate(zeros) if z.real / (2 * abs(z.imag)) >= Q_MIN]
            sobra_x = [v for i, v in enumerate(xs) if i not in {cas[j] for j in cas if j not in bons}]
            z2 = [zeros[j] for j in bons]
            if len(sobra_x) == len(z2) == len(casar(sobra_x, z2, esp_f)):
                r.update(regra="bate_Q2", x=[z.real for z in z2]); saida.append(r); continue
        _, xp = maximos(E, x, esp_f, prom=PROM_WL, w=Wl[l])
        r.update(regra="proeminencia" if zeros is not None else "sem_polos", x=list(refina_posicoes(E, x, xp)))
        saida.append(r)
    return saida


def derivada_D(l, z, cur, p):
    """F'(z_k) com F = z^p D_l e^{-s(z_k)} (diferenca central), e s(z_k)."""
    F = F_analitica(l, cur, p, z); h = 1e-7 * max(abs(z), 1e-4)
    return (F(z + h) - F(z - h)) / (2 * h), float(escala_s(np.array([z]), cur, l)[0])


def separacao_laurent(polos, cur, p, n=400):
    """Proposta do Gilberto: perto de dois polos vizinhos, 1/D_l ~ A1/(x-z1) + A2/(x-z2), A = 1/D_l'(z). Devolve a
    distancia entre os maximos desse modelo de dois polos (nan se os picos se fundem). Polos do mesmo l somam na
    amplitude; de l diferentes somam na energia. A escala e^{-s} de cada A e' desfeita em relacao ao meio do par."""
    if len(polos) < 2: return np.array([]), np.array([])
    pol = sorted(polos, key=lambda q: q["polo_real"])
    z = np.array([complex(q["polo_real"], q["polo_imag"]) for q in pol]); l = np.array([q["l"] for q in pol])
    A, sA, h = np.empty(len(z), complex), np.empty(len(z)), np.empty(len(z))
    for k, zk in enumerate(z):
        d, sA[k] = derivada_D(int(l[k]), zk, cur, p[int(l[k])]); A[k] = 1 / d
        gam = abs(zk.imag); s_ = max(gam, 1e-6 * zk.real)
        h[k] = energia_l(int(l[k]), np.array([zk.real + s_]), cur)[0] * (s_**2 + gam**2) / gam**2
    z1, z2 = z[:-1], z[1:]; x1, x2 = z1.real, z2.real; sm = 0.5 * (sA[:-1] + sA[1:])
    A1, A2 = A[:-1] * np.exp(sm - sA[:-1]), A[1:] * np.exp(sm - sA[1:])
    G1, G2 = np.abs(z1.imag)[:, None], np.abs(z2.imag)[:, None]
    g1, g2 = np.maximum(G1[:, 0], 1e3 * np.finfo(float).eps * x1), np.maximum(G2[:, 0], 1e3 * np.finfo(float).eps * x2)
    u = np.geomspace(1e-3, 30, n // 4)
    X = np.sort(np.concatenate([x1[:, None] - g1[:, None] * u[::-1], x1[:, None] + g1[:, None] * u, np.linspace(x1, x2, n // 4 + 2, axis=1)[:, 1:-1],
                                x2[:, None] - g2[:, None] * u[::-1], x2[:, None] + g2[:, None] * u], axis=1), axis=1)
    mesmo = (l[:-1] == l[1:])[:, None]
    with np.errstate(all="ignore"):
        W = np.where(mesmo, np.abs(A1[:, None] / (X - z1[:, None]) + A2[:, None] / (X - z2[:, None])) ** 2,
                     h[:-1, None] * G1**2 / ((X - x1[:, None]) ** 2 + G1**2) + h[1:, None] * G2**2 / ((X - x2[:, None]) ** 2 + G2**2))
    mx = (W[:, 1:-1] > W[:, :-2]) & (W[:, 1:-1] >= W[:, 2:]) & np.isfinite(W[:, 1:-1]); sep = np.full(len(z1), np.nan)
    for i in np.flatnonzero(mx.sum(axis=1) >= 2):
        q = X[i, 1:-1][mx[i]]; sep[i] = q.max() - q.min()
    return 0.5 * (x1 + x2), sep


# ---------------------------------------------------------------------------
# Busca de polos (principio do argumento + Beyn + Newton)
# ---------------------------------------------------------------------------
# Caixas de busca: trapezios de lados verticais, caixa = (xa, xb, ya0, yb0, ya1, yb1), com o fundo indo de (xa, ya0) a
# (xb, yb0) e o topo de (xa, ya1) a (xb, yb1). O retangulo e' o caso ya0 = yb0, ya1 = yb1.
def retangulo(xa, xb, ya, yb): return (xa, xb, ya, ya, yb, yb)


def dominio(k):
    """Regiao de busca: x em [x_min, x_max], topo em +1e-3 e fundo na reta y = -k x. Um polo so conta se Q = x/(2 gamma)
    >= 2, isto e', gamma <= x/4; o fundo fica em k = 1/(2 Q_BORDA) = 1/3, abaixo disso com folga. Zeros super
    amortecidos (Q < 1,5), que nunca contariam e no solido com perda podem ser centenas, ficam de fora, e o principio
    do argumento confere exatamente o conjunto que interessa."""
    return (X_MIN, X_MAX, -k * X_MIN, -k * X_MAX, IMAG_MAX, IMAG_MAX)


def faixa(caixa, xa, xb):
    """Pedaco [xa, xb] da caixa (mesmas retas de fundo e de topo)."""
    x0, x1, a0, b0, a1, b1 = caixa; y = lambda u, v, x: u + (v - u) * (x - x0) / (x1 - x0)
    return (xa, xb, y(a0, b0, xa), y(a0, b0, xb), y(a1, b1, xa), y(a1, b1, xb))


def centro(caixa):
    xa, xb, a0, b0, a1, b1 = caixa; return complex(0.5 * (xa + xb), 0.25 * (a0 + b0 + a1 + b1))


def contorno(caixa, n):
    xa, xb, a0, b0, a1, b1 = caixa; t = np.linspace(0, 1, max(int(n), 4), endpoint=False)
    return np.concatenate([xa + (xb - xa) * t + 1j * (a0 + (b0 - a0) * t), xb + 1j * (b0 + (b1 - b0) * t),
                           xb - (xb - xa) * t + 1j * (b1 + (a1 - b1) * t), xa + 1j * (a1 + (a0 - a1) * t)])


def _voltas_de(v):
    """Voltas de v em torno de 0 ao longo do contorno. So vale se o contorno estiver resolvido: se o argumento pular
    mais de ANG_MAX entre dois pontos vizinhos, a contagem pode estar errada por aliasing (e duas resolucoes podem
    concordar no inteiro errado), entao devolve None e o chamador refina."""
    v = np.asarray(v, dtype=complex)
    if not np.all(np.isfinite(v)) or np.any(v == 0): return None
    with np.errstate(all="ignore"): r = np.roll(v, -1) / v
    if not np.all(np.isfinite(r)): return None
    a = np.angle(r)
    return float(np.sum(a) / (2 * np.pi)) if np.max(np.abs(a)) <= ANG_MAX else None


def voltas(F, caixa, n): return _voltas_de(F(contorno(caixa, n)))


def conta_zeros(F, caixa, niveis=NIVEIS_CONTORNO):
    """Numero de zeros na caixa; so aceita quando duas resolucoes seguidas dao o mesmo inteiro."""
    antes = None
    for n in niveis:
        w = voltas(F, caixa, n); k = None if w is None else int(round(w))
        if k is None or k < 0 or abs(w - k) > TOL_INTEIRO: antes = None; continue
        if k == antes: return k
        antes = k
    return None


def auditoria_todos(cur, p, caixas, achados, prazo):
    """Principio do argumento no retangulo inteiro, com 256 a 32768 pontos por lado, todos os l de uma vez (as
    tabelas de Bessel do contorno sao comuns). Passa quando dois niveis seguidos dao o mesmo inteiro e ele bate
    com o numero de zeros achados. Devolve {l: (contagem, ok)}."""
    est = {l: [0, None, False] for l in caixas}
    for n in NIVEIS_AUDITORIA:
        pend = [l for l in caixas if not est[l][2]]
        if not pend or time.time() >= prazo: break
        for cx in sorted({caixas[l] for l in pend}):
            grupo = [l for l in pend if caixas[l] == cx]
            v = D_contorno(contorno(cx, n), cur, grupo, p)
            for k, l in enumerate(grupo):
                w = _voltas_de(v[k]); kk = None if w is None else int(round(w))
                if kk is None or kk < 0 or abs(w - kk) > TOL_INTEIRO: est[l][0] = 0; continue
                est[l][1] = kk; est[l][0] = est[l][0] + 1 if kk == achados[l] else 0
                if est[l][0] >= 2: est[l][2] = True
    return {l: (est[l][1], est[l][2]) for l in caixas}


def dentro(z, caixa, tol=1e-10):
    xa, xb, a0, b0, a1, b1 = caixa
    if not xa - tol <= z.real <= xb + tol: return False
    t = (z.real - xa) / (xb - xa); return a0 + (b0 - a0) * t - tol <= z.imag <= a1 + (b1 - a1) * t + tol


def residuo(F, z):
    v = F(z); h = 1e-7 * max(1.0, abs(z))
    viz = np.asarray([F(z + h), F(z - h), F(z + 1j * h), F(z - 1j * h)], dtype=complex); viz = viz[np.isfinite(viz)]
    return abs(v) / max(np.median(np.abs(viz)), np.finfo(float).tiny) if np.isfinite(v) and len(viz) else np.inf


def polir(l, cur, p, z0, caixa):
    """Newton (secante) em F = z^p D_l e^{-s(z0)}; aceita se ficar no retangulo e |F| < 1e-7 x mediana dos vizinhos."""
    F = F_analitica(l, cur, p[l], z0)
    try: z = complex(newton(F, z0, tol=1e-13, maxiter=150))
    except Exception: return None
    return z if np.isfinite(z) and dentro(z, caixa) and residuo(F, z) <= RESIDUO_MAX else None


def beyn(l, cur, p, caixa, r):
    """Momentos da integral de contorno de F'/F -> autovalores = zeros. So para caixas com 2 a 8 zeros."""
    Lb = r + BEYN_FOLGA; c = centro(caixa); F = F_analitica(l, cur, p[l], c)
    z = contorno(caixa, BEYN_N); fz = F(z)
    if not np.all(np.isfinite(fz)) or np.any(fz == 0): return None
    dlog = (F(z + 1e-6) - F(z - 1e-6)) / 2e-6 / fz; dz = np.roll(z, -1) - z; w = z - c
    m = [np.sum(0.5 * ((w**k) * dlog + np.roll((w**k) * dlog, -1)) * dz) / (2j * np.pi) for k in range(2 * Lb)]
    H0 = np.array([[m[i + j] for j in range(Lb)] for i in range(Lb)]); H1 = np.array([[m[i + j + 1] for j in range(Lb)] for i in range(Lb)])
    try:
        U, S, Vh = la.svd(H0)
        if np.any(S[:r] <= 0): return None
        return la.eigvals(U[:, :r].conj().T @ H1 @ Vh[:r].conj().T @ np.diag(1 / S[:r])) + c
    except Exception:
        return None


def sem_duplicatas(zs):
    out = []
    for z in sorted(zs, key=lambda z: (z.real, z.imag)):
        if not out or min(abs(z - u) for u in out) > TOL_DUPLICATA: out.append(z)
    return out


def zeros_na_caixa(l, cur, p, caixa, prazo, prof=0):
    if time.time() >= prazo: return []
    Fc = lambda z: D_contorno(z, cur, [l], p)[0]
    k = conta_zeros(Fc, caixa); xa, xb, a0, b0, a1, b1 = caixa
    if k == 0: return []
    if k == 1:
        z = polir(l, cur, p, centro(caixa), caixa)
        if z is not None: return [z]
    elif k is not None and k <= BEYN_MAX:
        cand = beyn(l, cur, p, caixa, k)
        if cand is not None and len(cand) == k and all(dentro(c, caixa) for c in cand):
            pol = [polir(l, cur, p, c, caixa) for c in cand]
            if all(q is not None for q in pol) and len(sem_duplicatas(pol)) == k: return sem_duplicatas(pol)
    if prof >= PROF_MAX: return []
    xm, bm, tm = 0.5 * (xa + xb), 0.5 * (a0 + b0), 0.5 * (a1 + b1); ma, mm, mb = 0.5 * (a0 + a1), 0.5 * (bm + tm), 0.5 * (b0 + b1); achados = []
    for sub in ((xa, xm, a0, bm, ma, mm), (xm, xb, bm, b0, mm, mb), (xa, xm, ma, mm, a1, tm), (xm, xb, mm, mb, tm, b1)):
        achados += zeros_na_caixa(l, cur, p, sub, prazo, prof + 1)
    return sem_duplicatas(achados)


def no_eixo(z):
    """Zero com 0 <= Im z < GAMMA_RAD |z|: polo colado no eixo (o sinal de Im e' ruido); vira Im negativo minimo."""
    return complex(z.real, -1e-300) if 0 <= z.imag < GAMMA_RAD * abs(z) else z


def polo_estreito(l, z, cur, p):
    """Polo com gamma/x < GAMMA_RAD: refaz gamma so com valores no eixo real.
    D_l(x0 - i gamma) ~ D_l(x0) - i gamma D_l'(x0) = 0  ->  z0 = x0 - D_l(x0)/D_l'(x0); itera x0 ate Re(D/D') = 0 e
    gamma = Im(D/D'). No core-shell isso exigia D_l = D_j + i D_y com h -> j e h -> y separados, porque so h era
    complexo no eixo. Aqui o material tambem e' complexo e a separacao nao vale, mas tambem nao e' necessaria:
    no eixo real as Bessel internas saem por Taylor com Bessel real (parte imaginaria exata) e o meio externo tem
    j, y reais separados, entao Re D e Im D sao calculados cada um com a precisao do double. Conferido com mpmath."""
    x = z.real
    if -z.imag > GAMMA_RAD * x: return z
    F = F_analitica(l, cur, p, complex(x)); q = complex(np.nan)
    for _ in range(50):
        h = 1e-7 * x; q = complex(F(x) / ((F(x + h) - F(x - h)) / (2 * h))); x -= q.real
        if abs(q.real) < 1e-15 * x: break
    return complex(x, -q.imag) if np.isfinite(q) and q.imag > 0 else z


def malha_rapida(cur):
    """Malha do passo rapido: passo de fase pi/200 (Delta_W/200 com Delta_W local) somada a 2e5 pontos uniformes;
    se passar de 2e6 pontos, o passo de fase aumenta ate caber."""
    xg = np.unique(np.concatenate([np.geomspace(X_MIN, 0.1, 4000), np.linspace(0.1, X_MAX, 16000)]))
    _, Phi = fase(cur, xg); th = np.concatenate([[0.0], np.cumsum(0.5 * (Phi[1:] + Phi[:-1]) * np.diff(xg))])
    n = int(min(N_RAPIDA_MAX - N_RAPIDA_MIN, PASSOS_POR_ESP * th[-1] / np.pi))
    xf = np.interp(np.linspace(0, th[-1], max(n, 2)), th, xg)
    return np.unique(np.concatenate([np.linspace(X_MIN, X_MAX, N_RAPIDA_MIN), xf]))


def buscar_polos(cur, prazo, lmax=LMAX, previstas=None):
    """Zeros de z^p D_l para cada l no retangulo [x_min, x_max] x [y_min, 1e-3].
    1. passo rapido: minimos de |z^p D_l| no eixo real (malha_rapida) e Newton a partir de cada um, com Im inicial
       -1e-12, -1e-9, -1e-6, -1e-3;
    2. auditoria pelo principio do argumento no retangulo inteiro;
    3. se nao fechar, busca completa: faixas com borda nos maximos de |D_l| no eixo, subdivisao ate 1 zero (Newton)
       ou ate 8 (Beyn), junta com o passo 1 e audita de novo.
    No fim, polos com gamma/x < 1e-10 tem gamma refeito pelo eixo real (polo_estreito).
    previstas = {l: (x_k, Q_k) previstos pela recorrencia}: entram como sementes extras do passo 1 (o Newton parte do
    zero complexo x_k - i x_k/(2 Q_k); pegam polo estreito que caia entre dois pontos da malha rapida)."""
    ls = list(range(lmax + 1)); p = expoentes_p(cur, ls); info = {"imag_min": {}, "p": p}
    caixas, pend, k = {}, list(ls), 1 / (2 * Q_BORDA)
    while pend and k * X_MAX > 1e-4:          # D_l zera ou estoura no contorno: sobe o fundo pela metade
        v = D_todos(contorno(dominio(k), 4096), cur, pend, p)
        ok = [l for i, l in enumerate(pend) if np.all(np.isfinite(v[i])) and np.all(v[i] != 0)]
        for l in ok: caixas[l] = dominio(k)
        pend = [l for l in pend if l not in ok]; k *= 0.5
    for l in pend: caixas[l] = dominio(k)
    info["imag_min"] = {l: caixas[l][3] for l in ls}
    xr = malha_rapida(cur); A = np.abs(D_todos(xr, cur, ls, p)); zeros_por_l, polos, achados = {}, [], {}
    filtra = lambda zs, cx: [z for z in sem_duplicatas([no_eixo(z) for z in zs]) if z.imag < 0 and dentro(z, cx, tol=0)]
    for k, l in enumerate(ls):
        with np.errstate(all="ignore"): la_ = -np.log10(np.maximum(A[k], np.finfo(float).tiny))
        sem = xr[find_peaks(la_, prominence=PROM_SEMENTE)[0]]; zs = []
        if previstas is not None and previstas.get(l) is not None:   # zeros previstos: Newton direto do zero complexo
            for xk, qk in zip(*previstas[l]):
                z = polir(l, cur, p, complex(xk, -xk / (2 * qk)), caixas[l]) if xk <= X_MAX else None
                if z is not None: zs.append(z)
        for s0 in sem:
            if time.time() >= prazo: break
            for y0 in (-1e-12, -1e-9, -1e-6, -1e-3):
                z = polir(l, cur, p, complex(s0, y0), caixas[l])
                if z is not None: zs.append(z); break
        if time.time() >= prazo: zeros_por_l[l] = None; info[l] = "sem_tempo"; continue
        achados[l] = filtra(zs, caixas[l]); info[l] = "rapida"
    def audita(l, zs):
        """Auditoria da regiao inteira; se nao bater, tenta de novo com o fundo em Q = 1,75 e Q = 1,9. Um zero colado
        no fundo (Q ~ 1,5) pode escapar da subdivisao sem importar: se a regiao Q >= 1,75 (ou 1,9) fecha, todos os
        polos que podem contar (Q >= 2) ja foram achados."""
        esperado, ok = auditoria_todos(cur, p, {l: caixas[l]}, {l: len(zs)}, prazo)[l]; det[l] = (Q_BORDA, esperado, len(zs))
        for qb in Q_BORDA_EXTRA:
            if ok or time.time() >= prazo: break
            cx = caixas[l][:2] + tuple(max(u, v) for u, v in zip(caixas[l][2:4], dominio(1 / (2 * qb))[2:4])) + caixas[l][4:]
            n_in = sum(dentro(z, cx, 0) for z in zs); esperado, ok = auditoria_todos(cur, p, {l: cx}, {l: n_in}, prazo)[l]; det[l] = (qb, esperado, n_in)
        return esperado, ok
    vivos = {l: caixas[l] for l in ls if info.get(l) == "rapida"}; det = {}
    aud = auditoria_todos(cur, p, vivos, {l: len(achados[l]) for l in vivos}, prazo) if vivos else {}
    for l in vivos: det[l] = (Q_BORDA, aud[l][0], len(achados[l]))
    for l in vivos:
        esperado, ok = aud[l]; zs, via = achados[l], "rapida"
        if not ok: esperado, ok = audita(l, zs)
        if not ok:
            if time.time() >= prazo: zeros_por_l[l] = None; info[l] = "sem_tempo"; continue
            a = A[ls.index(l)]; topo = np.flatnonzero((a[1:-1] > a[:-2]) & (a[1:-1] > a[2:])) + 1
            bordas = np.unique(np.r_[X_MIN, xr[topo], X_MAX]); mais = []
            for xa, xb in zip(bordas[:-1], bordas[1:]):
                mais += zeros_na_caixa(l, cur, p, faixa(caixas[l], xa, xb), prazo)
                if time.time() >= prazo: break
            if time.time() >= prazo: zeros_por_l[l] = None; info[l] = "sem_tempo"; continue
            zs = filtra(zs + mais, caixas[l]); esperado, ok = audita(l, zs); via = "completa"
        zs = [polo_estreito(l, z, cur, p[l]) for z in zs]
        info[l] = "ok" if ok else f"auditoria_falhou ({esperado} esperados, {len(zs)} achados)"
        zeros_por_l[l] = zs if ok else None
        polos += [{"l": l, "polo_real": z.real, "polo_imag": z.imag, "Gamma_polo": 2 * abs(z.imag), "Q_polo": z.real / (2 * abs(z.imag)),
                   "contagem_argumento_l": esperado, "auditoria_ok": ok, "busca": via, "pico_na_malha": False, "xM_pico_malha": np.nan} for z in zs]
    info["detalhe"] = {l: {"Q_fundo": v[0], "contagem_argumento": v[1], "achados_na_regiao": v[2]} for l, v in det.items()}
    return polos, zeros_por_l, info, p


# ---------------------------------------------------------------------------
# Peso, fundo e fracao de cada polo
# ---------------------------------------------------------------------------
def peso_polo(l, z, cur, p):
    """Perto do polo z_k = x_k - i gamma: a_l ~ R_a/(x - z_k), b_l ~ R_b/(x - z_k), R = N(z_k)/D_l'(z_k). Entao
        W_l(x) ~ S/((x - x_k)^2 + gamma^2),   S = Re[R^H M(x_k) R] (Z_S/Z_M),
    com M a forma da energia de campos (ja tem o fator 2l+1) avaliada no eixo em x_k. Altura S/gamma^2, FWHM = 2 gamma,
    area pi S/gamma. As escalas e^{-|Im z|} sao refeitas para x_k: R_a leva e^{|Im z_L(x_k)| - |Im z_L(z_k)|} e R_b o
    analogo com z_T, para casar com M(x_k).
    Validade: a lorentziana supoe que o resto (N, M, as ondas internas) quase nao muda entre z_k e x_k. Num solido
    lento, a onda interna muda de amplitude como e^{|Im z_T|}, e quando |Im z_j(x_k)| - |Im z_j(z_k)| passa de
    DELTA_ESCALA_MAX (gamma maior que ~1/Phi, isto e', largura maior que o espacamento entre modos vizinhos do mesmo l)
    o 'pico' do polo nao existe como lorentziana propria: S sai NaN e o polo nao entra no modelo nem na contagem.
    Devolve (S, delta)."""
    z = complex(z); x = np.array([z.real]); F = F_analitica(l, cur, p, z); h = 1e-7 * max(abs(z), 1e-4)
    with np.errstate(all="ignore"):
        dF = (F(z + h) - F(z - h)) / (2 * h) / z**p          # derivada de D_l e^{-s(z_k)} (o z^p sai porque F(z_k) = 0)
        rz = campos(np.array([z]), cur, ls=[l], lmax=l); rx = campos(x, cur, ls=[l], energia=True, lmax=l)
        Ra = rz[l]["Na"][0] / dF * np.exp(rx["sL"][0] - rz["sL"][0]); Rb = rz[l]["Nb"][0] / dF * np.exp(rx["sT"][0] - rz["sT"][0])
        M = [m[0] for m in rx[l]["M"]]; delta = float(max(abs(rx["sL"][0] - rz["sL"][0]), abs(rx["sT"][0] - rz["sT"][0])))
        S = np.real(np.conj(Ra) * (M[0] * Ra + M[1] * Rb) + np.conj(Rb) * (M[2] * Ra + M[3] * Rb)) * peso_Z(rx["cL"], cur)[0]
    return (float(S) if delta <= DELTA_ESCALA_MAX else np.nan), delta


def medir_polos(polos, cur, p, x, w, esp_f):
    """S, area pi S/gamma, altura S/gamma^2, fundo (mediana de W_T da malha base na janela de largura Delta_W em volta
    do polo) e fracao = area / (soma das areas dos polos na janela, com o arctan truncado, + fundo x largura)."""
    if not polos: return
    for q in polos:
        z = complex(q["polo_real"], q["polo_imag"]); g = -z.imag; S, dlt = peso_polo(q["l"], z, cur, p[q["l"]])
        q.update(S_polo=S, area_polo=np.pi * S / g, altura_polo=S / g**2, delta_escala=dlt)
    ordem = np.argsort([q["polo_real"] for q in polos]); X = np.array([polos[i]["polo_real"] for i in ordem])
    G = np.array([-polos[i]["polo_imag"] for i in ordem]); A = np.nan_to_num(np.maximum(np.array([polos[i]["area_polo"] for i in ordem]), 0))
    for q in polos:
        xk = q["polo_real"]; e = float(esp_f(xk)); lo, hi = max(xk - 0.5 * e, X_MIN), min(xk + 0.5 * e, X_MAX)
        k = (x >= lo) & (x <= hi) & np.isfinite(w); fundo = float(np.median(w[k])) if k.sum() >= 3 else float(np.interp(xk, x, w))
        i0, i1 = np.searchsorted(X, xk - 2 * e), np.searchsorted(X, xk + 2 * e)
        Ij = np.sum(A[i0:i1] * (np.arctan((hi - X[i0:i1]) / G[i0:i1]) - np.arctan((lo - X[i0:i1]) / G[i0:i1])) / np.pi)
        I_jan = Ij + max(fundo, 0) * (hi - lo)
        q.update(fundo_polo=fundo, frac_area_local=float(q["area_polo"] / I_jan) if I_jan > 0 else np.nan)
    proeminencia_analitica(polos)


def proeminencia_analitica(polos, n_theta=41, bloco=100):
    """Proeminencia de cada polo no modelo analitico da curva, sem malha:
        W_mod(x) = fundo(x) + soma_k S_k/((x - x_k)^2 + gamma_k^2),
    com fundo(x) interpolado entre os fundos dos polos. W_mod e' avaliado numa malha adaptada aos polos
    (x_k + gamma_k tan(theta)) e os seus maximos com proeminencia >= 0,1 em log10 sao os picos que uma malha
    infinitamente fina veria. Cada maximo vai para o polo que mais contribui nele; polos que nao viram maximo
    (escondidos sob um vizinho largo, ou fundidos com um vizinho) ficam com proeminencia 0. O nivel de base do pico
    (topo/10^proeminencia) entra no dx_nec no lugar do fundo, o que leva em conta os vizinhos."""
    P = [q for q in polos if np.isfinite(q.get("S_polo", np.nan)) and q["S_polo"] > 0 and np.isfinite(q.get("fundo_polo", np.nan))]
    for q in polos: q.update(prom_log10_polo=0.0, base_pico=np.nan)
    if not P: return
    P.sort(key=lambda q: q["polo_real"]); X = np.array([q["polo_real"] for q in P]); g = np.array([-q["polo_imag"] for q in P])
    S = np.array([q["S_polo"] for q in P]); F = np.maximum(np.array([q["fundo_polo"] for q in P]), np.finfo(float).tiny)
    th = np.linspace(-np.pi / 2 + 0.02, np.pi / 2 - 0.02, n_theta)
    grade = np.unique(np.clip(np.concatenate([np.linspace(X_MIN, X_MAX, 20001), (X[:, None] + g[:, None] * np.tan(th)[None, :]).ravel()]), X_MIN, X_MAX))
    W = np.exp(np.interp(grade, X, np.log(F)))
    for i in range(0, len(P), bloco): W = W + np.sum(S[i:i + bloco, None] / ((grade[None, :] - X[i:i + bloco, None]) ** 2 + g[i:i + bloco, None] ** 2), axis=0)
    idx, pr = find_peaks(np.log10(W), prominence=PROM_WT)
    for i, prom in zip(idx, pr["prominences"]):
        j = int(np.argmax(S / ((grade[i] - X) ** 2 + g**2)))
        if prom > P[j]["prom_log10_polo"]: P[j].update(prom_log10_polo=float(prom), base_pico=float(W[i] / 10**prom))


VAZIO_POLO = {"l": None, "polo_real": np.nan, "polo_imag": np.nan, "Gamma_polo": np.nan, "Q_polo": np.nan, "l_associados": "", "N_polos_associados": 0}

def associa_polos(res, polos, esp_f):
    """Pendura em cada pico os polos que caem dentro dele (o mais proximo vira o principal)."""
    P = np.array([q["polo_real"] for q in polos]) if polos else np.array([])
    for r in res:
        tol = max(r["largura"] if np.isfinite(r["largura"]) else 0, 0.25 * float(esp_f(r["xM_pico"])))
        perto = [polos[j] for j in np.flatnonzero(np.abs(P - r["xM_pico"]) <= tol)] if len(P) else []
        if not perto: r.update(VAZIO_POLO); continue
        q = min(perto, key=lambda u: abs(u["polo_real"] - r["xM_pico"]))
        r.update(l=q["l"], polo_real=q["polo_real"], polo_imag=q["polo_imag"], Gamma_polo=q["Gamma_polo"], Q_polo=q["Q_polo"],
                 l_associados=",".join(str(u) for u in sorted({u["l"] for u in perto})), N_polos_associados=len(perto))
        for u in perto: u["pico_na_malha"] = True; u["xM_pico_malha"] = r["xM_pico"]


def ressonancia_de_polo(q):
    """Ressonancia lida direto do polo: x_k, FWHM = 2 gamma, Q = x_k/(2 gamma), altura e area analiticas."""
    x, g, fundo = q["polo_real"], -q["polo_imag"], q["fundo_polo"]; topo = q["altura_polo"] + fundo
    return {"xM_pico": x, "altura": topo, "fundo": fundo, "razao_pico_fundo": topo / fundo if fundo > 0 else np.inf,
            "proeminencia_log10": q["prom_log10_polo"], "largura": 2 * g, "Q": x / (2 * g), "area": q["area_polo"],
            "frac_area_local": q["frac_area_local"], "origem": "polo", "l": q["l"], "polo_real": x, "polo_imag": q["polo_imag"],
            "Gamma_polo": 2 * g, "Q_polo": x / (2 * g), "l_associados": str(q["l"]), "N_polos_associados": 1}


K_VIS = 10**PROM_WT - 1

def passo_necessario(q):
    """Maior passo uniforme que garante ver o pico: com o ponto mais proximo a dx/2 do topo,
        S/(gamma^2 + (dx/2)^2) >= k fundo  ->  dx_nec = 2 sqrt(S/(k fundo) - gamma^2),  k = 10^0.1 - 1,
    com 'fundo' = nivel de base do pico no modelo analitico (proeminencia_analitica); para pico isolado e' o fundo
    da janela. Polo que nao vira pico no modelo nao tem dx_nec."""
    S, fu, g = q.get("S_polo", np.nan), q.get("base_pico", np.nan), abs(q["polo_imag"])
    if not (np.isfinite(S) and np.isfinite(fu) and S > 0 and fu > 0): return np.nan
    return float(2 * np.sqrt(max(S / (K_VIS * fu) - g * g, 0.0)))


def malha_dx(x, alvos, esp_f, laurent=None, x_max=X_MAX):
    """Malha base + um trecho uniforme em volta de cada grupo de picos (vizinhos a menos de Delta_W/2), com o passo do
    pico mais exigente (ou metade da separacao de Laurent, se menor). Os pontos comecam na borda do trecho, nunca em
    cima do pico. Se nao couber em N_MAX_PONTOS, sobe um piso de passo (bissecao em log) ate caber."""
    A = sorted([a for a in alvos if np.isfinite(a[1]) and a[1] > 0], key=lambda a: a[0])
    if not A: return x, np.nan
    P = np.array([a[0] for a in A]); Dn = np.array([a[1] for a in A]); ext = min(float(esp_f(x_max)), 0.05 * x_max); E = esp_f(P); seg = []
    for gi in np.split(np.arange(len(P)), np.flatnonzero(np.diff(P) > 0.5 * E[:-1]) + 1):
        lo, hi = max(P[gi[0]] - 0.25 * E[gi[0]], X_MIN), min(P[gi[-1]] + 0.25 * E[gi[-1]], x_max + ext); d = float(Dn[gi].min())
        if laurent is not None and len(laurent[0]):
            k = (laurent[0] >= lo) & (laurent[0] <= hi) & np.isfinite(laurent[1]) & (laurent[1] > 0)
            if k.any(): d = min(d, 0.5 * float(laurent[1][k].min()))
        seg.append((lo, hi, d, gi))
    livre = max(N_MAX_PONTOS - len(x), 1000); custo = lambda piso: sum((hi - lo) / max(d, piso) for lo, hi, d, _ in seg); piso = DX_MIN
    if custo(piso) > livre:
        a, b = DX_MIN, max(hi - lo for lo, hi, _, _ in seg)
        for _ in range(60):
            m = np.sqrt(a * b); a, b = (m, b) if custo(m) > livre else (a, m)
        piso = b
    extras = []
    for lo, hi, d, gi in seg:
        h = max(d, piso); extras.append(lo + h * np.arange(int((hi - lo) / h) + 1))
        for i in gi:
            if A[i][2] is not None: A[i][2]["dx_malha_usado"] = h
    return np.unique(np.concatenate([x] + extras)), float(piso)


# ---------------------------------------------------------------------------
# Uma curva
# ---------------------------------------------------------------------------
def lmax_efetivo(Wl, razao_minima=1e-3):
    """Mesmo criterio do codigo antigo: maior l cuja contribuicao maxima passa de 1e-3 da maior contribuicao."""
    m = np.max(np.abs(Wl), axis=1); g = m.max() if m.max() > 0 else 1.0; ok = np.flatnonzero(m / g > razao_minima)
    return int(ok.max()) if len(ok) else 0


def polo_conta(q, frac_area=FRAC_AREA):
    """Criterio da contagem final pelos polos: Q >= 2 e (visivel no modelo analitico, proeminencia >= 0,1, ou fracao
    de area >= frac_area). Polo sem peso S valido (delta_escala > 1) fica com area NaN e proeminencia 0: nao conta."""
    fa, pr = q.get("frac_area_local", np.nan), q.get("prom_log10_polo", np.nan)
    return bool(q["Q_polo"] >= Q_MIN_POLO and ((np.isfinite(fa) and fa >= frac_area) or (np.isfinite(pr) and pr >= PROM_WT)))


def contagem_polos(saida, frac_area=FRAC_AREA):
    """Contagem final pelos polos (busca fechada em todos os l): marca contado_area/sobreposto nos candidatos, preenche
    N_res_polos_area/N_res_polos_picos e devolve a lista de ressonancias. Usada tambem por --recontar."""
    conta = [q for q in sorted(saida["candidatos"], key=lambda u: u["polo_real"]) if polo_conta(q, frac_area)]
    ids = {id(q) for q in conta}
    for q in saida["candidatos"]: q["contado_area"] = id(q) in ids
    Xc, Gc = np.array([q["polo_real"] for q in conta]), np.array([-q["polo_imag"] for q in conta])
    junta = np.diff(Xc) < Gc[1:] + Gc[:-1] if len(conta) > 1 else np.array([], bool)
    for i, q in enumerate(conta): q["sobreposto"] = bool((i > 0 and junta[i - 1]) or (i < len(junta) and junta[i]))
    saida.update(N_res_polos_area=len(conta), N_res_polos_picos=int(len(conta) - junta.sum()), area_polos_total=float(sum(q["area_polo"] for q in saida["candidatos"])))
    return [ressonancia_de_polo(q) for q in conta]


def processa_curva(cur, prazo_polos, forcar_polos=False, frac_area=FRAC_AREA, n_zeros_max=N_ZEROS_MAX):
    t0 = time.time(); x, esp_f = malha_base(cur); WT = lambda v: energia(v, cur)
    w, Wl = energia(x, cur, por_l=True); lmx = lmax_efetivo(Wl)
    risco, n_prev, prev, pinfo = curva_de_risco(cur); risco_rec = risco; risco = risco or forcar_polos
    qprev = pinfo.get("Q", {l: np.full(len(v), np.nan) for l, v in prev.items() if v is not None})
    _, xs_base = maximos(WT, x, esp_f, prom=PROM_WT, w=w)
    saida = {"risco": risco, "risco_recorrencia": risco_rec, "risco_fase_v1": risco_fase(cur), "N_previsto_recorrencia": n_prev,
             "N_previsto_ate_xmax": int(sum(np.sum(np.asarray(v) <= X_MAX) for v in prev.values() if v is not None)) if not pinfo["aproximado"] else None,
             "previsao_aproximada": pinfo["aproximado"], "previstas": {l: (None if v is None else np.asarray(v).tolist()) for l, v in prev.items()},
             "Q_previstas": {l: np.asarray(v).tolist() for l, v in qprev.items()},
             "candidatos": [], "busca_polos": "nao_feita", "N_res_malha_base": len(xs_base), "N_res_malha_refinada": None,
             "N_modos": None, "pontos_malha": len(x), "dx_min": None, "l_max_efetivo": lmx, "N_zeros_estimado": estimativa_zeros(cur)}
    if not risco:
        dx = np.interp(xs_base, x[1:], np.diff(x)) if len(xs_base) else None
        res = medir_picos(WT, x, xs_base, esp_f, escala=dx)
        for r in res: r["origem"] = "malha_base"
        associa_polos(res, [], esp_f)
        saida.update(modo="malha_base", ressonancias=res, ressonancias_malha=res, runtime_s=time.time() - t0)
        return saida

    # curva de risco: polos, modos por l, malha refinada
    if saida["N_zeros_estimado"] > n_zeros_max:
        zeros_por_l, p = {l: None for l in range(LMAX + 1)}, None; saida["busca_polos"] = "pulada_densidade"
    else:
        sementes = {l: (np.asarray(prev[l]), np.asarray(qprev[l])) for l in prev if prev[l] is not None and l in qprev and np.all(np.isfinite(qprev[l]))}
        polos, zeros_por_l, info, p = buscar_polos(cur, time.time() + prazo_polos, previstas=sementes)
        imag_min = min(info.pop("imag_min").values(), default=np.nan); info.pop("p", None); det = info.pop("detalhe", {})
        saida.update(candidatos=polos, auditoria=info, auditoria_detalhe=det, imag_min_busca=imag_min, p_por_l=p,
                     busca_polos="incompleta" if "sem_tempo" in info.values() else "completa")
    modos = modos_por_l(cur, zeros_por_l, x, Wl, esp_f)
    bons = [q for q in saida["candidatos"] if q["auditoria_ok"]]
    if saida["candidatos"]: medir_polos(saida["candidatos"], cur, p, x, w, esp_f)
    vis = [q for q in bons if q["Q_polo"] >= Q_MIN_POLO and np.isfinite(q.get("prom_log10_polo", np.nan)) and q["prom_log10_polo"] >= PROM_WT]
    laurent = separacao_laurent(vis, cur, p) if len(vis) >= 2 else None
    if laurent is not None:
        d_polos = np.diff(np.sort([q["polo_real"] for q in vis]))
        saida.update(N_pares_fundidos_laurent=int(np.sum(np.isnan(laurent[1]))), dx_min_polos=float(0.5 * d_polos.min()),
                     dx_min_laurent=float(0.5 * np.nanmin(laurent[1])) if np.isfinite(laurent[1]).any() else None)
    for q in saida["candidatos"]: q["dx_nec"] = passo_necessario(q) if "S_polo" in q else np.nan
    alvos = [(q["polo_real"], q["dx_nec"], q) for q in vis]
    for m in modos:                                   # l sem polos confiaveis (auditoria falhou ou busca pulada): modos da malha
        if m.get("N_zeros") is None:
            P = np.sort(np.asarray(m["x"], float)); viz = np.minimum(np.r_[np.inf, np.diff(P)], np.r_[np.diff(P), np.inf]) if len(P) else []
            alvos += [(v, 0.5 * min(d, 0.25 * float(esp_f(v))), None) for v, d in zip(P, viz)]
    xr, piso = malha_dx(x, alvos, esp_f, laurent); coube = not (np.isfinite(piso) and piso > DX_MIN)
    # l sem polos confiaveis: refino guiado pela recorrencia. Cada zero previsto (x_k, Q_k) e' um polo de D_l com posicao
    # e largura conhecidas, entao basta por pontos em x_k + gamma_k {0, +-1/2, +-1, +-2, +-4, +-8} (gamma = x/(2Q)):
    # o topo e os dois lados do pico ficam amostrados qualquer que seja gamma, sem malha uniforme fina em volta.
    extra = [xk + xk / (2 * qk) * np.r_[0, 0.5, -0.5, 1, -1, 2, -2, 4, -4, 8, -8] for m in modos if m.get("N_zeros") is None
             for xk, qk in zip(prev.get(m["l"]) if prev.get(m["l"]) is not None else [], qprev.get(m["l"], [])) if xk <= X_MAX and np.isfinite(qk)]
    if extra: xr = np.unique(np.concatenate([xr] + extra)); xr = xr[xr >= X_MIN]
    saida["pontos_recorrencia"] = int(sum(len(e) for e in extra)); dx_min = float(np.min(np.diff(xr)))
    dn = np.array([q["dx_nec"] for q in vis if np.isfinite(q["dx_nec"])])
    saida.update(N_polos_visiveis=len(vis), dx_necessario=float(dn.min()) if len(dn) else None, dx_90=float(np.percentile(dn, 10)) if len(dn) else None,
                 dx_piso_malha=piso if np.isfinite(piso) else None,
                 N_previsto_malha=int(sum(q.get("dx_malha_usado", np.inf) <= q["dx_nec"] for q in vis if np.isfinite(q["dx_nec"]))))
    wr = WT(xr); _, xs = maximos(WT, xr, esp_f, prom=PROM_WT if coube else None, w=wr)
    escala = np.interp(xs, xr[1:], np.diff(xr)) if len(xs) else np.array([])
    if saida["candidatos"] and len(xs):
        P = np.array([q["polo_real"] for q in saida["candidatos"]]); G = np.array([q["Gamma_polo"] for q in saida["candidatos"]])
        j = np.argmin(np.abs(P[None, :] - xs[:, None]), axis=1); perto = np.abs(P[j] - xs) <= 0.25 * esp_f(xs)
        escala = np.where(perto, G[j], escala)
    res = medir_picos(WT, xr, xs, esp_f, escala=escala)
    for r in res: r["origem"] = "malha_refinada" if coube else "malha_limite"
    associa_polos(res, saida["candidatos"], esp_f)
    modo = "malha_refinada" if coube else "malha_limite"; saida["ressonancias_malha"] = list(res)

    # contagem final pelos polos, quando a busca fechou em todos os l
    completa = saida["busca_polos"] == "completa" and all(v == "ok" for v in saida.get("auditoria", {}).values())
    if completa: res, modo = contagem_polos(saida, frac_area), "polos_area"
    saida.update(modo=modo, ressonancias=res, N_res_malha_refinada=len(xs), N_modos=sum(len(m["x"]) for m in modos),
                 N_polos_estreitos=sum(-q["polo_imag"] < GAMMA_RAD * q["polo_real"] for q in saida["candidatos"]),
                 regras_l={m["l"]: m["regra"] for m in modos}, pontos_malha=len(xr), dx_min=dx_min, runtime_s=time.time() - t0)
    return saida


# ---------------------------------------------------------------------------
# Sorteio das curvas (hipercubo, como nos datasets antigos, mas reprodutivel por indice)
# ---------------------------------------------------------------------------
def sorteia(modelo, indice, seed, faixas):
    """alpha = rho_M/rho_S e beta = c_M/c_L0 log-uniformes, c_T0 uniforme em [1e-3, 0.999 (sqrt3/2) c_L0] (variavel
    genuina, como antes) e tau log-uniforme. Cada indice tem o seu gerador, entao a curva i e' sempre a mesma e uma
    rodada nova so acrescenta indices."""
    rng = np.random.default_rng([seed, indice, {"voigt": 0, "maxwell": 1, "elastico": 2}[modelo]]); F = faixas
    alpha = 10 ** rng.uniform(*np.log10(F["alpha"])); beta = 10 ** rng.uniform(*np.log10(F["beta"]))
    cL = CM_PADRAO / beta; cT = rng.uniform(CT_MIN, LIMITE_FISICO * 0.999 * cL)
    tau = 10 ** rng.uniform(*np.log10(F["tau"])) if "tau" in F else 0.0     # elastico: sem perda
    return {"modelo": modelo, "alpha": alpha, "beta": beta, "cL": cL, "cT": cT, "rhoS": RHOM_PADRAO / alpha, "tau": tau}


def metadados(indice, cur):
    """Colunas antigas do dataset (mesmo nome e significado) + tau_adim. tanL_ref, tanT_ref: tan d de cada canal em
    x = xM_ref = 0.5 (tan d = -Im(c^2)/Re(c^2)); na tese a perda vem toda de tau_adim."""
    tL, tT = tan_perdas(np.array([XM_REF]), cur)
    return {"_indice": indice, "modelo": cur["modelo"], "alpha": cur["alpha"], "beta": cur["beta"], "eta": cur["cL"] / cur["cT"],
            "alpha_vezes_beta": cur["alpha"] * cur["beta"], "cL0": cur["cL"], "cT0": cur["cT"], "cL": cur["cL"], "cT": cur["cT"], "rhoS": cur["rhoS"],
            "tau_adim": cur["tau"], "tanL_ref": tL, "tanT_ref": tT, "xM_ref": XM_REF, "cM": CM_PADRAO, "rhoM": RHOM_PADRAO}


class Estourou(Exception): pass

def _alarme(*_): raise Estourou()


def trabalho(args):
    """Roda uma curva dentro de um limite duro de tempo (SIGALRM). Nunca derruba o processo."""
    chave, cur, prazo_polos, limite, frac, norm, forcar, nzmax, rec = args
    global NORMALIZACAO, RECORRENCIA; NORMALIZACAO, RECORRENCIA = norm, rec
    signal.signal(signal.SIGALRM, _alarme); signal.setitimer(signal.ITIMER_REAL, limite)
    try:
        r = processa_curva(cur, prazo_polos, forcar_polos=forcar, frac_area=frac, n_zeros_max=nzmax); r["status"] = "ok"
    except Estourou:
        r = {"status": "timeout"}
    except Exception as e:
        r = {"status": "erro", "erro": f"{type(e).__name__}: {e}"}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    r["cur"] = cur
    return chave, r


# ---------------------------------------------------------------------------
# Checkpoint e CSV
# ---------------------------------------------------------------------------
def caminhos(raiz, modelo):
    nome = f"viscoelastico_lmax{LMAX}_{modelo}"; pasta = os.path.join(os.path.abspath(raiz), nome); os.makedirs(pasta, exist_ok=True)
    j = lambda s: os.path.join(pasta, s)
    return {"ckpt": j(f"checkpoint_{nome}.npz"), "resumo": j(f"resumo_{nome}.csv"), "res": j(f"ressonancias_{nome}.csv"),
            "cand": j(f"candidatos_{nome}.csv"), "res_malha": j(f"ressonancias_malha_{nome}.csv"), "leiame": j("LEIAME.txt"),
            "param": j(f"parametros_{nome}.json"), "hiper": j(f"hipercubo_{modelo}.csv")}


def salva(cam, curvas, modelo, seed):
    tmp = cam["ckpt"] + f".tmp_{uuid.uuid4().hex}.npz"
    np.savez_compressed(tmp, versao=VERSAO, modelo=modelo, seed=seed, lmax=LMAX, curves=np.array(curvas, dtype=object))
    os.replace(tmp, cam["ckpt"])


def carrega(cam, modelo, seed):
    if not os.path.exists(cam["ckpt"]): return {}
    with np.load(cam["ckpt"], allow_pickle=True) as d:
        if int(d["versao"]) != VERSAO or str(d["modelo"]) != modelo or int(d["seed"]) != seed:
            print("Checkpoint de outra versao/configuracao, comecando do zero."); return {}
        return d["curves"].item()


LEIAME = """Gerador da esfera solida ({modelo}), versao {versao}, metodologia do gerador core-shell v8.

Arquivos desta pasta
  checkpoint_*.npz          tudo o que foi calculado, curva a curva (dict 'curves'); os CSVs saem dele
  parametros_*.json         constantes, faixas do sorteio e opcoes de linha de comando desta rodada
  resumo_*.csv              uma linha por curva
  ressonancias_*.csv        uma linha por ressonancia da contagem final (N_res)
  ressonancias_malha_*.csv  uma linha por pico achado na malha (base nas curvas normais, refinada nas de risco),
                            mesmo quando a contagem final vem dos polos
  candidatos_*.csv          uma linha por polo (zero de D_l) achado na busca
  hipercubo_{modelo}.csv/.npz  formato antigo (hipercubo_voigt.csv / hipercubo_maxwell.csv): uma linha por pico da
                            contagem final, e uma linha com campos de pico em NaN para curva sem ressonancia

Fisica (secao 'Esfera Viscoelastica' da tese)
  Elastico: c_L, c_T reais e constantes (dataset8; cL, cT = cL0, cT0), tau_adim = 0, tanL_ref = tanT_ref = 0.
  So o cisalhamento dissipa (K real). tau = tau_adim = tau_T c_M/a.
  Voigt:   tan d_T = tau x, tan d_L = 4/(3 eta0^2) tan d_T, c_j = c_j0 sqrt(1 - i tan d_j).
  Maxwell: tan d_T = 1/(tau x), c_T = c_Tinf/sqrt(1 + i tan d_T), c_L = c_Linf sqrt(1 - 4/(3 eta^2) [1 - 1/(1 + i tan d_T)]).
  cL0, cT0, eta das tabelas sao as velocidades de referencia de cada modelo (c_j0 no Voigt, c_jinf no Maxwell).
  Coeficientes, denominador e energia: os mesmos do voigt.py/maxwell.py; D_l foi escrito sem divisoes
  (a_l = i rho_M c_M^2 x den_R/D_l, b_l = i rho_M c_M^2 x num_R/D_l) e as Bessel internas entram escaladas por
  e^{{-|Im z|}}, o que elimina os estouros do codigo antigo sem mudar nenhum resultado.
  Energia: W_T = (Z_S/Z_M) W~_S, Z_S = rho_S |c_L(x)| (normalizacao '{norm}'; 'separada' = W~_S das versoes antigas).
  Delta_W(x) = pi/Phi(x), Phi = max(d Re z_L/dx, d Re z_T/dx) entre as ondas que propagam (Q da onda >= 1,5):
  espacamento local entre modos de um mesmo l ('esp'); sem perda, pi c_T/c_M (Eq. deltaW_elastica da tese).

Como cada curva e' contada
  Curva normal (versao 2): a condicao de recorrencia da tese, kappa_l E_l + (rho_M/rho_S) z_L^2 G_l = 0 com
    kappa_l = -x h'/h e as grandezas complexas, nao tem zero com Q >= 1 em x <= 1,1 (zeros por Newton complexo a
    partir dos minimos de |f/f'| no eixo real; sao os polos de a_l).
    Resposta = maximos de W_T na malha base (5000 pontos) com proeminencia >= 0,1 em log10.
    (Versao 1: nenhuma onda que propaga mais lenta que o meio; fica na coluna risco_fase_v1.)
  Curva de risco:
    1. zeros de z^p D_l, l = 0..10, na regiao x em [1e-4, 1], -x/3 <= y <= 1e-3 (Q >= 1,5; os polos que contam tem
       Q >= 2) por Newton (sementes: minimos de |D_l| e zeros previstos) + principio do argumento (com a rotacao de
       fase das ondas muito amortecidas desfeita, ver D_contorno), com busca completa por subdivisao/Beyn se nao
       fechar; D_0 sem o fator espurio T_0;
       gamma/x < 1e-10: gamma refeito pelo eixo real;
       se a estimativa de zeros passar de {nzmax}, a busca e' pulada (busca_polos = pulada_densidade);
    2. malha refinada com passo dx_nec = 2 sqrt(S/(k base) - gamma^2), k = 10^0,1 - 1 (base = nivel sob o pico no
       modelo analitico, que para pico isolado e' o fundo da janela), por grupo de picos
       (ou metade da separacao de Laurent); teto de {nmax} pontos (acima disso sobe um piso de passo);
    2b. l sem polos confiaveis (busca pulada ou auditoria falhou): pontos extras em x_k + gamma_k {{0, +-1/2, ..., +-8}}
       em volta de cada zero previsto pela recorrencia (pontos_recorrencia);
    3. se a busca fechou em todos os l: ressonancias = polos com Q >= {qmin} que sao visiveis no modelo analitico
       (proeminencia >= 0,1) ou tem area >= {frac} da energia da janela Delta_W. Senao: picos da malha refinada.

resumo
  colunas antigas: _indice, alpha, beta, eta, alpha_vezes_beta, cL0, cT0, rhoS, tanL_ref, tanT_ref, xM_ref, cM, rhoM,
                   status, runtime_s, erro, N_res (contagem final; NaN se a curva nao terminou)
  modelo, tau_adim        modelo e tempo de relaxacao adimensional da tese (log-uniforme em [1e-4, 1e3] nos dois)
  curva_risco, modo_contagem   malha_base | malha_refinada | malha_limite | polos_area
  risco_recorrencia       criterio da versao 2 (decide a curva_risco, salvo --forcar-polos)
  risco_fase_v1           criterio da versao 1, so para comparacao
  N_previsto_recorrencia  zeros da condicao de recorrencia com Q >= 1 em x <= 1,1, somando os l (se
                          previsao_aproximada: 11 x fase acumulada/pi, solido lento demais para listar os zeros)
  N_previsto_ate_xmax     idem em x <= 1
  pontos_recorrencia      pontos extras da malha em volta dos zeros previstos (l sem polos confiaveis)
  N_res_malha_base        maximos de W_T na malha base
  N_res_malha_refinada    maximos de W_T na malha refinada
  N_res_polos_area        polos contados (um por modo)
  N_res_polos_picos       idem, juntando polos sobrepostos (|x1 - x2| < gamma1 + gamma2)
  N_polos_visiveis        polos com Q >= 2 e proeminencia analitica >= 0,1 (maximos do modelo fundo + soma das
                          lorentzianas de todos os polos: o que uma malha infinitamente fina veria)
  N_previsto_malha        quantos desses o passo usado deveria ver (dx_malha_usado <= dx_nec)
  frac_malha_refinada     N_res_malha_refinada / N_polos_visiveis
  dx_necessario, dx_90    passo para ver todos / 90% dos polos visiveis
  dx_piso_malha           piso de passo da malha refinada (> 1e-9 quando nao coube)
  N_polos_estreitos       polos com gamma/x < 1e-10 (gamma refeito pelo eixo real)
  N_zeros_estimado        estimativa de zeros usada para decidir se a busca roda
  imag_min_busca          fundo da regiao de busca em x = 1 (-1/3, ou mais raso se precisou subir)
  busca_polos, audit_all_ok    se a busca de polos fechou (nao_feita | completa | incompleta | pulada_densidade)
  l_max_efetivo           maior l com contribuicao > 1e-3 da maior (mesmo criterio antigo)
  runtime_s               tempo da curva

ressonancias e ressonancias_malha
  colunas antigas: xM_pico (posicao), altura (W_T no pico), largura (FWHM), Q (= x/FWHM), l, l_max_efetivo,
                   delta_x (distancia ao pico seguinte), origem (polo | malha_base | malha_refinada | malha_limite)
  fundo, razao_pico_fundo, proeminencia_log10
  area (pi S/gamma, so origem polo), frac_area_local (area / energia integrada na janela Delta_W)
  polo_real, polo_imag, Gamma_polo, Q_polo, l_associados, N_polos_associados: polo associado (o mais proximo)
  + metadados da curva

candidatos
  l, polo_real, polo_imag (z = x - i gamma), Gamma_polo = 2 gamma, Q_polo = x/(2 gamma)
  contagem_argumento_l, auditoria_ok, busca (rapida | completa)
  S_polo (W_l ~ S/((x - x_k)^2 + gamma^2)), area_polo, altura_polo, fundo_polo, frac_area_local
  prom_log10_polo, base_pico   proeminencia do polo no modelo analitico (0 se nao vira maximo) e nivel sob o pico
  delta_escala            max(|Im z_j(x_k)| - |Im z_j(z_k)|); acima de 1 a lorentziana do residuo nao vale e S = NaN
  contado_area (entrou na contagem final), sobreposto, dx_nec, dx_malha_usado, pico_na_malha, xM_pico_malha
"""

COL_PICO_ANTIGAS = ["xM_pico", "altura", "largura", "Q", "l", "l_max_efetivo", "delta_x", "origem"]
COL_CURVA_ANTIGAS = ["_indice", "alpha", "beta", "eta", "alpha_vezes_beta", "cL0", "cT0", "rhoS", "tanL_ref", "tanT_ref", "xM_ref", "cM", "rhoM", "status", "runtime_s", "erro", "N_res"]


def exporta(cam, curvas, args=None, modelo=None, seed=None):
    resumo, res, cand, res_malha, hiper = [], [], [], [], []
    for chave in sorted(curvas, key=int):
        c = curvas[chave]; ok = c["status"] == "ok"; rr, cc = c.get("ressonancias", []), c.get("candidatos", [])
        base = {**metadados(int(chave), c["cur"]), "status": c["status"], "runtime_s": c.get("runtime_s", np.nan), "erro": c.get("erro", ""),
                "N_res": len(rr) if ok else np.nan, "tau_adim": c["cur"]["tau"]}
        Q = [r["Q"] for r in rr if np.isfinite(r["Q"])]; aud = c.get("auditoria", {})
        resumo.append({**{k: base[k] for k in COL_CURVA_ANTIGAS}, "modelo": c["cur"]["modelo"], "tau_adim": c["cur"]["tau"],
                       "curva_risco": c.get("risco"), "risco_recorrencia": c.get("risco_recorrencia"), "risco_fase_v1": c.get("risco_fase_v1"),
                       "N_previsto_recorrencia": c.get("N_previsto_recorrencia"), "N_previsto_ate_xmax": c.get("N_previsto_ate_xmax"),
                       "previsao_aproximada": c.get("previsao_aproximada"), "pontos_recorrencia": c.get("pontos_recorrencia"), "modo_contagem": c.get("modo"), "busca_polos": c.get("busca_polos"),
                       "audit_all_ok": all(v == "ok" for v in aud.values()) if aud else np.nan,
                       "N_res_malha_base": c.get("N_res_malha_base"), "N_res_malha_refinada": c.get("N_res_malha_refinada"),
                       "N_res_polos_area": c.get("N_res_polos_area"), "N_res_polos_picos": c.get("N_res_polos_picos"),
                       "N_polos_visiveis": c.get("N_polos_visiveis"), "N_previsto_malha": c.get("N_previsto_malha"),
                       "frac_malha_refinada": (c["N_res_malha_refinada"] / c["N_polos_visiveis"]) if c.get("N_polos_visiveis") else None,
                       "dx_necessario": c.get("dx_necessario"), "dx_90": c.get("dx_90"), "dx_piso_malha": c.get("dx_piso_malha"),
                       "N_polos_estreitos": c.get("N_polos_estreitos"), "N_zeros_estimado": c.get("N_zeros_estimado"),
                       "N_polos_total": len(cc) if ok else np.nan, "l_max_efetivo": c.get("l_max_efetivo"),
                       "x_primeira_res": rr[0]["xM_pico"] if rr else np.nan, "Q_max": max(Q) if Q else np.nan,
                       "pontos_malha": c.get("pontos_malha"), "dx_min_refino": c.get("dx_min"), "dx_min_laurent": c.get("dx_min_laurent"),
                       "N_pares_fundidos_laurent": c.get("N_pares_fundidos_laurent"), "imag_min_busca": c.get("imag_min_busca")})
        for lista, destino in ((rr, res), (c.get("ressonancias_malha", []), res_malha)):
            xs = [r["xM_pico"] for r in lista]
            for i, r in enumerate(lista):
                destino.append({**{k: r.get(k) for k in COL_PICO_ANTIGAS if k not in ("l_max_efetivo", "delta_x")}, "l_max_efetivo": c.get("l_max_efetivo"),
                                "delta_x": xs[i + 1] - xs[i] if i < len(xs) - 1 else np.nan, **r, **base, "res_num": i + 1})
        extra = ["modelo", "tau_adim", "cL", "cT"]          # cL, cT: nomes do dataset elastico antigo (= cL0, cT0)
        if rr: hiper += [{k: d[k] for k in COL_PICO_ANTIGAS + COL_CURVA_ANTIGAS + extra} for d in res[-len(rr):]]
        else: hiper.append({**{k: np.nan for k in COL_PICO_ANTIGAS}, "origem": "", **{k: base[k] for k in COL_CURVA_ANTIGAS + extra}})
        cand += [{**base, "candidato_num": i + 1, **q} for i, q in enumerate(cc)]
    ordem = lambda df, prim: df[[k for k in prim if k in df.columns] + [k for k in df.columns if k not in prim]] if len(df) else df
    for nome, linhas, prim in (("resumo", resumo, COL_CURVA_ANTIGAS), ("res", res, COL_PICO_ANTIGAS), ("res_malha", res_malha, COL_PICO_ANTIGAS), ("cand", cand, ["_indice"]), ("hiper", hiper, COL_PICO_ANTIGAS)):
        ordem(pd.DataFrame(linhas), prim).to_csv(cam[nome], index=False)
    dfh = pd.DataFrame(hiper); np.savez_compressed(cam["hiper"].rsplit(".", 1)[0] + ".npz", **{k: dfh[k].to_numpy() for k in dfh.columns})
    with open(cam["leiame"], "w", encoding="utf-8") as fh:
        fh.write(LEIAME.format(modelo=modelo, versao=VERSAO, nmax=N_MAX_PONTOS, qmin=Q_MIN_POLO, frac=getattr(args, "frac_area", FRAC_AREA),
                               nzmax=getattr(args, "n_zeros_max", N_ZEROS_MAX), norm=NORMALIZACAO))
    const = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str, tuple, dict, np.ndarray)) and k != "LEIAME"}
    with open(cam["param"], "w", encoding="utf-8") as fh:
        json.dump({"versao": VERSAO, "modelo": modelo, "seed": seed, "argumentos": vars(args) if args is not None else None, "constantes": const,
                   "curvas_ok": sum(c["status"] == "ok" for c in curvas.values()), "curvas_total": len(curvas)}, fh, indent=1, default=str)
    print(f"CSVs salvos em {os.path.dirname(cam['resumo'])}")


# ---------------------------------------------------------------------------
# Disparo
# ---------------------------------------------------------------------------
def roda(args, modelo):
    cam = caminhos(args.saida, modelo); curvas = carrega(cam, modelo, args.seed); faixas = dict(FAIXAS[modelo])
    if args.recontar:   # refaz so a contagem final (criterio de polo_conta, --frac-area) a partir do checkpoint
        n = 0
        for c in curvas.values():
            if c.get("status") == "ok" and c.get("modo") == "polos_area": c["ressonancias"] = contagem_polos(c, args.frac_area); n += 1
        print(f"{modelo}: contagem refeita em {n} curvas"); salva(cam, curvas, modelo, args.seed); exporta(cam, curvas, args, modelo, args.seed); return
    if args.tau is not None and modelo != "elastico": faixas["tau"] = tuple(args.tau)
    pend = [(str(i), sorteia(modelo, i, args.seed, faixas), args.prazo_polos, args.limite_curva, args.frac_area, args.normalizacao, args.forcar_polos, args.n_zeros_max, args.recorrencia)
            for i in range(args.n_curvas) if str(i) not in curvas or (args.refazer_falhas and curvas[str(i)]["status"] != "ok")]
    if args.max_curvas: pend = pend[: args.max_curvas]
    prontas = sum(c["status"] == "ok" for c in curvas.values())
    print(f"{modelo}: {prontas} de {args.n_curvas} curvas ja feitas, {len(pend)} para rodar agora com {args.workers} workers")
    t0, feitas = time.time(), 0
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for fut in as_completed([ex.submit(trabalho, t) for t in pend]):
                chave, r = fut.result(); curvas[chave] = r; feitas += 1; cu = r["cur"]
                resto = (time.time() - t0) / feitas * (len(pend) - feitas) / 3600
                print(f"[{feitas}/{len(pend)}] {chave} a={cu['alpha']:.3g} b={cu['beta']:.3g} cT={cu['cT']:.3g} tau={cu['tau']:.3g} {r['status']} {r.get('modo', '')} "
                      f"N_res={len(r.get('ressonancias', [])) if r['status'] == 'ok' else 'NA'} (base {r.get('N_res_malha_base', '-')}) polos={len(r.get('candidatos', []))} "
                      f"t={r.get('runtime_s', 0):.1f}s falta~{resto:.1f}h", flush=True)
                if feitas % args.salvar_a_cada == 0: salva(cam, curvas, modelo, args.seed)
    except KeyboardInterrupt:
        print("Interrompido, salvando o que ja foi feito...")
    salva(cam, curvas, modelo, args.seed); exporta(cam, curvas, args, modelo, args.seed)


def main():
    ap = argparse.ArgumentParser(description="Ressonancias da esfera viscoelastica (Voigt/Maxwell): malha + polos nas curvas de risco.")
    ap.add_argument("--modelo", nargs="+", choices=("elastico", "voigt", "maxwell"), required=True)
    ap.add_argument("--n-curvas", type=int, default=8000, help="indices 0..N-1 do hipercubo (rodar de novo com N maior acrescenta curvas)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--saida", default="./resultados_viscoelastico")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tau", type=float, nargs=2, default=None, help="faixa de tau_adim (log-uniforme); padrao em FAIXAS")
    ap.add_argument("--prazo-polos", type=float, default=120.0, help="tempo maximo (s) da busca de polos por curva, somando todos os l")
    ap.add_argument("--limite-curva", type=float, default=600.0, help="limite duro (s) por curva")
    ap.add_argument("--salvar-a-cada", type=int, default=25)
    ap.add_argument("--normalizacao", choices=("ZM", "separada"), default=NORMALIZACAO, help="ZM: W_T = (Z_S/Z_M) W~_S (padrao); separada: W~_S como nas versoes antigas")
    ap.add_argument("--recorrencia", choices=("exata", "texto"), default=RECORRENCIA, help="kappa da condicao de recorrencia: -Re(x h'/h) (padrao) ou l+1 (texto da tese)")
    ap.add_argument("--frac-area", type=float, default=FRAC_AREA, help="fracao minima da energia da janela para um polo contar")
    ap.add_argument("--n-zeros-max", type=float, default=N_ZEROS_MAX, help="acima dessa estimativa de zeros a busca de polos e' pulada")
    ap.add_argument("--forcar-polos", action="store_true", help="faz a busca de polos tambem nas curvas normais (calibracao)")
    ap.add_argument("--max-curvas", type=int, default=None, help="roda so as primeiras N pendentes (teste)")
    ap.add_argument("--refazer-falhas", action="store_true", help="roda de novo as curvas com timeout ou erro")
    ap.add_argument("--recontar", action="store_true", help="so refaz a contagem final pelos polos (criterio atual, --frac-area) no checkpoint e exporta")
    args = ap.parse_args(); globals()["NORMALIZACAO"] = args.normalizacao; globals()["RECORRENCIA"] = args.recorrencia
    for modelo in args.modelo: roda(args, modelo)


if __name__ == "__main__":
    main()
