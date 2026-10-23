#!/usr/bin/env python3
"""
Dataset de ressonancias de uma esfera core-shell fluida por detector hibrido
(polos complexos + validacao pela energia interna).

Filosofia do detector:
    1) Para cada ordem parcial l = 0,...,lmax, procura zeros complexos do
       denominador D_l(x_M) no semiplano inferior pelo principio do argumento.
       Esses zeros sao tratados como CANDIDATOS MODAIS, nao como ressonancias
       energeticas automaticamente.
    2) Cada candidato e levado de volta ao eixo real. Em uma janela adaptativa
       centrada em Re(x_p), a energia interna total W_T e analisada em alta
       resolucao, com pontos adicionais na escala Gamma_p = 2|Im(x_p)|.
    3) So e contado em N_res_energy quando existe um maximo local de W_T
       associado ao candidato. Assim, um zero de D_l que seja cancelado pelo
       numerador ou que nao produza resposta energetica nao entra na contagem
       principal de ressonancias.
    4) Candidatos diferentes que convergem para o mesmo maximo de energia sao
       agrupados em uma unica ressonancia energetica, preservando a lista de
       ordens l associadas.

Para cada candidato ficam salvos, entre outros:
    - l, parte real e imaginaria do zero de D_l;
    - Gamma_pole e Q_pole;
    - se foi encontrado um maximo local de energia;
    - posicao, intensidade, contraste, FWHM e Q medidos na energia.

Para cada curva ficam separados:
    - N_Dzeros_total: quantidade de zeros de D_l encontrados no dominio;
    - N_res_energy: quantidade de picos unicos realmente observados em W_T.

Importante:
    - A energia interna continua sendo o observavel fisico que define a
      contagem principal de ressonancias.
    - O plano complexo serve para nao perder picos estreitos e para localizar
      de forma confiavel onde a energia deve ser examinada.
    - A busca de zeros e completa apenas dentro do dominio complexo definido
      por [x_min, x_max] x [imag_min, imag_max].
    - Por padrao, cada curva tem no maximo 300 s (5 min). Timeout e falha de
      auditoria sao estados separados e nunca sao tratados como "zero pico".

Saidas por regime:
    1) checkpoint .npz (salvo atomicamente durante a execucao)
    2) candidatos .csv (uma linha por zero de D_l, com validacao energetica)
    3) ressonancias .csv (uma linha por pico unico de energia confirmado)
    4) resumo .csv (uma linha por curva, inclusive zero picos/timeout/erro)

Exemplos no cluster:
    python -u core_shell_ressonancia_hibrida_cluster.py --regime velocity
    python -u core_shell_ressonancia_hibrida_cluster.py --regime mixed

Teste curto antes da producao:
    python -u core_shell_ressonancia_hibrida_cluster.py --regime velocity --max-curves 2
"""

import argparse
import os
import signal
import time
import uuid
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.linalg as la
from scipy.optimize import brentq, minimize_scalar, newton
from scipy.signal import find_peaks, peak_widths
from scipy.special import spherical_jn, spherical_yn


# ============================================================================
# 1. CONFIGURACOES GERAIS
# ============================================================================
LMAX_DEFAULT = 10
X_MIN_DEFAULT = 1e-4
X_MAX_DEFAULT = 1.0
IMAG_MIN_DEFAULT = -0.5
IMAG_MAX_DEFAULT = 1e-9
CURVE_TIMEOUT_DEFAULT = 300.0  # 5 minutos por curva
CHECKPOINT_EVERY_DEFAULT = 5
CHECKPOINT_VERSION = 5

DELTA_VALUES = np.array(
    [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60,
     0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.35, 1.50],
    dtype=float,
)
BINS_ETA = np.linspace(-1.0, 1.0, 21)
TARGET_PER_BIN = 20

# Principio do argumento
CONTOUR_SIDE_LEVELS = (64, 128, 256)
TOP_AUDIT_SIDE_LEVELS = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
INTEGER_TOLERANCE = 0.05
MAX_RECURSION_DEPTH = 40

# Acelerador por momentos/Hankel. So e aceito apos validacao.
BEYN_R_MAXIMUM = 8
BEYN_SLACK = 4
BEYN_N_SIDE = 500
BEYN_RELATIVE_RESIDUAL_MAX = 1e-7

# Duplicatas numericas dentro de uma mesma ordem l
ROOT_DEDUP_TOL = 1e-8

# Validacao do zero de D_l pela energia interna no eixo real.
# A janela usa a largura inferida do polo, mas nunca fica menor/maior que os
# limites abaixo. Nao ha um limiar fisico arbitrario de "pico forte": a
# classificacao exige apenas um maximo local numericamente resolvido. A razao
# pico/fundo e salva para filtros posteriores.
ENERGY_WINDOW_GAMMA = 10.0
ENERGY_WINDOW_MIN = 2e-5
ENERGY_WINDOW_MAX = 2e-2
ENERGY_LINEAR_POINTS = 161
ENERGY_FWHM_POINTS_PER_SIDE = 96
ENERGY_LOCAL_MAX_REL_TOL = 1e-6
ENERGY_PEAK_MATCH_FRACTION = 0.60
ENERGY_MERGE_WIDTH_FRACTION = 0.20
ENERGY_MERGE_ABS_TOL = 1e-9

RHO_WATER_REF = 1000.0
C_WATER_REF = 1480.0
Z_WATER = RHO_WATER_REF * C_WATER_REF

# Camada 0: triagem grosseira (vetorizada, 1 chamada) para decidir se vale a
# pena rodar a busca completa de polos complexos em uma curva. Criterio
# deliberadamente frouxo (falso positivo custa alguns segundos a mais; falso
# negativo perderia uma ressonancia de verdade, o que nao pode acontecer).
# Calibrado empiricamente: casos reais confirmados tem razao pico/fundo >=
# ~747; espurios numericos ficam em ~7-41 (ver notas do projeto). A malha usa
# a mesma estrategia log-perto-da-origem + linear ja adotada no gerador da
# esfera, para nao perder estrutura fina perto de x_min.
SCREEN_LOG_POINTS = 1500
SCREEN_LINEAR_POINTS = 1500
SCREEN_LOG_LINEAR_SPLIT = 0.1
SCREEN_RATIO_FLOOR = 5.0

# Metodo estabelecido do projeto (usado nos geradores anteriores) para
# contar/medir picos direto na malha grosseira, sem depender da busca de
# polos: find_peaks no log10 da energia (estavel em faixas com Q muito
# grande), largura a meia altura via peak_widths. Serve de resultado
# alternativo (sempre disponivel, barato) quando a busca de polos nao
# termina a tempo -- ver POLE_SEARCH_TIMEOUT_DEFAULT em detectar_coreshell.
MESH_PEAK_PROMINENCE_MIN = 0.75

# Abaixo deste FWHM a precisao do refino vetorizado (grade + parabola) nao e
# mais confiavel frente ao piso de ruido numerico ja identificado no projeto
# (~1e-11-1e-12). Nesses casos raros (tipicamente so as ordens l mais altas
# de uma curva muito rica) o candidato e re-validado pelo solver escalar
# original (brentq/minimize_scalar), que e mais lento mas ja e o metodo
# auditado para esse regime extremo.
VALIDATION_FALLBACK_FWHM_TRUST = 1e-10

# Limiar de relevancia energetica: um l cuja falha de auditoria (winding
# number nao fechou) so importa de verdade se a contribuicao maxima desse l
# for pelo menos essa fracao do pico da energia TOTAL (todos os l somados).
# Abaixo disso o l e' fisicamente irrelevante para a curva observavel, entao
# uma falha ali nao derruba a curva inteira (ver polos_por_l e
# detectar_coreshell). Calibrado com folga: no caso testado, l relevantes
# (0-3) ficam em 5e-2 a 5e-1; l irrelevantes (8,10) ficam em 1e-8 a 1e-11.
L_SIGNIFICANCE_THRESHOLD = 1e-4


# ============================================================================
# 2. FUNCOES ESFERICAS E MODELO CORE-SHELL
# ============================================================================
def J(n, x):
    if n >= 0:
        return spherical_jn(n, x)
    m = -n - 1
    return (-1) ** (m + 1) * spherical_yn(m, x)


def Y(n, x):
    if n >= 0:
        return spherical_yn(n, x)
    m = -n - 1
    return (-1) ** m * spherical_jn(m, x)


def dJ(n, x):
    if n >= 0:
        return spherical_jn(n, x, derivative=True)
    m = -n - 1
    return (-1) ** (m + 1) * spherical_yn(m, x, derivative=True)


def dY(n, x):
    if n >= 0:
        return spherical_yn(n, x, derivative=True)
    m = -n - 1
    return (-1) ** m * spherical_jn(m, x, derivative=True)


def H(n, x):
    if n >= 0:
        return spherical_jn(n, x) + 1j * spherical_yn(n, x)
    m = -n - 1
    return 1j * (-1) ** m * (
        spherical_jn(m, x) + 1j * spherical_yn(m, x)
    )


def dH(n, x):
    if n >= 0:
        return (
            spherical_jn(n, x, derivative=True)
            + 1j * spherical_yn(n, x, derivative=True)
        )
    m = -n - 1
    return 1j * (-1) ** m * (
        spherical_jn(m, x, derivative=True)
        + 1j * spherical_yn(m, x, derivative=True)
    )


def Fjj(l, xB, yB):
    def F(x):
        return x**3 * (J(l, x) ** 2 - J(l - 1, x) * J(l + 1, x))
    return F(yB) - F(xB)


def Fyy(l, xB, yB):
    def F(x):
        return x**3 * (Y(l, x) ** 2 - Y(l - 1, x) * Y(l + 1, x))
    return F(yB) - F(xB)


def Fjy(l, xB, yB):
    def F(x):
        return 0.5 * x**3 * (
            2 * J(l, x) * Y(l, x)
            - J(l + 1, x) * Y(l - 1, x)
            - J(l - 1, x) * Y(l + 1, x)
        )
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
    return 0.5 * xA**3 * (
        J(l, xA) ** 2 - J(l - 1, xA) * J(l + 1, xA)
    )


def BA(l, xA):
    return xA**2 * J(l, xA) * dJ(l, xA)


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

    return (
        (L * R - M * Q) * (F * HH - B * JJ)
        + (L * S - P * Q) * (A * JJ - F * G)
    )


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

    denominator = (
        (L * R - M * Q) * (F * HH - B * JJ)
        + (L * S - P * Q) * (A * JJ - F * G)
    )
    X = 1j * rho_MB * k_MB / yM**2

    b_l = (M * S - P * R) * X / denominator
    c_l = (P * Q - L * S) * X / denominator
    d_l = (L * R - M * Q) * X / denominator
    return b_l, c_l, d_l


def W_total_parts_vs_xM(
    xM_vals, delta, rho0, rho1, rho2, c0, c1, c2,
    lmax_core=LMAX_DEFAULT, lmax_shell=LMAX_DEFAULT,
):
    xM_vals = np.asarray(xM_vals, dtype=float)
    lmax_core = int(lmax_core)
    lmax_shell = int(lmax_shell)
    lmax_total = max(lmax_core, lmax_shell)

    WK_core = np.zeros_like(xM_vals, dtype=float)
    WP_core = np.zeros_like(xM_vals, dtype=float)
    WK_shell = np.zeros_like(xM_vals, dtype=float)
    WP_shell = np.zeros_like(xM_vals, dtype=float)

    xA = (c0 / c1) * xM_vals
    xB = (c0 / c2) * xM_vals
    yM = (1.0 + delta) * xM_vals
    yB = (1.0 + delta) * xB

    with np.errstate(all="ignore"):
        for l in range(lmax_total + 1):
            bl, cl, dl = coeffs(
                l, xA, xB, yM, yB,
                rho0, rho1, rho2, c0, c1, c2,
            )
            factor = 2 * l + 1

            if l <= lmax_core:
                bl2 = np.abs(bl) ** 2
                fa = FA(l, xA)
                ba = BA(l, xA)
                WP_core += np.real(bl2 * factor * fa)
                WK_core += np.real(bl2 * factor * (fa + ba))

            if l <= lmax_shell:
                cl2 = np.abs(cl) ** 2
                dl2 = np.abs(dl) ** 2
                re_cd = np.real(np.conj(cl) * dl)
                potential = (
                    0.5 * cl2 * Fjj(l, xB, yB)
                    + 0.5 * dl2 * Fyy(l, xB, yB)
                    + re_cd * Fjy(l, xB, yB)
                )
                boundary = (
                    cl2 * Bjj(l, xB, yB)
                    + dl2 * Byy(l, xB, yB)
                    + 2.0 * re_cd * Bjy(l, xB, yB)
                )
                WP_shell += np.real(factor * potential)
                WK_shell += np.real(factor * (potential + boundary))

    return WK_core, WP_core, WK_shell, WP_shell


def energia_total_coreshell(
    xM_vals, delta, rho0, rho1, rho2, c0, c1, c2,
    lmax=LMAX_DEFAULT,
):
    parts = W_total_parts_vs_xM(
        xM_vals, delta, rho0, rho1, rho2, c0, c1, c2,
        lmax_core=lmax, lmax_shell=lmax,
    )
    WT = sum(parts)
    return np.where(np.isfinite(WT) & (WT > 0.0), WT, 0.0)


def _screening_grid(x_min, x_max):
    """Malha grosseira log (perto da origem) + linear, para a camada 0."""
    lo = max(float(x_min), 1e-9)
    split = min(SCREEN_LOG_LINEAR_SPLIT, float(x_max))
    log_part = np.geomspace(lo, split, SCREEN_LOG_POINTS) if split > lo else np.array([lo])
    lin_part = np.linspace(split, float(x_max), SCREEN_LINEAR_POINTS)
    return np.unique(np.concatenate([log_part, lin_part]))


def tem_potencial_ressonancia(
    delta, materials, x_min, x_max, lmax,
    ratio_floor=SCREEN_RATIO_FLOOR, x_grid=None,
):
    """Camada 0 do detector: 1 chamada vetorizada da energia total numa malha
    grosseira, seguida de find_peaks. Decide SO se vale a pena rodar a busca
    completa de polos complexos por l (a parte cara). Nao mede posicao/Q com
    precisao nenhuma -- isso continua por conta do detector completo.

    Retorna (potencialmente_ressonante, razao_pico_fundo, x_grid, w_grid).
    """
    if x_grid is None:
        x_grid = _screening_grid(x_min, x_max)
    rho0, rho1, rho2, c0, c1, c2 = materials
    w_grid = energia_total_coreshell(
        x_grid, delta, rho0, rho1, rho2, c0, c1, c2, lmax=lmax
    )
    finite = np.isfinite(w_grid) & (w_grid > 0.0)
    if not np.any(finite):
        return False, 0.0, x_grid, w_grid

    peaks, _ = find_peaks(w_grid, prominence=0.0)
    if len(peaks) == 0:
        return False, 0.0, x_grid, w_grid

    background = max(float(np.median(w_grid[finite])), np.finfo(float).tiny)
    ratio = float(np.max(w_grid[peaks]) / background)
    return ratio > ratio_floor, ratio, x_grid, w_grid


def _picos_por_malha_grosseira(x_grid, w_grid, prominence_min=MESH_PEAK_PROMINENCE_MIN):
    """Conta e mede picos direto na malha grosseira (metodo ja estabelecido
    no projeto): find_peaks no log10(W) com proeminencia minima, largura a
    meia altura via peak_widths na escala linear. Nao depende da busca de
    polos no plano complexo; serve como resultado sempre disponivel, usado
    tanto para a triagem quanto como fallback quando a busca de polos e
    interrompida por tempo (ver POLE_SEARCH_TIMEOUT_DEFAULT)."""
    x_grid = np.asarray(x_grid, dtype=float)
    w_grid = np.asarray(w_grid, dtype=float)
    finite = np.isfinite(w_grid) & (w_grid > 0.0)
    if np.count_nonzero(finite) < 5:
        return []

    floor = max(float(np.min(w_grid[finite])) * 1e-14, np.finfo(float).tiny)
    w_safe = np.where(finite, w_grid, floor)
    ylog = np.log10(np.maximum(w_safe, floor))

    idx, _ = find_peaks(ylog, prominence=prominence_min)
    idx = idx[(idx >= 3) & (idx <= len(x_grid) - 4)]
    if len(idx) == 0:
        return []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, lips, rips = peak_widths(w_grid, idx, rel_height=0.5)

    eixo = np.arange(len(x_grid), dtype=float)
    xl = np.interp(lips, eixo, x_grid)
    xr = np.interp(rips, eixo, x_grid)

    picos = []
    for k, i in enumerate(idx):
        x0 = float(x_grid[i])
        fwhm = float(xr[k] - xl[k])
        wt_peak = float(w_grid[i])
        if not (np.isfinite(fwhm) and fwhm > 0.0 and np.isfinite(wt_peak) and wt_peak > 0.0):
            continue
        picos.append({
            "xM_peak": x0, "FWHM": fwhm, "Q": x0 / fwhm, "WT_peak": wt_peak,
        })
    return picos


# ============================================================================
# 3. MATERIAIS E REGIMES
# ============================================================================
def make_materials_from_alpha_beta(
    alpha,
    beta,
    ZM=Z_WATER,
    rho_ref=RHO_WATER_REF,
    c_ref=C_WATER_REF,
    contrast_in="mixed",
    mix_factor=0.5,
):
    alpha = float(alpha)
    beta = float(beta)

    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("alpha e beta precisam ser positivos.")

    ZA = alpha * ZM
    ZB = beta * ZM

    if contrast_in == "sound_speed":
        rho0 = rho1 = rho2 = rho_ref
        c0 = ZM / rho0
        c1 = ZA / rho1
        c2 = ZB / rho2
    elif contrast_in == "density":
        c0 = c1 = c2 = c_ref
        rho0 = ZM / c0
        rho1 = ZA / c1
        rho2 = ZB / c2
    elif contrast_in == "mixed":
        mix_factor = float(mix_factor)
        rho0 = rho_ref
        c0 = ZM / rho0
        rho1 = rho_ref * alpha**mix_factor
        c1 = c_ref * alpha ** (1.0 - mix_factor)
        rho2 = rho_ref * beta**mix_factor
        c2 = c_ref * beta ** (1.0 - mix_factor)
    else:
        raise ValueError('contrast_in deve ser "sound_speed", "density" ou "mixed".')

    return rho0, rho1, rho2, c0, c1, c2


def eta_from_alpha_beta(alpha, beta):
    return (float(alpha) - float(beta)) / (float(alpha) + float(beta))


def contrast_from_regime(regime):
    if regime == "velocity":
        return "sound_speed"
    if regime == "mixed":
        return "mixed"
    if regime == "density":
        return "density"
    raise ValueError(f"Regime desconhecido: {regime}")


# ============================================================================
# 4. DETECTOR DE POLOS: PRINCIPIO DO ARGUMENTO
# ============================================================================
class PoleAuditError(RuntimeError):
    def __init__(self, message, audit=None):
        super().__init__(message)
        self.audit = audit or {}


def _contour_points(x_lo, x_hi, y_lo, y_hi, n_side):
    """Contorno retangular anti-horario; o ponto inicial nao e repetido."""
    n_side = max(int(n_side), 4)
    bottom = np.linspace(x_lo, x_hi, n_side, endpoint=False) + 1j * y_lo
    right = x_hi + 1j * np.linspace(y_lo, y_hi, n_side, endpoint=False)
    top = np.linspace(x_hi, x_lo, n_side, endpoint=False) + 1j * y_hi
    left = x_lo + 1j * np.linspace(y_hi, y_lo, n_side, endpoint=False)
    return np.concatenate([bottom, right, top, left])


def _winding_number(D_l, x_lo, x_hi, y_lo, y_hi, n_side):
    z = _contour_points(x_lo, x_hi, y_lo, y_hi, n_side)
    values = np.asarray(D_l(z), dtype=complex)

    if not np.all(np.isfinite(values)):
        return None
    if np.any(np.abs(values) == 0.0):
        return None

    values_next = np.roll(values, -1)
    with np.errstate(all="ignore"):
        ratios = values_next / values
    if not np.all(np.isfinite(ratios)):
        return None

    increments = np.angle(ratios)
    return float(np.sum(increments) / (2.0 * np.pi))


def _reliable_zero_count(D_l, x_lo, x_hi, y_lo, y_hi):
    """Contagem local usada na subdivisao.

    Usa duas resolucoes sucessivas concordando no mesmo inteiro. A auditoria
    final de cada ordem l e feita separadamente, com contorno externo muito
    mais refinado.
    """
    previous_integer = None
    for n_side in CONTOUR_SIDE_LEVELS:
        winding = _winding_number(D_l, x_lo, x_hi, y_lo, y_hi, n_side)
        if winding is None:
            previous_integer = None
            continue

        n_integer = int(round(winding))
        if n_integer < 0 or abs(winding - n_integer) > INTEGER_TOLERANCE:
            previous_integer = None
            continue

        if previous_integer is not None and n_integer == previous_integer:
            return n_integer, True
        previous_integer = n_integer

    return None, False

def _audit_top_count_against_found(D_l, x_lo, x_hi, y_lo, y_hi, found_count):
    """Auditoria independente do contorno externo, com refinamento forte.

    A busca de raizes e feita pela arvore de subdivisao. Depois dela, o
    contorno externo e reavaliado com resolucoes progressivamente maiores.
    A auditoria so passa quando a contagem do principio do argumento coincide
    com ``found_count`` em duas resolucoes sucessivas.
    """
    streak = 0
    previous = None
    last_valid = None

    for n_side in TOP_AUDIT_SIDE_LEVELS:
        winding = _winding_number(D_l, x_lo, x_hi, y_lo, y_hi, n_side)
        if winding is None:
            streak = 0
            previous = None
            continue

        n_integer = int(round(winding))
        if n_integer < 0 or abs(winding - n_integer) > INTEGER_TOLERANCE:
            streak = 0
            previous = None
            continue

        last_valid = n_integer
        if n_integer == found_count:
            if previous == n_integer:
                streak += 1
            else:
                streak = 1
            if streak >= 2:
                return n_integer, True
        else:
            streak = 0
        previous = n_integer

    return last_valid, False


def _inside(z, x_lo, x_hi, y_lo, y_hi, tol=1e-10):
    return (
        x_lo - tol <= z.real <= x_hi + tol
        and y_lo - tol <= z.imag <= y_hi + tol
    )


def _contour_integral(values, z_values):
    z_next = np.roll(z_values, -1)
    values_next = np.roll(values, -1)
    return np.sum((values + values_next) * 0.5 * (z_next - z_values))


def _relative_root_residual(D_l, root):
    value = D_l(root)
    if not np.isfinite(value):
        return np.inf

    h = 1e-7 * max(1.0, abs(root))
    probes = np.array(
        [root + h, root - h, root + 1j * h, root - 1j * h],
        dtype=complex,
    )
    probe_values = np.asarray(D_l(probes), dtype=complex)
    finite = np.isfinite(probe_values)
    if not np.any(finite):
        return np.inf

    scale = float(np.median(np.abs(probe_values[finite])))
    scale = max(scale, np.finfo(float).tiny)
    return float(abs(value) / scale)


def _try_beyn(D_l, x_lo, x_hi, y_lo, y_hi, r):
    L = int(r) + BEYN_SLACK
    center = 0.5 * (x_lo + x_hi) + 0.5j * (y_lo + y_hi)
    z = _contour_points(x_lo, x_hi, y_lo, y_hi, BEYN_N_SIDE)
    w = z - center

    fz = np.asarray(D_l(z), dtype=complex)
    if not np.all(np.isfinite(fz)) or np.any(np.abs(fz) == 0.0):
        return None

    h = 1e-6
    dfz = (D_l(z + h) - D_l(z - h)) / (2.0 * h)
    log_derivative = dfz / fz

    moments = [
        _contour_integral((w**k) * log_derivative, z) / (2j * np.pi)
        for k in range(2 * L)
    ]
    H0 = np.array([[moments[i + j] for j in range(L)] for i in range(L)])
    H1 = np.array([[moments[i + j + 1] for j in range(L)] for i in range(L)])

    try:
        U, S, Vh = la.svd(H0)
        U_r = U[:, :r]
        S_r = S[:r]
        V_r = Vh[:r, :].conj().T
        if np.any(S_r <= 0.0) or not np.all(np.isfinite(S_r)):
            return None
        B = (U_r.conj().T @ H1 @ V_r) @ np.diag(1.0 / S_r)
        return la.eigvals(B) + center
    except Exception:
        return None


def _polish_root(D_l, seed, x_lo, x_hi, y_lo, y_hi):
    try:
        root = complex(newton(D_l, seed, tol=1e-13, maxiter=150))
    except Exception:
        return None

    if not _inside(root, x_lo, x_hi, y_lo, y_hi):
        return None
    if _relative_root_residual(D_l, root) > BEYN_RELATIVE_RESIDUAL_MAX:
        return None
    return root


def _validate_beyn_batch(D_l, candidates, x_lo, x_hi, y_lo, y_hi):
    polished = []
    for candidate in candidates:
        if not _inside(candidate, x_lo, x_hi, y_lo, y_hi):
            return None
        root = _polish_root(D_l, candidate, x_lo, x_hi, y_lo, y_hi)
        if root is None:
            return None
        polished.append(root)

    for i in range(len(polished)):
        for j in range(i + 1, len(polished)):
            if abs(polished[i] - polished[j]) <= ROOT_DEDUP_TOL:
                return None
    return polished


def _quarters(x_lo, x_hi, y_lo, y_hi):
    xm = 0.5 * (x_lo + x_hi)
    ym = 0.5 * (y_lo + y_hi)
    return [
        (x_lo, xm, y_lo, ym),
        (xm, x_hi, y_lo, ym),
        (x_lo, xm, ym, y_hi),
        (xm, x_hi, ym, y_hi),
    ]


def _deduplicate_roots(roots, tol=ROOT_DEDUP_TOL):
    roots = [complex(z) for z in roots]
    if not roots:
        return []
    roots.sort(key=lambda z: (z.real, z.imag))
    unique = []
    for z in roots:
        if not unique or min(abs(z - u) for u in unique) > tol:
            unique.append(z)
    return unique


def _subdivide_and_recurse(D_l, x_lo, x_hi, y_lo, y_hi, depth, deadline=None):
    roots = []
    for xa, xb, ya, yb in _quarters(x_lo, x_hi, y_lo, y_hi):
        roots.extend(find_zeros_in_rectangle(D_l, xa, xb, ya, yb, depth + 1, deadline))
    return _deduplicate_roots(roots)


def find_zeros_in_rectangle(D_l, x_lo, x_hi, y_lo, y_hi, depth=0, deadline=None):
    if deadline is not None and time.time() >= deadline:
        return []

    n_zeros, reliable = _reliable_zero_count(D_l, x_lo, x_hi, y_lo, y_hi)

    if not reliable:
        if depth >= MAX_RECURSION_DEPTH:
            return []
        return _subdivide_and_recurse(D_l, x_lo, x_hi, y_lo, y_hi, depth, deadline)

    if n_zeros == 0:
        return []

    if n_zeros == 1:
        seed = 0.5 * (x_lo + x_hi) + 0.5j * (y_lo + y_hi)
        root = _polish_root(D_l, seed, x_lo, x_hi, y_lo, y_hi)
        if root is not None:
            return [root]
        if depth >= MAX_RECURSION_DEPTH:
            return []
        return _subdivide_and_recurse(D_l, x_lo, x_hi, y_lo, y_hi, depth, deadline)

    if 2 <= n_zeros <= BEYN_R_MAXIMUM:
        candidates = _try_beyn(D_l, x_lo, x_hi, y_lo, y_hi, n_zeros)
        if candidates is not None and len(candidates) == n_zeros:
            validated = _validate_beyn_batch(
                D_l, candidates, x_lo, x_hi, y_lo, y_hi
            )
            if validated is not None and len(validated) == n_zeros:
                return _deduplicate_roots(validated)

    if depth >= MAX_RECURSION_DEPTH:
        return []
    return _subdivide_and_recurse(D_l, x_lo, x_hi, y_lo, y_hi, depth, deadline)


def _D_l_complexo(l, z, delta, materials):
    rho0, rho1, rho2, c0, c1, c2 = materials
    z = np.asarray(z, dtype=complex)
    xA = (c0 / c1) * z
    xB = (c0 / c2) * z
    yM = (1.0 + delta) * z
    yB = (1.0 + delta) * xB
    return D_l_coreshell(
        l, xA, xB, yM, yB,
        rho0, rho1, rho2, c0, c1, c2,
    )


def polos_por_l(l, delta, materials, x_min, x_max, imag_min, imag_max, deadline=None):
    def D_l(z):
        return _D_l_complexo(l, z, delta, materials)

    roots = find_zeros_in_rectangle(
        D_l, x_min, x_max, imag_min, imag_max, deadline=deadline
    )
    roots = _deduplicate_roots(
        [
            z for z in roots
            if np.isfinite(z.real)
            and np.isfinite(z.imag)
            and imag_min <= z.imag < 0.0
            and x_min <= z.real <= x_max
        ]
    )

    if deadline is not None and time.time() >= deadline:
        # Tempo esgotado antes mesmo de auditar: o que foi achado ate aqui
        # fica registrado, mas nao da pra confirmar contagem esperada.
        return roots, None, False

    expected_count, audit_ok = _audit_top_count_against_found(
        D_l, x_min, x_max, imag_min, imag_max, len(roots)
    )
    return roots, expected_count, audit_ok


# Alvo de raizes por faixa ao estimar quantas faixas horizontais usar como
# ponto de partida da busca (ver polos_por_l_particionado). Nao afeta
# corretude, so o quanto a busca comeca ja subdividida.
STRIP_TARGET_ROOTS_PER_STRIP = 4
STRIP_MAX_COUNT = 96


def _estimate_strip_count(
    l, delta, materials, x_min, x_max,
    target_roots_per_strip=STRIP_TARGET_ROOTS_PER_STRIP,
    max_strips=STRIP_MAX_COUNT, x_probe=None,
):
    """Estimativa barata (so eixo real, 1 avaliacao vetorizada de D_l) de
    quantas faixas horizontais vale a pena usar como ponto de partida da
    busca de polos dessa ordem l especifica, contando trocas de sinal de
    Re(D_l) ao longo do eixo real. Isso NUNCA decide se ha ou nao raiz, so
    orienta a granularidade inicial de find_zeros_in_rectangle (que continua
    exatamente a mesma busca rigorosa de sempre); uma estimativa ruim so
    custa tempo a mais ou a menos, nunca corretude, porque cada faixa ainda
    roda a auditoria de contagem por winding number normalmente."""
    if x_probe is None:
        x_probe = _screening_grid(x_min, x_max)
    z_probe = x_probe.astype(complex)
    d_vals = _D_l_complexo(l, z_probe, delta, materials)
    re = np.real(d_vals)
    finite = np.isfinite(re) & (re != 0.0)
    if np.count_nonzero(finite) < 2:
        return 1
    sinais = np.sign(re[finite])
    trocas_de_sinal = int(np.sum(np.diff(sinais) != 0))
    n_strips = max(1, int(np.ceil(trocas_de_sinal / target_roots_per_strip)))
    return min(n_strips, max_strips)


def _contribuicao_energia_l(l, delta, materials, x_vals):
    """Contribuicao de UM l isolado para a energia total, nos mesmos termos
    somados em W_total_parts_vs_xM. Usada so para decidir se uma falha de
    auditoria (winding number) nesse l chega a importar para a curva de
    energia observavel (ver L_SIGNIFICANCE_THRESHOLD); nunca substitui o
    calculo da energia total, que soma todos os l."""
    rho0, rho1, rho2, c0, c1, c2 = materials
    x_vals = np.asarray(x_vals, dtype=float)
    xA = (c0 / c1) * x_vals
    xB = (c0 / c2) * x_vals
    yM = (1.0 + delta) * x_vals
    yB = (1.0 + delta) * xB
    factor = 2 * l + 1
    with np.errstate(all="ignore"):
        bl, cl, dl = coeffs(l, xA, xB, yM, yB, rho0, rho1, rho2, c0, c1, c2)
        bl2 = np.abs(bl) ** 2
        fa_l = FA(l, xA)
        ba_l = BA(l, xA)
        w_core = np.real(bl2 * factor * (fa_l + ba_l))

        cl2 = np.abs(cl) ** 2
        dl2 = np.abs(dl) ** 2
        re_cd = np.real(np.conj(cl) * dl)
        potential = (
            0.5 * cl2 * Fjj(l, xB, yB)
            + 0.5 * dl2 * Fyy(l, xB, yB)
            + re_cd * Fjy(l, xB, yB)
        )
        boundary = (
            cl2 * Bjj(l, xB, yB)
            + dl2 * Byy(l, xB, yB)
            + 2.0 * re_cd * Bjy(l, xB, yB)
        )
        w_shell = np.real(factor * (potential + boundary))

    w_l = w_core + w_shell
    return np.where(np.isfinite(w_l), w_l, 0.0)


def _relevancia_energetica_l(l, delta, materials, x_grid, w_grid_total):
    """Razao entre o pico da contribuicao isolada do l e o pico da energia
    TOTAL (todos os l), na mesma malha. Usada so quando esse l falha
    auditoria, para decidir se a falha e' fisicamente relevante."""
    w_l = _contribuicao_energia_l(l, delta, materials, x_grid)
    finite_l = np.isfinite(w_l)
    finite_t = np.isfinite(w_grid_total) & (w_grid_total > 0.0)
    if not np.any(finite_l) or not np.any(finite_t):
        return np.inf  # sem info confiavel: trata como relevante, por seguranca
    max_l = float(np.max(w_l[finite_l]))
    max_total = float(np.max(w_grid_total[finite_t]))
    if max_total <= 0.0:
        return np.inf
    return max_l / max_total


# Limiares calibrados empiricamente (ver notas do projeto): acima disso, a
# busca de polos no plano complexo estoura float64 (ou pior, cancela tudo
# silenciosamente com precisao insuficiente). O fator de amplificacao vem
# da razao entre a velocidade do som na agua e no nucleo/casca -- e o que
# transforma uma profundidade pequena no plano complexo (fixa, ate
# IMAG_MIN_DEFAULT) num argumento gigante para as funcoes de Bessel/Hankel.
OVERFLOW_RISK_SHELL_THRESHOLD = 700.0
OVERFLOW_RISK_CORE_THRESHOLD = 1000.0


def risco_overflow_busca_polos(
    alpha, beta, delta, regime,
    shell_threshold=OVERFLOW_RISK_SHELL_THRESHOLD,
    core_threshold=OVERFLOW_RISK_CORE_THRESHOLD,
):
    """Estimativa fechada (nao roda nada, so calcula), de se a busca de
    polos no plano complexo provavelmente vai estourar float64 para esses
    parametros. So importa onde a velocidade do som varia entre os meios
    (regimes "velocity" e "mixed"); no regime "density" c1=c2=c0 sempre, o
    problema nao existe. Retorna (risco: bool, amp_casca, amp_nucleo)."""
    contrast_in = contrast_from_regime(regime)
    if contrast_in == "sound_speed":
        amp_shell = (1.0 + float(delta)) / float(beta)
        amp_core = 1.0 / float(alpha)
    elif contrast_in == "mixed":
        amp_shell = (1.0 + float(delta)) / np.sqrt(float(beta))
        amp_core = 1.0 / np.sqrt(float(alpha))
    else:  # "density": c1 = c2 = c0 sempre, nunca amplifica
        amp_shell = 0.0
        amp_core = 0.0
    risco = (amp_shell > shell_threshold) or (amp_core > core_threshold)
    return bool(risco), float(amp_shell), float(amp_core)


def _resonancias_fallback_malha(picos_malha):
    """Converte os picos da malha grosseira (find_peaks + peak_widths) para
    o mesmo formato de _merge_energy_resonances, marcados com
    origem="malha_grosseira" -- usado quando a busca de polos e' pulada por
    risco de overflow (ver risco_overflow_busca_polos). Sem posicao/Q de
    polo complexo por l, so o que a malha consegue medir direto."""
    resonancias = []
    for p in picos_malha:
        resonancias.append({
            "xM_peak": float(p["xM_peak"]),
            "WT_peak": float(p["WT_peak"]),
            "WT_background": np.nan,
            "energy_ratio": np.nan,
            "energy_log10_prominence": np.nan,
            "FWHM_energy": float(p["FWHM"]),
            "Q_energy": float(p["Q"]),
            "primary_l": None,
            "primary_pole_real": np.nan,
            "primary_pole_imag": np.nan,
            "primary_Gamma_pole": np.nan,
            "primary_Q_pole": np.nan,
            "associated_l": "",
            "N_associated_Dzeros": 0,
            "associated_pole_real": "",
            "associated_pole_imag": "",
            "origem": "malha_grosseira",
        })
    return resonancias


def _audit_entry_aceitavel(v):
    """Uma entrada de auditoria conta como aceitavel se o winding number
    fechou (ok=True) OU se falhou mas foi marcada como energeticamente
    irrelevante (ver L_SIGNIFICANCE_THRESHOLD). So uma falha relevante
    (que de fato afeta a curva observavel) conta contra audit_all_ok --
    e essa, alias, ja teria levantado PoleAuditError antes de chegar aqui."""
    if v.get("ok", False):
        return True
    return v.get("energeticamente_relevante", True) is False


def polos_por_l_particionado(
    l, delta, materials, x_min, x_max, imag_min, imag_max,
    target_roots_per_strip=STRIP_TARGET_ROOTS_PER_STRIP,
    max_strips=STRIP_MAX_COUNT, x_probe=None,
):
    """Mesmo contrato e mesmo resultado de polos_por_l (mesma
    find_zeros_in_rectangle, mesma _audit_top_count_against_found), mas
    comeca a busca ja dividida em faixas no eixo real em vez de deixar a
    recursao dividir o retangulo inteiro do zero. O numero de faixas vem de
    _estimate_strip_count (barato, vetorizado). Faixas vizinhas se sobrepoem
    um pouco para nao perder raiz perto da fronteira entre duas faixas;
    duplicatas sao removidas no final com a mesma tolerancia de sempre."""
    def D_l(z):
        return _D_l_complexo(l, z, delta, materials)

    n_strips = _estimate_strip_count(
        l, delta, materials, x_min, x_max,
        target_roots_per_strip=target_roots_per_strip,
        max_strips=max_strips, x_probe=x_probe,
    )

    if n_strips <= 1:
        roots = find_zeros_in_rectangle(D_l, x_min, x_max, imag_min, imag_max)
    else:
        edges = np.linspace(x_min, x_max, n_strips + 1)
        width = (x_max - x_min) / n_strips
        overlap = 0.05 * width
        roots = []
        for i in range(n_strips):
            lo = max(x_min, edges[i] - overlap)
            hi = min(x_max, edges[i + 1] + overlap)
            roots.extend(find_zeros_in_rectangle(D_l, lo, hi, imag_min, imag_max))

    roots = _deduplicate_roots(
        [
            z for z in roots
            if np.isfinite(z.real)
            and np.isfinite(z.imag)
            and imag_min <= z.imag < 0.0
            and x_min <= z.real <= x_max
        ]
    )

    expected_count, audit_ok = _audit_top_count_against_found(
        D_l, x_min, x_max, imag_min, imag_max, len(roots)
    )
    return roots, expected_count, audit_ok


def _energia_scalar(x, delta, materials, lmax):
    """Energia total em um unico ponto real, com protecao numerica."""
    rho0, rho1, rho2, c0, c1, c2 = materials
    value = energia_total_coreshell(
        np.array([float(x)], dtype=float),
        delta, rho0, rho1, rho2, c0, c1, c2,
        lmax=lmax,
    )[0]
    if not np.isfinite(value) or value < 0.0:
        return np.nan
    return float(value)


def _adaptive_energy_grid(x0, gamma, lo, hi):
    """Malha real que combina cobertura uniforme e pontos na escala do polo."""
    linear = np.linspace(lo, hi, ENERGY_LINEAR_POINTS)
    points = [linear, np.array([x0], dtype=float)]

    if np.isfinite(gamma) and gamma > 0.0:
        scales = np.array(
            [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0],
            dtype=float,
        )
        offsets = gamma * scales
        around = np.concatenate([x0 - offsets, [x0], x0 + offsets])
        around = around[(around >= lo) & (around <= hi)]
        points.append(around)

    x = np.unique(np.concatenate(points))
    return x[(x >= lo) & (x <= hi)]


def _local_maxima_indices(x, w):
    """Indices de maximos locais estritos/semiestritos em malha nao uniforme."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    if len(x) < 3:
        return np.array([], dtype=int)

    finite = np.isfinite(w)
    good = finite[1:-1] & finite[:-2] & finite[2:]
    mid = w[1:-1]
    left = w[:-2]
    right = w[2:]
    maxima = good & (mid >= left) & (mid >= right) & ((mid > left) | (mid > right))
    return np.flatnonzero(maxima) + 1


def _refine_energy_peak(x_left, x_seed, x_right, delta, materials, lmax):
    """Refina um maximo de W_T minimizando -log(W_T) em um pequeno intervalo."""
    tiny = np.finfo(float).tiny

    def objective(xx):
        value = _energia_scalar(xx, delta, materials, lmax)
        if not np.isfinite(value) or value <= 0.0:
            return 1e300
        return -np.log(max(value, tiny))

    try:
        result = minimize_scalar(
            objective,
            bounds=(float(x_left), float(x_right)),
            method="bounded",
            options={"xatol": 1e-13, "maxiter": 250},
        )
        if result.success and np.isfinite(result.x):
            x_peak = float(result.x)
            w_peak = _energia_scalar(x_peak, delta, materials, lmax)
            if np.isfinite(w_peak):
                return x_peak, w_peak
    except Exception:
        pass

    w_seed = _energia_scalar(x_seed, delta, materials, lmax)
    if np.isfinite(w_seed):
        return float(x_seed), float(w_seed)
    return np.nan, np.nan


def _background_energy(x_grid, w_grid, x_peak, gamma, lo, hi):
    """Fundo robusto usando a parte externa da janela ao redor do pico."""
    x_grid = np.asarray(x_grid, dtype=float)
    w_grid = np.asarray(w_grid, dtype=float)
    finite = np.isfinite(w_grid) & (w_grid >= 0.0)
    if not np.any(finite):
        return np.nan

    half_window = max(x_peak - lo, hi - x_peak)
    exclusion = max(4.0 * gamma if np.isfinite(gamma) else 0.0, 0.20 * half_window)
    outer = finite & (np.abs(x_grid - x_peak) >= exclusion)

    if np.count_nonzero(outer) < 6:
        qlo = lo + 0.20 * (hi - lo)
        qhi = hi - 0.20 * (hi - lo)
        outer = finite & ((x_grid <= qlo) | (x_grid >= qhi))

    values = w_grid[outer]
    if len(values) == 0:
        return np.nan
    return float(np.median(values))


def _find_half_height_crossing(
    x_peak, w_peak, background, gamma, lo, hi, delta, materials, lmax, side
):
    """Procura cruzamento de meia proeminencia usando amostragem geometrica."""
    if not (np.isfinite(w_peak) and np.isfinite(background) and w_peak > background):
        return np.nan

    level = background + 0.5 * (w_peak - background)
    max_distance = (x_peak - lo) if side < 0 else (hi - x_peak)
    if max_distance <= 0.0:
        return np.nan

    base = gamma if np.isfinite(gamma) and gamma > 0.0 else max_distance / 100.0
    d_min = max(
        np.finfo(float).eps * max(1.0, abs(x_peak)) * 32.0,
        min(base * 1e-3, max_distance * 1e-3),
    )
    if d_min >= max_distance:
        d_min = max_distance * 1e-6
    if d_min <= 0.0:
        return np.nan

    distances = np.geomspace(d_min, max_distance, ENERGY_FWHM_POINTS_PER_SIDE)
    xs = x_peak + side * distances
    xs = np.clip(xs, lo, hi)
    xs = np.unique(xs)
    if side < 0:
        xs = xs[::-1]

    rho0, rho1, rho2, c0, c1, c2 = materials
    ws = energia_total_coreshell(
        xs, delta, rho0, rho1, rho2, c0, c1, c2, lmax=lmax
    )
    vals = np.asarray(ws, dtype=float) - level

    # Comecamos no ponto mais proximo do pico e caminhamos para fora.
    if side < 0:
        order = np.argsort(np.abs(xs - x_peak))
    else:
        order = np.argsort(np.abs(xs - x_peak))
    xs_ord = xs[order]
    vals_ord = vals[order]

    prev_x = float(x_peak)
    prev_v = float(w_peak - level)
    for xx, vv in zip(xs_ord, vals_ord):
        if not np.isfinite(vv):
            continue
        xx = float(xx)
        vv = float(vv)
        if vv == 0.0:
            return xx
        if prev_v > 0.0 and vv < 0.0:
            a, b = sorted((prev_x, xx))

            def f(root_x):
                value = _energia_scalar(root_x, delta, materials, lmax)
                return value - level if np.isfinite(value) else np.nan

            try:
                return float(brentq(f, a, b, xtol=1e-13, rtol=1e-12, maxiter=100))
            except Exception:
                return np.nan
        prev_x, prev_v = xx, vv

    return np.nan


def _energia_vetor(x_vals, delta, materials, lmax):
    """energia_total_coreshell com a assinatura (materials desempacotado)
    ja resolvida, para reaproveitar em lote nos refinamentos abaixo."""
    rho0, rho1, rho2, c0, c1, c2 = materials
    w = energia_total_coreshell(
        np.asarray(x_vals, dtype=float), delta, rho0, rho1, rho2, c0, c1, c2, lmax=lmax
    )
    return np.asarray(w, dtype=float)


def _refine_energy_peak_vetorizado(
    x_left, x_seed, x_right, delta, materials, lmax, n_sub=61, n_zoom=61, zoom_factor=0.08
):
    """Mesmo objetivo de _refine_energy_peak, mas com 2 chamadas vetorizadas
    (grade grossa + zoom) em vez de ~20-30 avaliacoes escalares sequenciais
    do minimize_scalar. Termina com ajuste parabolico de 3 pontos para
    precisao sub-grade."""
    x0, x1 = float(x_left), float(x_right)
    if not (x1 > x0):
        w_seed = _energia_scalar(x_seed, delta, materials, lmax)
        return (float(x_seed), w_seed) if np.isfinite(w_seed) else (np.nan, np.nan)

    xs = np.linspace(x0, x1, n_sub)
    ws = _energia_vetor(xs, delta, materials, lmax)
    finite = np.isfinite(ws)
    if not np.any(finite):
        return np.nan, np.nan
    i = int(np.argmax(np.where(finite, ws, -np.inf)))
    x_center = float(xs[i])

    zlo = max(x0, x_center - zoom_factor * (x1 - x0))
    zhi = min(x1, x_center + zoom_factor * (x1 - x0))
    if not (zhi > zlo):
        return x_center, float(ws[i])
    xz = np.linspace(zlo, zhi, n_zoom)
    wz = _energia_vetor(xz, delta, materials, lmax)
    finite_z = np.isfinite(wz)
    if not np.any(finite_z):
        return x_center, float(ws[i])
    j = int(np.argmax(np.where(finite_z, wz, -np.inf)))

    if j <= 0 or j >= n_zoom - 1:
        return float(xz[j]), float(wz[j])

    xa, xb, xc = xz[j - 1], xz[j], xz[j + 1]
    ya, yb, yc = wz[j - 1], wz[j], wz[j + 1]
    denom = (xa - xb) * (xa - xc) * (xb - xc)
    if denom == 0.0 or not np.isfinite(denom):
        return float(xb), float(yb)
    A = (xc * (yb - ya) + xb * (ya - yc) + xa * (yc - yb)) / denom
    if A >= 0 or not np.isfinite(A):
        return float(xb), float(yb)
    B = (xc**2 * (ya - yb) + xb**2 * (yc - ya) + xa**2 * (yb - yc)) / denom
    x_peak = -B / (2 * A)
    if not (xa <= x_peak <= xc):
        return float(xb), float(yb)
    w_peak = _energia_scalar(x_peak, delta, materials, lmax)
    if not np.isfinite(w_peak):
        return float(xb), float(yb)
    return float(x_peak), float(w_peak)


def _find_half_height_crossing_vetorizado(
    x_peak, w_peak, background, gamma, lo, hi, delta, materials, lmax, side,
    n_scan=ENERGY_FWHM_POINTS_PER_SIDE, n_zoom=61,
):
    """Mesmo objetivo de _find_half_height_crossing, mas troca o brentq final
    (varias dezenas de avaliacoes escalares) por 1 zoom vetorizado +
    interpolacao linear entre os 2 pontos que cercam o cruzamento."""
    if not (np.isfinite(w_peak) and np.isfinite(background) and w_peak > background):
        return np.nan

    level = background + 0.5 * (w_peak - background)
    max_distance = (x_peak - lo) if side < 0 else (hi - x_peak)
    if max_distance <= 0.0:
        return np.nan

    base = gamma if np.isfinite(gamma) and gamma > 0.0 else max_distance / 100.0
    d_min = max(
        np.finfo(float).eps * max(1.0, abs(x_peak)) * 32.0,
        min(base * 1e-3, max_distance * 1e-3),
    )
    if d_min >= max_distance:
        d_min = max_distance * 1e-6
    if d_min <= 0.0:
        return np.nan

    distances = np.geomspace(d_min, max_distance, n_scan)
    xs = x_peak + side * distances
    xs = np.clip(xs, lo, hi)
    xs = np.unique(xs)
    if side < 0:
        xs = xs[::-1]

    ws = _energia_vetor(xs, delta, materials, lmax)
    vals = ws - level

    order = np.argsort(np.abs(xs - x_peak))
    xs_ord, vals_ord = xs[order], vals[order]

    prev_x, prev_v = float(x_peak), float(w_peak - level)
    for xx, vv in zip(xs_ord, vals_ord):
        if not np.isfinite(vv):
            continue
        xx, vv = float(xx), float(vv)
        if vv == 0.0:
            return xx
        if prev_v > 0.0 and vv < 0.0:
            a, b = sorted((prev_x, xx))
            xz = np.linspace(a, b, n_zoom)
            wz = _energia_vetor(xz, delta, materials, lmax)
            vz = wz - level
            finite_z = np.isfinite(vz)
            if not np.any(finite_z):
                return 0.5 * (a + b)
            cross = np.flatnonzero(
                finite_z[:-1] & finite_z[1:] & (vz[:-1] > 0) & (vz[1:] <= 0)
            )
            if len(cross) == 0:
                k = int(np.argmin(np.abs(np.where(finite_z, vz, np.inf))))
                return float(xz[k])
            k = int(cross[0])
            x1_, x2_ = xz[k], xz[k + 1]
            v1_, v2_ = vz[k], vz[k + 1]
            if v1_ == v2_:
                return float(x1_)
            frac = v1_ / (v1_ - v2_)
            return float(x1_ + frac * (x2_ - x1_))
        prev_x, prev_v = xx, vv
    return np.nan


def _validate_candidate_by_energy(candidate, delta, materials, x_min, x_max, lmax):
    """Valida um zero de D_l como pico da energia interna no eixo real."""
    x0 = float(candidate["pole_real"])
    gamma = float(candidate["Gamma_pole"])

    half_window = ENERGY_WINDOW_GAMMA * gamma if np.isfinite(gamma) else ENERGY_WINDOW_MIN
    half_window = float(np.clip(half_window, ENERGY_WINDOW_MIN, ENERGY_WINDOW_MAX))
    lo = max(float(x_min), x0 - half_window)
    hi = min(float(x_max), x0 + half_window)

    result = {
        "energy_peak_found": False,
        "xM_energy_peak": np.nan,
        "WT_peak": np.nan,
        "WT_background": np.nan,
        "energy_ratio": np.nan,
        "energy_log10_prominence": np.nan,
        "FWHM_energy": np.nan,
        "Q_energy": np.nan,
        "energy_peak_shift": np.nan,
        "energy_window_lo": lo,
        "energy_window_hi": hi,
    }

    if not (hi > lo):
        return result

    x_grid = _adaptive_energy_grid(x0, gamma, lo, hi)
    rho0, rho1, rho2, c0, c1, c2 = materials
    w_grid = energia_total_coreshell(
        x_grid, delta, rho0, rho1, rho2, c0, c1, c2, lmax=lmax
    )
    w_grid = np.asarray(w_grid, dtype=float)

    maxima = _local_maxima_indices(x_grid, w_grid)
    if len(maxima) == 0:
        return result

    # O maximo associado ao polo e o maximo local mais proximo de Re(x_p),
    # nao necessariamente o maior pico da janela.
    i = int(maxima[np.argmin(np.abs(x_grid[maxima] - x0))])
    if i <= 0 or i >= len(x_grid) - 1:
        return result

    x_peak, w_peak = _refine_energy_peak_vetorizado(
        x_grid[i - 1], x_grid[i], x_grid[i + 1], delta, materials, lmax
    )
    if not (np.isfinite(x_peak) and np.isfinite(w_peak) and w_peak > 0.0):
        return result

    shift = abs(x_peak - x0)
    if shift > ENERGY_PEAK_MATCH_FRACTION * (hi - lo) * 0.5:
        return result

    # Verificacao local independente para evitar classificar ponto de sela ou
    # flutuacao numerica como maximo.
    eps = max(
        gamma * 0.05 if np.isfinite(gamma) and gamma > 0.0 else 0.0,
        (hi - lo) * 1e-6,
        np.finfo(float).eps * max(1.0, abs(x_peak)) * 64.0,
    )
    xl = max(lo, x_peak - eps)
    xr = min(hi, x_peak + eps)
    wl = _energia_scalar(xl, delta, materials, lmax)
    wr = _energia_scalar(xr, delta, materials, lmax)
    if not (np.isfinite(wl) and np.isfinite(wr)):
        return result

    scale = max(abs(w_peak), abs(wl), abs(wr), np.finfo(float).tiny)
    local_margin = w_peak - max(wl, wr)
    if local_margin <= ENERGY_LOCAL_MAX_REL_TOL * scale:
        return result

    background = _background_energy(x_grid, w_grid, x_peak, gamma, lo, hi)
    if np.isfinite(background) and background > 0.0:
        ratio = float(w_peak / background)
        log_prominence = float(np.log10(ratio)) if ratio > 0.0 else np.nan
    elif np.isfinite(background) and background == 0.0 and w_peak > 0.0:
        ratio = np.inf
        log_prominence = np.inf
    else:
        ratio = np.nan
        log_prominence = np.nan

    x_left = _find_half_height_crossing_vetorizado(
        x_peak, w_peak, background, gamma, lo, hi,
        delta, materials, lmax, side=-1,
    )
    x_right = _find_half_height_crossing_vetorizado(
        x_peak, w_peak, background, gamma, lo, hi,
        delta, materials, lmax, side=+1,
    )
    if np.isfinite(x_left) and np.isfinite(x_right) and x_right > x_left:
        fwhm_energy = float(x_right - x_left)
        q_energy = float(x_peak / fwhm_energy) if fwhm_energy > 0.0 else np.inf
    else:
        fwhm_energy = np.nan
        q_energy = np.nan

    # Reforco de precisao, raro: so dispara quando a largura estimada cai no
    # regime abaixo do piso de confianca do refino vetorizado (perto do
    # ruido numerico de ponto flutuante, ~1e-11-1e-12) ou quando o zoom
    # vetorizado nao chegou a bracketear o cruzamento de meia altura. Repete
    # com o solver escalar original (brentq/minimize_scalar), mais lento
    # porem ja e o metodo auditado nesse regime extremo.
    needs_fallback = (not np.isfinite(fwhm_energy)) or (fwhm_energy < VALIDATION_FALLBACK_FWHM_TRUST)
    if needs_fallback:
        x_peak_precise, w_peak_precise = _refine_energy_peak(
            x_grid[i - 1], x_grid[i], x_grid[i + 1], delta, materials, lmax
        )
        if np.isfinite(x_peak_precise) and np.isfinite(w_peak_precise) and w_peak_precise > 0.0:
            x_peak, w_peak = x_peak_precise, w_peak_precise
            background_precise = _background_energy(x_grid, w_grid, x_peak, gamma, lo, hi)
            if np.isfinite(background_precise):
                background = background_precise
                if background > 0.0:
                    ratio = float(w_peak / background)
                    log_prominence = float(np.log10(ratio)) if ratio > 0.0 else np.nan
                elif background == 0.0 and w_peak > 0.0:
                    ratio, log_prominence = np.inf, np.inf
            x_left = _find_half_height_crossing(
                x_peak, w_peak, background, gamma, lo, hi,
                delta, materials, lmax, side=-1,
            )
            x_right = _find_half_height_crossing(
                x_peak, w_peak, background, gamma, lo, hi,
                delta, materials, lmax, side=+1,
            )
            if np.isfinite(x_left) and np.isfinite(x_right) and x_right > x_left:
                fwhm_energy = float(x_right - x_left)
                q_energy = float(x_peak / fwhm_energy) if fwhm_energy > 0.0 else np.inf
            else:
                fwhm_energy = np.nan
                q_energy = np.nan

    result.update(
        {
            "energy_peak_found": True,
            "xM_energy_peak": float(x_peak),
            "WT_peak": float(w_peak),
            "WT_background": background,
            "energy_ratio": ratio,
            "energy_log10_prominence": log_prominence,
            "FWHM_energy": fwhm_energy,
            "Q_energy": q_energy,
            "energy_peak_shift": float(x_peak - x0),
        }
    )
    return result


def _candidate_width_for_merge(candidate):
    width = candidate.get("FWHM_energy", np.nan)
    if np.isfinite(width) and width > 0.0:
        return float(width)
    gamma = candidate.get("Gamma_pole", np.nan)
    if np.isfinite(gamma) and gamma > 0.0:
        return float(gamma)
    return ENERGY_MERGE_ABS_TOL


def _merge_energy_resonances(candidates):
    """Agrupa polos que apontam para o mesmo maximo de energia total."""
    active = [c for c in candidates if c.get("energy_peak_found", False)]
    active.sort(key=lambda c: c["xM_energy_peak"])
    if not active:
        return []

    groups = []
    current = [active[0]]
    for candidate in active[1:]:
        prev = current[-1]
        dx = abs(candidate["xM_energy_peak"] - prev["xM_energy_peak"])
        width_scale = min(
            _candidate_width_for_merge(candidate),
            _candidate_width_for_merge(prev),
        )
        tol = max(ENERGY_MERGE_ABS_TOL, ENERGY_MERGE_WIDTH_FRACTION * width_scale)
        if dx <= tol:
            current.append(candidate)
        else:
            groups.append(current)
            current = [candidate]
    groups.append(current)

    resonances = []
    for group in groups:
        # Representante: maior razao pico/fundo; se indisponivel, menor
        # deslocamento entre Re(polo) e maximo da energia.
        def score(c):
            ratio = c.get("energy_ratio", np.nan)
            ratio_score = float(ratio) if np.isfinite(ratio) else -np.inf
            shift_score = -abs(float(c.get("energy_peak_shift", np.inf)))
            return (ratio_score, shift_score)

        representative = max(group, key=score)
        ls = sorted({int(c["l"]) for c in group})
        pole_reals = [float(c["pole_real"]) for c in group]
        pole_imags = [float(c["pole_imag"]) for c in group]

        resonances.append(
            {
                "xM_peak": float(representative["xM_energy_peak"]),
                "WT_peak": float(representative["WT_peak"]),
                "WT_background": representative.get("WT_background", np.nan),
                "energy_ratio": representative.get("energy_ratio", np.nan),
                "energy_log10_prominence": representative.get("energy_log10_prominence", np.nan),
                "FWHM_energy": representative.get("FWHM_energy", np.nan),
                "Q_energy": representative.get("Q_energy", np.nan),
                "primary_l": int(representative["l"]),
                "primary_pole_real": float(representative["pole_real"]),
                "primary_pole_imag": float(representative["pole_imag"]),
                "primary_Gamma_pole": float(representative["Gamma_pole"]),
                "primary_Q_pole": float(representative["Q_pole"]),
                "associated_l": ",".join(str(v) for v in ls),
                "N_associated_Dzeros": int(len(group)),
                "associated_pole_real": ";".join(f"{v:.16g}" for v in pole_reals),
                "associated_pole_imag": ";".join(f"{v:.16g}" for v in pole_imags),
                "origem": "polo_validado",
            }
        )

    resonances.sort(key=lambda r: r["xM_peak"])
    return resonances


POLE_SEARCH_TIMEOUT_DEFAULT = 120.0

# Fase 2 (refinamento automatico ao final da grade, ver _refinar_casos_pendentes):
# orcamento bem maior, ja que so sobram os casos sinalizados como risco de
# overflow -- poucos, entao da pra gastar mais tempo em cada um sem
# comprometer o prazo geral do dataset.
REFINE_POLE_SEARCH_TIMEOUT_DEFAULT = 3600.0  # 60 min por curva
REFINE_CURVE_TIMEOUT_DEFAULT = 3900.0  # 65 min (rede de seguranca dura)


def detectar_coreshell(
    delta,
    alpha,
    beta,
    regime,
    mix_factor=0.5,
    x_min=X_MIN_DEFAULT,
    x_max=X_MAX_DEFAULT,
    imag_min=IMAG_MIN_DEFAULT,
    imag_max=IMAG_MAX_DEFAULT,
    lmax=LMAX_DEFAULT,
    use_screening=True,
    screen_ratio_floor=SCREEN_RATIO_FLOOR,
    pole_search_timeout_s=POLE_SEARCH_TIMEOUT_DEFAULT,
    skip_overflow_risk=True,
    l_state_inicial=None,
):
    """Retorna (candidatos_Dzero, ressonancias_energia, auditoria).

    A auditoria do principio do argumento deve passar para todas as ordens l
    (ou a falha precisa ser energeticamente irrelevante, ver
    L_SIGNIFICANCE_THRESHOLD). Zeros de D_l sao candidatos modais. A
    contagem principal de ressonancias e feita apenas depois da confirmacao
    por maximo local de energia interna.

    Se use_screening=True (padrao), uma camada 0 barata (1 avaliacao
    vetorizada da energia numa malha grosseira + find_peaks) decide antes de
    tudo se a curva tem qualquer chance de ser ressonante. So quando ela
    aponta "sim" e que a busca completa de polos complexos por l (a parte
    cara) e realmente disparada.

    Se skip_overflow_risk=True (padrao), antes mesmo de tentar a busca de
    polos, uma conta fechada (risco_overflow_busca_polos) decide se essa
    combinacao de alpha/beta/delta provavelmente estoura float64 no plano
    complexo. Se sim, a busca de polos e pulada inteiramente (nao gasta nem
    o tempo de pole_search_timeout_s tentando) e o resultado vem direto da
    malha grosseira (picos_malha), marcado com origem="malha_grosseira" em
    cada ressonancia -- mais rapido para rodar a grade inteira agora, com
    esses poucos casos sinalizados para reprocessar com calma depois (ver
    L_SIGNIFICANCE_THRESHOLD e o modulo de precisao arbitraria). Use
    skip_overflow_risk=False para forcar a busca completa mesmo nesses
    casos (usado na segunda passada, dedicada, sem pressa de tempo).

    A contagem/medida direto na malha grosseira (metodo ja estabelecido no
    projeto: find_peaks no log10 da energia, proeminencia minima, FWHM via
    peak_widths) e' sempre calculada e guardada em audit["screening"]
    ["picos_malha"], nao so quando a curva e' descartada ou pulada -- serve
    de resultado alternativo, sempre disponivel.

    pole_search_timeout_s limita o tempo TOTAL gasto tentando encontrar os
    zeros de D_l (soma de todos os l, nao por l), usando um deadline
    (time.time() absoluto) passado por dentro da recursao -- nao usa sinal,
    entao pode conviver com o curve_timeout externo (baseado em SIGALRM) sem
    conflito. Se o tempo acabar no meio da busca, o que ja foi encontrado
    ate ali fica registrado normalmente (validado contra a energia como
    qualquer outro candidato); os l ainda nao tentados ficam marcados como
    "tempo_esgotado" em vez de contar como falha de auditoria. Use None ou
    <= 0 para desligar esse limite (busca sem prazo, como antes).

    l_state_inicial permite retomar de uma tentativa anterior: um dict
    {l: {"roots":[...], "expected_count":.., "audit_ok":..}} com os l's que
    ja fecharam antes (ver audit["l_state"] no retorno de uma chamada
    anterior). Esses l's sao reaproveitados direto, sem busca de novo, e o
    orcamento de pole_search_timeout_s vale so para os l's que ainda
    faltam -- nao precisa refazer do zero o que ja tinha fechado. Um l que
    tinha dado tempo_esgotado NAO entra em l_state (nao da pra retomar no
    meio da recursao com seguranca), entao e refeito do zero na proxima.
    """
    if not (x_min > 0.0 and x_max > x_min):
        raise ValueError("E necessario ter 0 < x_min < x_max.")
    if not (imag_min < 0.0 <= imag_max):
        raise ValueError("Use imag_min < 0 e imag_max >= 0.")

    contrast_in = contrast_from_regime(regime)
    materials = make_materials_from_alpha_beta(
        alpha, beta, contrast_in=contrast_in, mix_factor=mix_factor
    )

    audit = {}

    potencialmente_ressonante, screen_ratio, screen_x_grid, screen_w_grid = tem_potencial_ressonancia(
        delta, materials, x_min, x_max, int(lmax), ratio_floor=screen_ratio_floor,
    )
    picos_malha = _picos_por_malha_grosseira(screen_x_grid, screen_w_grid)
    audit["screening"] = {
        "ok": True,
        "ran": True,
        "ratio": float(screen_ratio),
        "potencialmente_ressonante": bool(potencialmente_ressonante),
        "picos_malha": picos_malha,
    }
    if use_screening and not potencialmente_ressonante:
        return [], [], audit

    if skip_overflow_risk:
        risco, amp_shell, amp_core = risco_overflow_busca_polos(alpha, beta, delta, regime)
        audit["risco_overflow"] = {
            "ok": True,
            "risco": risco,
            "amp_shell": amp_shell,
            "amp_core": amp_core,
        }
        if risco:
            return [], _resonancias_fallback_malha(picos_malha), audit

    raw = []
    l_state = {}
    ls_prontos = set()
    if l_state_inicial:
        for l_str, info in l_state_inicial.items():
            l = int(l_str)
            if l > int(lmax):
                continue
            roots_l = [complex(z) for z in info["roots"]]
            expected_l = info.get("expected_count")
            audit_ok = bool(info.get("audit_ok", False))
            audit[l] = {
                "expected_count": expected_l,
                "found_count": len(roots_l),
                "ok": audit_ok,
                "retomado_de_tentativa_anterior": True,
            }
            if info.get("energeticamente_relevante") is not None:
                audit[l]["energeticamente_relevante"] = info["energeticamente_relevante"]
                audit[l]["contribuicao_relativa"] = info.get("contribuicao_relativa", np.nan)
            l_state[l] = {
                "roots": roots_l, "expected_count": expected_l, "audit_ok": audit_ok,
                "energeticamente_relevante": info.get("energeticamente_relevante"),
                "contribuicao_relativa": info.get("contribuicao_relativa"),
            }
            ls_prontos.add(l)
            for z in roots_l:
                raw.append((l, complex(z), expected_l))

    lmax_int = int(lmax)
    pole_search_deadline = (
        time.time() + float(pole_search_timeout_s)
        if pole_search_timeout_s is not None and float(pole_search_timeout_s) > 0.0
        else None
    )
    busca_polos_interrompida = False

    for l in range(lmax_int + 1):
        if l in ls_prontos:
            continue

        if pole_search_deadline is not None and time.time() >= pole_search_deadline:
            audit[l] = {
                "expected_count": None, "found_count": 0, "ok": False,
                "tempo_esgotado": True,
            }
            busca_polos_interrompida = True
            continue

        roots_l, expected_l, audit_ok = polos_por_l(
            l, delta, materials, x_min, x_max, imag_min, imag_max,
            deadline=pole_search_deadline,
        )

        if expected_l is None and not audit_ok:
            # Tempo esgotado no meio da busca (ou da auditoria) desse l.
            # O que foi encontrado ate aqui fica, mas nao auditado, e esse l
            # NAO entra em l_state -- uma proxima tentativa refaz do zero,
            # ja que nao da pra retomar no meio da recursao com seguranca.
            audit[l] = {
                "expected_count": None,
                "found_count": int(len(roots_l)),
                "ok": False,
                "tempo_esgotado": True,
            }
            busca_polos_interrompida = True
            for z in roots_l:
                raw.append((int(l), complex(z), None))
            continue

        audit[l] = {
            "expected_count": int(expected_l),
            "found_count": int(len(roots_l)),
            "ok": bool(audit_ok),
        }
        energ_relevante = None
        contrib_ratio_val = None
        if not audit_ok:
            contrib_ratio = _relevancia_energetica_l(
                l, delta, materials, screen_x_grid, screen_w_grid
            )
            contrib_ratio_val = float(contrib_ratio)
            energ_relevante = bool(contrib_ratio >= L_SIGNIFICANCE_THRESHOLD)
            audit[l]["contribuicao_relativa"] = contrib_ratio_val
            audit[l]["energeticamente_relevante"] = energ_relevante
            if energ_relevante:
                raise PoleAuditError(
                    f"Falha de auditoria em l={l} (contribuicao relativa "
                    f"{contrib_ratio:.2e}, acima do limiar de relevancia "
                    f"{L_SIGNIFICANCE_THRESHOLD:.0e}): expected={expected_l}, "
                    f"found={len(roots_l)}",
                    audit=audit,
                )
            # contribuicao desprezivel: segue em frente com o que foi
            # encontrado nesse l, sem derrubar a curva inteira.
        # l com resultado definitivo (audit_ok, ou audit_ok=False mas ja
        # aceito como irrelevante): entra em l_state, uma proxima tentativa
        # nao precisa refazer esse l.
        l_state[l] = {
            "roots": [complex(z) for z in roots_l],
            "expected_count": int(expected_l),
            "audit_ok": bool(audit_ok),
            "energeticamente_relevante": energ_relevante,
            "contribuicao_relativa": contrib_ratio_val,
        }
        for z in roots_l:
            raw.append((int(l), complex(z), int(expected_l)))

    audit["l_state"] = l_state

    audit["busca_polos"] = {
        "ok": True,
        "completa": not busca_polos_interrompida,
        "timeout_s": None if pole_search_timeout_s is None else float(pole_search_timeout_s),
    }

    raw.sort(key=lambda item: (item[1].real, item[0], item[1].imag))

    candidates = []
    for l, z, expected_l in raw:
        x_res = float(z.real)
        pole_imag = float(z.imag)
        gamma = float(2.0 * abs(pole_imag))
        q_pole = float(x_res / gamma) if gamma > 0.0 else np.inf

        candidate = {
            "l": int(l),
            "pole_real": x_res,
            "pole_imag": pole_imag,
            "Gamma_pole": gamma,
            "Q_pole": q_pole,
            "argument_count_l": None if expected_l is None else int(expected_l),
            "argument_audit_ok": bool(audit.get(l, {}).get("ok", False)),
        }
        candidate.update(
            _validate_candidate_by_energy(
                candidate, delta, materials, x_min, x_max, int(lmax)
            )
        )
        candidates.append(candidate)

    resonances = _merge_energy_resonances(candidates)
    return candidates, resonances, audit


# ============================================================================
# 5. TIMEOUT POR CURVA
# ============================================================================
class CurveTimeoutError(TimeoutError):
    pass


def _timeout_handler(signum, frame):
    raise CurveTimeoutError("Tempo maximo da curva excedido.")


class curve_timeout:
    """Context manager Linux/Unix para limite de wall-clock por curva."""
    def __init__(self, seconds):
        self.seconds = float(seconds)
        self.old_handler = None

    def __enter__(self):
        if self.seconds > 0.0:
            if not hasattr(signal, "SIGALRM"):
                raise RuntimeError("Timeout por SIGALRM requer Linux/Unix.")
            self.old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.seconds > 0.0:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            if self.old_handler is not None:
                signal.signal(signal.SIGALRM, self.old_handler)
        return False


# ============================================================================
# 6. GRADE BALANCEADA EM ETA
# ============================================================================
def build_eta_grid(target_per_bin=TARGET_PER_BIN, seed=42):
    p1 = np.linspace(0.001, 0.009, 10)
    p2 = np.linspace(0.01, 0.09, 10)
    p3 = np.linspace(0.1, 0.9, 10)
    p4 = np.linspace(1.0, 9.0, 10)
    p5 = np.linspace(10.0, 90.0, 10)
    p6 = np.linspace(100.0, 900.0, 10)
    vals = np.unique(
        np.concatenate([p1, p2, p3, p4, p5, p6, [1000.0, 2000.0, 3000.0]])
    )

    alpha_mesh, beta_mesh = np.meshgrid(vals, vals, indexing="ij")
    alpha_all = alpha_mesh.ravel()
    beta_all = beta_mesh.ravel()
    eta_all = (alpha_all - beta_all) / (alpha_all + beta_all)

    bin_idx = np.searchsorted(BINS_ETA, eta_all, side="left") - 1
    bin_idx[eta_all == BINS_ETA[0]] = 0
    bin_idx = np.clip(bin_idx, 0, len(BINS_ETA) - 2)

    alphas = []
    betas = []
    rng = np.random.default_rng(seed)

    for i in range(len(BINS_ETA) - 1):
        mask = bin_idx == i
        aa = alpha_all[mask]
        bb = beta_all[mask]
        n_available = aa.size

        if n_available > target_per_bin:
            idxs = np.linspace(0, n_available - 1, target_per_bin, dtype=int)
            aa_sel = aa[idxs]
            bb_sel = bb[idxs]
        else:
            aa_sel = aa.copy()
            bb_sel = bb.copy()
            missing = target_per_bin - n_available
            if missing > 0:
                eta_mid = 0.5 * (BINS_ETA[i] + BINS_ETA[i + 1])
                scales = rng.choice(vals, size=missing, replace=True)
                aa_extra = scales * (1.0 + eta_mid) / 2.0
                bb_extra = scales * (1.0 - eta_mid) / 2.0
                aa_sel = np.concatenate([aa_sel, aa_extra])
                bb_sel = np.concatenate([bb_sel, bb_extra])

        alphas.append(aa_sel)
        betas.append(bb_sel)

    return np.concatenate(alphas), np.concatenate(betas)


# ============================================================================
# 7. CHECKPOINT E EXPORTACAO
# ============================================================================
def make_paths(output_root, regime, lmax):
    save_dir = os.path.join(
        os.path.abspath(output_root),
        f"resultados_ressonancias_coreshell_lmax{int(lmax)}_{regime}",
    )
    export_dir = os.path.join(save_dir, "exports")
    os.makedirs(export_dir, exist_ok=True)
    run_name = f"ressonancias_coreshell_lmax{int(lmax)}_{regime}_final"
    return {
        "save_dir": save_dir,
        "checkpoint": os.path.join(save_dir, f"checkpoint_{run_name}.npz"),
        "candidates_csv": os.path.join(export_dir, f"candidatos_{run_name}.csv"),
        "resonances_csv": os.path.join(export_dir, f"ressonancias_{run_name}.csv"),
        "summary_csv": os.path.join(export_dir, f"resumo_{run_name}.csv"),
    }


def safe_savez(final_file, **arrays):
    os.makedirs(os.path.dirname(final_file), exist_ok=True)
    tmp = os.path.join(
        os.path.dirname(final_file),
        "." + os.path.basename(final_file) + f".tmp_{uuid.uuid4().hex}.npz",
    )
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, final_file)


def save_checkpoint(
    paths,
    curves,
    done,
    alpha_grid,
    beta_grid,
    regime,
    lmax,
    x_min,
    x_max,
    imag_min,
    imag_max,
    timeout_seconds,
):
    safe_savez(
        paths["checkpoint"],
        checkpoint_version=CHECKPOINT_VERSION,
        detector=np.array("argument_principle_plus_internal_energy_validation"),
        regime=np.array(regime),
        lmax=int(lmax),
        x_min=float(x_min),
        x_max=float(x_max),
        imag_min=float(imag_min),
        imag_max=float(imag_max),
        timeout_seconds=float(timeout_seconds),
        curves=np.array(curves, dtype=object),
        done=np.asarray(done, dtype=bool),
        alpha_grid=np.asarray(alpha_grid, dtype=float),
        beta_grid=np.asarray(beta_grid, dtype=float),
        delta_values=DELTA_VALUES,
    )


def load_checkpoint(paths, total, regime, lmax, x_min, x_max, imag_min, imag_max):
    if not os.path.exists(paths["checkpoint"]):
        return {}, np.zeros(total, dtype=bool), None, None

    with np.load(paths["checkpoint"], allow_pickle=True) as data:
        compatible = (
            int(data.get("checkpoint_version", -1)) == CHECKPOINT_VERSION
            and str(data.get("regime", "")) == regime
            and int(data.get("lmax", -1)) == int(lmax)
            and np.isclose(float(data.get("x_min", np.nan)), x_min)
            and np.isclose(float(data.get("x_max", np.nan)), x_max)
            and np.isclose(float(data.get("imag_min", np.nan)), imag_min)
            and np.isclose(float(data.get("imag_max", np.nan)), imag_max)
        )
        if not compatible:
            print("Checkpoint encontrado, mas incompativel com esta configuracao. Ignorando.")
            return {}, np.zeros(total, dtype=bool), None, None

        curves = data["curves"].item()
        done = np.asarray(data["done"], dtype=bool)
        alpha_grid = np.asarray(data["alpha_grid"], dtype=float)
        beta_grid = np.asarray(data["beta_grid"], dtype=float)

    if len(done) != total:
        print("Checkpoint com tamanho de grade diferente. Ignorando.")
        return {}, np.zeros(total, dtype=bool), None, None

    return curves, done, alpha_grid, beta_grid


def export_csv(paths, curves, lmax):
    candidate_rows = []
    resonance_rows = []
    summary_rows = []

    keys = sorted(
        curves.keys(),
        key=lambda key: tuple(int(v) for v in key.split("_")),
    )

    for key in keys:
        item = curves[key]
        idd, ip = (int(v) for v in key.split("_"))
        status = item["status"]
        candidates = item.get("candidatos", []) if status == "ok" else []
        resonances = item.get("ressonancias", []) if status == "ok" else []

        if status == "ok":
            n_dzeros = len(candidates)
            n_energy_candidates = sum(bool(c.get("energy_peak_found", False)) for c in candidates)
            n_res = len(resonances)
            n_l_res = len({
                int(lval)
                for r in resonances
                for lval in str(r.get("associated_l", "")).split(",")
                if str(lval).strip() != ""
            })
            x_first = resonances[0]["xM_peak"] if resonances else np.nan
            q_energy = [r["Q_energy"] for r in resonances if np.isfinite(r.get("Q_energy", np.nan))]
            ratios = [r["energy_ratio"] for r in resonances if np.isfinite(r.get("energy_ratio", np.nan))]
            q_max = max(q_energy) if q_energy else np.nan
            ratio_max = max(ratios) if ratios else np.nan
            has_resonance = bool(n_res > 0)
        else:
            # Timeout/falha nunca vira artificialmente "zero ressonancias".
            n_dzeros = np.nan
            n_energy_candidates = np.nan
            n_res = np.nan
            n_l_res = np.nan
            x_first = np.nan
            q_max = np.nan
            ratio_max = np.nan
            has_resonance = np.nan

        screening = item.get("audit", {}).get("screening", {}) if status == "ok" else {}
        screened_out = bool(screening.get("ran", False) and not screening.get("potencialmente_ressonante", True))
        screening_ratio = screening.get("ratio", np.nan)
        risco_overflow_info = item.get("audit", {}).get("risco_overflow", {}) if status == "ok" else {}
        resolvido_por_malha = bool(risco_overflow_info.get("risco", False))

        summary_rows.append(
            {
                "idd": idd,
                "ip": ip,
                "delta": item["delta"],
                "alpha": item["alpha"],
                "beta": item["beta"],
                "eta": item["eta"],
                "regime": item["regime"],
                "lmax": int(lmax),
                "status": status,
                "N_Dzeros_total": n_dzeros,
                "N_Dzeros_com_maximo_energia": n_energy_candidates,
                "N_res_energy": n_res,
                "N_l_res_energy": n_l_res,
                "has_resonance": has_resonance,
                "x_primeira_res_energy": x_first,
                "Q_energy_max": q_max,
                "energy_ratio_max": ratio_max,
                "audit_all_ok": bool(item.get("audit_all_ok", False)) if status == "ok" else False,
                "screened_out": screened_out,
                "screening_ratio": screening_ratio,
                "resolvido_por_malha": resolvido_por_malha,
                "runtime_s": item.get("runtime_s", np.nan),
                "error": item.get("error", ""),
            }
        )

        for candidate_number, c in enumerate(candidates, start=1):
            candidate_rows.append(
                {
                    "idd": idd,
                    "ip": ip,
                    "delta": item["delta"],
                    "alpha": item["alpha"],
                    "beta": item["beta"],
                    "eta": item["eta"],
                    "regime": item["regime"],
                    "candidate_num": candidate_number,
                    "l": c["l"],
                    "pole_real": c["pole_real"],
                    "pole_imag": c["pole_imag"],
                    "Gamma_pole": c["Gamma_pole"],
                    "Q_pole": c["Q_pole"],
                    "argument_count_l": c["argument_count_l"],
                    "argument_audit_ok": c["argument_audit_ok"],
                    "energy_peak_found": c.get("energy_peak_found", False),
                    "xM_energy_peak": c.get("xM_energy_peak", np.nan),
                    "WT_peak": c.get("WT_peak", np.nan),
                    "WT_background": c.get("WT_background", np.nan),
                    "energy_ratio": c.get("energy_ratio", np.nan),
                    "energy_log10_prominence": c.get("energy_log10_prominence", np.nan),
                    "FWHM_energy": c.get("FWHM_energy", np.nan),
                    "Q_energy": c.get("Q_energy", np.nan),
                    "energy_peak_shift": c.get("energy_peak_shift", np.nan),
                    "energy_window_lo": c.get("energy_window_lo", np.nan),
                    "energy_window_hi": c.get("energy_window_hi", np.nan),
                }
            )

        for resonance_number, r in enumerate(resonances, start=1):
            resonance_rows.append(
                {
                    "idd": idd,
                    "ip": ip,
                    "delta": item["delta"],
                    "alpha": item["alpha"],
                    "beta": item["beta"],
                    "eta": item["eta"],
                    "regime": item["regime"],
                    "res_num": resonance_number,
                    "xM_peak": r["xM_peak"],
                    "WT_peak": r["WT_peak"],
                    "WT_background": r.get("WT_background", np.nan),
                    "energy_ratio": r.get("energy_ratio", np.nan),
                    "energy_log10_prominence": r.get("energy_log10_prominence", np.nan),
                    "FWHM_energy": r.get("FWHM_energy", np.nan),
                    "Q_energy": r.get("Q_energy", np.nan),
                    "primary_l": r["primary_l"],
                    "primary_pole_real": r["primary_pole_real"],
                    "primary_pole_imag": r["primary_pole_imag"],
                    "primary_Gamma_pole": r["primary_Gamma_pole"],
                    "primary_Q_pole": r["primary_Q_pole"],
                    "associated_l": r["associated_l"],
                    "N_associated_Dzeros": r["N_associated_Dzeros"],
                    "associated_pole_real": r["associated_pole_real"],
                    "associated_pole_imag": r["associated_pole_imag"],
                    "origem": r.get("origem", "polo_validado"),
                }
            )

    pd.DataFrame(candidate_rows).to_csv(paths["candidates_csv"], index=False)
    pd.DataFrame(resonance_rows).to_csv(paths["resonances_csv"], index=False)
    pd.DataFrame(summary_rows).to_csv(paths["summary_csv"], index=False)
    print(f"CSV de candidatos:   {paths['candidates_csv']}")
    print(f"CSV de ressonancias: {paths['resonances_csv']}")
    print(f"CSV de resumo:       {paths['summary_csv']}")


# ============================================================================
# 8. DISPARADOR DO DATASET
# ============================================================================
def run_regime(args, regime):
    alpha_new, beta_new = build_eta_grid(
        target_per_bin=args.target_per_bin,
        seed=args.seed,
    )
    n_pairs = len(alpha_new)
    total = n_pairs * len(DELTA_VALUES)

    paths = make_paths(args.output_root, regime, args.lmax)
    curves, done, alpha_saved, beta_saved = load_checkpoint(
        paths,
        total,
        regime,
        args.lmax,
        args.x_min,
        args.x_max,
        args.imag_min,
        args.imag_max,
    )

    if alpha_saved is not None and len(alpha_saved) == n_pairs:
        alpha_grid = alpha_saved
        beta_grid = beta_saved
    else:
        alpha_grid = alpha_new
        beta_grid = beta_new

    if args.retry_failed:
        for key, item in curves.items():
            if item.get("status") in {"timeout", "error", "audit_failed"}:
                idd, ip = (int(v) for v in key.split("_"))
                flat = idd * n_pairs + ip
                if 0 <= flat < len(done):
                    done[flat] = False

    global_start = time.time()
    runtimes_ok = []
    processed_this_run = 0

    print("=" * 78)
    print(f"Regime: {regime}")
    print(f"Curvas totais: {total} ({n_pairs} pares x {len(DELTA_VALUES)} deltas)")
    print(f"l = 0,...,{args.lmax}")
    print(f"Dominio real: [{args.x_min}, {args.x_max}]")
    print(f"Dominio imaginario: [{args.imag_min}, {args.imag_max}]")
    print(f"Timeout por curva: {args.curve_timeout:.1f} s")
    print(f"Checkpoint: {paths['checkpoint']}")
    print("=" * 78)

    try:
        stop_requested = False
        for idd, delta in enumerate(DELTA_VALUES):
            if stop_requested:
                break
            for ip in range(n_pairs):
                flat = idd * n_pairs + ip
                if done[flat]:
                    continue

                if args.max_curves is not None and processed_this_run >= args.max_curves:
                    stop_requested = True
                    break

                alpha = float(alpha_grid[ip])
                beta = float(beta_grid[ip])
                eta = float(eta_from_alpha_beta(alpha, beta))
                key = f"{idd}_{ip}"
                t0 = time.time()

                status = "ok"
                error_message = ""
                audit = {}
                candidates = []
                resonances = []

                try:
                    with curve_timeout(args.curve_timeout):
                        candidates, resonances, audit = detectar_coreshell(
                            delta=float(delta),
                            alpha=alpha,
                            beta=beta,
                            regime=regime,
                            mix_factor=args.mix_factor,
                            x_min=args.x_min,
                            x_max=args.x_max,
                            imag_min=args.imag_min,
                            imag_max=args.imag_max,
                            lmax=args.lmax,
                            use_screening=not args.no_screening,
                            screen_ratio_floor=args.screen_ratio_floor,
                            pole_search_timeout_s=args.pole_search_timeout,
                            skip_overflow_risk=not args.no_skip_overflow_risk,
                        )
                except CurveTimeoutError:
                    status = "timeout"
                    error_message = f"curve_timeout>{args.curve_timeout:.1f}s"
                except PoleAuditError as exc:
                    status = "audit_failed"
                    error_message = str(exc)
                    audit = exc.audit
                except Exception as exc:
                    status = "error"
                    error_message = f"{type(exc).__name__}: {exc}"

                runtime = float(time.time() - t0)
                if status == "ok":
                    runtimes_ok.append(runtime)

                curves[key] = {
                    "delta": float(delta),
                    "alpha": alpha,
                    "beta": beta,
                    "eta": eta,
                    "regime": regime,
                    "status": status,
                    "candidatos": candidates if status == "ok" else [],
                    "ressonancias": resonances if status == "ok" else [],
                    "audit": audit,
                    "audit_all_ok": bool(
                        status == "ok"
                        and all(_audit_entry_aceitavel(v) for v in audit.values())
                    ),
                    "runtime_s": runtime,
                    "error": error_message,
                }
                done[flat] = True
                processed_this_run += 1

                n_done = int(np.sum(done))
                avg = float(np.mean(runtimes_ok)) if runtimes_ok else np.nan
                if np.isfinite(avg):
                    eta_seconds = avg * int(np.sum(~done))
                    avg_text = f"avg_ok={avg:.2f}s ETA~{eta_seconds / 3600:.1f}h"
                else:
                    avg_text = "avg_ok=n/a"

                n_candidates_text = len(candidates) if status == "ok" else "NA"
                n_res_text = len(resonances) if status == "ok" else "NA"
                busca_polos = audit.get("busca_polos", {}) if status == "ok" else {}
                risco_overflow = audit.get("risco_overflow", {}) if status == "ok" else {}
                if status == "ok" and risco_overflow.get("risco", False):
                    status_text = "ok(malha)"
                    n_picos_malha = len(audit.get("screening", {}).get("picos_malha", []))
                    extra_text = f" picos_malha={n_picos_malha} (busca de polos pulada, risco overflow)"
                elif status == "ok" and not busca_polos.get("completa", True):
                    status_text = "ok(parcial)"
                    n_picos_malha = len(audit.get("screening", {}).get("picos_malha", []))
                    extra_text = f" picos_malha={n_picos_malha}"
                else:
                    status_text = status
                    extra_text = ""
                print(
                    f"[{n_done}/{total}] delta={delta:.2f} eta={eta:+.3f} "
                    f"status={status_text} Dzeros={n_candidates_text} res_energy={n_res_text}"
                    f"{extra_text} t={runtime:.2f}s {avg_text}",
                    flush=True,
                )

                if processed_this_run % args.checkpoint_every == 0:
                    save_checkpoint(
                        paths, curves, done, alpha_grid, beta_grid,
                        regime, args.lmax,
                        args.x_min, args.x_max,
                        args.imag_min, args.imag_max,
                        args.curve_timeout,
                    )

    except KeyboardInterrupt:
        print("\nInterrompido pelo usuario. Salvando checkpoint e CSV parcial...", flush=True)
        save_checkpoint(
            paths, curves, done, alpha_grid, beta_grid,
            regime, args.lmax,
            args.x_min, args.x_max,
            args.imag_min, args.imag_max,
            args.curve_timeout,
        )
        export_csv(paths, curves, args.lmax)
        raise

    save_checkpoint(
        paths, curves, done, alpha_grid, beta_grid,
        regime, args.lmax,
        args.x_min, args.x_max,
        args.imag_min, args.imag_max,
        args.curve_timeout,
    )
    export_csv(paths, curves, args.lmax)

    statuses = {}
    for item in curves.values():
        statuses[item.get("status", "unknown")] = statuses.get(item.get("status", "unknown"), 0) + 1

    print("-" * 78)
    print(f"Finalizado este disparo em {(time.time() - global_start) / 3600:.2f} h")
    print(f"Status acumulados no checkpoint: {statuses}")
    print(f"Checkpoint NPZ: {paths['checkpoint']}")
    print(f"Candidatos CSV:   {paths['candidates_csv']}")
    print(f"Ressonancias CSV: {paths['resonances_csv']}")
    print(f"Resumo CSV:       {paths['summary_csv']}")
    print("-" * 78)

    dataset_completo = bool(np.all(done)) and not stop_requested
    if not dataset_completo:
        return

    n_completo = sum(
        1 for item in curves.values()
        if item.get("status") == "ok"
        and item.get("audit", {}).get("busca_polos", {}).get("completa", False)
        and not item.get("audit", {}).get("risco_overflow", {}).get("risco", False)
    )
    n_pendente = sum(
        1 for item in curves.values()
        if item.get("status") == "ok"
        and (
            item.get("audit", {}).get("risco_overflow", {}).get("risco", False)
            or not item.get("audit", {}).get("busca_polos", {}).get("completa", True)
        )
    )
    print("=" * 78)
    print("[DATASET COMPLETO] Todas as curvas foram processadas.")
    print(f"  {n_completo} resolvidas pela busca de polos completa.")
    if n_pendente:
        print(f"  {n_pendente} ficaram pendentes (puladas por risco de overflow ou "
              f"incompletas por tempo) -- todas entram no refinamento, sem distincao "
              f"entre as duas causas.")
    if args.no_auto_refine or n_pendente == 0:
        print("  Nenhum refinamento automatico sera' feito." if n_pendente == 0 else
              "  --no-auto-refine ativo: nao vai refinar automaticamente agora.")
        print("=" * 78)
        return
    print(f"  Iniciando refinamento desses {n_pendente} casos (busca completa de "
          f"polos, mais tempo, sem pular ninguem por risco), sem precisar reiniciar "
          f"nada.")
    print("=" * 78)

    _refinar_casos_pendentes(args, regime, paths, curves, done, alpha_grid, beta_grid, total)


def _refinar_casos_pendentes(args, regime, paths, curves, done, alpha_grid, beta_grid, total):
    """Fase 2, disparada automaticamente ao final da fase 1 (ver
    dataset_completo em run_regime): reprocessa, no MESMO disparo do
    script, TODAS as curvas que nao fecharam a busca de polos completa --
    tanto as puladas por risco de overflow quanto as que tentaram e nao
    deram tempo -- sem distinguir por formula qual "vale a pena" tentar de
    novo. skip_overflow_risk=False forca a tentativa mesmo nas marcadas
    como risco.

    Cada tentativa acumula progresso por l: os l's que ja fecharam numa
    chamada anterior (curves[key]["l_state_parcial"], salvo no proprio
    checkpoint da curva) sao passados como l_state_inicial e NAO sao
    refeitos -- so os l's que ainda faltam ganham o orcamento de tempo
    dessa vez. Isso significa que rodar o script de novo mais tarde
    continua de onde parou, em vez de recomecar do zero (e' assim que se
    da mais uma rodada de tempo num caso que ainda nao fechou: so roda de
    novo). Excecao: se o limite duro externo
    (--refine-curve-timeout, via SIGALRM) interromper a chamada, o
    progresso dessa tentativa especifica se perde (nao da pra recuperar o
    estado interno de uma chamada interrompida por sinal) -- so o que ja
    estava salvo de tentativas anteriores fica.

    So substitui o resultado FINAL da curva no checkpoint quando a busca
    completa realmente fecha (busca_polos.completa=True); enquanto isso,
    mantem o resultado publicado como estava (malha ou parcial) e so
    atualiza o l_state_parcial por baixo dos panos."""
    chaves_para_refinar = [
        key for key, item in curves.items()
        if item.get("status") == "ok"
        and (
            item.get("audit", {}).get("risco_overflow", {}).get("risco", False)
            or not item.get("audit", {}).get("busca_polos", {}).get("completa", True)
        )
    ]
    n_total_refino = len(chaves_para_refinar)
    n_melhorados = 0
    n_ainda_incompleto = 0
    refino_start = time.time()

    try:
        for i, key in enumerate(chaves_para_refinar, start=1):
            idd, ip = (int(v) for v in key.split("_"))
            delta = float(DELTA_VALUES[idd])
            alpha = float(alpha_grid[ip])
            beta = float(beta_grid[ip])
            eta = float(eta_from_alpha_beta(alpha, beta))

            t0 = time.time()
            status = "ok"
            error_message = ""
            audit = {}
            candidates = []
            resonances = []
            l_state_inicial = curves[key].get("l_state_parcial")
            try:
                with curve_timeout(args.refine_curve_timeout):
                    candidates, resonances, audit = detectar_coreshell(
                        delta=delta, alpha=alpha, beta=beta, regime=regime,
                        mix_factor=args.mix_factor,
                        x_min=args.x_min, x_max=args.x_max,
                        imag_min=args.imag_min, imag_max=args.imag_max,
                        lmax=args.lmax,
                        use_screening=not args.no_screening,
                        screen_ratio_floor=args.screen_ratio_floor,
                        pole_search_timeout_s=args.refine_pole_search_timeout,
                        skip_overflow_risk=False,
                        l_state_inicial=l_state_inicial,
                    )
            except CurveTimeoutError:
                status = "timeout"
                error_message = f"refine_curve_timeout>{args.refine_curve_timeout:.1f}s"
                # o SIGALRM externo interrompe a chamada sem devolver nada;
                # o progresso feito nessa tentativa se perde (nao ha como
                # recuperar o estado local da funcao interrompida). O que
                # ja estava salvo de tentativas anteriores continua intacto.
            except PoleAuditError as exc:
                status = "audit_failed"
                error_message = str(exc)
                audit = getattr(exc, "audit", {})
            except Exception as exc:
                status = "error"
                error_message = f"{type(exc).__name__}: {exc}"
            runtime = time.time() - t0

            novo_l_state = audit.get("l_state")
            if novo_l_state:
                curves[key]["l_state_parcial"] = novo_l_state
                n_l_prontos = sum(1 for v in novo_l_state.values() if v.get("audit_ok"))

            completou = status == "ok" and audit.get("busca_polos", {}).get("completa", False)
            if completou:
                curves[key] = {
                    "delta": delta, "alpha": alpha, "beta": beta, "eta": eta,
                    "regime": regime, "status": status,
                    "candidatos": candidates, "ressonancias": resonances,
                    "audit": audit,
                    "audit_all_ok": bool(
                        all(_audit_entry_aceitavel(v) for v in audit.values())
                    ),
                    "runtime_s": runtime, "error": error_message,
                }
                n_melhorados += 1
                tag = "MELHORADO"
            else:
                n_ainda_incompleto += 1
                tag = (
                    f"continua parcial ({len(novo_l_state)}/{args.lmax + 1} l prontos, "
                    f"acumulado)" if novo_l_state else "continua parcial (sem progresso novo)"
                )

            print(
                f"[refino {i}/{n_total_refino}] delta={delta:.2f} eta={eta:+.3f} "
                f"status={status} N_res={len(resonances)} t={runtime:.1f}s -> {tag}",
                flush=True,
            )

            if i % max(1, args.checkpoint_every) == 0:
                save_checkpoint(
                    paths, curves, done, alpha_grid, beta_grid,
                    regime, args.lmax,
                    args.x_min, args.x_max, args.imag_min, args.imag_max,
                    args.curve_timeout,
                )
    except KeyboardInterrupt:
        print("\nInterrompido durante o refinamento. Salvando checkpoint e CSV...", flush=True)
        save_checkpoint(
            paths, curves, done, alpha_grid, beta_grid,
            regime, args.lmax,
            args.x_min, args.x_max, args.imag_min, args.imag_max,
            args.curve_timeout,
        )
        export_csv(paths, curves, args.lmax)
        raise

    save_checkpoint(
        paths, curves, done, alpha_grid, beta_grid,
        regime, args.lmax,
        args.x_min, args.x_max, args.imag_min, args.imag_max,
        args.curve_timeout,
    )
    export_csv(paths, curves, args.lmax)

    print("=" * 78)
    print(f"[REFINO COMPLETO] {n_melhorados} de {n_total_refino} casos melhorados "
          f"(busca de polos fechou de verdade); {n_ainda_incompleto} continuam "
          f"parciais mesmo com mais tempo.")
    print(f"Tempo total do refinamento: {(time.time() - refino_start) / 3600:.2f} h")
    print("=" * 78)


# ============================================================================
# 9. CLI
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Gera dataset core-shell por zeros complexos de D_l validados pela energia interna."
    )
    parser.add_argument(
        "--regime",
        nargs="+",
        choices=("velocity", "mixed", "density"),
        required=True,
        help="Um ou mais regimes. Para rodar velocity e mixed em paralelo, "
             "use duas sessoes/processos separados.",
    )
    parser.add_argument("--output-root", default="./resultados_rigorosos")
    parser.add_argument("--lmax", type=int, default=LMAX_DEFAULT)
    parser.add_argument("--x-min", type=float, default=X_MIN_DEFAULT)
    parser.add_argument("--x-max", type=float, default=X_MAX_DEFAULT)
    parser.add_argument("--imag-min", type=float, default=IMAG_MIN_DEFAULT)
    parser.add_argument("--imag-max", type=float, default=IMAG_MAX_DEFAULT)
    parser.add_argument("--mix-factor", type=float, default=0.5)
    parser.add_argument(
        "--curve-timeout",
        type=float,
        default=CURVE_TIMEOUT_DEFAULT,
        help="Tempo maximo por curva em segundos (rede de seguranca dura, "
             "via SIGALRM, cobre a curva inteira). Padrao: 300 s.",
    )
    parser.add_argument(
        "--pole-search-timeout",
        type=float,
        default=POLE_SEARCH_TIMEOUT_DEFAULT,
        help="Tempo maximo gasto tentando encontrar os zeros de D_l (soma "
             "de todos os l, nao por l). Se estourar, o que ja foi "
             "encontrado ate ali fica registrado e validado normalmente; "
             "os l ainda nao tentados ficam marcados como tempo_esgotado "
             "em vez de derrubar a curva. Use 0 para desligar (sem prazo, "
             "como antes). Precisa ser menor que --curve-timeout, que "
             "continua sendo o limite duro pra curva inteira. Padrao: "
             f"{POLE_SEARCH_TIMEOUT_DEFAULT:.0f} s.",
    )
    parser.add_argument(
        "--no-skip-overflow-risk",
        action="store_true",
        help="Desliga a checagem previa de risco de overflow (formula "
             "fechada em alpha/beta/delta/regime, ver risco_overflow_"
             "busca_polos) e forca a busca completa de polos mesmo nos "
             "casos sinalizados como arriscados. Normalmente nao precisa "
             "usar isso na mao: o refinamento automatico (ver "
             "--no-auto-refine) ja faz isso sozinho depois que a grade "
             "inteira termina.",
    )
    parser.add_argument(
        "--no-auto-refine",
        action="store_true",
        help="Ao terminar a grade inteira (todas as curvas processadas), o "
             "padrao e' continuar automaticamente, no mesmo disparo, "
             "reprocessando TODAS as curvas que nao fecharam a busca de "
             "polos completa (puladas por risco de overflow ou apenas "
             "incompletas por tempo), sem distincao, com um orcamento maior "
             "(--refine-pole-search-timeout / --refine-curve-timeout). Use "
             "esta flag pra desligar isso e so parar quando a grade "
             "principal terminar.",
    )
    parser.add_argument(
        "--refine-pole-search-timeout",
        type=float,
        default=REFINE_POLE_SEARCH_TIMEOUT_DEFAULT,
        help="Orcamento de tempo (por curva) para a busca de polos na fase "
             "de refinamento automatico (ver --no-auto-refine). Bem maior "
             "que --pole-search-timeout, ja que aqui so sobram os casos "
             "dificeis e da pra deixar rodando por mais tempo sem pressa. "
             f"Padrao: {REFINE_POLE_SEARCH_TIMEOUT_DEFAULT:.0f} s "
             f"({REFINE_POLE_SEARCH_TIMEOUT_DEFAULT/60:.0f} min).",
    )
    parser.add_argument(
        "--refine-curve-timeout",
        type=float,
        default=REFINE_CURVE_TIMEOUT_DEFAULT,
        help="Limite duro (SIGALRM) por curva na fase de refinamento "
             "automatico; precisa ser maior que --refine-pole-search-"
             f"timeout. Padrao: {REFINE_CURVE_TIMEOUT_DEFAULT:.0f} s "
             f"({REFINE_CURVE_TIMEOUT_DEFAULT/60:.0f} min).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=CHECKPOINT_EVERY_DEFAULT,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-per-bin", type=int, default=TARGET_PER_BIN)
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Reprocessa curvas previamente marcadas timeout/error/audit_failed.",
    )
    parser.add_argument(
        "--max-curves",
        type=int,
        default=None,
        help="Limita quantas curvas novas rodar neste disparo; util para teste.",
    )
    parser.add_argument(
        "--no-screening",
        action="store_true",
        help="Desliga a camada 0 (triagem resonante/nao-resonante) e roda a "
             "busca completa de polos em toda curva, como antes. Use para "
             "auditoria/comparacao; o padrao (triagem ligada) e' equivalente "
             "em resultado e muito mais rapido nas curvas sem ressonancia.",
    )
    parser.add_argument(
        "--screen-ratio-floor",
        type=float,
        default=SCREEN_RATIO_FLOOR,
        help=f"Razao pico/fundo minima na triagem grosseira para considerar "
             f"a curva potencialmente ressonante (padrao {SCREEN_RATIO_FLOOR}, "
             f"bem abaixo do menor caso real ja confirmado no projeto).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.lmax < 0:
        raise ValueError("lmax precisa ser >= 0.")
    if args.curve_timeout <= 0:
        raise ValueError("curve-timeout precisa ser > 0.")
    if args.pole_search_timeout < 0:
        raise ValueError("pole-search-timeout precisa ser >= 0 (0 desliga o limite).")
    if 0.0 < args.pole_search_timeout and args.pole_search_timeout >= args.curve_timeout:
        raise ValueError(
            "pole-search-timeout precisa ser menor que curve-timeout (a "
            "validacao por energia depois da busca de polos tambem precisa "
            "de tempo dentro do limite duro da curva)."
        )
    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint-every precisa ser > 0.")
    if args.target_per_bin <= 0:
        raise ValueError("target-per-bin precisa ser > 0.")
    if args.refine_pole_search_timeout <= 0:
        raise ValueError("refine-pole-search-timeout precisa ser > 0.")
    if args.refine_curve_timeout <= 0:
        raise ValueError("refine-curve-timeout precisa ser > 0.")
    if args.refine_pole_search_timeout >= args.refine_curve_timeout:
        raise ValueError(
            "refine-pole-search-timeout precisa ser menor que "
            "refine-curve-timeout, pelo mesmo motivo do par normal."
        )

    for regime in args.regime:
        run_regime(args, regime)


if __name__ == "__main__":
    main()
