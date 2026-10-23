# -*- coding: utf-8 -*-
"""
DATASET8 -- CASO VOIGT (viscoelastico), reparametrizado em
(alpha, beta, eta, tan_delta_ref).

Arquivo 100% AUTOCONTIDO -- so' bibliotecas padrao.

Modelo de perda (Voigt): tan_delta(xM) = tan_delta_ref * (xM/xM_ref),
aplicado independentemente aos canais L e T (tanT_ref = 3*tanL_ref,
convencao usada o dia inteiro). xM_ref FIXO em 0.5.

c_L(xM), c_T(xM) ficam COMPLEXOS (com perda) -- por isso G(z,l)
precisa dos 3 ramos completos (real/imaginario/complexo geral), ao
contrario do caso elastico onde so' o ramo real era alcancavel.

Reparametrizacao (mesma orientacao meio/solido dos dois):
    alpha = rho_M / rho_S
    beta  = c_M / c_L0     (c_L0 = velocidade longitudinal DE REFERENCIA,
                             antes de aplicar a perda)
    eta   = c_L0 / c_T0

Hipercubo 4D: alpha, beta, c_T0, tan_delta_ref (todos sorteados juntos).
"""
import os
import signal
import time
import numpy as np
import pandas as pd
from scipy.special import spherical_jn, spherical_yn, spherical_in
from scipy.optimize import newton
from scipy.signal import find_peaks, peak_widths


CM_PADRAO = 1500.0
RHOM_PADRAO = 1000.0
LIMITE_FISICO = np.sqrt(3) / 2


# =================================================================
# 0b. Timeout por curva (mesmo mecanismo do detector core-shell):
# limite duro de wall-clock via SIGALRM, pra uma curva individual
# travada (ou so' muito lenta) nao comprometer o disparo inteiro.
# =================================================================
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
XM_REF = 0.5  # ancora do Voigt, FIXA (ver discussao extensa sobre nao variar isso sem querer)


# =================================================================
# 1. Bessel/Neumann/Hankel esfericas -- aceitam argumento COMPLEXO
#    (testado: scipy.special.spherical_jn funciona com complexo)
# =================================================================
def J(n, x):
    if n >= 0:
        return spherical_jn(n, x)
    m = -n - 1
    return (-1)**(m + 1) * spherical_yn(m, x)


def dJ(n, x):
    if n >= 0:
        return spherical_jn(n, x, derivative=True)
    m = -n - 1
    return (-1)**(m + 1) * spherical_yn(m, x, derivative=True)


def H(n, x):
    if n >= 0:
        return spherical_jn(n, x) + 1j * spherical_yn(n, x)
    m = -n - 1
    return 1j * (-1)**m * (spherical_jn(m, x) + 1j * spherical_yn(m, x))


def dH(n, x):
    if n >= 0:
        return spherical_jn(n, x, derivative=True) + 1j * spherical_yn(n, x, derivative=True)
    m = -n - 1
    return 1j * (-1)**m * (
        spherical_jn(m, x, derivative=True) + 1j * spherical_yn(m, x, derivative=True)
    )


def ddJ(n, z):
    z = np.asarray(z, dtype=complex)
    escalar = (z.ndim == 0)
    z = np.atleast_1d(z)
    pequeno = np.isclose(z, 0.0)
    z_seguro = np.where(pequeno, 1.0, z)
    out = (
        -(2.0 / z_seguro) * dJ(n, z_seguro)
        + (n * (n + 1) / z_seguro**2 - 1.0) * J(n, z_seguro)
    )
    if np.any(pequeno):
        limite = -1.0 / 3.0 if n == 0 else (2.0 / 15.0 if n == 2 else 0.0)
        out = np.where(pequeno, limite + 0.0j, out)
    return out[0] if escalar else out


# =================================================================
# 2. Bessel esfericas MODIFICADAS (I, dI) -- so' pro ramo imaginario
#    do G(z,l). So' precisamos de n>=-1 (l-1 com l=0 da' n=-1).
#    i_{-1}(x) = cosh(x)/x (identidade padrao).
# =================================================================
def I(n, x):
    if n >= 0:
        return spherical_in(n, x)
    if n == -1:
        return np.cosh(x) / x
    raise NotImplementedError(f"I(n,x) para n={n} nao implementado")


def dI(n, x):
    if n >= 0:
        return spherical_in(n, x, derivative=True)
    if n == -1:
        return np.sinh(x) / x - np.cosh(x) / x**2
    raise NotImplementedError(f"dI(n,x) para n={n} nao implementado")


# =================================================================
# 3. G(z,l): peca de Lommel, 3 ramos (real / imaginario / complexo)
# =================================================================
def G(z, l, tol=1e-6):
    x = np.abs(z)
    if abs(z.imag) < tol * x:
        return x*J(l,x)*dJ(l,x) + (x**2/2)*(J(l,x)**2 - J(l-1,x)*J(l+1,x))
    elif abs(z.real) < tol * x:
        return x*I(l,x)*dI(l,x) - (x**2/2)*(I(l,x)**2 - I(l-1,x)*I(l+1,x))
    else:
        zc = np.conj(z)
        den = z**2 - zc**2
        return (x**2/den) * (z*J(l,z)*dJ(l,zc) - zc*dJ(l,z)*J(l,zc))


# =================================================================
# 4. Modelo Voigt: tan_delta(xM) e c_L(xM), c_T(xM) complexos
# =================================================================
def tan_delta_voigt(xM, xM_ref, tan_delta_ref):
    return tan_delta_ref * (xM / xM_ref)


def cL_cT_voigt(xM, cL0, cT0, xM_ref, tanL_ref, tanT_ref):
    """c_L(xM), c_T(xM) complexos -- Voigt: tan_delta cresce linear
    com xM, ancorado em xM_ref."""
    tanL = tan_delta_voigt(xM, xM_ref, tanL_ref)
    tanT = tan_delta_voigt(xM, xM_ref, tanT_ref)
    cL = cL0 * np.sqrt(1.0 - 1j * tanL)
    cT = cT0 * np.sqrt(1.0 - 1j * tanT)
    return cL, cT


# =================================================================
# 5. Denominador de a_l -- versao Voigt (c_L, c_T complexos, f(xM))
# =================================================================
def den_a_l_voigt(xM, l, cL0, cT0, cM, rhoS, rhoM, xM_ref, tanL_ref, tanT_ref):
    cL, cT = cL_cT_voigt(xM, cL0, cT0, xM_ref, tanL_ref, tanT_ref)
    zL = xM * (cM / cL)
    eta = cL / cT
    zT = eta * zL
    zM = xM
    mu = rhoS * cT**2
    lamb = rhoS * cL**2 - 2 * mu

    den_M = rhoM * (cM**2) * (zM**2) * H(l, zM)
    A_l = (lamb * (zL**2) * J(l, zL) - 2 * mu * (zL**2) * ddJ(l, zL)) / den_M
    B_l = -(2 * mu * l * (l + 1) * (zT * dJ(l, zT) - J(l, zT))) / den_M
    den_M_tilde = zM * dH(l, zM)
    A_l_tilde = (zL * dJ(l, zL)) / den_M_tilde
    B_l_tilde = (l * (l + 1) * J(l, zT)) / den_M_tilde
    num_R = 2 * (J(l, zL) - zL * dJ(l, zL))
    den_R = (zT**2) * ddJ(l, zT) + (l * (l + 1) - 2) * J(l, zT)
    R_l = num_R / den_R
    return (A_l - A_l_tilde) - (B_l_tilde - B_l) * R_l


def coefs_voigt(xM, l_max, cL0, cT0, cM, rhoS, rhoM, xM_ref, tanL_ref, tanT_ref):
    cL, cT = cL_cT_voigt(xM, cL0, cT0, xM_ref, tanL_ref, tanT_ref)
    zL = xM * (cM / cL)
    eta = cL / cT
    zT = eta * zL
    zM = xM
    mu = rhoS * (cT**2)
    lamb = rhoS * (cL**2) - 2 * mu

    a_l = np.zeros(l_max + 1, dtype=complex)
    b_l = np.zeros(l_max + 1, dtype=complex)

    for l in range(l_max + 1):
        den_M = rhoM * (cM**2) * (zM**2) * H(l, zM)
        A_l = (lamb*(zL**2)*J(l,zL) - 2*mu*(zL**2)*ddJ(l,zL)) / den_M
        B_l = -(2*mu*l*(l+1)*(zT*dJ(l,zT)-J(l,zT))) / den_M
        C_l = -J(l,zM)/H(l,zM)

        den_M_tilde = zM*dH(l,zM)
        A_l_tilde = (zL*dJ(l,zL))/den_M_tilde
        B_l_tilde = (l*(l+1)*J(l,zT))/den_M_tilde
        C_l_tilde = -dJ(l,zM)/dH(l,zM)

        num_R = 2*(J(l,zL) - zL*dJ(l,zL))
        den_R = (zT**2)*ddJ(l,zT) + (l*(l+1)-2)*J(l,zT)
        R_l = num_R/den_R if den_R != 0 else 0j

        num_a = C_l_tilde - C_l
        den_a = (A_l - A_l_tilde) - (B_l_tilde - B_l)*R_l
        a_l[l] = num_a/den_a if den_a != 0 else 0j
        b_l[l] = R_l * a_l[l]

    return a_l, b_l


def energia_total_voigt(xM_arr, l_max, cL0, cT0, cM, rhoS, rhoM, xM_ref, tanL_ref, tanT_ref):
    """Energia de Clapeyron, caso Voigt -- xM_arr deve ser REAL (essa
    e' a curva observavel; a busca de raiz complexa usa den_a_l_voigt
    direto, nao essa funcao)."""
    xM_arr = np.atleast_1d(xM_arr).astype(float)
    saida = np.zeros(len(xM_arr))
    for idx, xM in enumerate(xM_arr):
        cL, cT = cL_cT_voigt(xM, cL0, cT0, xM_ref, tanL_ref, tanT_ref)
        a_l, b_l = coefs_voigt(xM, l_max, cL0, cT0, cM, rhoS, rhoM, xM_ref, tanL_ref, tanT_ref)
        zL = (cM / cL) * xM
        xL = np.abs(zL)
        eta = cL / cT
        zT = eta * zL
        mu = rhoS * cT**2
        lamb = rhoS * cL**2 - 2*mu

        dois_T = 0j
        IS_sum = 0j
        for l in range(l_max + 1):
            a, b = a_l[l], b_l[l]
            L = l * (l + 1)
            zLc, zTc = np.conj(zL), np.conj(zT)
            termo_a = xL * G(zL, l)
            termo_b = xL * (J(l, zT)*J(l, zTc) + G(zT, l))
            termo_cruzado = xL * J(l, zL) * J(l, zTc)
            dois_T += 2*(2*l + 1) * (
                np.abs(a)**2 * termo_a
                + np.abs(b)**2 * L * termo_b
                + 2*L * np.real(a * np.conj(b) * termo_cruzado)
            )
            fator1 = np.conj(a)*dJ(l, zLc) + np.conj(b)*(L/zLc)*J(l, zTc)
            fator2 = a*zL**2*(lamb*J(l,zL) - 2*mu*ddJ(l,zL)) - 2*mu*L*b*(zT*dJ(l,zT) - J(l,zT))
            IS_sum += (2*l + 1) * fator1 * fator2

        IS_termo = -np.real(IS_sum / (rhoS * np.abs(cL) * np.conj(cL)))
        saida[idx] = np.real(dois_T) + IS_termo
    return saida


def contribuicoes_por_l_voigt(xM_arr, l_max, cL0, cT0, cM, rhoS, rhoM, xM_ref, tanL_ref, tanT_ref):
    xM_arr = np.atleast_1d(xM_arr).astype(float)
    n = len(xM_arr)
    contrib = np.zeros((l_max + 1, n))
    for i, xM in enumerate(xM_arr):
        cL, cT = cL_cT_voigt(xM, cL0, cT0, xM_ref, tanL_ref, tanT_ref)
        a_l, b_l = coefs_voigt(xM, l_max, cL0, cT0, cM, rhoS, rhoM, xM_ref, tanL_ref, tanT_ref)
        zL = (cM / cL) * xM
        xL = abs(zL)
        eta = cL / cT
        zT = eta * zL
        mu = rhoS * cT**2
        lamb = rhoS * cL**2 - 2 * mu
        zLc, zTc = np.conj(zL), np.conj(zT)
        for l in range(l_max + 1):
            a, b = a_l[l], b_l[l]
            L = l * (l + 1)
            termo_a = xL * G(zL, l)
            termo_b = xL * (J(l, zT) * J(l, zTc) + G(zT, l))
            termo_cruzado = xL * J(l, zL) * J(l, zTc)
            dois_T = 2 * (2 * l + 1) * (
                abs(a)**2 * termo_a + abs(b)**2 * L * termo_b
                + 2 * L * np.real(a * np.conj(b) * termo_cruzado)
            )
            fator1 = np.conj(a) * dJ(l, zLc) + np.conj(b) * (L / zLc) * J(l, zTc)
            fator2 = a * zL**2 * (lamb * J(l, zL) - 2 * mu * ddJ(l, zL)) - 2 * mu * L * b * (zT * dJ(l, zT) - J(l, zT))
            IS_termo = -np.real((2 * l + 1) * fator1 * fator2 / (rhoS * np.abs(cL) * np.conj(cL)))
            contrib[l, i] = np.real(dois_T) + IS_termo
    return contrib


def _malha_e_lmax_efetivo_voigt(cL0, cT0, rhoS, tanL_ref, tanT_ref, xM_ref, cM=CM_PADRAO,
                                rhoM=RHOM_PADRAO, l_maximo_absoluto=10,
                                xM_min=1e-3, xM_max=1.0, n_scan=3000, razao_minima=1e-3):
    """Faz a varredura de contribuicoes_por_l_voigt UMA VEZ so' e devolve
    a malha, a energia total (soma exata dos l) e o l_max efetivo -- evita
    rodar a parte cara (coefs por ponto) duas vezes so' pra separar as
    duas informacoes."""
    xM_vals = np.linspace(xM_min, xM_max, n_scan)
    contrib = contribuicoes_por_l_voigt(xM_vals, l_maximo_absoluto, cL0, cT0, cM, rhoS, rhoM,
                                          xM_ref, tanL_ref, tanT_ref)
    w_total = contrib.sum(axis=0)
    maximo_por_l = np.max(np.abs(contrib), axis=1)
    maximo_global = maximo_por_l.max() if maximo_por_l.max() > 0 else 1.0
    ls_significativos = np.where(maximo_por_l / maximo_global > razao_minima)[0]
    l_max_efetivo = int(ls_significativos.max()) if len(ls_significativos) > 0 else 0
    return xM_vals, w_total, l_max_efetivo


def descobrir_l_max_por_energia_voigt(cL0, cT0, rhoS, tanL_ref, tanT_ref, xM_ref, cM=CM_PADRAO,
                                       rhoM=RHOM_PADRAO, l_maximo_absoluto=10,
                                       xM_min=1e-3, xM_max=1.0, n_scan=3000, razao_minima=1e-3):
    _, _, l_max_efetivo = _malha_e_lmax_efetivo_voigt(
        cL0, cT0, rhoS, tanL_ref, tanT_ref, xM_ref, cM, rhoM,
        l_maximo_absoluto, xM_min, xM_max, n_scan, razao_minima,
    )
    return l_max_efetivo


# =================================================================
# 5b. Camada 0: triagem + identificacao de picos direto na malha
# (mesma ideia do detector core-shell: reaproveita a varredura acima,
# find_peaks em cima, sem precisar de raiz complexa por l.)
# =================================================================
RAZAO_TRIAGEM_MINIMA = 5.0
PROMINENCIA_MALHA_MINIMA = 0.75


def tem_potencial_ressonancia_voigt(xM_vals, w_total, razao_minima=RAZAO_TRIAGEM_MINIMA):
    """Decide, a partir da energia total ja calculada numa malha grosseira,
    se vale a pena rodar a busca cara (raiz complexa por l)."""
    finito = np.isfinite(w_total) & (np.abs(w_total) > 0.0)
    if not np.any(finito):
        return False, 0.0
    picos_idx, _ = find_peaks(np.abs(w_total), prominence=0.0)
    if len(picos_idx) == 0:
        return False, 0.0
    fundo = max(float(np.median(np.abs(w_total)[finito])), np.finfo(float).tiny)
    razao = float(np.max(np.abs(w_total)[picos_idx]) / fundo)
    return razao > razao_minima, razao


def picos_por_malha_grosseira_voigt(xM_vals, w_total, prominencia_minima=PROMINENCIA_MALHA_MINIMA):
    """Conta/mede picos direto na malha (find_peaks no log10 de |energia|,
    largura a meia altura via peak_widths) -- resultado sempre disponivel,
    sem depender da raiz complexa por l."""
    w_abs = np.abs(w_total)
    finito = np.isfinite(w_abs) & (w_abs > 0.0)
    if np.count_nonzero(finito) < 5:
        return []
    piso = max(float(np.min(w_abs[finito])) * 1e-14, np.finfo(float).tiny)
    w_seguro = np.where(finito, w_abs, piso)
    log_w = np.log10(np.maximum(w_seguro, piso))

    idx, _ = find_peaks(log_w, prominence=prominencia_minima)
    idx = idx[(idx >= 3) & (idx <= len(xM_vals) - 4)]
    if len(idx) == 0:
        return []

    _, _, lips, rips = peak_widths(w_abs, idx, rel_height=0.5)
    eixo = np.arange(len(xM_vals), dtype=float)
    xl = np.interp(lips, eixo, xM_vals)
    xr = np.interp(rips, eixo, xM_vals)

    picos = []
    for k, i in enumerate(idx):
        largura = float(xr[k] - xl[k])
        altura = float(w_abs[i])
        if not (np.isfinite(largura) and largura > 0.0 and np.isfinite(altura) and altura > 0.0):
            continue
        picos.append({
            "xM_pico": float(xM_vals[i]), "altura": altura, "largura": largura,
            "Q": float(xM_vals[i]) / largura, "origem": "malha_grosseira",
        })
    return picos


# =================================================================
# 6. Raiz complexa do denominador -- Voigt, com os 4 criterios
# =================================================================
def candidatos_reais_para_l_voigt(l, cL0, cT0, rhoS, tanL_ref, tanT_ref, xM_ref, cM=CM_PADRAO,
                                   rhoM=RHOM_PADRAO, xM_min=1e-3, xM_max=1.0,
                                   n_scan=3000, prominencia=0.02):
    xM_vals = np.linspace(xM_min, xM_max, n_scan)
    modulos = np.array([abs(den_a_l_voigt(x, l, cL0, cT0, cM, rhoS, rhoM, xM_ref, tanL_ref, tanT_ref))
                         for x in xM_vals])
    log_inv = -np.log10(np.clip(modulos, 1e-300, None))
    minimos_idx, _ = find_peaks(log_inv, prominence=prominencia)
    return xM_vals[minimos_idx]


def _significativo_voigt(raiz, l_max_total, cL0, cT0, rhoS, tanL_ref, tanT_ref, xM_ref, cM=CM_PADRAO,
                          rhoM=RHOM_PADRAO, fator_janela=8.0, razao_minima=1.5):
    largura_esperada = max(abs(raiz.imag) * 2, 1e-12)
    janela = fator_janela * largura_esperada
    xM_centro = raiz.real
    xM_todos = np.array([xM_centro - janela, xM_centro, xM_centro + janela])
    try:
        W_todos = energia_total_voigt(xM_todos, l_max_total, cL0, cT0, cM, rhoS, rhoM,
                                        xM_ref, tanL_ref, tanT_ref)
    except Exception:
        return False
    altura_centro = abs(W_todos[1])
    altura_bordas = max(abs(W_todos[0]), abs(W_todos[2]), 1e-300)
    return (altura_centro / altura_bordas) > razao_minima


def raizes_complexas_para_l_voigt(l, l_max_total, cL0, cT0, rhoS, tanL_ref, tanT_ref, xM_ref,
                                   cM=CM_PADRAO, rhoM=RHOM_PADRAO, xM_min=1e-3, xM_max=1.0,
                                   n_scan=3000, offset_im=-1e-4, tol_aceitar=0.01,
                                   prominencia=0.02, tol_convergencia=1e-6):
    candidatos = candidatos_reais_para_l_voigt(l, cL0, cT0, rhoS, tanL_ref, tanT_ref, xM_ref, cM, rhoM,
                                                xM_min, xM_max, n_scan, prominencia)
    raizes = []
    for xM_c in candidatos:
        try:
            z0 = xM_c + offset_im * 1j
            raiz = newton(lambda z: den_a_l_voigt(z, l, cL0, cT0, cM, rhoS, rhoM,
                                                    xM_ref, tanL_ref, tanT_ref),
                           z0, tol=1e-12, maxiter=100)
            residuo = abs(den_a_l_voigt(raiz, l, cL0, cT0, cM, rhoS, rhoM, xM_ref, tanL_ref, tanT_ref))
            escala = abs(den_a_l_voigt(xM_c, l, cL0, cT0, cM, rhoS, rhoM, xM_ref, tanL_ref, tanT_ref))
            convergiu = residuo < tol_convergencia * max(escala, 1e-300)
            im_ok = raiz.imag < -1e-10
            perto = abs(raiz.real - xM_c) < tol_aceitar
            if not (convergiu and im_ok and perto):
                continue
            if not _significativo_voigt(raiz, l_max_total, cL0, cT0, rhoS, tanL_ref, tanT_ref, xM_ref, cM, rhoM):
                continue
            raizes.append(raiz)
        except Exception:
            continue
    return sorted(set(raizes), key=lambda r: r.real)


def _deduplicar(picos):
    """Ordena os picos por xM, sem fundir nada por proximidade -- qualquer
    tolerancia de fusao (relativa ou absoluta) corre o risco de juntar por
    engano duas ressonancias distintas que calham de ficar proximas em xM
    (por exemplo l's diferentes cujas raizes quase se cruzam). Cada raiz
    que passou nos criterios de aceitacao conta como um pico proprio."""
    if not picos:
        return picos
    return sorted(picos, key=lambda p: p["xM_pico"])


# =================================================================
# 7. FUNCAO PRINCIPAL: conta os picos de um material Voigt
# =================================================================
def contar_picos_voigt(cL0, cT0, rhoS, tanL_ref, tanT_ref=None, xM_ref=0.5, cM=CM_PADRAO,
                        rhoM=RHOM_PADRAO, l_max=10, xM_min=1e-3, xM_max=1.0, usar_triagem=True,
                        evitar_overflow=True):
    """cL0, cT0: velocidades DE REFERENCIA (elasticas, antes da perda).
    tanT_ref: se None, usa 3*tanL_ref (convencao do dia todo).
    xM_ref: ancora do Voigt -- agora e' PARAMETRO, nao mais fixo em 0.5
    (default 0.5 so' pra manter compatibilidade se ninguem passar nada).

    Se usar_triagem=True (padrao), reaproveita a MESMA varredura que ja
    calcula l_max_efetivo (nao roda de novo) pra decidir se vale a pena
    continuar; se nao ha ressonancia nenhuma, retorna rapido, sem procurar
    raiz complexa.

    Se evitar_overflow=True (padrao), antes de tentar a raiz complexa,
    risco_overflow_voigt (formula fechada em cL0, cT0) decide se essa
    combinacao provavelmente estoura float64 durante o Newton (mesmo
    mecanismo do core-shell: Bessel/Hankel crescem exponencialmente longe
    do eixo real). Se sim, pula a busca de raiz e devolve o resultado da
    malha grosseira (so' usa xM real, nao sofre desse problema), marcado
    origem="malha_grosseira". Em qualquer caso, os picos da malha ficam
    sempre anexados em cada pico como picos[i]["picos_malha"]."""
    if tanT_ref is None:
        tanT_ref = 3.0 * tanL_ref

    xM_vals, w_total, l_max_efetivo = _malha_e_lmax_efetivo_voigt(
        cL0, cT0, rhoS, tanL_ref, tanT_ref, xM_ref, cM, rhoM,
        l_maximo_absoluto=l_max, xM_min=xM_min, xM_max=xM_max,
    )
    picos_malha = picos_por_malha_grosseira_voigt(xM_vals, w_total)

    if usar_triagem:
        potencialmente_ressonante, _ = tem_potencial_ressonancia_voigt(xM_vals, w_total)
        if not potencialmente_ressonante:
            return []

    if evitar_overflow:
        risco, max_im = risco_overflow_voigt(cL0, cT0, xM_ref, tanL_ref, tanT_ref, xM_vals, cM)
        if risco:
            fracao_finita = np.mean(np.isfinite(w_total))
            if fracao_finita < FRACAO_MINIMA_MALHA_UTIL:
                # nem a malha (fallback) e' confiavel aqui -- a maior
                # parte do dominio quebrou (perda + c_T pequeno demais
                # juntos). Falha honesta: N_res=0 explicito, marcado como
                # "sem dado confiavel", nao um resultado inventado.
                return []
            picos_fallback = [dict(p) for p in picos_malha]
            for p in picos_fallback:
                p["l"] = None
                p["l_max_efetivo"] = l_max_efetivo
            picos_fallback.sort(key=lambda p: p["xM_pico"])
            for i, p in enumerate(picos_fallback):
                p["delta_x"] = (picos_fallback[i + 1]["xM_pico"] - p["xM_pico"]
                                if i < len(picos_fallback) - 1 else float("nan"))
                p["picos_malha"] = picos_malha
            return picos_fallback

    picos = []
    for l in range(l_max_efetivo + 1):
        raizes = raizes_complexas_para_l_voigt(l, l_max, cL0, cT0, rhoS, tanL_ref, tanT_ref, xM_ref,
                                                cM, rhoM, xM_min=xM_min, xM_max=xM_max)
        for r in raizes:
            altura = float(np.real(energia_total_voigt(
                np.array([r.real]), l_max, cL0, cT0, cM, rhoS, rhoM, xM_ref, tanL_ref, tanT_ref)[0]))
            Q = abs(r.real / (2 * r.imag))
            largura = abs(r.real / Q) if Q not in (0, float('inf')) else float('nan')
            picos.append({"xM_pico": r.real, "altura": altura, "largura": largura, "Q": Q,
                           "l": l, "l_max_efetivo": l_max_efetivo, "origem": "polo_validado"})
    picos = _deduplicar(picos)
    picos.sort(key=lambda p: p["xM_pico"])
    for i, p in enumerate(picos):
        p["delta_x"] = picos[i + 1]["xM_pico"] - p["xM_pico"] if i < len(picos) - 1 else float("nan")
        p["picos_malha"] = picos_malha
    return picos


# =================================================================
# 6b. Risco de overflow na busca de polo complexo -- mesmo cuidado do
# core-shell e do caso elastico, mas aqui e' mais delicado: como c_L(xM)
# e c_T(xM) ja SAO complexos por causa do modelo de perda (mesmo com xM
# real), o argumento das funcoes de Bessel/Hankel pode ter parte
# imaginaria grande mesmo sem nenhuma busca no plano complexo acontecer
# -- inclusive a PROPRIA malha grosseira (o fallback) pode ser afetada,
# ao contrario do caso elastico onde a malha era sempre segura.
#
# Calibrado direto (nao mais por uma formula fechada em cL0/cT0 sozinhos,
# porque o modelo de perda muda o quadro: Maxwell diverge perto de
# xM_min, Voigt cresce ate' xM_max): o estouro comeca quando |Im(z_T)|
# (ou |Im(z_L)|) passa de ~650-750 em QUALQUER ponto do dominio -- mesmo
# valor calibrado no core-shell e no caso elastico, reforcando que e' a
# mesma causa fisica (Bessel/Hankel crescendo exponencialmente longe do
# eixo real), so' que aqui a "distancia do eixo real" vem do proprio
# modelo de perda, nao de uma busca.
# =================================================================
LIMIAR_IM_OVERFLOW = 500.0
FRACAO_MINIMA_MALHA_UTIL = 0.5


def risco_overflow_voigt(cL0, cT0, xM_ref, tanL_ref, tanT_ref, xM_vals,
                        cM=CM_PADRAO, limiar=LIMIAR_IM_OVERFLOW):
    """Reaproveita a malha ja calculada (xM_vals) pra achar o PIOR caso de
    |Im(z_T)|/|Im(z_L)| ao longo do dominio inteiro -- nao so' num ponto,
    porque Maxwell e Voigt tem perfis de perda bem diferentes (um diverge
    perto de xM_min, o outro cresce ate' xM_max). Sem rodar nada novo
    alem de cL_cT_voigt vetorizado na malha que ja existe."""
    with np.errstate(all="ignore"):
        cL, cT = cL_cT_voigt(xM_vals, cL0, cT0, xM_ref, tanL_ref, tanT_ref)
        zL = xM_vals * (cM / cL)
        eta = cL / cT
        zT = eta * zL
    im_zL = np.abs(np.imag(zL))
    im_zT = np.abs(np.imag(zT))
    finito = np.isfinite(im_zL) & np.isfinite(im_zT)
    if not np.any(finito):
        return True, float("inf")  # nem da pra avaliar -- trata como risco maximo
    max_im = float(max(np.max(im_zL[finito]), np.max(im_zT[finito])))
    nao_finito_em_algum_lugar = not np.all(finito)
    risco = (max_im > limiar) or nao_finito_em_algum_lugar
    return bool(risco), max_im


# =================================================================
# 8. Reparametrizacao (alpha, beta) -- mesma orientacao meio/solido
# =================================================================
def alpha_beta_para_cL_rhoS(alpha, beta, cM=CM_PADRAO, rhoM=RHOM_PADRAO):
    rhoS = rhoM / alpha
    cL = cM / beta
    return cL, rhoS


# =================================================================
# 9. HIPERCUBO 4D: alpha, beta, c_T0, tan_delta_ref -- roda por dias
# =================================================================
def _salvar_checkpoint(caminho, linhas):
    df = pd.DataFrame(linhas)
    df.to_csv(caminho, index=False)
    caminho_npz = caminho.rsplit(".", 1)[0] + ".npz"
    np.savez_compressed(caminho_npz, **{c: df[c].to_numpy() for c in df.columns})


def _carregar_indices_prontos(caminho):
    if caminho is not None and os.path.exists(caminho):
        df = pd.read_csv(caminho)
        if "_indice" in df.columns:
            return set(df["_indice"].unique()), df.to_dict("records")
        return set(), df.to_dict("records")
    return set(), []


def grid_hipercubo_voigt(n_curvas=8000, cM=CM_PADRAO, rhoM=RHOM_PADRAO, l_max=10,
                          alpha_min=1e-3, alpha_max=1e3,
                          beta_min=1.0/3000.0, beta_max=1000.0,
                          tan_delta_min=1e-3, tan_delta_max=1.0,
                          xM_ref_min=0.05, xM_ref_max=0.95,
                          caminho_saida=None, salvar_a_cada=200,
                          timeout_por_curva=60.0, timeout_refino=600.0, auto_refino=True):
    """Sorteia n_curvas combinacoes JUNTAS de (alpha, beta, c_T0,
    tan_delta_ref, xM_ref) -- hipercubo 5D. c_T0 sorteado DIRETO
    (variavel genuina), respeitando o limite fisico do c_L0 DAQUELE
    ponto. tan_delta_ref (magnitude da perda, canal L -- T usa 3x
    isso) log-uniforme entre tan_delta_min e tan_delta_max. xM_ref
    (ancora do Voigt) sorteado UNIFORME entre xM_ref_min e xM_ref_max
    -- ja NAO fica mais fixo em 0.5, pra ver o efeito dele de verdade.
    Cada execucao SOMA n_curvas novas ao arquivo.

    Cada curva individual roda sob um limite de tempo (timeout_por_curva,
    padrao 60s): se estourar ou der erro, fica marcada status="timeout"/
    "error" (N_res=NaN, nao 0 -- 0 e' "confirmado sem ressonancia", NaN e'
    "nao sabemos ainda") e o disparo CONTINUA pra proxima curva. Ao
    terminar as n_curvas, se auto_refino=True (padrao), reprocessa
    automaticamente so' as que ficaram timeout/error, com timeout_refino
    (bem maior, padrao 600s) -- so' substitui o resultado quando a curva
    realmente fecha dessa vez; senao mantem marcada como estava."""
    indices_prontos, linhas = _carregar_indices_prontos(caminho_saida)
    indice_inicial = (max(indices_prontos) + 1) if indices_prontos else 0
    if indices_prontos:
        print(f"[hipercubo_voigt] {len(indices_prontos)} curvas ja no arquivo -- somando {n_curvas} novas")

    rng = np.random.default_rng(None)
    alpha_vals = 10.0 ** rng.uniform(np.log10(alpha_min), np.log10(alpha_max), n_curvas)
    beta_vals = 10.0 ** rng.uniform(np.log10(beta_min), np.log10(beta_max), n_curvas)
    tanL_vals = 10.0 ** rng.uniform(np.log10(tan_delta_min), np.log10(tan_delta_max), n_curvas)
    xM_ref_vals = rng.uniform(xM_ref_min, xM_ref_max, n_curvas)

    parametros_por_indice = {}
    status_por_indice = {}
    contagem_status = {"ok": 0, "timeout": 0, "error": 0}
    t0 = time.time()

    def processa_uma_curva(i, alpha, beta, tanL_ref, xM_ref, cT0, timeout_s):
        cL0, rhoS = alpha_beta_para_cL_rhoS(alpha, beta, cM, rhoM)
        tanT_ref = 3.0 * tanL_ref
        status, erro_msg, picos = "ok", "", []
        t_ini = time.time()
        try:
            with curve_timeout(timeout_s):
                picos = contar_picos_voigt(cL0, cT0, rhoS, tanL_ref, tanT_ref, xM_ref, cM, rhoM, l_max)
        except CurveTimeoutError:
            status = "timeout"
            erro_msg = f"timeout>{timeout_s:.0f}s"
        except Exception as exc:
            status = "error"
            erro_msg = f"{type(exc).__name__}: {exc}"
        runtime = time.time() - t_ini

        base = {"_indice": i, "alpha": alpha, "beta": beta, "eta": cL0 / cT0,
                "alpha_vezes_beta": alpha * beta, "cL0": cL0, "cT0": cT0, "rhoS": rhoS,
                "tanL_ref": tanL_ref, "tanT_ref": tanT_ref, "xM_ref": xM_ref,
                "cM": cM, "rhoM": rhoM, "status": status, "runtime_s": runtime, "erro": erro_msg,
                "N_res": len(picos) if status == "ok" else float("nan")}

        novas_linhas = []
        if picos:
            for p in picos:
                linha = dict(p)
                linha.update(base)
                novas_linhas.append(linha)
        else:
            linha = {"xM_pico": float("nan"), "altura": float("nan"), "largura": float("nan"),
                      "Q": float("nan"), "l": None, "l_max_efetivo": None, "delta_x": float("nan")}
            linha.update(base)
            novas_linhas.append(linha)
        return status, novas_linhas

    for k in range(n_curvas):
        i = indice_inicial + k
        alpha, beta, tanL_ref, xM_ref = (float(alpha_vals[k]), float(beta_vals[k]),
                                          float(tanL_vals[k]), float(xM_ref_vals[k]))
        cL0_tmp, _ = alpha_beta_para_cL_rhoS(alpha, beta, cM, rhoM)
        cT0_maximo_local = LIMITE_FISICO * 0.999 * cL0_tmp
        cT0 = float(rng.uniform(1e-3, cT0_maximo_local))

        parametros_por_indice[i] = (alpha, beta, tanL_ref, xM_ref, cT0)
        status, novas_linhas = processa_uma_curva(i, alpha, beta, tanL_ref, xM_ref, cT0, timeout_por_curva)
        status_por_indice[i] = status
        contagem_status[status] += 1
        linhas.extend(novas_linhas)

        if caminho_saida and (k + 1) % salvar_a_cada == 0:
            _salvar_checkpoint(caminho_saida, linhas)
            decorrido = time.time() - t0
            restante = decorrido / (k + 1) * (n_curvas - k - 1)
            total_picos_de_verdade = sum(1 for l in linhas if l.get("N_res", 0) > 0 and not pd.isna(l.get("xM_pico")))
            print(f"[hipercubo_voigt] {k+1}/{n_curvas} curvas ({contagem_status['ok']} ok, "
                  f"{contagem_status['timeout']} timeout, {contagem_status['error']} error), "
                  f"{total_picos_de_verdade} picos, "
                  f"{decorrido/3600:.1f}h decorridas, ~{restante/3600:.1f}h restantes")

    if caminho_saida:
        _salvar_checkpoint(caminho_saida, linhas)

    print("-" * 78)
    print(f"[hipercubo_voigt] disparo principal terminado em {(time.time()-t0)/3600:.2f}h: "
          f"{contagem_status}")

    indices_pendentes = [i for i, s in status_por_indice.items() if s in ("timeout", "error")]

    if not auto_refino or not indices_pendentes:
        print("[hipercubo_voigt] nenhum refinamento automatico sera' feito."
              if not indices_pendentes else
              "[hipercubo_voigt] auto_refino=False -- nao refinando agora.")
        print("-" * 78)
        return pd.DataFrame(linhas)

    print(f"[hipercubo_voigt] iniciando refinamento de {len(indices_pendentes)} curvas "
          f"timeout/error, com timeout_refino={timeout_refino:.0f}s cada")
    t_refino = time.time()
    n_melhoradas = 0
    for j, i in enumerate(indices_pendentes, start=1):
        alpha, beta, tanL_ref, xM_ref, cT0 = parametros_por_indice[i]
        status, novas_linhas = processa_uma_curva(i, alpha, beta, tanL_ref, xM_ref, cT0, timeout_refino)
        if status == "ok":
            n_melhoradas += 1
            linhas = [l for l in linhas if l.get("_indice") != i]
            linhas.extend(novas_linhas)
        print(f"[refino {j}/{len(indices_pendentes)}] indice={i} -> status={status}", flush=True)
        if caminho_saida and j % max(1, salvar_a_cada // 10) == 0:
            _salvar_checkpoint(caminho_saida, linhas)

    if caminho_saida:
        _salvar_checkpoint(caminho_saida, linhas)
    print(f"[hipercubo_voigt] refinamento terminado em {(time.time()-t_refino)/3600:.2f}h: "
          f"{n_melhoradas} de {len(indices_pendentes)} curvas fecharam dessa vez.")
    print("-" * 78)
    return pd.DataFrame(linhas)


if __name__ == "__main__":
    print("=== teste de sanidade: tan_delta -> 0 deve recuperar o caso elastico ===")
    print("(esperado: proximo dos 5 picos elasticos conhecidos, l=0,2,3,4,5)")
    picos = contar_picos_voigt(1500.0, 900.0, 56.0, tanL_ref=1e-6, l_max=8)
    print(f"{len(picos)} picos (tan_delta=1e-6, quase elastico):")
    for p in picos:
        print(f"  xM={p['xM_pico']:.4f}  l={p['l']}  Q={p['Q']:.1f}")
    print()

    print("=== disparando o hipercubo completo ===")
    grid_hipercubo_voigt(
        n_curvas=8000,
        l_max=10,
        caminho_saida="resultados_voigt/hipercubo_voigt.csv",
        salvar_a_cada=200,
        timeout_por_curva=60.0,
        timeout_refino=600.0,
        auto_refino=True,
    )
