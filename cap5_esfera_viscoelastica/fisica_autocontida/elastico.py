# -*- coding: utf-8 -*-
"""
DATASET8 -- ELASTICO, REPARAMETRIZADO EM (alpha, beta, eta).

Arquivo 100% AUTOCONTIDO -- nao importa nada de nenhum outro codigo
seu, so' bibliotecas padrao (numpy, scipy, matplotlib, pandas).

Reparametrizacao:
    alpha = rho_M / rho_S   (razao de densidades -- "caso densidade")
    beta  = c_M / c_L       (razao de velocidades -- "caso velocidade")
    eta   = c_L / c_T       (split longitudinal/transversal, como sempre)

c_M=1500 e rho_M=1000 (agua) continuam FIXOS -- alpha e beta sao so'
outra forma de escrever rho_S e c_L em relacao a eles (mais facil de
variar e de plotar), nao uma mudanca fisica.

A malha de x_M continua 0 a 1, 3000 pontos, como sempre.

FIX (vs versao anterior): curvas com N_res=0 agora SAO salvas (uma
linha com campos de pico em NaN e N_res=0 explicito). Antes, quando
`picos` vinha vazio o loop `for p in picos` nao rodava nenhuma vez e
aquele indice simplesmente desaparecia do CSV -- os mapas de estado
ficavam cheios de buracos (nao eh N_res baixo, eh ausencia de dado).
"""
import os
import signal
import time
import numpy as np
import pandas as pd
from scipy.special import spherical_jn, spherical_yn
from scipy.optimize import newton
from scipy.signal import find_peaks, peak_widths


CM_PADRAO = 1500.0   # velocidade do som na agua (m/s)
RHOM_PADRAO = 1000.0  # densidade da agua (kg/m^3)
LIMITE_FISICO = np.sqrt(3) / 2  # c_T/c_L <= isso (nu >= -1)


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


# =================================================================
# 1. Bessel/Neumann/Hankel esfericas
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
# 2. Denominador de a_l (o polo = ressonancia de verdade)
# =================================================================
def den_a_l(xM, l, cL, cT, cM, rhoS, rhoM):
    xL = xM * (cM / cL)
    eta = cL / cT
    zL = xL
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


# =================================================================
# 3. G(z,l) e coefs -- so' o ramo real (elastico: z sempre real)
# =================================================================
def G(z, l):
    x = np.abs(z)
    return x*J(l, x)*dJ(l, x) + (x**2/2)*(J(l, x)**2 - J(l-1, x)*J(l+1, x))


def coefs(xM, l_max, cL, cT, cM, rhoS, rhoM):
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


def energia_total_elastica(xM_arr, l_max, cL, cT, cM, rhoS, rhoM):
    xM_arr = np.atleast_1d(xM_arr)
    saida = np.zeros(len(xM_arr))
    for idx, xM in enumerate(xM_arr):
        a_l, b_l = coefs(xM, l_max, cL, cT, cM, rhoS, rhoM)
        zL = (cM / cL) * xM
        xL = np.abs(zL)
        eta = cL / cT
        zT = eta * zL
        mu = rhoS * cT**2
        lamb = rhoS * cL**2 - 2*mu

        dois_T = 0.0
        IS_sum = 0.0
        for l in range(l_max + 1):
            a, b = a_l[l], b_l[l]
            L = l * (l + 1)
            termo_a = xL * G(zL, l)
            termo_b = xL * (J(l, zT)*J(l, zT) + G(zT, l))
            termo_cruzado = xL * J(l, zL) * J(l, zT)
            dois_T += 2*(2*l + 1) * (
                np.abs(a)**2 * termo_a
                + np.abs(b)**2 * L * termo_b
                + 2*L * np.real(a * np.conj(b) * termo_cruzado)
            )
            fator1 = np.conj(a)*dJ(l, zL) + np.conj(b)*(L/zL)*J(l, zT)
            fator2 = a*zL**2*(lamb*J(l,zL) - 2*mu*ddJ(l,zL)) - 2*mu*L*b*(zT*dJ(l,zT) - J(l,zT))
            IS_sum += np.real((2*l + 1) * fator1 * fator2)

        IS_termo = -IS_sum / (rhoS * cL * cL)
        saida[idx] = np.real(dois_T) + IS_termo
    return saida


def contribuicoes_por_l_elastico(xM_arr, l_max, cL, cT, cM, rhoS, rhoM):
    xM_arr = np.atleast_1d(xM_arr)
    n = len(xM_arr)
    contrib = np.zeros((l_max + 1, n))
    for i, xM in enumerate(xM_arr):
        a_l, b_l = coefs(xM, l_max, cL, cT, cM, rhoS, rhoM)
        zL = (cM / cL) * xM
        xL = abs(zL)
        eta = cL / cT
        zT = eta * zL
        mu = rhoS * cT**2
        lamb = rhoS * cL**2 - 2 * mu
        for l in range(l_max + 1):
            a, b = a_l[l], b_l[l]
            L = l * (l + 1)
            termo_a = xL * G(zL, l)
            termo_b = xL * (J(l, zT) * J(l, zT) + G(zT, l))
            termo_cruzado = xL * J(l, zL) * J(l, zT)
            dois_T = 2 * (2 * l + 1) * (
                abs(a)**2 * termo_a + abs(b)**2 * L * termo_b
                + 2 * L * np.real(a * np.conj(b) * termo_cruzado)
            )
            fator1 = np.conj(a) * dJ(l, zL) + np.conj(b) * (L / zL) * J(l, zT)
            fator2 = a * zL**2 * (lamb * J(l, zL) - 2 * mu * ddJ(l, zL)) - 2 * mu * L * b * (zT * dJ(l, zT) - J(l, zT))
            IS_termo = -np.real((2 * l + 1) * fator1 * fator2 / (rhoS * cL * cL))
            contrib[l, i] = np.real(dois_T) + IS_termo
    return contrib


def _malha_e_lmax_efetivo(cL, cT, rhoS, cM=CM_PADRAO, rhoM=RHOM_PADRAO, l_maximo_absoluto=10,
                           xM_min=1e-3, xM_max=1.0, n_scan=3000, razao_minima=1e-3):
    """Faz a varredura de contribuicoes_por_l_elastico UMA VEZ so' e devolve
    tudo que da pra tirar dela: a malha, a energia total (soma exata dos l,
    conferido numericamente == energia_total_elastica), e o l_max efetivo.
    Isso evita rodar duas vezes a parte mais cara (coefs() ponto a ponto)
    so' pra descobrir l_max de um lado e a curva de energia do outro."""
    xM_vals = np.linspace(xM_min, xM_max, n_scan)
    contrib = contribuicoes_por_l_elastico(xM_vals, l_maximo_absoluto, cL, cT, cM, rhoS, rhoM)
    w_total = contrib.sum(axis=0)
    maximo_por_l = np.max(np.abs(contrib), axis=1)
    maximo_global = maximo_por_l.max() if maximo_por_l.max() > 0 else 1.0
    ls_significativos = np.where(maximo_por_l / maximo_global > razao_minima)[0]
    l_max_efetivo = int(ls_significativos.max()) if len(ls_significativos) > 0 else 0
    return xM_vals, w_total, l_max_efetivo


def descobrir_l_max_por_energia(cL, cT, rhoS, cM=CM_PADRAO, rhoM=RHOM_PADRAO, l_maximo_absoluto=10,
                                 xM_min=1e-3, xM_max=1.0, n_scan=3000, razao_minima=1e-3):
    _, _, l_max_efetivo = _malha_e_lmax_efetivo(
        cL, cT, rhoS, cM, rhoM, l_maximo_absoluto, xM_min, xM_max, n_scan, razao_minima
    )
    return l_max_efetivo


# =================================================================
# 3b. Camada 0: triagem + identificacao de picos direto na malha
# (mesma ideia usada no detector core-shell: 1 varredura vetorizada
# ja calculada acima, find_peaks em cima, sem precisar da raiz
# complexa por l. So' decide/mede rapido; a raiz complexa continua
# sendo quem valida de verdade cada pico, mais abaixo.)
# =================================================================
RAZAO_TRIAGEM_MINIMA = 5.0
PROMINENCIA_MALHA_MINIMA = 0.75


def tem_potencial_ressonancia_elastico(xM_vals, w_total, razao_minima=RAZAO_TRIAGEM_MINIMA):
    """Decide, a partir da energia total ja calculada numa malha grosseira,
    se vale a pena rodar a busca cara (raiz complexa por l). Criterio
    frouxo de proposito (mesmo espirito do core-shell): melhor um falso
    positivo (gasta tempo a toa) do que um falso negativo (perde pico)."""
    finito = np.isfinite(w_total) & (np.abs(w_total) > 0.0)
    if not np.any(finito):
        return False, 0.0
    picos_idx, _ = find_peaks(np.abs(w_total), prominence=0.0)
    if len(picos_idx) == 0:
        return False, 0.0
    fundo = max(float(np.median(np.abs(w_total)[finito])), np.finfo(float).tiny)
    razao = float(np.max(np.abs(w_total)[picos_idx]) / fundo)
    return razao > razao_minima, razao


def picos_por_malha_grosseira_elastico(xM_vals, w_total, prominencia_minima=PROMINENCIA_MALHA_MINIMA):
    """Conta/mede picos direto na malha (find_peaks no log10 de |energia|,
    largura a meia altura via peak_widths) -- resultado sempre disponivel,
    sem depender da raiz complexa por l. Mesmo metodo ja usado no detector
    core-shell (ver _picos_por_malha_grosseira la')."""
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
# 4. Raiz complexa do denominador, com os 4 criterios de aceitacao
# =================================================================
def candidatos_reais_para_l(l, cL, cT, cM, rhoS, rhoM, xM_min=1e-3, xM_max=1.0,
                             n_scan=3000, prominencia=0.02):
    xM_vals = np.linspace(xM_min, xM_max, n_scan)
    modulos = np.array([abs(den_a_l(x, l, cL, cT, cM, rhoS, rhoM)) for x in xM_vals])
    log_inv = -np.log10(np.clip(modulos, 1e-300, None))
    minimos_idx, _ = find_peaks(log_inv, prominence=prominencia)
    return xM_vals[minimos_idx]


def _significativo(raiz, l_max_total, cL, cT, cM, rhoS, rhoM,
                    fator_janela=8.0, razao_minima=1.5):
    largura_esperada = max(abs(raiz.imag) * 2, 1e-12)
    janela = fator_janela * largura_esperada
    xM_centro = raiz.real
    xM_todos = np.array([xM_centro - janela, xM_centro, xM_centro + janela])
    try:
        W_todos = energia_total_elastica(xM_todos, l_max_total, cL, cT, cM, rhoS, rhoM)
    except Exception:
        return False
    altura_centro = abs(W_todos[1])
    altura_bordas = max(abs(W_todos[0]), abs(W_todos[2]), 1e-300)
    return (altura_centro / altura_bordas) > razao_minima


def raizes_complexas_para_l(l, l_max_total, cL, cT, cM, rhoS, rhoM, xM_min=1e-3, xM_max=1.0,
                             n_scan=3000, offset_im=-1e-4, tol_aceitar=0.01,
                             prominencia=0.02, tol_convergencia=1e-6):
    candidatos = candidatos_reais_para_l(l, cL, cT, cM, rhoS, rhoM, xM_min, xM_max,
                                          n_scan, prominencia)
    raizes = []
    for xM_c in candidatos:
        try:
            z0 = xM_c + offset_im * 1j
            raiz = newton(lambda z: den_a_l(z, l, cL, cT, cM, rhoS, rhoM),
                           z0, tol=1e-12, maxiter=100)
            residuo = abs(den_a_l(raiz, l, cL, cT, cM, rhoS, rhoM))
            escala = abs(den_a_l(xM_c, l, cL, cT, cM, rhoS, rhoM))
            convergiu = residuo < tol_convergencia * max(escala, 1e-300)
            im_ok = raiz.imag < -1e-10
            perto = abs(raiz.real - xM_c) < tol_aceitar
            if not (convergiu and im_ok and perto):
                continue
            if not _significativo(raiz, l_max_total, cL, cT, cM, rhoS, rhoM):
                continue
            raizes.append(raiz)
        except Exception:
            continue
    return sorted(set(raizes), key=lambda r: r.real)


def _deduplicar(picos):
    """Ordena os picos por xM, sem fundir nada por proximidade.

    A versao anterior (com tolerancia, relativa ou absoluta) sempre corre
    o risco de fundir por engano duas ressonancias genuinamente distintas
    que calham de ficar proximas em xM (por exemplo, l's diferentes cujas
    raizes quase se cruzam) -- qualquer tolerancia > 0 tem essa
    possibilidade, so' muda o quao raro e'. Sem tolerancia nenhuma, cada
    raiz encontrada (e que passou em todos os criterios de aceitacao em
    raizes_complexas_para_l: convergencia, parte imaginaria, proximidade
    do candidato, significancia energetica) conta como um pico proprio."""
    if not picos:
        return picos
    return sorted(picos, key=lambda p: p["xM_pico"])


def contar_picos_elastico(cL, cT, rhoS, cM=CM_PADRAO, rhoM=RHOM_PADRAO, l_max=10,
                           xM_min=1e-3, xM_max=1.0, usar_triagem=True, evitar_overflow=True):
    """Malha de x_M: 0 a 1, 3000 pontos (mantido como sempre).
    Pode retornar lista vazia (N_res=0) -- isso eh um resultado
    valido e tem que ser tratado explicitamente por quem chama.

    Se usar_triagem=True (padrao), antes de rodar a busca de raiz
    complexa (cara: Newton por candidato, por l), reaproveita a MESMA
    varredura que ja calcula l_max_efetivo (nao roda de novo) e decide se
    vale a pena continuar. Se a triagem disser que nao ha ressonancia
    nenhuma, retorna rapido -- sem procurar raiz nenhuma.

    Se evitar_overflow=True (padrao), antes de tentar a raiz complexa,
    risco_overflow_elastico decide (formula fechada em cL, cT) se essa
    combinacao provavelmente estoura float64 durante o Newton (mesmo
    mecanismo do core-shell: as funcoes de Bessel/Hankel crescem
    exponencialmente longe do eixo real). Se sim, a busca de raiz e'
    pulada e o resultado vem direto da malha grosseira (mesma marcada
    origem="malha_grosseira"), que so' usa xM real e nao sofre desse
    problema. Nos dois casos (raiz validada ou fallback pela malha), os
    picos identificados direto na malha ficam sempre anexados em cada
    pico como picos[i]["picos_malha"] (mesmo metodo do core-shell)."""
    xM_vals, w_total, l_max_efetivo = _malha_e_lmax_efetivo(
        cL, cT, rhoS, cM, rhoM, l_maximo_absoluto=l_max,
        xM_min=xM_min, xM_max=xM_max,
    )
    picos_malha = picos_por_malha_grosseira_elastico(xM_vals, w_total)

    if usar_triagem:
        potencialmente_ressonante, razao_triagem = tem_potencial_ressonancia_elastico(xM_vals, w_total)
        if not potencialmente_ressonante:
            return []

    if evitar_overflow:
        risco, amp_L, amp_T = risco_overflow_elastico(cL, cT, cM)
        if risco:
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
        raizes = raizes_complexas_para_l(l, l_max, cL, cT, cM, rhoS, rhoM,
                                          xM_min=xM_min, xM_max=xM_max)
        for r in raizes:
            altura = float(np.real(energia_total_elastica(
                np.array([r.real]), l_max, cL, cT, cM, rhoS, rhoM)[0]))
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
# 3c. Risco de overflow na busca de polo complexo (mesmo cuidado do
# core-shell): as funcoes de Bessel/Hankel crescem exponencialmente
# longe do eixo real, e o quanto uma excursao imaginaria pequena vira
# um argumento grande depende de cM/cT (canal transversal) e cM/cL
# (canal longitudinal). Calibrado empiricamente (den_a_l com
# Im(z)=-0.01, Q~25 a xM~0.5, uma ressonancia bem realista): o estouro
# comeca entre amp=60000 (ok) e 75000 (estoura). Limiar com folga.
# =================================================================
LIMIAR_AMP_OVERFLOW = 40000.0


def risco_overflow_elastico(cL, cT, cM=CM_PADRAO, limiar=LIMIAR_AMP_OVERFLOW):
    """Estimativa fechada (nao roda nada) de se a busca de raiz complexa
    provavelmente estoura float64 (ou pior, cancela tudo silenciosamente)
    para esses cL, cT. amp_L = cM/cL, amp_T = cM/cT -- quanto maior, mais
    uma pequena excursao no plano complexo vira um argumento gigante nas
    funcoes de Bessel/Hankel. Na pratica, dentro do hipercubo (beta<=1000,
    logo amp_L<=1000, bem abaixo do limiar), quem manda e' amp_T, porque
    c_T e' sorteado direto e pode chegar bem perto de 0."""
    amp_L = cM / cL
    amp_T = cM / cT
    risco = (amp_L > limiar) or (amp_T > limiar)
    return bool(risco), float(amp_L), float(amp_T)


# =================================================================
# 5. REPARAMETRIZACAO: alpha = rho_M/rho_S, beta = c_M/c_L
#
# Consistencia: os dois na mesma orientacao (meio sobre solido) --
# alpha = densidade do meio / densidade do solido;
# beta  = velocidade do meio / velocidade do solido.
#
# c_M, rho_M continuam FIXOS (agua) -- alpha e beta sao so' outra
# forma de escrever rho_S e c_L em relacao a eles.
# =================================================================
def alpha_beta_para_cL_rhoS(alpha, beta, cM=CM_PADRAO, rhoM=RHOM_PADRAO):
    """Converte (alpha, beta) de volta pra (c_L, rho_S) absolutos.
    alpha = rho_M/rho_S, beta = c_M/c_L (meio sobre solido, os dois)."""
    rhoS = rhoM / alpha
    cL = cM / beta
    return cL, rhoS


def gerar_pontos_por_decada(target_per_bin, exp_min=-3, exp_max=3, extras=None, semente=None):
    rng = np.random.default_rng(semente)
    partes = [rng.uniform(10.0**e, 9*10.0**e, target_per_bin) for e in range(exp_min, exp_max)]
    if extras:
        partes.append(np.array(extras, dtype=float))
    return np.unique(np.concatenate(partes))


# =================================================================
# 6. HIPERCUBO em (alpha, beta, c_T) -- roda por dias, um arquivo so'
#
# FIX: toda combinacao (alpha, beta, cT) sorteada gera PELO MENOS UMA
# linha no CSV, mesmo que N_res=0 -- senao os mapas de estado ficam
# com buracos onde na verdade ha' dado (N_res=0 eh informacao, nao
# ausencia de informacao).
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


def _linha_metadados(i, alpha, beta, cL, cT, rhoS, cM, rhoM):
    """Campos comuns a toda linha de uma curva, tenha ela picos ou nao."""
    return {"_indice": i, "alpha": alpha, "beta": beta, "eta": cL / cT,
            "alpha_vezes_beta": alpha * beta, "cL": cL, "cT": cT, "rhoS": rhoS,
            "cM": cM, "rhoM": rhoM}


def grid_hipercubo_alpha_beta(n_curvas=8000, cM=CM_PADRAO, rhoM=RHOM_PADRAO, l_max=10,
                               alpha_min=1e-3, alpha_max=1e3,
                               beta_min=1.0/3000.0, beta_max=1000.0, caminho_saida=None,
                               salvar_a_cada=200, timeout_por_curva=60.0,
                               timeout_refino=600.0, auto_refino=True):
    """Sorteia n_curvas combinacoes JUNTAS de (alpha, beta, c_T) --
    cada combinacao = UMA curva de energia completa, analisada pra achar
    todos os picos dela --
    SEM recorte de "plausibilidade fisica": alpha=rho_M/rho_S e
    beta=c_M/c_L cobrem a faixa TODA (razao 0.001 a 3000, esquema
    original), de proposito -- qualquer recorte "realista" sempre
    exclui alguma coisa (ex.: uma esfera de ar), entao aqui pega tudo.
    c_T sorteado DIRETO (variavel genuina, nao razao), de quase 0 ate
    o teto fisico DAQUELE c_L especifico. Cada execucao SOMA n_curvas
    novos ao arquivo.

    Curvas com N_res=0 SAO salvas (uma linha com campos de pico em
    NaN e N_res=0 explicito) -- ver _linha_metadados / N_res abaixo.

    Cada curva individual roda sob um limite de tempo (timeout_por_curva,
    padrao 60s): se estourar ou der erro, a curva fica marcada
    status="timeout"/"error" (N_res=NaN, nao 0 -- 0 e' "confirmado sem
    ressonancia", NaN e' "nao sabemos ainda") e o disparo CONTINUA pra
    curva seguinte, sem travar tudo. Ao terminar as n_curvas, se
    auto_refino=True (padrao), reprocessa automaticamente so' as que
    ficaram timeout/error, agora com timeout_refino (bem maior, padrao
    600s) -- so' substitui o resultado da curva quando ela realmente
    fecha dessa vez; senao mantem marcada como estava."""
    indices_prontos, linhas = _carregar_indices_prontos(caminho_saida)
    indice_inicial = (max(indices_prontos) + 1) if indices_prontos else 0
    if indices_prontos:
        print(f"[hipercubo] {len(indices_prontos)} curvas ja no arquivo -- somando {n_curvas} novas")

    rng = np.random.default_rng(None)
    log_alpha = rng.uniform(np.log10(alpha_min), np.log10(alpha_max), n_curvas)
    alpha_vals = 10.0 ** log_alpha

    log_beta = rng.uniform(np.log10(beta_min), np.log10(beta_max), n_curvas)
    beta_vals = 10.0 ** log_beta

    parametros_por_indice = {}
    status_por_indice = {}
    contagem_status = {"ok": 0, "timeout": 0, "error": 0}
    t0 = time.time()

    def processa_uma_curva(i, alpha, beta, cT, timeout_s):
        cL, rhoS = alpha_beta_para_cL_rhoS(alpha, beta, cM, rhoM)
        status, erro_msg, picos = "ok", "", []
        t_ini = time.time()
        try:
            with curve_timeout(timeout_s):
                picos = contar_picos_elastico(cL, cT, rhoS, cM, rhoM, l_max)
        except CurveTimeoutError:
            status = "timeout"
            erro_msg = f"timeout>{timeout_s:.0f}s"
        except Exception as exc:
            status = "error"
            erro_msg = f"{type(exc).__name__}: {exc}"
        runtime = time.time() - t_ini

        meta = _linha_metadados(i, alpha, beta, cL, cT, rhoS, cM, rhoM)
        meta["status"] = status
        meta["runtime_s"] = runtime
        meta["erro"] = erro_msg
        n_res = len(picos) if status == "ok" else float("nan")

        novas_linhas = []
        if picos:
            for p in picos:
                linha = dict(p)
                linha.update(meta)
                linha["N_res"] = n_res
                novas_linhas.append(linha)
        else:
            linha_vazia = {"xM_pico": np.nan, "altura": np.nan, "largura": np.nan,
                            "Q": np.nan, "l": np.nan, "l_max_efetivo": np.nan,
                            "delta_x": np.nan}
            linha_vazia.update(meta)
            linha_vazia["N_res"] = n_res
            novas_linhas.append(linha_vazia)
        return status, novas_linhas

    for k in range(n_curvas):
        i = indice_inicial + k
        alpha = float(alpha_vals[k])
        beta = float(beta_vals[k])
        cL_tmp, _ = alpha_beta_para_cL_rhoS(alpha, beta, cM, rhoM)
        cT_maximo_local = LIMITE_FISICO * 0.999 * cL_tmp
        cT = float(rng.uniform(1e-3, cT_maximo_local))  # c_T: variavel genuina, sorteada direto

        parametros_por_indice[i] = (alpha, beta, cT)
        status, novas_linhas = processa_uma_curva(i, alpha, beta, cT, timeout_por_curva)
        status_por_indice[i] = status
        contagem_status[status] += 1
        linhas.extend(novas_linhas)

        if caminho_saida and (k + 1) % salvar_a_cada == 0:
            _salvar_checkpoint(caminho_saida, linhas)
            decorrido = time.time() - t0
            restante = decorrido / (k + 1) * (n_curvas - k - 1)
            print(f"[hipercubo] {k+1}/{n_curvas} curvas ({contagem_status['ok']} ok, "
                  f"{contagem_status['timeout']} timeout, {contagem_status['error']} error), "
                  f"{len(linhas)} linhas no total, "
                  f"{decorrido/3600:.1f}h decorridas, ~{restante/3600:.1f}h restantes")

    if caminho_saida:
        _salvar_checkpoint(caminho_saida, linhas)

    print("-" * 78)
    print(f"[hipercubo] disparo principal terminado em {(time.time()-t0)/3600:.2f}h: "
          f"{contagem_status}")

    indices_pendentes = [i for i, s in status_por_indice.items() if s in ("timeout", "error")]

    if not auto_refino or not indices_pendentes:
        print("[hipercubo] nenhum refinamento automatico sera' feito."
              if not indices_pendentes else
              "[hipercubo] auto_refino=False -- nao refinando agora.")
        print("-" * 78)
        return pd.DataFrame(linhas)

    print(f"[hipercubo] iniciando refinamento de {len(indices_pendentes)} curvas "
          f"timeout/error, com timeout_refino={timeout_refino:.0f}s cada")
    t_refino = time.time()
    n_melhoradas = 0
    for j, i in enumerate(indices_pendentes, start=1):
        alpha, beta, cT = parametros_por_indice[i]
        status, novas_linhas = processa_uma_curva(i, alpha, beta, cT, timeout_refino)
        if status == "ok":
            n_melhoradas += 1
            linhas = [l for l in linhas if l.get("_indice") != i]
            linhas.extend(novas_linhas)
        print(f"[refino {j}/{len(indices_pendentes)}] indice={i} -> status={status}", flush=True)
        if caminho_saida and j % max(1, salvar_a_cada // 10) == 0:
            _salvar_checkpoint(caminho_saida, linhas)

    if caminho_saida:
        _salvar_checkpoint(caminho_saida, linhas)
    print(f"[hipercubo] refinamento terminado em {(time.time()-t_refino)/3600:.2f}h: "
          f"{n_melhoradas} de {len(indices_pendentes)} curvas fecharam dessa vez.")
    print("-" * 78)
    return pd.DataFrame(linhas)


# =================================================================
# 7. OS TRES MAPAS DE ESTADO (N_res como cor, sempre vs c_T)
#    1) alpha        vs c_T  -- "caso densidade"
#    2) alpha*beta   vs c_T  -- "caso misto"
#    3) beta         vs c_T  -- "caso velocidade"
#
# FIX: N_res agora vem direto da coluna N_res (gravada por linha,
# inclusive nas linhas de curvas com 0 picos) em vez de contar
# "xM_pico" -- antes, indices com 0 picos nem apareciam no groupby
# porque nao existia linha nenhuma pra eles.
# =================================================================
def plotar_tres_mapas(caminho_csv, arquivo_saida="tres_mapas_estado.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(caminho_csv)
    agregacoes = {"alpha": ("alpha", "first"), "beta": ("beta", "first"),
                  "cT": ("cT", "first"), "alpha_vezes_beta": ("alpha_vezes_beta", "first")}
    if "N_res" in df.columns:
        agregacoes["N_res"] = ("N_res", "first")
        contagem = df.groupby("_indice").agg(**agregacoes).reset_index()
    else:
        # compatibilidade com CSVs antigos (sem coluna N_res e sem linhas N_res=0)
        agregacoes["N_res"] = ("xM_pico", "count")
        contagem = df.groupby("_indice").agg(**agregacoes).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    eixos_y = [("alpha", "alpha (rho_M/rho_S) -- caso densidade"),
               ("alpha_vezes_beta", "alpha x beta -- caso misto"),
               ("beta", "beta (c_M/c_L) -- caso velocidade")]

    for ax, (coluna_y, titulo) in zip(axes, eixos_y):
        sc = ax.scatter(contagem["cT"], contagem[coluna_y], c=contagem["N_res"],
                         cmap="viridis", s=25, edgecolors="black", linewidths=0.3)
        ax.set_xlabel("c_T")
        ax.set_ylabel(coluna_y)
        ax.set_yscale("log")
        ax.set_title(titulo, fontsize=10)
        fig.colorbar(sc, ax=ax, label="N_res")

    fig.tight_layout()
    fig.savefig(arquivo_saida, dpi=150, bbox_inches="tight")
    print(f"salvo: {arquivo_saida}")


if __name__ == "__main__":
    print("=== teste de sanidade: 5 picos esperados (l=0,2,3,4,5) ===")
    picos = contar_picos_elastico(1500.0, 900.0, 56.0, l_max=8)
    print(f"{len(picos)} picos")
    for p in picos:
        print(f"  xM={p['xM_pico']:.4f}  l={p['l']}  Q={p['Q']:.1f}")
    print()

    print("=== disparando o hipercubo completo ===")
    grid_hipercubo_alpha_beta(
        n_curvas=8000,
        l_max=10,
        caminho_saida="resultados_elastico/hipercubo_elastico.csv",
        salvar_a_cada=200,
        timeout_por_curva=60.0,
        timeout_refino=600.0,
        auto_refino=True,
    )
