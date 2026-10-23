import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn, eval_legendre
import time

# ============================================================
# CONFIGURAÇÃO
# ============================================================
# Este código mantém a normalização da sua derivação:
#
#   N_Q = 2 f0 <Q> / (Z_s |A|^2),   com 2*pi*f0 = omega
#
# A superfície é calculada em duas formas equivalentes:
#
#   I_S = -4*pi*|A|^2*kL*/omega^2 * S_modal
#
#   N_S = (2 f0)/(Z_s |A|^2) * I_S/4
#       = -kL*/(Z_s*omega) * S_modal
#
# Portanto, W_ana normalizada é:
#
#   N_W^(ana) = 2*N_T + N_S
#
# Sem recolocar nem remover arbitrariamente o fator de superfície.

N_RADIAL_GAUSS = 260
N_R_TEST = 90
N_MU_TEST = 80
EPS = 1e-14


# ============================================================
# BESSEL ESFÉRICA E DERIVADAS
# ============================================================
def J(n, x):
    if n >= 0:
        return spherical_jn(n, x)
    m = -n - 1
    return (-1) ** (m + 1) * spherical_yn(m, x)


def dJ(n, x):
    if n >= 0:
        return spherical_jn(n, x, derivative=True)
    m = -n - 1
    return (-1) ** (m + 1) * spherical_yn(m, x, derivative=True)


def H(n, x):
    if n >= 0:
        return spherical_jn(n, x) + 1j * spherical_yn(n, x)
    m = -n - 1
    return 1j * (-1) ** m * (spherical_jn(m, x) + 1j * spherical_yn(m, x))


def dH(n, x):
    if n >= 0:
        return spherical_jn(n, x, derivative=True) + 1j * spherical_yn(n, x, derivative=True)
    m = -n - 1
    return 1j * (-1) ** m * (
        spherical_jn(m, x, derivative=True) + 1j * spherical_yn(m, x, derivative=True)
    )


def ddJ(n, z):
    """Segunda derivada de j_n(z) em relação ao argumento z."""
    z = np.asarray(z)
    out = np.empty_like(z, dtype=complex)

    mask0 = np.abs(z) < 1e-12
    mask = ~mask0

    if np.any(mask):
        zm = z[mask]
        out[mask] = (
            n * (n - 1) / (zm * zm) * J(n, zm)
            - (2 * n + 1) / zm * J(n + 1, zm)
            + J(n + 2, zm)
        )

    if np.any(mask0):
        if n == 0:
            out[mask0] = -1.0 / 3.0
        elif n == 1:
            out[mask0] = 0.0
        elif n == 2:
            out[mask0] = 2.0 / 15.0
        else:
            out[mask0] = 0.0

    if out.ndim == 0:
        return complex(out)
    return out


# ============================================================
# COEFICIENTES DE ESPALHAMENTO
# ============================================================
def coefs(xM, l_max, cL, cT, cM, rhoS, rhoM):
    xL_abs = xM * (cM / np.abs(cL))
    eta = cL / cT
    phi = np.angle(1.0 / cL)

    # x_L = k_L a e x_T = k_T a = eta*x_L.
    zL = xL_abs * np.exp(1j * phi)
    zT = eta * zL
    zM = xM

    mu = rhoS * cT**2
    lamb = rhoS * cL**2 - 2.0 * mu

    a_l = np.zeros(l_max + 1, dtype=complex)
    b_l = np.zeros(l_max + 1, dtype=complex)

    for l in range(l_max + 1):
        den_M = rhoM * cM**2 * zM**2 * H(l, zM)

        A_l = (lamb * zL**2 * J(l, zL) - 2.0 * mu * zL**2 * ddJ(l, zL)) / den_M
        B_l = -2.0 * mu * l * (l + 1) * (zT * dJ(l, zT) - J(l, zT)) / den_M
        C_l = -J(l, zM) / H(l, zM)

        den_M_tilde = zM * dH(l, zM)
        A_l_tilde = zL * dJ(l, zL) / den_M_tilde
        B_l_tilde = l * (l + 1) * J(l, zT) / den_M_tilde
        C_l_tilde = -dJ(l, zM) / dH(l, zM)

        num_R = 2.0 * (J(l, zL) - zL * dJ(l, zL))
        den_R = zT**2 * ddJ(l, zT) + (l * (l + 1) - 2.0) * J(l, zT)
        R_l = num_R / den_R

        num_a = C_l_tilde - C_l
        den_a = (A_l - A_l_tilde) - (B_l_tilde - B_l) * R_l

        a_l[l] = num_a / den_a
        b_l[l] = R_l * a_l[l]

    return a_l, b_l


# ============================================================
# QUADRATURA RADIAL E FUNÇÕES MODAIS
# ============================================================
def gauss_interval(a, b, n):
    nodes, weights = np.polynomial.legendre.leggauss(n)
    x = 0.5 * (b - a) * nodes + 0.5 * (b + a)
    w = 0.5 * (b - a) * weights
    return x, w


def modal_functions(l, xi, a_l, b_l, eta, phi):
    """A_l, B_l e derivadas em x=e^(i phi) xi."""
    L = l * (l + 1)
    x = np.exp(1j * phi) * xi
    xt = eta * x

    jl = J(l, x)
    djl = dJ(l, x)
    ddjl = ddJ(l, x)

    jlt = J(l, xt)
    djlt = dJ(l, xt)
    ddjlt = ddJ(l, xt)

    A = a_l * djl + b_l * L * jlt / x
    B = a_l * jl / x + b_l * (jlt / x + eta * djlt)

    dA = a_l * ddjl + b_l * L * (eta * djlt / x - jlt / x**2)
    dB = (
        a_l * (djl / x - jl / x**2)
        + b_l * (eta * djlt / x - jlt / x**2 + eta**2 * ddjlt)
    )

    return A, B, dA, dB


# ============================================================
# ENERGIAS DE VOLUME: SOMATÓRIOS ADIMENSIONAIS
# ============================================================
def calc_modal_volume_sums(xM, l_max, cL, cT, cM, rhoS, rhoM, n_radial=N_RADIAL_GAUSS):
    """
    Retorna os cinco somatórios radiais sem prefatores físicos.

    A formulação usa diretamente A_l, B_l, A'_l e B'_l. Assim, não há
    conjugação dupla: cada produto é F*conj(F) no mesmo ponto xi.
    """
    a_coeff, b_coeff = coefs(xM, l_max, cL, cT, cM, rhoS, rhoM)

    xL_abs = xM * cM / np.abs(cL)
    phi = np.angle(1.0 / cL)
    eta = cL / cT

    xi, w = gauss_interval(0.0, xL_abs, n_radial)

    sums = {
        "T_dim": 0.0,
        "U_lambda_dim": 0.0,
        "U_rr_dim": 0.0,
        "U_rtheta_dim": 0.0,
        "U_tang_dim": 0.0,
    }

    for l in range(l_max + 1):
        L = l * (l + 1)
        fact2 = (l + 2) * (l + 1) * l * (l - 1) if l >= 2 else 0.0

        A, B, dA, dB = modal_functions(
            l, xi, a_coeff[l], b_coeff[l], eta, phi
        )

        # Energia cinética: xi^2 (|A|^2 + L |B|^2).
        T_int = np.sum(w * xi**2 * (np.abs(A)**2 + L * np.abs(B)**2))

        # Parte volumétrica: div u só contém o potencial longitudinal.
        x = np.exp(1j * phi) * xi
        div_int = np.sum(w * xi**2 * np.abs(a_coeff[l] * J(l, x))**2)

        # epsilon_rr.
        rr_int = np.sum(w * xi**2 * np.abs(dA)**2)

        # epsilon_rtheta: B' + A/x - B/x.
        Q = dB + A / x - B / x
        rtheta_int = np.sum(w * xi**2 * np.abs(Q)**2)

        # epsilon_thetatheta + epsilon_phiphi e a diferença m=2.
        S = 2.0 * A - L * B
        tang_int = np.sum(w * (np.abs(S)**2 + fact2 * np.abs(B)**2))

        pref_mode = 2 * l + 1
        sums["T_dim"] += pref_mode * T_int
        sums["U_lambda_dim"] += pref_mode * div_int
        sums["U_rr_dim"] += pref_mode * rr_int
        sums["U_rtheta_dim"] += pref_mode * L * rtheta_int
        sums["U_tang_dim"] += pref_mode * tang_int

    return sums, a_coeff, b_coeff


# ============================================================
# SUPERFÍCIE: I_S FÍSICO E A MESMA PARCELA NORMALIZADA
# ============================================================
def calc_surface_terms(xM, l_max, cL, cT, cM, rhoS, rhoM, amplitude=1.0):
    a_coeff, b_coeff = coefs(xM, l_max, cL, cT, cM, rhoS, rhoM)

    omega = xM * cM
    kL = omega / cL
    eta = cL / cT
    zL = kL
    zT = eta * zL
    Zs = rhoS * cL

    mu = rhoS * cT**2
    lamb = rhoS * cL**2 - 2.0 * mu

    modal_sum = 0.0j
    shear_bc_values = []

    for l in range(l_max + 1):
        L = l * (l + 1)

        A_boundary, B_boundary, dA_boundary, dB_boundary = modal_functions(
            l,
            np.asarray([np.abs(zL)]),
            a_coeff[l],
            b_coeff[l],
            eta,
            np.angle(kL),
        )
        A_boundary = A_boundary[0]
        B_boundary = B_boundary[0]
        dA_boundary = dA_boundary[0]
        dB_boundary = dB_boundary[0]

        # Forma explícita da sua derivação para C_l.
        C_l = (
            a_coeff[l] * zL**2 * (lamb * J(l, zL) - 2.0 * mu * ddJ(l, zL))
            - 2.0 * mu * L * b_coeff[l] * (zT * dJ(l, zT) - J(l, zT))
        )

        modal_sum += (2 * l + 1) * C_l * np.conj(A_boundary)

        # sigma_rtheta proporcional a B'_l + A_l/x - B_l/x na fronteira.
        shear_bc_values.append(dB_boundary + A_boundary / zL - B_boundary / zL)

    # Integral FÍSICA: este fator é necessário e fica explícito aqui.
    I_S_physical = (
        -4.0 * np.pi * np.abs(amplitude)**2 * np.conj(kL) / omega**2 * modal_sum
    )

    # O fator inverso recupera exatamente o somatório modal da superfície.
    modal_recovered = (
        -omega**2
        / (4.0 * np.pi * np.abs(amplitude)**2 * np.conj(kL))
        * I_S_physical
    )

    # Depois de normalizar 1/4 I_S por 2 f0/(Z_s |A|^2),
    # obtém-se diretamente a parcela que entra em N_W.
    surface_norm_from_IS = (
        omega / (np.pi * Zs * np.abs(amplitude)**2)
    ) * 0.25 * I_S_physical

    surface_norm_direct = -np.conj(kL) / (Zs * omega) * modal_sum

    shear_bc_values = np.asarray(shear_bc_values)

    return {
        "modal_sum": modal_sum,
        "I_S_physical": I_S_physical,
        "modal_recovered": modal_recovered,
        "surface_norm_from_IS": surface_norm_from_IS,
        "surface_norm_direct": surface_norm_direct,
        "shear_bc_values": shear_bc_values,
    }


# ============================================================
# ENERGIAS NORMALIZADAS: COMPONENTES E CLAPEYRON
# ============================================================
def calc_normalized_energies(xM, l_max, cL, cT, cM, rhoS, rhoM, n_radial=N_RADIAL_GAUSS):
    sums, a_coeff, b_coeff = calc_modal_volume_sums(
        xM, l_max, cL, cT, cM, rhoS, rhoM, n_radial=n_radial
    )

    omega = xM * cM
    kL = omega / cL
    abs_kL = np.abs(kL)
    Zs = rhoS * cL

    mu = rhoS * cT**2
    lamb = rhoS * cL**2 - 2.0 * mu

    # N_Q = 2 f0 <Q>/(Z_s |A|^2), com 2*pi*f0 = omega.
    N_T = omega * rhoS / (Zs * abs_kL) * sums["T_dim"]
    N_U_lambda = lamb * abs_kL / (Zs * omega) * sums["U_lambda_dim"]
    N_U_rr = 2.0 * mu * abs_kL / (Zs * omega) * sums["U_rr_dim"]
    N_U_rtheta = mu * abs_kL / (Zs * omega) * sums["U_rtheta_dim"]
    N_U_tang = mu * abs_kL / (Zs * omega) * sums["U_tang_dim"]

    N_U = N_U_lambda + N_U_rr + N_U_rtheta + N_U_tang
    N_W_volume = N_T + N_U

    surface = calc_surface_terms(xM, l_max, cL, cT, cM, rhoS, rhoM)
    N_surface = surface["surface_norm_direct"]
    N_W_clapeyron = 2.0 * N_T + N_surface

    scale_W = max(abs(N_W_volume), abs(N_W_clapeyron), np.finfo(float).tiny)
    scale_surface = max(abs(N_U - N_T), abs(N_surface), np.finfo(float).tiny)

    error_W = 100.0 * abs(N_W_volume - N_W_clapeyron) / scale_W
    error_clapeyron = 100.0 * abs((N_U - N_T) - N_surface) / scale_surface

    # Testes locais que precisam ficar próximos da precisão numérica.
    surface_factor_error = abs(surface["modal_recovered"] - surface["modal_sum"]) / max(
        abs(surface["modal_sum"]), np.finfo(float).tiny
    )
    normalized_surface_error = abs(
        surface["surface_norm_from_IS"] - surface["surface_norm_direct"]
    ) / max(abs(surface["surface_norm_direct"]), np.finfo(float).tiny)

    traction_scale = max(abs(a_coeff).max(), abs(b_coeff).max(), np.finfo(float).tiny)
    shear_bc_error = np.max(abs(surface["shear_bc_values"])) / traction_scale

    return {
        **sums,
        "N_T": N_T,
        "N_U_lambda": N_U_lambda,
        "N_U_rr": N_U_rr,
        "N_U_rtheta": N_U_rtheta,
        "N_U_tang": N_U_tang,
        "N_U": N_U,
        "N_W_volume": N_W_volume,
        "N_W_clapeyron": N_W_clapeyron,
        "N_surface": N_surface,
        "error_W_percent": error_W,
        "error_clapeyron_percent": error_clapeyron,
        "surface_factor_error": surface_factor_error,
        "normalized_surface_error": normalized_surface_error,
        "shear_bc_error": shear_bc_error,
        "I_S_physical": surface["I_S_physical"],
        "surface_modal_sum": surface["modal_sum"],
    }


# ============================================================
# TESTE INDEPENDENTE: INTEGRAÇÃO DIRETA DO CAMPO 2D
# ============================================================
def legendre_with_theta_derivatives(l, mu):
    P = eval_legendre(l, mu)
    if l == 0:
        dP_dmu = np.zeros_like(mu)
    else:
        P_prev = eval_legendre(l - 1, mu)
        dP_dmu = l * (mu * P - P_prev) / (mu**2 - 1.0)

    sin_theta = np.sqrt(1.0 - mu**2)
    P_theta = -sin_theta * dP_dmu
    P_thetatheta = -l * (l + 1) * P + mu * dP_dmu
    cot_Ptheta = -mu * dP_dmu
    return P, P_theta, P_thetatheta, cot_Ptheta


def calc_fullfield_energy_2d(xM, l_max, cL, cT, cM, rhoS, rhoM, n_r=N_R_TEST, n_mu=N_MU_TEST):
    """Verificação independente por quadratura em r e mu=cos(theta)."""
    omega = xM * cM
    kL = omega / cL
    abs_kL = abs(kL)
    phi = np.angle(kL)
    eta = cL / cT
    Zs = rhoS * cL
    mu_lame = rhoS * cT**2
    lamb = rhoS * cL**2 - 2.0 * mu_lame

    a_coeff, b_coeff = coefs(xM, l_max, cL, cT, cM, rhoS, rhoM)
    xL_abs = abs_kL

    xi, wx = gauss_interval(0.0, xL_abs, n_r)
    mu_nodes, wmu = np.polynomial.legendre.leggauss(n_mu)

    P_list = []
    Pt_list = []
    Ptt_list = []
    cotPt_list = []
    for l in range(l_max + 1):
        P, Pt, Ptt, cotPt = legendre_with_theta_derivatives(l, mu_nodes)
        P_list.append(P)
        Pt_list.append(Pt)
        Ptt_list.append(Ptt)
        cotPt_list.append(cotPt)

    pref_u = -kL / (1j * omega)
    pref_strain = -kL**2 / (1j * omega)

    T_phys = 0.0
    U_phys = 0.0j

    for xii, wi in zip(xi, wx):
        A_sum = np.zeros_like(mu_nodes, dtype=complex)
        Btheta_sum = np.zeros_like(mu_nodes, dtype=complex)
        dA_sum = np.zeros_like(mu_nodes, dtype=complex)
        Qtheta_sum = np.zeros_like(mu_nodes, dtype=complex)
        theta_theta_sum = np.zeros_like(mu_nodes, dtype=complex)
        phi_phi_sum = np.zeros_like(mu_nodes, dtype=complex)

        for l in range(l_max + 1):
            L = l * (l + 1)
            amp_l = (2 * l + 1) * (1j)**l
            A, B, dA, dB = modal_functions(
                l,
                np.asarray([xii]),
                a_coeff[l],
                b_coeff[l],
                eta,
                phi,
            )
            A = A[0]
            B = B[0]
            dA = dA[0]
            dB = dB[0]

            P = P_list[l]
            Pt = Pt_list[l]
            Ptt = Ptt_list[l]
            cotPt = cotPt_list[l]

            A_sum += amp_l * A * P
            Btheta_sum += amp_l * B * Pt
            dA_sum += amp_l * dA * P
            Qtheta_sum += amp_l * (dB + A / (np.exp(1j * phi) * xii) - B / (np.exp(1j * phi) * xii)) * Pt

            x_complex = np.exp(1j * phi) * xii
            theta_theta_sum += amp_l * (B * Ptt + A * P) / x_complex
            phi_phi_sum += amp_l * (A * P + B * cotPt) / x_complex

        ur = pref_u * A_sum
        uth = pref_u * Btheta_sum

        eps_rr = pref_strain * dA_sum
        eps_rtheta = 0.5 * pref_strain * Qtheta_sum
        eps_tt = pref_strain * theta_theta_sum
        eps_pp = pref_strain * phi_phi_sum
        div_u = eps_rr + eps_tt + eps_pp

        kinetic_density = 0.25 * rhoS * omega**2 * (np.abs(ur)**2 + np.abs(uth)**2)
        strain_contract = (
            eps_rr * np.conj(eps_rr)
            + eps_tt * np.conj(eps_tt)
            + eps_pp * np.conj(eps_pp)
            + 2.0 * eps_rtheta * np.conj(eps_rtheta)
        )
        potential_density = 0.25 * (
            lamb * div_u * np.conj(div_u) + 2.0 * mu_lame * strain_contract
        )

        dV_factor = 2.0 * np.pi * xii**2 / abs_kL**3
        T_phys += wi * dV_factor * np.sum(wmu * kinetic_density)
        U_phys += wi * dV_factor * np.sum(wmu * potential_density)

    norm_factor = omega / (np.pi * Zs)
    return {
        "N_T_full": norm_factor * T_phys,
        "N_U_full": norm_factor * U_phys,
        "N_W_full": norm_factor * (T_phys + U_phys),
    }


# ============================================================
# EXECUÇÃO
# ============================================================
if __name__ == "__main__":
    rhoM = 1000.0
    cM = 1500.0
    rhoS = 1200.0
    l_max = 5

    # Para a varredura completa; aumente depois se quiser.
    xm_values = np.linspace(0.1, 6.0, 35)

    cenarios = [
        {"title": "Caso 1: Real / Real\n(Elástico puro)", "cL": 2500.0 + 0j, "cT": 1100.0 + 0j},
        {"title": "Caso 2: Comp / Comp\n(Viscoelástico)", "cL": 2500.0 - 50j, "cT": 1100.0 - 25j},
        {"title": "Caso 3: Comp / Real\n(Misto: apenas cL atenua)", "cL": 2500.0 - 50j, "cT": 1100.0 + 0j},
    ]

    # --------------------------------------------------------
    # TESTE INDEPENDENTE 2D: somente no caso elástico.
    # --------------------------------------------------------
    print("\n" + "=" * 72)
    print("TESTE INDEPENDENTE 2D (campo completo) — CASO REAL / REAL")
    print("=" * 72)
    for xm_test in [0.5, 1.5, 3.0, 5.0]:
        out_modal = calc_normalized_energies(
            xm_test, l_max, cenarios[0]["cL"], cenarios[0]["cT"], cM, rhoS, rhoM
        )
        out_2d = calc_fullfield_energy_2d(
            xm_test, l_max, cenarios[0]["cL"], cenarios[0]["cT"], cM, rhoS, rhoM
        )

        err_full_vs_modal = 100.0 * abs(out_2d["N_W_full"] - out_modal["N_W_volume"]) / max(
            abs(out_2d["N_W_full"]), abs(out_modal["N_W_volume"]), np.finfo(float).tiny
        )
        err_full_vs_clap = 100.0 * abs(out_2d["N_W_full"] - out_modal["N_W_clapeyron"]) / max(
            abs(out_2d["N_W_full"]), abs(out_modal["N_W_clapeyron"]), np.finfo(float).tiny
        )

        print(
            f"xM={xm_test:4.1f} | "
            f"erro 2D-vs-modal={err_full_vs_modal:.3e}% | "
            f"erro 2D-vs-Clapeyron={err_full_vs_clap:.3e}% | "
            f"BC cisalhante={out_modal['shear_bc_error']:.3e}"
        )

    resultados = []

    # --------------------------------------------------------
    # VARREDURA PRINCIPAL
    # --------------------------------------------------------
    for caso in cenarios:
        print("\n" + "=" * 72)
        print(caso["title"].replace("\n", " — "))
        print("=" * 72)

        dados = {
            "xM": [],
            "N_T": [],
            "N_U_lambda": [],
            "N_U_rr": [],
            "N_U_rtheta": [],
            "N_U_tang": [],
            "N_W_volume": [],
            "N_W_clapeyron": [],
            "err_W": [],
            "err_clap": [],
            "err_surface_factor": [],
            "err_surface_norm": [],
            "err_shear_bc": [],
        }

        t0 = time.time()
        for xm in xm_values:
            out = calc_normalized_energies(
                xm, l_max, caso["cL"], caso["cT"], cM, rhoS, rhoM
            )

            for key in [
                "N_T", "N_U_lambda", "N_U_rr", "N_U_rtheta", "N_U_tang",
                "N_W_volume", "N_W_clapeyron",
            ]:
                dados[key].append(out[key])

            dados["xM"].append(xm)
            dados["err_W"].append(out["error_W_percent"])
            dados["err_clap"].append(out["error_clapeyron_percent"])
            dados["err_surface_factor"].append(out["surface_factor_error"])
            dados["err_surface_norm"].append(out["normalized_surface_error"])
            dados["err_shear_bc"].append(out["shear_bc_error"])

        for key in dados:
            dados[key] = np.asarray(dados[key])

        resultados.append(dados)

        print(f"Tempo: {time.time() - t0:.2f} s")
        print(
            "Erro W_volume vs W_Clapeyron (min/med/max): "
            f"{dados['err_W'].min():.3e}% / {dados['err_W'].mean():.3e}% / {dados['err_W'].max():.3e}%"
        )
        print(
            "Resíduo U-T vs superfície (min/med/max):    "
            f"{dados['err_clap'].min():.3e}% / {dados['err_clap'].mean():.3e}% / {dados['err_clap'].max():.3e}%"
        )
        print(
            "Teste do fator -omega^2/(4*pi|A|^2 kL*) (max): "
            f"{dados['err_surface_factor'].max():.3e}"
        )
        print(
            "Teste N_S via I_S versus forma direta (max):     "
            f"{dados['err_surface_norm'].max():.3e}"
        )
        print(
            "Resíduo da condição sigma_rtheta(a)=0 (max):      "
            f"{dados['err_shear_bc'].max():.3e}"
        )

    # --------------------------------------------------------
    # FIGURA 1: REAIS DAS ENERGIAS NORMALIZADAS
    # --------------------------------------------------------
    fig1, axes1 = plt.subplots(1, 3, figsize=(19, 6))

    for ax, caso, dados in zip(axes1, cenarios, resultados):
        x = dados["xM"]
        ax.plot(x, np.real(dados["N_T"]), label=r"$N_T$")
        ax.plot(x, np.real(dados["N_U_lambda"]), linestyle="--", label=r"$N_{U_\lambda}$")
        ax.plot(x, np.real(dados["N_U_rr"]), linestyle=":", label=r"$N_{U_{\mu,rr}}$")
        ax.plot(x, np.real(dados["N_U_rtheta"]), linestyle="-.", label=r"$N_{U_{\mu,r\theta}}$")
        ax.plot(x, np.real(dados["N_U_tang"]), alpha=0.75, label=r"$N_{U_{\mu,\theta\theta+\phi\phi}}$")
        ax.plot(x, np.real(dados["N_W_volume"]), "ko", markersize=4.0, label=r"$N_W$ volume")
        ax.plot(x, np.real(dados["N_W_clapeyron"]), linewidth=2.0, label=r"$N_W$ Clapeyron")
        ax.set_title(caso["title"], fontsize=11, fontweight="bold")
        ax.set_xlabel(r"$x_M$")
        ax.grid(True, which="both", linestyle="--", alpha=0.45)
        ax.legend(fontsize=7.5, loc="best")

    axes1[0].set_ylabel(r"Parte real de $2f_0\langle Q\rangle/(Z_s|A|^2)$")
    fig1.suptitle("Energia normalizada: volume modal versus superfície de Clapeyron", y=1.02)
    fig1.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # FIGURA 2: ERROS E TESTES
    # --------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 3, figsize=(19, 5), sharey=True)

    for ax, caso, dados in zip(axes2, cenarios, resultados):
        x = dados["xM"]
        ax.plot(x, np.maximum(dados["err_W"], np.finfo(float).tiny), "o-", label=r"$|W_V-W_C|$")
        ax.plot(x, np.maximum(dados["err_clap"], np.finfo(float).tiny), "s--", label=r"$|U-T-N_S|$")
        ax.set_yscale("log")
        ax.set_title(caso["title"], fontsize=11, fontweight="bold")
        ax.set_xlabel(r"$x_M$")
        ax.grid(True, which="both", linestyle="--", alpha=0.45)
        ax.legend(fontsize=8)

    axes2[0].set_ylabel("Erro relativo (%)")
    fig2.suptitle("Diagnóstico de Clapeyron — o fator de superfície permanece explícito", y=1.02)
    fig2.tight_layout()
    plt.show()
