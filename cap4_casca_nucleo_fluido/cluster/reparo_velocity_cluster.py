#!/usr/bin/env python3
# ============================================================
# REPARO DO CHECKPOINT VELOCITY (CLUSTER)
#
# Refaz a busca de raízes de D_l nas curvas velocity que passaram (ou
# deveriam ter passado) pela busca de polos, com duas correções:
#
# 1) Polo de ordem 4 em z = 0: busca feita em z^4 * D_l(z), que tem os
#    mesmos zeros no retângulo mas não tem o polo (a borda x_min = 1e-4
#    colada no polo fazia perder a raiz mais baixa de cada l).
#
# 2) Janela confiável no plano complexo: com amplificação ~1000-2350
#    (velocity), D_l na base J/Y da casca sofre cancelamento catastrófico
#    quando |Im(argumento)| passa de ~15-20 (J*Y' - Y*J' subtrai números
#    ~e^{2|Im|} para dar algo de ordem 1). Nessa região D_l vira ruído
#    (inclusive zero exato). Para cada curva a janela é escolhida
#    automaticamente: a mais profunda y em que, nas linhas Im = y e
#    Im = |y|, D_l é finito, sem zero exato e com |D_l| < 1e200 para
#    l = 0, 5, 10. As Bessel do scipy estão corretas (conferido contra
#    mpmath); o problema é só a fórmula de D_l nessa região.
#
# Busca: contagem por faixas em x (que se subdividem quando a contagem
# não fecha), aceitando só quando duas resoluções concordam E nenhum
# salto de fase entre amostras passa de pi/2; raízes já achadas são
# divididas da função (deflação); Newton + resíduo relativo < 1e-7.
# Validação por energia e merge: funções do próprio gerador.
#
# Nunca piora uma curva: se algum l energeticamente relevante não fechar
# a contagem, a curva fica EXATAMENTE como estava no checkpoint de
# entrada, com o resultado da tentativa guardado em "reparo_velocity".
#
# Uso típico:
#   python reparo_velocity_cluster.py --checkpoint CK.npz --gerador GER.py --workers 64
#
# Com job array (N partes em paralelo, depois juntar):
#   python reparo_velocity_cluster.py --checkpoint CK.npz --gerador GER.py \
#          --workers 32 --shard $SLURM_ARRAY_TASK_ID --nshards N
#   python reparo_velocity_cluster.py --checkpoint CK.npz --gerador GER.py \
#          --juntar --nshards N
#
# Só algumas curvas (ex.: casos da figura primeiro):
#   ... --chaves 2_40,7_40,15_40,2_340,7_340,15_340
#
# Curvas que não terminaram no tempo ficam como estavam (anotadas em
# "reparo_velocity"); para tentar de novo com mais tempo:
#   ... --refazer-incompletas --tempo-max-curva 14400
# ============================================================

import argparse
import copy
import glob
import importlib.util
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.optimize import newton

warnings.filterwarnings("ignore")

# ============================================================
# PARÂMETROS NUMÉRICOS
# ============================================================
NIVEIS = (128, 256, 512, 1024, 2048, 4096, 8192, 16384)
SALTO_FASE_MAX = np.pi / 2
TOL_INTEIRO = 0.05
PROF_MAX = 40
PROF_MAX_FAIXA = 9            # até 512 faixas em x
TOL_DUPLICATA = 1e-8
RESIDUO_REL_MAX = 1e-7

Y_TOP_MAX = 0.05              # topo do contorno (não há zeros com Im > 0)
JANELAS_CANDIDATAS = (
    -0.5, -0.4, -0.3, -0.2, -0.15, -0.1, -0.07, -0.05, -0.03, -0.02,
    -0.015, -0.01, -0.008, -0.006, -0.005, -0.004, -0.003,
)
LOG10_MAX_CONFIAVEL = 200.0
N_PONTOS_TESTE_JANELA = 20000

g = None  # módulo do gerador (carregado em cada processo)


# ============================================================
# GERADOR
# ============================================================
def carregar_gerador(path):
    spec = importlib.util.spec_from_file_location("gerador_coreshell", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Bessel esféricas: mesmos ufuncs do scipy, sem o wrapper lento
    # (valores idênticos bit a bit). O gerador sempre chama com n >= 0.
    try:
        from scipy.special._ufuncs import (
            _spherical_jn, _spherical_jn_d, _spherical_yn, _spherical_yn_d,
        )

        def _sph_jn(n, z, derivative=False):
            return _spherical_jn_d(n, z) if derivative else _spherical_jn(n, z)

        def _sph_yn(n, z, derivative=False):
            return _spherical_yn_d(n, z) if derivative else _spherical_yn(n, z)

        mod.spherical_jn = _sph_jn
        mod.spherical_yn = _sph_yn
    except Exception:
        pass
    return mod


def _init_worker(gerador_path):
    global g
    g = carregar_gerador(gerador_path)


# ============================================================
# D_l REGULARIZADO (z^4 aplicado uma única vez, qualquer versão do gerador)
# ============================================================
def z_4_vezes_D(l, z, delta, materials):
    rho0, rho1, rho2, c0, c1, c2 = materials
    z = np.asarray(z, dtype=complex)
    xA = (c0 / c1) * z
    xB = (c0 / c2) * z
    yM = (1.0 + delta) * z
    yB = (1.0 + delta) * xB
    with np.errstate(all="ignore"):
        return z**4 * g.D_l_coreshell(
            l, xA, xB, yM, yB, rho0, rho1, rho2, c0, c1, c2,
        )


# ============================================================
# JANELA CONFIÁVEL
# ============================================================
def _linha_confiavel(y, delta, materials, x_min, x_max, lmax):
    x = np.linspace(x_min, x_max, N_PONTOS_TESTE_JANELA) + 1j * y
    for l in sorted({0, int(lmax) // 2, int(lmax)}):
        v = z_4_vezes_D(l, x, delta, materials)
        if not np.all(np.isfinite(v)) or np.any(v == 0.0):
            return False
        if np.log10(np.max(np.abs(v))) > LOG10_MAX_CONFIAVEL:
            return False
    return True


def escolher_janela(delta, materials, x_min, x_max, imag_min, lmax):
    """Retorna (y_baixo, y_topo) ou None se nem a janela mais rasa serve."""
    for y in JANELAS_CANDIDATAS:
        if y < imag_min:
            continue
        topo = min(Y_TOP_MAX, abs(y))
        if _linha_confiavel(y, delta, materials, x_min, x_max, lmax) and \
           _linha_confiavel(topo, delta, materials, x_min, x_max, lmax):
            return float(y), float(topo)
    return None


# ============================================================
# CONTAGEM E BUSCA
# ============================================================
def _contorno(x_lo, x_hi, y_lo, y_hi, n):
    return g._contour_points(x_lo, x_hi, y_lo, y_hi, n)


def _enrolamento(F, caixa, n):
    z = _contorno(*caixa, n)
    v = np.asarray(F(z), dtype=complex)
    if not np.all(np.isfinite(v)) or np.any(v == 0.0):
        return None, None
    with np.errstate(all="ignore"):
        razao = np.roll(v, -1) / v
    if not np.all(np.isfinite(razao)):
        return None, None
    inc = np.angle(razao)
    return float(np.sum(inc) / (2.0 * np.pi)), float(np.max(np.abs(inc)))


def contar(F, caixa, deadline=None):
    anterior = None
    for n in NIVEIS:
        if deadline is not None and time.time() > deadline:
            return None
        w, salto = _enrolamento(F, caixa, n)
        if w is None:
            return None  # não finito: resolução maior não resolve
        k = int(round(w))
        if k < 0 or abs(w - k) > TOL_INTEIRO or salto > SALTO_FASE_MAX:
            anterior = None
            continue
        if anterior == k:
            return k
        anterior = k
    return None


def _semente_momento(F, caixa, n=2048):
    z = _contorno(*caixa, n)
    v = np.asarray(F(z), dtype=complex)
    z2 = np.roll(z, -1)
    v2 = np.roll(v, -1)
    with np.errstate(all="ignore"):
        dlog = np.log(np.abs(v2 / v)) + 1j * np.angle(v2 / v)
    return complex(np.sum(0.5 * (z + z2) * dlog) / (2j * np.pi))


def _dentro(z, caixa, tol=1e-12):
    x_lo, x_hi, y_lo, y_hi = caixa
    return (x_lo - tol <= z.real <= x_hi + tol) and (y_lo - tol <= z.imag <= y_hi + tol)


def _quartos(caixa):
    x_lo, x_hi, y_lo, y_hi = caixa
    xm = x_lo + 0.5137 * (x_hi - x_lo)
    ym = y_lo + 0.4871 * (y_hi - y_lo)
    return [
        (x_lo, xm, y_lo, ym), (xm, x_hi, y_lo, ym),
        (x_lo, xm, ym, y_hi), (xm, x_hi, ym, y_hi),
    ]


def _polir(G, semente):
    try:
        r = complex(newton(G, semente, tol=1e-14, maxiter=200))
    except Exception:
        return None
    if not (np.isfinite(r.real) and np.isfinite(r.imag)):
        return None
    return r


def buscar(F, G_reg, caixa, m, deadline, prof=0):
    """Acha os m zeros de F na caixa (m=None: contagem desconhecida)."""
    if m is not None and m <= 0:
        return []
    if deadline is not None and time.time() > deadline:
        return []

    if m == 1:
        try:
            s = _semente_momento(F, caixa)
        except Exception:
            s = None
        if s is not None and np.isfinite(s.real) and np.isfinite(s.imag):
            r = _polir(G_reg, s)
            if r is not None and _dentro(r, caixa):
                res = abs(G_reg(r))
                escala = max(abs(G_reg(r + 1e-6 * max(1e-4, abs(r)))), 1e-300)
                if res / escala < RESIDUO_REL_MAX:
                    return [r]

    if prof >= PROF_MAX:
        return []

    achadas = []
    for sub in _quartos(caixa):
        k = contar(F, sub, deadline)
        achadas.extend(buscar(F, G_reg, sub, k, deadline, prof + 1))
    return achadas


def raizes_na_janela(l, delta, materials, conhecidas, x_min, x_max,
                     y_baixo, y_topo, imag_max, deadline):
    """Todas as raízes de D_l em [x_min,x_max] x [y_baixo, imag_max].
    Retorna (novas, esperadas_total, ok)."""
    conhecidas = [complex(r) for r in conhecidas]
    arr = np.array(conhecidas, dtype=complex)

    def G_reg(z):
        return z_4_vezes_D(l, z, delta, materials)

    def F(z):
        z = np.asarray(z, dtype=complex)
        v = G_reg(z)
        if len(arr):
            v = v / np.prod(z[..., None] - arr[None, :], axis=-1)
        return v

    novas = []
    faltam_total = 0
    ok = True
    pilha = [(float(x_min), float(x_max), 0)]

    while pilha:
        a, b, prof = pilha.pop()
        if deadline is not None and time.time() > deadline:
            ok = False
            break
        caixa = (a, b, y_baixo, y_topo)
        m = contar(F, caixa, deadline)
        if m is None:
            if prof >= PROF_MAX_FAIXA:
                ok = False
                continue
            meio = a + 0.5137 * (b - a)
            pilha.append((meio, b, prof + 1))
            pilha.append((a, meio, prof + 1))
            continue
        if m == 0:
            continue
        faltam_total += m

        achadas = []
        for r in buscar(F, G_reg, caixa, m, deadline):
            if not (a - 1e-12 <= r.real <= b + 1e-12):
                continue
            # Im <= imag_max: raízes quase reais de l alto podem voltar
            # com Im ~ +1e-17 por arredondamento
            if not (x_min <= r.real <= x_max and y_baixo <= r.imag <= imag_max):
                continue
            if len(arr) and np.min(np.abs(arr - r)) <= TOL_DUPLICATA:
                continue
            if any(abs(r - q) <= TOL_DUPLICATA for q in achadas + novas):
                continue
            achadas.append(r)
        if len(achadas) != m:
            ok = False
        novas.extend(achadas)

    esperadas = len(conhecidas) + faltam_total if ok else None
    return novas, esperadas, ok


# ============================================================
# UMA CURVA
# ============================================================
def _candidato(l, z, esperadas, ok_l, delta, materials, x_min, x_max, lmax):
    gamma = float(2.0 * abs(z.imag))
    c = {
        "l": int(l),
        "pole_real": float(z.real),
        "pole_imag": float(z.imag),
        "Gamma_pole": gamma,
        "Q_pole": float(z.real / gamma) if gamma > 0.0 else np.inf,
        "argument_count_l": None if esperadas is None else int(esperadas),
        "argument_audit_ok": bool(ok_l),
    }
    c.update(g._validate_candidate_by_energy(c, delta, materials, x_min, x_max, int(lmax)))
    return c


def _passou_pela_busca_ou_deveria(item):
    """Triagem descartou (sem potencial ressonante): fica como está.
    Todo o resto (busca completa, incompleta, risco de overflow,
    audit_failed, timeout) é refeito."""
    audit = item.get("audit", {}) or {}
    screening = audit.get("screening", {}) or {}
    if screening.get("ran") and not screening.get("potencialmente_ressonante", True):
        return False
    return True


def reparar_curva(item, regime, lmax, x_min, x_max, imag_min, imag_max, tempo_max):
    original = item
    item = copy.deepcopy(item)
    if not _passou_pela_busca_ou_deveria(item):
        return original, {"acao": "sem_busca_triagem", "novas": 0}

    delta = float(item["delta"])
    materials = g.make_materials_from_alpha_beta(
        item["alpha"], item["beta"], contrast_in=g.contrast_from_regime(regime)
    )
    t0 = time.time()
    deadline = t0 + tempo_max

    janela = escolher_janela(delta, materials, x_min, x_max, imag_min, lmax)
    if janela is None:
        novo = original
        novo = copy.deepcopy(original)
        novo["reparo_velocity"] = {"ok": False, "motivo": "sem_janela_confiavel"}
        return novo, {"acao": "sem_janela", "novas": 0}
    y_baixo, y_topo = janela

    # raízes já conhecidas DENTRO da janela confiável (fora dela, D_l era
    # ruído e o que foi achado lá não é confiável)
    audit_antigo = item.get("audit", {}) or {}
    l_state = audit_antigo.get("l_state", {}) or {}
    candidatos_antigos = item.get("candidatos", []) or []

    candidatos = []
    novo_l_state = {}
    audit_l = {}
    ls_ruins = []
    novas_total = 0
    grade_energia = None

    for l in range(int(lmax) + 1):
        if l in l_state:
            conhecidas = [complex(z) for z in l_state[l]["roots"]]
        else:
            conhecidas = [
                complex(c["pole_real"], c["pole_imag"])
                for c in candidatos_antigos if int(c["l"]) == l
            ]
        conhecidas = [
            z for z in conhecidas
            if y_baixo <= z.imag <= imag_max and x_min <= z.real <= x_max
        ]

        novas, esperadas, ok_l = raizes_na_janela(
            l, delta, materials, conhecidas, x_min, x_max,
            y_baixo, y_topo, imag_max, deadline,
        )
        novas_total += len(novas)
        todas = conhecidas + novas

        entrada = {
            "expected_count": esperadas,
            "found_count": len(todas),
            "ok": bool(ok_l),
            "reparo_velocity": True,
            "janela_im": [y_baixo, imag_max],
            "raizes_recuperadas": len(novas),
        }
        relevante = None
        contrib = None
        if not ok_l:
            try:
                if grade_energia is None:
                    xg = g._screening_grid(x_min, x_max)
                    grade_energia = (xg, g.energia_total_coreshell(xg, delta, *materials, lmax=int(lmax)))
                xg, wg = grade_energia
                contrib = float(g._relevancia_energetica_l(l, delta, materials, xg, wg))
                relevante = bool(contrib >= g.L_SIGNIFICANCE_THRESHOLD)
            except Exception:
                relevante = True
            entrada["contribuicao_relativa"] = contrib
            entrada["energeticamente_relevante"] = relevante
            if relevante:
                ls_ruins.append(l)
        audit_l[l] = entrada
        novo_l_state[l] = {
            "roots": todas,
            "expected_count": esperadas,
            "audit_ok": bool(ok_l),
            "energeticamente_relevante": relevante,
            "contribuicao_relativa": contrib,
        }

        antigos_l = {
            (round(c["pole_real"], 12), round(c["pole_imag"], 15)): c
            for c in candidatos_antigos if int(c["l"]) == l
        }
        for z in todas:
            chave = (round(z.real, 12), round(z.imag, 15))
            if chave in antigos_l and antigos_l[chave].get("energy_window_lo") is not None:
                c = dict(antigos_l[chave])
                c["argument_count_l"] = None if esperadas is None else int(esperadas)
                c["argument_audit_ok"] = bool(ok_l)
            else:
                c = _candidato(l, z, esperadas, ok_l, delta, materials, x_min, x_max, lmax)
            candidatos.append(c)

    completo = time.time() <= deadline
    info_tentativa = {
        "ok": (not ls_ruins) and completo,
        "janela_im": [y_baixo, imag_max],
        "y_topo_contorno": y_topo,
        "ls_relevantes_sem_fechar": list(ls_ruins),
        "runtime_s": float(time.time() - t0),
        "raizes_recuperadas": int(novas_total),
    }

    if ls_ruins or not completo:
        # nunca piora: devolve a curva original, só anota a tentativa
        novo = copy.deepcopy(original)
        novo["reparo_velocity"] = info_tentativa
        return novo, {"acao": "incompleta", "novas": 0, "ls_ruins": ls_ruins}

    candidatos.sort(key=lambda c: (c["pole_real"], c["l"], c["pole_imag"]))
    audit = dict(audit_antigo)
    for k in list(audit.keys()):
        if isinstance(k, (int, np.integer)):
            del audit[k]
    audit.update(audit_l)
    audit["l_state"] = novo_l_state
    audit["busca_polos"] = {
        "ok": True, "completa": True, "timeout_s": float(tempo_max),
        "reparo_velocity": True, "janela_im": [y_baixo, imag_max],
    }

    status_antigo = item.get("status")
    item["candidatos"] = candidatos
    item["ressonancias"] = g._merge_energy_resonances(candidatos)
    item["audit"] = audit
    item["audit_all_ok"] = all(g._audit_entry_aceitavel(audit[l]) for l in range(int(lmax) + 1))
    item["status"] = "ok"
    item["error"] = ""
    item["reparo_velocity"] = info_tentativa
    return item, {
        "acao": "reparada" if status_antigo == "ok" else f"refeita_{status_antigo}",
        "novas": novas_total,
    }


def _trabalho(args):
    chave, item, regime, lmax, x_min, x_max, imag_min, imag_max, tempo_max = args
    try:
        novo, info = reparar_curva(item, regime, lmax, x_min, x_max, imag_min, imag_max, tempo_max)
        return chave, novo, info, None
    except Exception as e:
        return chave, item, {"acao": "erro", "novas": 0}, repr(e)


# ============================================================
# ARQUIVOS
# ============================================================
def _ler_checkpoint(path):
    with np.load(path, allow_pickle=True) as d:
        meta = {k: d[k] for k in d.files if k not in ("curves", "done", "chaves_reparadas")}
        meta = {k: (v.item() if np.ndim(v) == 0 else v) for k, v in meta.items()}
        curves = d["curves"].item()
        done = np.asarray(d["done"], dtype=bool) if "done" in d.files else None
        feitas = set(str(k) for k in d["chaves_reparadas"]) if "chaves_reparadas" in d.files else set()
    return meta, curves, done, feitas


def _salvar(path, meta, curves, done, feitas):
    tmp = path + ".tmp.npz"
    np.savez_compressed(
        tmp,
        checkpoint_version=meta["checkpoint_version"],
        detector=np.array(str(meta["detector"]).split("+")[0] + "+reparo_z4_velocity"),
        regime=np.array(meta["regime"]),
        lmax=int(meta["lmax"]),
        x_min=float(meta["x_min"]),
        x_max=float(meta["x_max"]),
        imag_min=float(meta["imag_min"]),
        imag_max=float(meta["imag_max"]),
        timeout_seconds=float(meta["timeout_seconds"]),
        curves=np.array(curves, dtype=object),
        done=np.asarray(done, dtype=bool),
        alpha_grid=np.asarray(meta["alpha_grid"], dtype=float),
        beta_grid=np.asarray(meta["beta_grid"], dtype=float),
        delta_values=np.asarray(meta["delta_values"], dtype=float),
        chaves_reparadas=np.array(sorted(feitas), dtype=object),
    )
    os.replace(tmp, path)


def _caminho_saida(checkpoint, shard, nshards):
    base, _ = os.path.splitext(checkpoint)
    if nshards > 1:
        return f"{base}_reparado_z4_parte{shard:03d}de{nshards:03d}.npz"
    return f"{base}_reparado_z4.npz"


def _ordenar_chaves(keys):
    def k(s):
        try:
            return tuple(int(v) for v in s.split("_"))
        except Exception:
            return (10**9, s)
    return sorted(keys, key=k)


# ============================================================
# EXECUÇÃO
# ============================================================
def rodar(args):
    global g
    g = carregar_gerador(args.gerador)

    meta, curves_in, done, _ = _ler_checkpoint(args.checkpoint)
    regime = str(meta["regime"])
    lmax = int(meta["lmax"])
    x_min, x_max = float(meta["x_min"]), float(meta["x_max"])
    imag_min, imag_max = float(meta["imag_min"]), float(meta["imag_max"])

    todas = _ordenar_chaves(curves_in.keys())
    if args.chaves:
        pedidas = [c.strip() for c in args.chaves.split(",") if c.strip()]
        faltando = [c for c in pedidas if c not in curves_in]
        if faltando:
            print(f"Aviso: chaves inexistentes ignoradas: {faltando}")
        todas = [c for c in todas if c in pedidas]
    minhas = [c for i, c in enumerate(todas) if i % args.nshards == args.shard]

    saida = _caminho_saida(args.checkpoint, args.shard, args.nshards)
    if args.chaves and args.nshards == 1:
        base, _ = os.path.splitext(args.checkpoint)
        saida = f"{base}_reparado_z4_selecao.npz"

    # parte: guarda só as curvas desta parte; completo: todas
    if os.path.exists(saida):
        _, curves_out, _, feitas = _ler_checkpoint(saida)
        print(f"Retomando {saida}: {len(feitas)} curvas já processadas")
    else:
        curves_out = {} if args.nshards > 1 else dict(curves_in)
        feitas = set()
        if args.nshards > 1:
            for c in minhas:
                curves_out[c] = curves_in[c]

    if args.refazer_incompletas:
        reabrir = [
            c for c in minhas
            if c in feitas and not (curves_out.get(c, {}).get("reparo_velocity") or {}).get("ok", True)
        ]
        for c in reabrir:
            feitas.discard(c)
        if reabrir:
            print(f"Reabrindo {len(reabrir)} curvas com tentativa incompleta")
    pendentes = [c for c in minhas if c not in feitas]
    # triagem descartou: nada a fazer, marca direto
    triagem = [c for c in pendentes if not _passou_pela_busca_ou_deveria(curves_in[c])]
    for c in triagem:
        feitas.add(c)
    pendentes = [c for c in pendentes if c not in feitas]

    print("=" * 90)
    print(f"{os.path.basename(args.checkpoint)} | regime={regime} | parte {args.shard+1}/{args.nshards}")
    print(f"curvas da parte: {len(minhas)} | sem busca (triagem): {len(triagem)} | a processar: {len(pendentes)}")
    print(f"workers: {args.workers} | tempo máx por curva: {args.tempo_max_curva:.0f} s | saída: {saida}")
    print("=" * 90, flush=True)

    stats = {"reparada": 0, "refeita": 0, "incompleta": 0, "sem_janela": 0, "erro": 0, "novas": 0}
    t0 = time.time()
    n_feitas = 0
    tarefas = [
        (c, curves_in[c], regime, lmax, x_min, x_max, imag_min, imag_max, args.tempo_max_curva)
        for c in pendentes
    ]
    done_out = done if done is not None else np.ones(len(curves_in), dtype=bool)

    def registrar(chave, novo, info, erro):
        nonlocal n_feitas
        n_feitas += 1
        if erro is not None:
            stats["erro"] += 1
            print(f"  [erro] {chave}: {erro}", flush=True)
            return  # não marca como feita: tenta de novo na retomada
        curves_out[chave] = novo
        feitas.add(chave)
        stats["novas"] += info.get("novas", 0)
        a = info["acao"]
        if a == "reparada":
            stats["reparada"] += 1
        elif a.startswith("refeita"):
            stats["refeita"] += 1
        elif a == "incompleta":
            stats["incompleta"] += 1
        elif a == "sem_janela":
            stats["sem_janela"] += 1
        el = time.time() - t0
        print(f"  [{n_feitas}/{len(tarefas)}] {chave} {a} | raízes novas {info.get('novas', 0)} | "
              f"{el/3600:.2f} h", flush=True)
        if n_feitas % args.salvar_a_cada == 0:
            _salvar(saida, meta, curves_out, done_out, feitas)

    if args.workers <= 1:
        for t in tarefas:
            registrar(*_trabalho(t))
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers, initializer=_init_worker, initargs=(args.gerador,)
        ) as ex:
            futuros = [ex.submit(_trabalho, t) for t in tarefas]
            for f in as_completed(futuros):
                registrar(*f.result())

    _salvar(saida, meta, curves_out, done_out, feitas)
    print("-" * 90)
    for k, v in stats.items():
        print(f"{k:>12}: {v}")
    print(f"tempo: {(time.time()-t0)/3600:.2f} h | saída: {saida}", flush=True)

    if args.nshards == 1 and not args.chaves:
        _exportar_csv(saida, curves_out, lmax)
    return saida


def juntar(args):
    global g
    g = carregar_gerador(args.gerador)
    meta, curves, done, _ = _ler_checkpoint(args.checkpoint)
    feitas_total = set()
    base, _ = os.path.splitext(args.checkpoint)
    partes = sorted(glob.glob(f"{base}_reparado_z4_parte*de{args.nshards:03d}.npz"))
    if len(partes) != args.nshards:
        print(f"Aviso: esperava {args.nshards} partes, achei {len(partes)}: {partes}")
    for p in partes:
        _, cp, _, fp = _ler_checkpoint(p)
        for k in fp:
            curves[k] = cp[k]
        feitas_total |= fp
        print(f"  {os.path.basename(p)}: {len(fp)} curvas")
    saida = f"{base}_reparado_z4.npz"
    _salvar(saida, meta, curves, done if done is not None else np.ones(len(curves), bool), feitas_total)
    print(f"Juntado: {len(feitas_total)} de {len(curves)} curvas processadas -> {saida}")
    _exportar_csv(saida, curves, int(meta["lmax"]))
    _relatorio(curves)


def _exportar_csv(saida, curves, lmax):
    base, _ = os.path.splitext(saida)
    paths = {
        "candidates_csv": base + "_candidatos.csv",
        "resonances_csv": base + "_ressonancias.csv",
        "summary_csv": base + "_resumo.csv",
    }
    try:
        g.export_csv(paths, curves, lmax)
    except Exception as e:
        print(f"Aviso: CSV não exportado ({e!r})")


def _relatorio(curves):
    n_rep = n_inc = n_malha_falta = 0
    for item in curves.values():
        rv = item.get("reparo_velocity")
        if rv is None:
            continue
        if rv.get("ok"):
            n_rep += 1
        else:
            n_inc += 1
        pm = [p["xM_peak"] for p in ((item.get("audit") or {}).get("screening") or {}).get("picos_malha", []) or []]
        rs = np.array([r["xM_peak"] for r in item.get("ressonancias", [])])
        if rv.get("ok") and any(len(rs) == 0 or np.min(np.abs(rs - x)) > 0.01 * x for x in pm):
            n_malha_falta += 1
    print(f"Curvas reparadas com sucesso: {n_rep}")
    print(f"Curvas com tentativa incompleta (mantidas como estavam): {n_inc}")
    print(f"Reparadas com pico da malha sem ressonância: {n_malha_falta}")


def main():
    ap = argparse.ArgumentParser(description="Reparo do checkpoint velocity (z^4 + janela confiável)")
    ap.add_argument("--checkpoint", required=True, help="checkpoint velocity de entrada (.npz)")
    ap.add_argument("--gerador", required=True, help="core_shell_ressonancia_hibrida_cluster.py")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--tempo-max-curva", type=float, default=7200.0,
                    help="segundos por curva (padrão 2 h; típico 10-15 min)")
    ap.add_argument("--salvar-a-cada", type=int, default=20)
    ap.add_argument("--shard", type=int, default=0, help="índice desta parte (0..nshards-1)")
    ap.add_argument("--nshards", type=int, default=1, help="número total de partes")
    ap.add_argument("--chaves", default="", help="lista de chaves idd_ip separadas por vírgula")
    ap.add_argument("--juntar", action="store_true", help="junta as partes num checkpoint final")
    ap.add_argument("--refazer-incompletas", action="store_true",
                    help="tenta de novo as curvas que ficaram incompletas (use com --tempo-max-curva maior)")
    args = ap.parse_args()

    if not (0 <= args.shard < args.nshards):
        sys.exit("--shard precisa estar entre 0 e nshards-1")
    if args.juntar:
        juntar(args)
    else:
        rodar(args)


if __name__ == "__main__":
    main()
