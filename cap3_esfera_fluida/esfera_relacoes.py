# ============================================================
# ESFERA FLUIDA HOMOGÊNEA: MAPA DE FASE, RELAÇÃO DE EXISTÊNCIA E RELAÇÃO DE RECORRÊNCIA
# Um código só, uma pasta só. Lê a saída do gerador da esfera (gerador_esfera.py, versão 3) e não refaz física nenhuma.
#
# USO
#   no PC (Windows, Linux):  python esfera_relacoes.py <pasta_dos_dados>
#   no Colab, sem argumento: monta o Drive e lê MyDrive/resultados_esfera/esfera_fluida_lmax10
#   a pasta precisa ter resumo_*.csv; para a recorrência, também ressonancias_*.csv (ou checkpoint_*.npz).
#
# NOTAÇÃO (esfera A de raio a imersa no meio M, sem casca)
#   x_M = k_M a,  m = c_M/c_S,  rho = rho_S/rho_M,  m_t = Z_M/Z_S = m/rho,  faixa 1e-4 <= x_M <= 1,  l = 0 a 10
#   (nas figuras em inglês, do artigo, o índice da esfera é A; nas em português, da tese, é S)
#   variáveis das relações:  u = ln m,  v = ln rho
#   tamanho da esfera frente ao comprimento de onda interno:  x_S = k_S a = m x_M
#     esfera pequena: x_S < pi (abaixo do primeiro modo assintótico); esfera grande: x_S >= pi
#   ressonâncias: picos observados na energia interna total W_T (máximos com proeminência >= 0,1 em log10),
#   na malha base nas curvas normais (m <= 1) e na malha refinada nas curvas de risco (m > 1), isto é,
#   as linhas de ressonancias_malha_*.csv (colunas N_res_malha_base e N_res_malha_refinada do resumo)
#
# PARTE 0. O QUE TEM NOS DADOS (campos, curvas, ressonâncias, faixas)
#
# PARTE 1. MAPA DE FASE: número de picos de W_T até x_M = 1, em escala log, no plano (m = c_M/c_S, rho_S/rho_M),
#   com a estética do mapa Z_A x Z_B do artigo do core-shell (figuras_artigo.ipynb): mesma paleta, fonte, molduras,
#   barra de cores, caixas de rótulo e preenchimento dos bins. Caixinhas vermelhas: água em água e gotas de óleo em água.
#   Versões em inglês (artigo) e em português (tese).
#
# PARTE 2. RELAÇÃO DE EXISTÊNCIA (logística, forma do core-shell)
#   P_res = 1/(1 + exp(-g)),   g = c0 + c1 ln m + c2 ln rho + c3 (ln m)^2 + c4 ln m ln rho + c5 (ln rho)^2
#   classe ressonante: pelo menos um pico de W_T em 1e-4 <= x_M <= 1
#   treino/teste por curva inteira (20 % de teste, semente 42); a fronteira P_res = 0,5 (g = 0) vai para o mapa.
#
# PARTE 3. RELAÇÃO DE RECORRÊNCIA, nas MESMAS curvas de teste da Parte 2
#   A(u,v) = a0 + a1 u + a2 v + a3 u^2 + a4 u v + a5 v^2,   B(u,v) = b0 + b1 u + b2 v + b3 u^2 + b4 u v + b5 v^2
#   âncora da esfera (forma da tese):  x_{n+1} = x_ref A(u,v) + B(u,v) x_n,   x_ref = sqrt(3/(m m_t)) = sqrt(3 rho)/m (Minnaert)
#   âncora no espaçamento:             x_{n+1} = (pi/m) A(u,v) + B(u,v) x_n, pi/m = espaçamento assintótico de um l
#   (12 coeficientes em cada uma, mínimos quadrados)
#   3a. nos picos consecutivos de W_T (todas as famílias de l misturadas, como no core-shell);
#   3b. por família de l: picos consecutivos de W_T com o mesmo l (o l do polo de D_l associado a cada pico pelo
#       gerador, primary_l). Além das duas âncoras, o passo de McMahon (1 coeficiente S):
#         x_{l,n+1} = x_{l,n} + (S/m) [z_{l-1}(s+1) - z_{l-1}(s)],   s = nint(m x_{l,n}/pi - l/2 + 1/2)
#         z_{l-1}(s) = beta - (mu - 1)/(8 beta) - 4 (mu - 1)(7 mu - 31)/(3 (8 beta)^3),  beta = (s + l/2 - 1/2) pi,  mu = (2l - 1)^2
#       (zeros de j_{l-1} pela expansão de McMahon: fórmula fechada, nada de buscar raiz por configuração).
#   métricas no teste, em gráficos, separadas entre esfera pequena (x_S < pi) e grande (x_S >= pi):
#     R^2 e MAPE de x_{n+1} (passo a passo); erro relativo do passo x_{n+1} - x_n;
#     desvio da sequência gerada a partir do primeiro pico (por família: a partir do primeiro de cada l);
#     erro no número de ressonâncias até x_M = 1.
#
# PARTE 4. CUSTO: tempo do cálculo completo por configuração (runtime_s do gerador) contra o tempo das relações.
#
# SAÍDAS: <pasta>/relacoes_esfera_fluida/  (figuras em PNG, PDF e SVG, com sufixo _en e _pt;
#         tabelas de coeficientes em PNG, PDF, SVG, CSV e .tex; números em CSV; resumo em texto)
# ============================================================
import os
import sys
import glob
import time
import shutil
import subprocess
import importlib.util

PASTA_COLAB = "/content/drive/MyDrive/resultados_esfera/esfera_fluida_lmax10"
PASTA = next((a for a in sys.argv[1:] if os.path.isdir(a)), os.environ.get("ESFERA_DADOS", PASTA_COLAB))
if PASTA.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")

if importlib.util.find_spec("sklearn") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scikit-learn"])
if shutil.which("latex") is None and os.path.exists("/content"):
    print("Instalando LaTeX (só na primeira vez, alguns minutos)...")
    subprocess.run("apt-get update -qq && apt-get install -y -qq texlive texlive-latex-extra "
                   "texlive-fonts-recommended dvipng cm-super", shell=True, check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import numpy as np
import pandas as pd
import matplotlib
if not os.environ.get("DISPLAY") and os.name != "nt" and not os.path.exists("/content"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

USA_LATEX = shutil.which("latex") is not None and (shutil.which("dvipng") is not None or os.name == "nt")
plt.rcParams.update({"text.usetex": USA_LATEX, "font.family": "sans-serif"})
PCT = r"\%" if USA_LATEX else "%"

SAIDA = os.path.join(PASTA, "relacoes_esfera_fluida")
DIRS = {k: os.path.join(SAIDA, k) for k in ("mapa", "existencia", "recorrencia", "custo")}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

X_MIN, X_MAX = 1e-4, 1.0
TEST_FRACTION, RANDOM_SEED = 0.20, 42
LMAX = 10
IDIOMAS = ("en", "pt")
LOG = []


def diz(*partes):
    texto = " ".join(str(p) for p in partes)
    print(texto, flush=True)
    LOG.append(texto)


# ============================================================
# IDIOMA E FORMATAÇÃO (vírgula decimal na tese, ponto no artigo)
# ============================================================
IDIOMA = {"atual": "pt"}


def sep():
    return ("{,}" if USA_LATEX else ",") if IDIOMA["atual"] == "pt" else "."


def T(en, pt):
    return pt if IDIOMA["atual"] == "pt" else en


def fdec(value, decimals=None):
    value = float(value)
    texto = f"{value:g}" if decimals is None else f"{value:.{decimals}f}"
    return texto.replace(".", sep())


def latex_sci(value, sig=3):
    if not np.isfinite(value):
        return "--"
    if np.isclose(value, 0.0):
        return "0"
    e = int(np.floor(np.log10(abs(value))))
    return rf"${fdec(value / 10.0 ** e, sig - 1)}\times10^{{{e}}}$"


def fmt_pm(value, error):
    """Valor ± incerteza, incerteza com 2 algarismos significativos (formato das tabelas do artigo)."""
    if not np.isfinite(value):
        return "--"
    if not (np.isfinite(error) and error > 0.0):
        return f"${fdec(value, 4)}$"
    dec = max(0, -int(np.floor(np.log10(error))) + 1)
    return rf"${f'{value:+.{dec}f}'.replace('.', sep())}\pm{f'{error:.{dec}f}'.replace('.', sep())}$"


def salvar(fig, base):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{base}.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.03, facecolor="white", edgecolor="white")
    plt.close(fig)


# ============================================================
# LEITURA
# ============================================================
def achar(padrao, evita=None):
    arqs = sorted(a for a in glob.glob(os.path.join(PASTA, "**", padrao), recursive=True)
                  if not (evita and evita in os.path.basename(a)) and "relacoes_esfera_fluida" not in a)
    return arqs[0] if arqs else None


ARQ_RESUMO = achar("resumo_*.csv")
ARQ_RES = achar("ressonancias_malha_*.csv")
ARQ_CKPT = achar("checkpoint_*.npz")
if ARQ_RESUMO is None:
    raise FileNotFoundError(f"Nenhum resumo_*.csv em {PASTA}")

resumo = pd.read_csv(ARQ_RESUMO)
resumo = resumo.loc[resumo["status"] == "ok"].copy()
resumo["u"] = np.log(resumo["m"])                            # u = ln m,  m = c_M/c_S
resumo["v"] = np.log(resumo["rho_ratio"])                    # v = ln rho, rho = rho_S/rho_M
resumo["N_polos"] = resumo["N_peaks"]                        # contagem final do gerador (polos, nas curvas de risco)
resumo["N"] = np.where(resumo["curva_risco"].astype(bool), resumo["N_res_malha_refinada"],
                       resumo["N_res_malha_base"]).astype(int)   # picos observados em W_T
resumo["ressonante"] = (resumo["N"] > 0).astype(int)
resumo = resumo.sort_values("global").reset_index(drop=True)


def ler_ressonancias():
    """Uma linha por pico observado em W_T (ressonancias_malha): global, x_M e o l do polo associado (primary_l)."""
    if ARQ_RES is not None:
        cols = ["global", "xM_peak", "primary_l"]
        partes = [p for p in pd.read_csv(ARQ_RES, usecols=cols, chunksize=2_000_000,
                                         dtype={"global": np.int32, "xM_peak": float, "primary_l": float})]
        df = pd.concat(partes, ignore_index=True)
    elif ARQ_CKPT is not None:
        with np.load(ARQ_CKPT, allow_pickle=True) as data:
            curvas = data["curves"].item()
        g, x, l = [], [], []
        for chave, c in curvas.items():
            if c.get("status") != "ok":
                continue
            for r in c.get("ressonancias_malha", []):
                g.append(int(chave) + 1)
                x.append(float(r["xM_peak"]))
                l.append(np.nan if r.get("primary_l") is None else float(r["primary_l"]))
        df = pd.DataFrame({"global": g, "xM_peak": x, "primary_l": l})
    else:
        return None
    df = df.loc[np.isfinite(df["xM_peak"]) & (df["xM_peak"] >= X_MIN) & (df["xM_peak"] <= X_MAX)]
    return df.sort_values(["global", "xM_peak"]).reset_index(drop=True)


# ############################################################################
# PARTE 0: O QUE TEM NOS DADOS
# ############################################################################
diz("=" * 96 + "\nPARTE 0. DADOS\n" + "=" * 96)
diz(f"pasta: {PASTA}")
diz(f"resumo: {os.path.basename(ARQ_RESUMO)} | ressonâncias: {os.path.basename(ARQ_RES) if ARQ_RES else '--'} | "
    f"checkpoint: {os.path.basename(ARQ_CKPT) if ARQ_CKPT else '--'}")
diz("colunas do resumo:", ", ".join(pd.read_csv(ARQ_RESUMO, nrows=0).columns))
if ARQ_RES is not None:
    diz("colunas das ressonâncias:", ", ".join(pd.read_csv(ARQ_RES, nrows=0).columns))
elif ARQ_CKPT is not None:                     # o checkpoint só é aberto quando falta o CSV (ele é grande)
    with np.load(ARQ_CKPT, allow_pickle=True) as data:
        diz("chaves do checkpoint:", ", ".join(data.files))
        exemplo = next(iter(data["curves"].item().values()))
    diz("campos de cada curva:", ", ".join(sorted(exemplo)))
diz(f"curvas ok: {len(resumo)} ({resumo['m'].nunique()} valores de m x {resumo['rho_ratio'].nunique()} de rho_S/rho_M)")
diz(f"m = c_M/c_S de {resumo['m'].min():.3g} a {resumo['m'].max():.4g} | rho_S/rho_M de "
    f"{resumo['rho_ratio'].min():.3g} a {resumo['rho_ratio'].max():.4g}")
diz(f"picos observados em W_T: {int(resumo['N'].sum())} | curvas com pelo menos um: {int(resumo['ressonante'].sum())} "
    f"({100 * resumo['ressonante'].mean():.1f} %) | máximo por curva: {int(resumo['N'].max())}")
diz(f"(para comparação, polos contados pelo gerador: {int(resumo['N_polos'].sum())}, curvas com pelo menos um: "
    f"{int((resumo['N_polos'] > 0).sum())})")
for risco, sub in resumo.groupby("curva_risco"):
    diz(f"  {'risco (m > 1)' if risco else 'normal (m <= 1)'}: {len(sub)} curvas, {int(sub['N'].sum())} picos de W_T, "
        f"modo {', '.join(sub['modo_contagem'].unique())}, tempo médio {sub['runtime_s'].mean():.2f} s por curva")
largos = resumo.loc[(~resumo["curva_risco"].astype(bool)) & (resumo["N"] > resumo["N_res_polos_area"].fillna(0))]
diz(f"curvas normais com pico de W_T sem polo de Q >= 2 (pico largo, Q < 2): {len(largos)}; contam, porque o pico é observado em W_T")


# ############################################################################
# PARTE 2 (antes do mapa, porque a fronteira P_res = 0,5 vai para o mapa): EXISTÊNCIA
# ############################################################################
TERMOS_EXIST = ["1", "ln m", "ln rho", "(ln m)^2", "ln m ln rho", "(ln rho)^2"]
TEX_EXIST = [r"$1$", r"$\ln m$", r"$\ln\rho$", r"$(\ln m)^2$", r"$\ln m\,\ln\rho$", r"$(\ln\rho)^2$"]


def termos_existencia(u, v):
    u, v = np.asarray(u, float), np.asarray(v, float)
    return np.column_stack([np.ones_like(u), u, v, u * u, u * v, v * v])


def split_curvas(df, frac, seed):
    """Separação por curva inteira (mesma rotina do core-shell)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_test = max(1, int(round(len(idx) * frac)))
    return df.iloc[idx[n_test:]].copy(), df.iloc[idx[:n_test]].copy()


treino_c, teste_c = split_curvas(resumo, TEST_FRACTION, RANDOM_SEED)
GLOBAIS_TESTE = set(teste_c["global"])

modelo_exist = Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)),
                         ("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=5000))])
modelo_exist.fit(treino_c[["u", "v"]].to_numpy(float), treino_c["ressonante"].to_numpy(int))
esc, lr = modelo_exist.named_steps["scale"], modelo_exist.named_steps["clf"]
COEF_EXIST = np.r_[lr.intercept_[0] - np.sum(lr.coef_[0] * esc.mean_ / esc.scale_), lr.coef_[0] / esc.scale_]


def p_res(u, v, coef=None):
    g = termos_existencia(u, v) @ (COEF_EXIST if coef is None else coef)
    return 1.0 / (1.0 + np.exp(-np.clip(g, -700, 700)))


def erros_fisher(df, coef):
    X = termos_existencia(df["u"], df["v"])
    p = p_res(df["u"], df["v"], coef)
    return np.sqrt(np.clip(np.diag(np.linalg.pinv(X.T @ (X * (p * (1 - p))[:, None]))), 0, None))


ERRO_EXIST = erros_fisher(treino_c, COEF_EXIST)


def metricas_exist(df):
    y = df["ressonante"].to_numpy(int)
    p = p_res(df["u"], df["v"])
    pred = (p >= 0.5).astype(int)
    return {"N": len(y), "Positive_rate": float(np.mean(y)), "Accuracy": accuracy_score(y, pred),
            "Precision": precision_score(y, pred, zero_division=0), "Recall": recall_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0), "ROC_AUC": roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
            "FP": int(np.sum((pred == 1) & (y == 0))), "FN": int(np.sum((pred == 0) & (y == 1)))}


MET_EXIST = {"treino": metricas_exist(treino_c), "teste": metricas_exist(teste_c)}
diz("\n" + "=" * 96 + "\nPARTE 2. EXISTÊNCIA: P_res = 1/(1 + exp(-g)), g = c0 + c1 ln m + c2 ln rho + c3 (ln m)^2 + c4 ln m ln rho + c5 (ln rho)^2\n" + "=" * 96)
diz(f"treino {len(treino_c)} curvas | teste {len(teste_c)} curvas")
for i, nome in enumerate(TERMOS_EXIST):
    aviso = "" if abs(COEF_EXIST[i]) >= 2 * ERRO_EXIST[i] else "   <- não significativo (|c| < 2 sigma), fica na fórmula"
    diz(f"  c{i} ({nome:>3s}) = {COEF_EXIST[i]:+.6e} ± {ERRO_EXIST[i]:.2e}{aviso}")
for parte, m in MET_EXIST.items():
    diz(f"  {parte}: fração ressonante {m['Positive_rate']:.4f} | acurácia {m['Accuracy']:.4f} | precisão {m['Precision']:.4f} | "
        f"revocação {m['Recall']:.4f} | F1 {m['F1']:.4f} | ROC AUC {m['ROC_AUC']:.4f} | FP {m['FP']} | FN {m['FN']}")


# ============================================================
# TABELAS DE COEFICIENTES (mesma estética das tabelas do artigo e da tese)
# ============================================================
def save_compact_table(filename, table_data, row_labels, columns, figsize, first_col_width=0.16,
                       header_label="Coef.", fontsize=17):
    with plt.rc_context({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 30,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "savefig.edgecolor": "white"}):
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
        ax.axis("off")
        n_cols = len(columns)
        table = ax.table(cellText=[[row_labels[i], *table_data[i]] for i in range(len(row_labels))],
                         colLabels=[header_label, *columns], cellLoc="center",
                         colWidths=[first_col_width] + [(1.0 - first_col_width) / n_cols] * n_cols,
                         bbox=[0.02, 0.02, 0.96, 0.96])
        table.auto_set_font_size(False)
        table.set_fontsize(fontsize)
        for (row, col), cell in table.get_celld().items():
            cell.visible_edges = ""
            cell.set_edgecolor("black")
            cell.set_linewidth(0.0)
            cell.set_facecolor("white")
            cell.PAD = 0.01
            text = cell.get_text()
            text.set_color("black")
            text.set_va("center")
            text.set_linespacing(0.90)
            text.set_ha("left" if col == 0 else "center")
            text.set_fontsize(fontsize)
        for col in range(n_cols + 1):
            table[(0, col)].visible_edges = "TB"
            table[(0, col)].set_linewidth(1.5)
            table[(0, col)].get_text().set_fontsize(fontsize + 1)
            table[(len(row_labels), col)].visible_edges = "B"
            table[(len(row_labels), col)].set_linewidth(1.5)
        table[(0, 0)].get_text().set_fontweight("bold")
        fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(f"{filename}.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.01, facecolor="white", edgecolor="white")
        plt.close(fig)


def salvar_tabela_coeficientes(base, linhas, colunas, valores, erros, figsize, first_col_width=0.16, fontsize=17):
    """valores/erros: matrizes (linhas x colunas). PNG, PDF, SVG, CSV e .tex."""
    dados = [[fmt_pm(valores[i][j], erros[i][j]) for j in range(len(colunas))] for i in range(len(linhas))]
    save_compact_table(base, dados, linhas, colunas, figsize, first_col_width, fontsize=fontsize)
    pd.DataFrame([{"coeficiente": linha.replace("$", ""), "coluna": coluna.replace("\n", " "), "valor": valores[i][j],
                   "erro_padrao": erros[i][j]} for i, linha in enumerate(linhas) for j, coluna in enumerate(colunas)]
                 ).to_csv(base + ".csv", index=False)
    with open(base + ".tex", "w", encoding="utf-8") as fh:
        fh.write("\\begin{tabular}{l" + "c" * len(colunas) + "}\n\\hline\n")
        fh.write("Coef. & " + " & ".join(c.replace("\n", " ") for c in colunas) + " \\\\\n\\hline\n")
        for i, linha in enumerate(linhas):
            fh.write(linha + " & " + " & ".join(dados[i]) + " \\\\\n")
        fh.write("\\hline\n\\end{tabular}\n")


for IDIOMA["atual"] in IDIOMAS:
    salvar_tabela_coeficientes(os.path.join(DIRS["existencia"], f"tabela_coeficientes_existencia_{IDIOMA['atual']}"),
                               [f"$c_{{{i}}}$  " + TEX_EXIST[i] for i in range(6)], [T("Value", "Valor")],
                               [[c] for c in COEF_EXIST], [[e] for e in ERRO_EXIST], figsize=(4.6, 4.4), first_col_width=0.45)

with open(os.path.join(DIRS["existencia"], "formula_existencia.txt"), "w", encoding="utf-8") as fh:
    fh.write("P_res(m, rho) = 1/(1 + exp[-g]),  m = c_M/c_S,  rho = rho_S/rho_M\n"
             "g = c0 + c1 ln m + c2 ln rho + c3 (ln m)^2 + c4 ln m ln rho + c5 (ln rho)^2;  existe ressonância em 1e-4 <= x_M <= 1 quando g >= 0\n\n")
    for i, nome in enumerate(TERMOS_EXIST):
        nao = "" if abs(COEF_EXIST[i]) >= 2 * ERRO_EXIST[i] else "   (não significativo: |c| < 2 sigma)"
        fh.write(f"c{i} ({nome}) = {COEF_EXIST[i]:+.12e} +- {ERRO_EXIST[i]:.3e}{nao}\n")
    for parte, m in MET_EXIST.items():
        fh.write(f"\n{parte}: " + ", ".join(f"{k} = {v:.6g}" for k, v in m.items()))


# ############################################################################
# PARTE 1: MAPA DE FASE (estética do mapa Z_A x Z_B do figuras_artigo.ipynb)
# ############################################################################
# bins de 0,25 década nos dois eixos (células quadradas, como no mapa do core-shell)
LOG_X = (-3.0, 3.5)                                   # log10 m,   m = c_M/c_S de 1e-3 a 3000
LOG_Y = (-4.0, 3.5)                                   # log10 rho, rho = rho_S/rho_M de 1e-4 a 3000
LARGURA_BIN = 0.25
NX, NY = int(round((LOG_X[1] - LOG_X[0]) / LARGURA_BIN)), int(round((LOG_Y[1] - LOG_Y[0]) / LARGURA_BIN))
BOX_COLOR = "#600B00"
BOX_LINEWIDTH = CBAR_BOX_LINEWIDTH = 1.8
CELL_LINECOLOR, CELL_LINEWIDTH = "white", 0.42
CENTER_LINE_COLOR, CENTER_LINE_STYLE, CENTER_LINE_WIDTH, CENTER_LINE_ALPHA = "#AEB9C5", ":", 1.35, 0.35
FRONT_OBS = dict(color="#7FA6D9", linestyle=(0, (5.0, 2.2)), linewidth=2.0, alpha=0.95)
FRONT_LOG = dict(color="#600B00", linestyle=(0, (4.5, 2.5)), linewidth=1.9, alpha=0.92)
LABEL_FONTSIZE, TICK_FONTSIZE, CBAR_LABEL_FONTSIZE, CBAR_TICK_FONTSIZE = 23, 15, 20, 15
LEGEND_FONTSIZE, NRES_BOX_FONTSIZE = 20.0, 24
RESONANCE_CMAP = LinearSegmentedColormap.from_list("resonance_count", ["white", "#D9E1EA", "#526F92", "black"], N=256)
RESONANCE_CMAP.set_bad("#D9D9D9")

# materiais (20 a 25 °C), em relação à água (997 kg/m^3, 1497 m/s; Garofalo, arXiv:1703.08059):
#   azeite 910 kg/m^3, 1450 m/s (Garofalo); azeite extravirgem 1453 m/s (Yan et al., Food Res. Int. 125, 108552, 2019);
#   óleo de silicone 950 kg/m^3, 1100 m/s (Garofalo); petróleo 915 kg/m^3, 1434 m/s a 15 °C (Loranger et al., JASA 144, 2018);
#   compilação de óleos comestíveis a 20 °C: Coupland e McClements, JAOCS 74, 1559 (1997).
#   m = c_água/c_óleo e rho = rho_óleo/rho_água
AGUA = dict(m=(1.0, 1.0), rho=(1.0, 1.0))
OLEO = dict(m=(1497 / 1453, 1497 / 1100), rho=(910 / 997, 950 / 997))
LADO_MIN_CAIXA = 0.35                                    # lado mínimo de cada caixinha, em bins (só para ficar visível)
COR_CAIXA = "red"


def celula_x(valor):
    """m (escala log) -> coordenada da célula no eixo horizontal (centro da célula i em i)."""
    return (np.log10(valor) - LOG_X[0]) / LARGURA_BIN - 0.5


def celula_y(valor):
    """rho (escala log) -> coordenada da célula no eixo vertical."""
    return (np.log10(valor) - LOG_Y[0]) / LARGURA_BIN - 0.5


def mapa_binado(df):
    """Máximo de N em cada bin de (log m, log rho); bins vazios pelo vizinho mais próximo (cKDTree), como no core-shell."""
    ex = np.linspace(LOG_X[0], LOG_X[1], NX + 1)
    ey = np.linspace(LOG_Y[0], LOG_Y[1], NY + 1)
    lx, ly, n = np.log10(df["m"].to_numpy()), np.log10(df["rho_ratio"].to_numpy()), df["N"].to_numpy(float)
    ix = np.clip(np.searchsorted(ex, lx, side="right") - 1, 0, NX - 1)
    iy = np.clip(np.searchsorted(ey, ly, side="right") - 1, 0, NY - 1)
    contagem = np.zeros((NX, NY), int)
    maximo = np.full((NX, NY), -np.inf)
    np.add.at(contagem, (ix, iy), 1)
    np.maximum.at(maximo, (ix, iy), n)
    vazio = contagem == 0
    if vazio.any():
        ga, gb = np.meshgrid(0.5 * (ex[:-1] + ex[1:]), 0.5 * (ey[:-1] + ey[1:]), indexing="ij")
        _, viz = cKDTree(np.column_stack([lx, ly])).query(np.column_stack([ga[vazio], gb[vazio]]))
        maximo[vazio] = n[viz]
    return maximo, contagem, (ex, ey)


def segmentos_fronteira(mask):
    n_a, n_b = mask.shape
    seg = []
    for i in range(n_a):
        for j in range(n_b):
            if not mask[i, j]:
                continue
            xl, xr, yb, yt = i - 0.5, i + 0.5, j - 0.5, j + 0.5
            if i == 0 or not mask[i - 1, j]: seg.append(((xl, yb), (xl, yt)))
            if i == n_a - 1 or not mask[i + 1, j]: seg.append(((xr, yb), (xr, yt)))
            if j == 0 or not mask[i, j - 1]: seg.append(((xl, yb), (xr, yb)))
            if j == n_b - 1 or not mask[i, j + 1]: seg.append(((xl, yt), (xr, yt)))
    return seg


def estilo_eixos(ax, cbar=None):
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, width=1.25, length=4.5, direction="out")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(BOX_COLOR)
        spine.set_linewidth(BOX_LINEWIDTH)
    if cbar is not None:
        cbar.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE, width=1.25, length=4.5, direction="out")
        for spine in cbar.ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(BOX_COLOR)
            spine.set_linewidth(CBAR_BOX_LINEWIDTH)


def caixinha(ax, faixa_x, faixa_y, rotulo, lado_texto):
    """Caixinha vermelha em coordenadas de célula, com lado mínimo LADO_MIN_CAIXA."""
    x0, x1 = celula_x(faixa_x[0]), celula_x(faixa_x[1])
    y0, y1 = celula_y(faixa_y[0]), celula_y(faixa_y[1])
    if x1 - x0 < LADO_MIN_CAIXA:
        xc = 0.5 * (x0 + x1); x0, x1 = xc - LADO_MIN_CAIXA / 2, xc + LADO_MIN_CAIXA / 2
    if y1 - y0 < LADO_MIN_CAIXA:
        yc = 0.5 * (y0 + y1); y0, y1 = yc - LADO_MIN_CAIXA / 2, yc + LADO_MIN_CAIXA / 2
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=COR_CAIXA, linewidth=1.8, zorder=30))
    if lado_texto == "acima":
        ax.text(x0 - 0.12, y1, rotulo, color=COR_CAIXA, fontsize=15, ha="right", va="bottom", zorder=31,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, boxstyle="square,pad=0.05"))
    else:
        ax.text(x1 + 0.12, y0, rotulo, color=COR_CAIXA, fontsize=15, ha="left", va="top", zorder=31,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, boxstyle="square,pad=0.05"))


def IDX():
    """Índice da esfera: S na tese (português), A no artigo (inglês)."""
    return "S" if IDIOMA["atual"] == "pt" else "A"


def rotulo_m():
    return rf"$m=c_M/c_{IDX()}$"


def rotulo_rho():
    return rf"$\rho_{IDX()}/\rho_M$"


def plot_mapa_fase(df, base):
    maximo, contagem, (ex, ey) = mapa_binado(df)
    mapa = np.where(np.isfinite(maximo), np.maximum(maximo, 0.0), np.nan)
    n_max = int(np.ceil(np.nanmax(mapa)))
    norm = Normalize(vmin=0.0, vmax=np.log10(1.0 + n_max))                     # escala log: log10(1 + N)
    ticks = np.unique(np.array([0] + [10 ** k for k in range(0, 6) if 10 ** k < 0.8 * n_max] + [n_max], dtype=int))

    altura = 0.70
    largura = altura * NX / NY
    fig = plt.figure(figsize=(10.5, 10.5))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.12, 0.13, largura, altura])
    cax = fig.add_axes([0.12 + largura + 0.02, 0.13, 0.03, altura])
    for i in range(NX):
        for j in range(NY):
            val = mapa[i, j]
            cor = RESONANCE_CMAP(norm(np.log10(1.0 + val))) if np.isfinite(val) else RESONANCE_CMAP.get_bad()
            ax.add_patch(Rectangle((i - 0.5, j - 0.5), 1.0, 1.0, facecolor=cor, edgecolor=CELL_LINECOLOR,
                                   linewidth=CELL_LINEWIDTH, zorder=1))

    # linhas de referência: m = 1 (c_S = c_M), rho = 1 (rho_S = rho_M) e rho = m (Z_S = Z_M, a diagonal)
    guia = dict(color=CENTER_LINE_COLOR, linestyle=CENTER_LINE_STYLE, linewidth=CENTER_LINE_WIDTH * 1.8,
                alpha=CENTER_LINE_ALPHA, zorder=6)
    ax.axvline(celula_x(1.0), **guia)
    ax.axhline(celula_y(1.0), **guia)
    d = np.array([10 ** LOG_X[0], 10 ** LOG_X[1]])
    ax.plot(celula_x(d), celula_y(d), **guia)

    # fronteira observada (bins com N >= 1) e fronteira da logística (g = 0)
    for (a, b) in segmentos_fronteira(mapa > 0):
        ax.plot([a[0], b[0]], [a[1], b[1]], solid_capstyle="butt", zorder=17, **FRONT_OBS)
    GX, GY = np.meshgrid(np.linspace(*LOG_X, 600), np.linspace(*LOG_Y, 600), indexing="xy")
    G = (termos_existencia(GX.ravel() * np.log(10), GY.ravel() * np.log(10)) @ COEF_EXIST).reshape(GX.shape)
    ax.contour(celula_x(10 ** GX), celula_y(10 ** GY), G, levels=[0.0], colors=[FRONT_LOG["color"]],
               linewidths=FRONT_LOG["linewidth"], linestyles=[FRONT_LOG["linestyle"]], alpha=FRONT_LOG["alpha"], zorder=18)

    caixinha(ax, AGUA["m"], AGUA["rho"], T("water", "água"), "acima")
    caixinha(ax, OLEO["m"], OLEO["rho"], T("oil", "óleo"), "abaixo")

    fmt = lambda t: f"{t:.4f}".rstrip("0").rstrip(".").replace(".", sep()) if t < 1 else f"{t:g}"
    tx = 10.0 ** np.arange(np.ceil(LOG_X[0]), LOG_X[1] + 1e-9)
    ty = 10.0 ** np.arange(np.ceil(LOG_Y[0]), LOG_Y[1] + 1e-9)
    ax.set_xticks(celula_x(tx)); ax.set_xticklabels([fmt(t) for t in tx])
    ax.set_yticks(celula_y(ty)); ax.set_yticklabels([fmt(t) for t in ty])
    ax.set_xlabel(rotulo_m(), fontsize=LABEL_FONTSIZE, labelpad=9)
    ax.set_ylabel(rotulo_rho(), fontsize=LABEL_FONTSIZE, labelpad=10)
    ax.set_xlim(-0.5, NX - 0.5); ax.set_ylim(-0.5, NY - 0.5)
    ax.set_aspect("equal", adjustable="box")

    handles = [Line2D([0], [0], label=T("Observed frontier", "Fronteira observada"), **FRONT_OBS),
               Line2D([0], [0], label=T(r"Logistic frontier: $P_{\mathrm{res}}=0.5$",
                                        r"Fronteira logística: $P_{\mathrm{res}}=0" + sep() + "5$"), **FRONT_LOG)]
    leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.03, 0.97), bbox_transform=ax.transAxes,
                    fontsize=LEGEND_FONTSIZE, frameon=True, framealpha=1.0, facecolor="white", edgecolor="none",
                    fancybox=False, borderpad=0.20, handlelength=3.0, handletextpad=1.0, labelspacing=0.45,
                    borderaxespad=0.0)
    leg.set_zorder(100)
    ax.text(0.955, 0.965, r"$N_{\mathrm{res}}$", transform=ax.transAxes, fontsize=NRES_BOX_FONTSIZE, color="white",
            va="top", ha="right", bbox=dict(facecolor=BOX_COLOR, edgecolor="none", alpha=1.0, boxstyle="square,pad=0.14"),
            zorder=111)

    mapper = cm.ScalarMappable(norm=norm, cmap=RESONANCE_CMAP)
    mapper.set_array([])
    cbar = fig.colorbar(mapper, cax=cax, ticks=np.log10(1.0 + ticks))
    cbar.ax.set_yticklabels([str(t) for t in ticks])
    cbar.set_label(T("Resonances", "Ressonâncias"), fontsize=CBAR_LABEL_FONTSIZE, labelpad=18, rotation=270, color="black")
    estilo_eixos(ax, cbar)
    for ext in ("png", "pdf", "eps"):
        fig.savefig(f"{base}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    np.savez_compressed(base + "_dados.npz", edges_log10_m=ex, edges_log10_rho=ey, mapa=mapa, amostras_por_bin=contagem,
                        n_max=n_max, ticks=ticks, coef_existencia=COEF_EXIST)


diz("\n" + "=" * 96 + "\nPARTE 1. MAPA DE FASE\n" + "=" * 96)
for IDIOMA["atual"] in IDIOMAS:
    plot_mapa_fase(resumo, os.path.join(DIRS["mapa"], f"mapa_fase_esfera_{IDIOMA['atual']}"))
diz(f"mapa salvo em {DIRS['mapa']}")


# ============================================================
# GRÁFICOS (mesma estética das figuras de relações da tese)
# ============================================================
RC_FIG = {"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 16,
          "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "savefig.edgecolor": "white"}


def moldura(ax, labelsize=15):
    ax.grid(False)
    ax.tick_params(direction="in", length=5.0, width=1.15, labelsize=labelsize, pad=4, which="both")
    for spine in ax.spines.values():
        spine.set_linewidth(1.15)
        spine.set_edgecolor("black")


def legenda(ax_ou_fig, **kw):
    leg = ax_ou_fig.legend(fontsize=kw.pop("fontsize", 10.5), frameon=True, fancybox=False, framealpha=1.0,
                           handlelength=2.0, borderpad=0.35, labelspacing=0.28, handletextpad=0.45, borderaxespad=0.55, **kw)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(1.0)
    leg.get_frame().set_facecolor("white")
    return leg


def caixa_texto(ax, texto):
    ax.text(0.035, 0.965, texto, transform=ax.transAxes, ha="left", va="top", fontsize=10.5, linespacing=1.05,
            bbox=dict(boxstyle="square,pad=0.28", facecolor="white", edgecolor="black", linewidth=1.0), zorder=10)


def plot_existencia(base):
    with plt.rc_context(RC_FIG):
        y = teste_c["ressonante"].to_numpy(float)
        p = p_res(teste_c["u"], teste_c["v"])
        xj = y + np.random.default_rng(RANDOM_SEED).uniform(-0.045, 0.045, size=len(y))
        fig, ax = plt.subplots(figsize=(6.5, 5.0))
        ax.scatter(xj[y == 0], p[y == 0], s=28, facecolors="#D9D9D9", edgecolors="#8A8A8A", linewidths=0.45, alpha=0.75,
                   zorder=2, label=T("No resonance", "Sem ressonância"))
        ax.scatter(xj[y == 1], p[y == 1], s=28, facecolors="#0f223d", edgecolors="black", linewidths=0.45, alpha=0.85,
                   zorder=3, label=T("With resonance", "Com ressonância"))
        ax.plot([0.0, 1.0], [0.0, 1.0], "--", color="0.35", linewidth=1.15, zorder=0, label="1:1")
        ax.axhline(0.5, color="black", linestyle=":", linewidth=1.15, zorder=1,
                   label=T("Threshold", "Limiar") + r" $P_{\mathrm{res}}=0" + sep() + "5$")
        m = MET_EXIST["teste"]
        caixa_texto(ax, f"$N={m['N']}$\n"
                        f"{T('Resonant fraction', 'Fração ressonante')} = {fdec(m['Positive_rate'], 3)}\n"
                        f"{T('Accuracy', 'Acurácia')} = {fdec(m['Accuracy'], 3)}\n"
                        f"{T('Precision', 'Precisão')} = {fdec(m['Precision'], 3)}\n"
                        f"{T('Recall', 'Revocação')} = {fdec(m['Recall'], 3)}\n"
                        f"ROC AUC = {fdec(m['ROC_AUC'], 4)}")
        ax.set_xlim(-0.12, 1.12); ax.set_ylim(-0.02, 1.02)
        ax.set_xticks([0.0, 1.0]); ax.set_xticklabels(["0", "1"], fontsize=15)
        ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
        ax.set_yticklabels(["0"] + [f"0{sep()}25", f"0{sep()}50", f"0{sep()}75", f"1{sep()}00"], fontsize=15)
        ax.set_xlabel(T("Observed existence of resonance", "Existência observada de ressonância"), fontsize=18, labelpad=7)
        ax.set_ylabel(T("Predicted probability", "Probabilidade prevista") + r" $P_{\mathrm{res}}$", fontsize=18, labelpad=7)
        moldura(ax)
        legenda(ax, loc="lower right")
        fig.subplots_adjust(left=0.18, right=0.97, bottom=0.17, top=0.97)
        salvar(fig, base)


for IDIOMA["atual"] in IDIOMAS:
    plot_existencia(os.path.join(DIRS["existencia"], f"grafico_existencia_{IDIOMA['atual']}"))
diz(f"existência salva em {DIRS['existencia']}")


# ############################################################################
# PARTE 3: RECORRÊNCIA
# ############################################################################
diz("\n" + "=" * 96 + "\nPARTE 3. RECORRÊNCIA (mesmas curvas de teste da existência)\n" + "=" * 96)
RES = ler_ressonancias()
if RES is None:
    diz("Sem ressonancias_*.csv nem checkpoint_*.npz na pasta: a recorrência e o custo ficam de fora.")
    with open(os.path.join(SAIDA, "resumo_resultados.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG))
    sys.exit(0)

# só curvas com todas as posições (o número de linhas bate com o número de picos de W_T do resumo)
por_curva = RES.groupby("global").size()
N_por_global = resumo.set_index("global")["N"]
completas = por_curva.index[por_curva.to_numpy() == N_por_global.reindex(por_curva.index).to_numpy()]
RES = RES.loc[RES["global"].isin(completas)].reset_index(drop=True)
CURVAS = resumo.loc[(resumo["N"] == 0) | resumo["global"].isin(completas)].set_index("global")
diz(f"ressonâncias lidas: {len(RES)} em {len(completas)} curvas ({len(CURVAS)} curvas usadas, contando as sem ressonância)")

PARAM = CURVAS[["m", "rho_ratio", "u", "v"]]


def x_ref(m, rho):
    """x_ref = sqrt(3/(m m_t)), m_t = m/rho  ->  x_ref = sqrt(3 rho)/m (limite de Minnaert)."""
    return np.sqrt(3.0 * rho) / m


def termos_uv(u, v):
    """[1, u, v, u^2, u v, v^2]: base de A(u,v) e de B(u,v)."""
    return termos_existencia(u, v)


def zeros_mcmahon(l, s):
    """s-ésimo zero positivo de j_{l-1} (ordem nu = l - 1/2 de J) pela expansão de McMahon:
       z = beta - (mu - 1)/(8 beta) - 4 (mu - 1)(7 mu - 31)/(3 (8 beta)^3),  beta = (s + l/2 - 1/2) pi,  mu = (2l - 1)^2.
       Para l = 0, j_{-1}(x) = cos(x)/x e a expansão é exata: z = (s - 1/2) pi."""
    l, s = np.asarray(l, float), np.asarray(s, float)
    beta = (s + 0.5 * l - 0.5) * np.pi
    mu = (2.0 * l - 1.0) ** 2
    return beta - (mu - 1.0) / (8.0 * beta) - 4.0 * (mu - 1.0) * (7.0 * mu - 31.0) / (3.0 * (8.0 * beta) ** 3)


def passo_mcmahon(l, m, x_n):
    """Distância entre zeros vizinhos de j_{l-1}, dividida por m, no índice s estimado de x_n (fórmula fechada)."""
    s = np.maximum(1.0, np.rint(m * x_n / np.pi - 0.5 * l + 0.5))
    return (zeros_mcmahon(l, s + 1) - zeros_mcmahon(l, s)) / m


class Modelo:
    """Passo x_n -> x_{n+1}.
    "ref" : x_{n+1} = x_ref A(u,v) + B(u,v) x_n      (12 coef., forma da tese)
    "esp" : x_{n+1} = (pi/m) A(u,v) + B(u,v) x_n     (12 coef.)
    "mcm" : x_{n+1} = x_n + S passo_mcmahon(l, m, x_n) (1 coef., mediana; só por família)"""

    def __init__(self, tipo, coef=None, erro=None):
        self.tipo, self.coef, self.erro = tipo, coef, erro

    def ancora(self, m, rho):
        return x_ref(m, rho) if self.tipo == "ref" else np.pi / m

    def matriz(self, m, rho, u, v, x_n):
        Tuv = termos_uv(u, v)
        return np.column_stack([self.ancora(m, rho)[:, None] * Tuv, np.asarray(x_n)[:, None] * Tuv])

    def ajusta(self, tr):
        m, rho, u, v, xn, xn1 = (tr[k].to_numpy(float) for k in ("m", "rho_ratio", "u", "v", "x_n", "x_np1"))
        if self.tipo == "mcm":
            # S = mediana de (x_{n+1} - x_n)/passo_mcmahon (robusta, como o S por família do core-shell);
            # incerteza da mediana por bootstrap (300 reamostragens)
            q = (xn1 - xn) / passo_mcmahon(tr["l"].to_numpy(float), m, xn)
            q = q[np.isfinite(q)]
            rng = np.random.default_rng(RANDOM_SEED)
            boot = [np.median(rng.choice(q, len(q))) for _ in range(300)]
            self.coef, self.erro = np.array([np.median(q)]), np.array([np.std(boot)])
            return self
        X = self.matriz(m, rho, u, v, xn)
        self.coef, _, _, _ = np.linalg.lstsq(X, xn1, rcond=None)
        res = xn1 - X @ self.coef
        s2 = float(res @ res) / max(len(xn1) - X.shape[1], 1)
        self.erro = np.sqrt(np.clip(np.diag(s2 * np.linalg.pinv(X.T @ X)), 0.0, None))
        return self

    def proximo(self, m, rho, u, v, x_n, l=None):
        m, rho, u, v, x_n = (np.atleast_1d(np.asarray(a, float)) for a in (m, rho, u, v, x_n))
        if self.tipo == "mcm":
            return x_n + self.coef[0] * passo_mcmahon(np.atleast_1d(np.asarray(l, float)), m, x_n)
        return self.matriz(m, rho, u, v, x_n) @ self.coef


def transicoes(df, por_familia):
    """Pares consecutivos (x_n, x_{n+1}) de cada curva (W_T) ou de cada (curva, l) (família)."""
    if por_familia:
        df = df.loc[np.isfinite(df["primary_l"])].sort_values(["global", "primary_l", "xM_peak"])
    g, x = df["global"].to_numpy(), df["xM_peak"].to_numpy(float)
    l = df["primary_l"].to_numpy(float)
    mesmo = g[1:] == g[:-1]
    if por_familia:
        mesmo &= l[1:] == l[:-1]
    tr = pd.DataFrame({"global": g[:-1][mesmo], "l": l[:-1][mesmo], "x_n": x[:-1][mesmo], "x_np1": x[1:][mesmo]})
    tr = tr.join(PARAM, on="global")
    tr["x_S"] = tr["m"] * tr["x_n"]                     # k_S a no ponto de partida
    tr["teste"] = tr["global"].isin(GLOBAIS_TESTE)
    return tr


TR = {"wt": transicoes(RES, False), "fam": transicoes(RES, True)}
for k, tr in TR.items():
    diz(f"transições {'de W_T' if k == 'wt' else 'por família de l'}: {len(tr)} (treino {int((~tr['teste']).sum())}, "
        f"teste {int(tr['teste'].sum())})")

MODELOS = {
    ("wt", "ref"): Modelo("ref").ajusta(TR["wt"].loc[~TR["wt"]["teste"]]),
    ("wt", "esp"): Modelo("esp").ajusta(TR["wt"].loc[~TR["wt"]["teste"]]),
    ("fam", "ref"): Modelo("ref").ajusta(TR["fam"].loc[~TR["fam"]["teste"]]),
    ("fam", "esp"): Modelo("esp").ajusta(TR["fam"].loc[~TR["fam"]["teste"]]),
    ("fam", "mcm"): Modelo("mcm").ajusta(TR["fam"].loc[~TR["fam"]["teste"]]),
}
NOMES_COEF = [f"a{i}" for i in range(6)] + [f"b{i}" for i in range(6)]
for (alvo, tipo), mod in MODELOS.items():
    diz(f"\n{alvo} / {tipo}:")
    nomes = ["S"] if tipo == "mcm" else NOMES_COEF
    for nome, c, e in zip(nomes, mod.coef, mod.erro):
        diz(f"  {nome:>3s} = {c:+.6e} ± {e:.2e}" + ("" if abs(c) >= 2 * e else "   <- |c| < 2 sigma"))


def rotulo_modelo(chave, curto=False):
    alvo, tipo = chave
    anc = {"ref": r"$x_{\mathrm{ref}}$", "esp": r"$\pi/m$", "mcm": "McMahon"}[tipo]
    if curto:
        return anc
    return (T("$W_T$ peaks", "Picos de $W_T$") if alvo == "wt" else T("Family $l$", "Família $l$")) + ", " + \
        (anc if tipo == "mcm" else T("anchor ", "âncora ") + anc)


ESTILO = {("wt", "ref"): dict(color="#0f223d", linestyle="-"), ("wt", "esp"): dict(color="#810000", linestyle="--"),
          ("fam", "ref"): dict(color="#526F92", linestyle="-."), ("fam", "esp"): dict(color="black", linestyle=":"),
          ("fam", "mcm"): dict(color="#8A8A8A", linestyle=(0, (5, 1.5, 1, 1.5)))}


# ============================================================
# MÉTRICAS NO TESTE
# ============================================================
def r2_mape(obs, pred):
    ok = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[ok], pred[ok]
    if len(obs) < 2:
        return np.nan, np.nan
    ss = np.sum((obs - obs.mean()) ** 2)
    return 1.0 - np.sum((obs - pred) ** 2) / ss if ss > 0 else np.nan, 100.0 * np.mean(np.abs((obs - pred) / obs))


def sequencias(mod, inicio):
    """inicio: DataFrame com global, l, x0 e N_lim. Gera x_{k+1} = f(x_k) a partir de x0 até passar de X_MAX
    (vetorizado em todas as sequências de uma vez; N_lim = 4 N_obs + 50 só evita laço infinito)."""
    m, rho, u, v = (inicio[k].to_numpy(float) for k in ("m", "rho_ratio", "u", "v"))
    l = inicio["l"].to_numpy(float)
    lim = inicio["N_lim"].to_numpy(int)
    atual = inicio["x0"].to_numpy(float).copy()
    tam = np.ones(len(atual), int)
    blocos_i, blocos_x = [np.arange(len(atual))], [atual.copy()]
    ativo = np.ones(len(atual), bool)
    while ativo.any():
        idx = np.flatnonzero(ativo)
        with np.errstate(all="ignore"):
            nxt = mod.proximo(m[idx], rho[idx], u[idx], v[idx], atual[idx], l[idx])
        ok = np.isfinite(nxt) & (nxt > atual[idx]) & (nxt <= X_MAX)
        blocos_i.append(idx[ok]); blocos_x.append(nxt[ok])
        atual[idx[ok]] = nxt[ok]
        tam[idx[ok]] += 1
        ativo[idx[~ok]] = False
        ativo &= tam < lim
    I, X = np.concatenate(blocos_i), np.concatenate(blocos_x)
    ordem = np.lexsort((X, I))
    return np.split(X[ordem], np.cumsum(np.bincount(I, minlength=len(atual)))[:-1])


TESTE_RES = RES.loc[RES["global"].isin(GLOBAIS_TESTE)]


def avaliar(chave):
    alvo, tipo = chave
    mod = MODELOS[chave]
    te = TR[alvo].loc[TR[alvo]["teste"]]
    pred = mod.proximo(te["m"], te["rho_ratio"], te["u"], te["v"], te["x_n"], te["l"])
    obs, xn = te["x_np1"].to_numpy(float), te["x_n"].to_numpy(float)
    passo_obs = obs - xn
    erro_passo = np.where(passo_obs > 0, np.abs((pred - xn) - passo_obs) / np.where(passo_obs > 0, passo_obs, 1), np.nan)
    pequena = (te["x_S"] < np.pi).to_numpy()
    out = {"m_trans": te["m"].to_numpy(float), "x_S": te["x_S"].to_numpy(float), "erro_passo": erro_passo}
    for grupo, mk in (("pequena", pequena), ("grande", ~pequena), ("todas", np.ones_like(pequena))):
        out[f"R2_{grupo}"], out[f"MAPE_{grupo}"] = r2_mape(obs[mk], pred[mk])
        e = erro_passo[mk][np.isfinite(erro_passo[mk])]
        out[f"passo_med_{grupo}"] = 100 * np.median(e) if len(e) else np.nan
        out[f"passo_p90_{grupo}"] = 100 * np.percentile(e, 90) if len(e) else np.nan
        out[f"n_{grupo}"] = int(mk.sum())

    # sequência a partir do primeiro pico (W_T) ou do primeiro de cada família, e contagem até x_M = 1
    df = TESTE_RES if alvo == "wt" else TESTE_RES.loc[np.isfinite(TESTE_RES["primary_l"])]
    chaves = ["global"] if alvo == "wt" else ["global", "primary_l"]
    grupos = df.groupby(chaves, sort=True)["xM_peak"]
    obs_seq = [np.sort(g.to_numpy(float)) for _, g in grupos]
    ini = grupos.first().reset_index().rename(columns={"xM_peak": "x0", "primary_l": "l"})
    if "l" not in ini:
        ini["l"] = np.nan
    ini = ini.join(PARAM, on="global")
    ini["N_lim"] = [4 * len(o) + 50 for o in obs_seq]
    geradas = sequencias(mod, ini)
    desv, xa_desv, n_pred = [], [], {}
    for g, o, s in zip(ini["global"], obs_seq, geradas):
        n = min(len(o), len(s))
        if n >= 2:
            desv.append(np.abs(s[1:n] - o[1:n]) / o[1:n])
            xa_desv.append(PARAM.at[g, "m"] * o[1:n])
        n_pred[g] = n_pred.get(g, 0) + len(s)
    desv = np.concatenate(desv) if desv else np.array([])
    xa_desv = np.concatenate(xa_desv) if xa_desv else np.array([])
    for grupo, mk in (("pequena", xa_desv < np.pi), ("grande", xa_desv >= np.pi), ("todas", np.ones(len(desv), bool))):
        out[f"seq_med_{grupo}"] = 100 * np.median(desv[mk]) if mk.any() else np.nan
    cont = pd.DataFrame({"global": list(n_pred), "N_pred": list(n_pred.values())}).join(CURVAS[["N", "m"]], on="global")
    cont["erro_N"] = 100 * np.abs(cont["N_pred"] - cont["N"]) / cont["N"]
    out["contagem"] = cont
    for grupo, mk in (("pequena", cont["m"] < np.pi), ("grande", cont["m"] >= np.pi), ("todas", cont["m"] > 0)):
        out[f"N_erro_{grupo}"] = float(cont.loc[mk, "erro_N"].median()) if mk.any() else np.nan
        out[f"N_exato_{grupo}"] = 100 * float((cont.loc[mk, "N_pred"] == cont.loc[mk, "N"]).mean()) if mk.any() else np.nan
    out["teste_df"] = pd.DataFrame({"x_n": xn, "obs": obs, "pred": pred})
    return out


AVAL = {}
for chave in MODELOS:
    t0 = time.time()
    AVAL[chave] = avaliar(chave)
    a = AVAL[chave]
    diz(f"\n{chave[0]} / {chave[1]} (teste, {time.time() - t0:.1f} s):")
    for grupo in ("pequena", "grande", "todas"):
        diz(f"  esfera {grupo:>7s}: n = {a[f'n_{grupo}']:>8d} | R2 = {a[f'R2_{grupo}']:.5f} | MAPE = {a[f'MAPE_{grupo}']:.3f} % | "
            f"erro do passo med = {a[f'passo_med_{grupo}']:.2f} %, p90 = {a[f'passo_p90_{grupo}']:.2f} % | "
            f"sequência med = {a[f'seq_med_{grupo}']:.3f} % | erro em N med = {a[f'N_erro_{grupo}']:.2f} % | "
            f"N exato = {a[f'N_exato_{grupo}']:.1f} %")

pd.DataFrame([{"alvo": k[0], "ancora": k[1], **{c: v for c, v in a.items() if np.isscalar(v)}} for k, a in AVAL.items()]
             ).to_csv(os.path.join(DIRS["recorrencia"], "metricas_recorrencia_teste.csv"), index=False)


# ============================================================
# TABELAS DE COEFICIENTES DA RECORRÊNCIA (formato da Tabela 3 do artigo)
# ============================================================
for IDIOMA["atual"] in IDIOMAS:
    for alvo, nome_arq in (("wt", "picos_WT"), ("fam", "familias_l")):
        colunas, valores, erros = [], [[] for _ in range(6)], [[] for _ in range(6)]
        for tipo in ("ref", "esp"):
            mod = MODELOS[(alvo, tipo)]
            for parte, desl in (("A", 0), ("B", 6)):
                colunas.append(T("Anchor ", "Âncora ") + rotulo_modelo((alvo, tipo), curto=True) + f"\n${parte}$")
                for i in range(6):
                    valores[i].append(mod.coef[desl + i])
                    erros[i].append(mod.erro[desl + i])
        salvar_tabela_coeficientes(os.path.join(DIRS["recorrencia"], f"tabela_coeficientes_recorrencia_{nome_arq}_{IDIOMA['atual']}"),
                                   [f"${i}$" for i in range(6)], colunas, valores, erros, figsize=(12.5, 4.8),
                                   first_col_width=0.08, fontsize=16)
    mcm = MODELOS[("fam", "mcm")]
    salvar_tabela_coeficientes(os.path.join(DIRS["recorrencia"], f"tabela_coeficiente_mcmahon_{IDIOMA['atual']}"),
                               [r"$S$"], [T("Value", "Valor")], [[mcm.coef[0]]], [[mcm.erro[0]]], figsize=(4.0, 1.6),
                               first_col_width=0.35)


# tabela combinada no formato da tese (Termo | a_i | b_i | c_i), recorrência na forma da tese (âncora x_ref, picos de W_T)
for IDIOMA["atual"] in IDIOMAS:
    mod = MODELOS[("wt", "ref")]
    with open(os.path.join(DIRS["recorrencia"], f"tabela_tese_recorrencia_existencia_{IDIOMA['atual']}.tex"), "w", encoding="utf-8") as fh:
        fh.write("\\begin{tabular}{lccc}\n\\toprule\n")
        fh.write(f"& \\multicolumn{{2}}{{c}}{{\\textbf{{{T('Recurrence', 'Recorrência')}}}}} & "
                 f"\\textbf{{{T('Existence', 'Existência')}}} \\\\\n\\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-4}}\n")
        fh.write(f"\\textbf{{{T('Term', 'Termo')}}} & $a_i$ & $b_i$ & $c_i$ \\\\\n\\midrule\n")
        for i in range(6):
            fh.write(f"{TEX_EXIST[i]} & {fmt_pm(mod.coef[i], mod.erro[i])} & {fmt_pm(mod.coef[6 + i], mod.erro[6 + i])} & "
                     f"{fmt_pm(COEF_EXIST[i], ERRO_EXIST[i])} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")


# ============================================================
# GRÁFICOS DA RECORRÊNCIA
# ============================================================
def plot_dispersao(chave, base):
    """Observado e previsto contra x_M(n), no teste (mesmo gráfico das recorrências da tese)."""
    a = AVAL[chave]
    df = a["teste_df"]
    with plt.rc_context(RC_FIG):
        fig, ax = plt.subplots(figsize=(6.5, 5.0))
        ax.scatter(df["x_n"], df["obs"], s=36, facecolors="#D9D9D9", edgecolors="#8A8A8A", linewidths=0.45, alpha=0.75,
                   zorder=1, label=T("Observed value", "Valor observado"), rasterized=True)
        ax.scatter(df["x_n"], df["pred"], s=28, facecolors="#0f223d", edgecolors="black", linewidths=0.45, alpha=0.9,
                   zorder=3, label=T("Predicted value", "Valor previsto"), rasterized=True)
        lims = [min(df["x_n"].min(), df["obs"].min()), max(df["x_n"].max(), df["obs"].max())]
        ax.plot(lims, lims, "--", color="0.35", linewidth=1.15, zorder=0, label="1:1")
        ax.set_xscale("log"); ax.set_yscale("log")                # posições de 1e-4 a 1: escala log
        caixa_texto(ax, rotulo_modelo(chave) + f"\n$N={a['n_todas']}$\n$R^2={fdec(a['R2_todas'], 4)}$\n"
                        f"MAPE = {fdec(a['MAPE_todas'], 2)}{PCT}\n"
                        f"{T('Step error (median)', 'Erro do passo (mediana)')} = {fdec(float(f"{a['passo_med_todas']:.2g}"))}{PCT}")
        ax.set_xlabel(r"$x_M(n)$", fontsize=18, labelpad=7)
        ax.set_ylabel(r"$x_M(n+1)$", fontsize=18, labelpad=7)
        moldura(ax)
        legenda(ax, loc="lower right")
        fig.subplots_adjust(left=0.18, right=0.97, bottom=0.17, top=0.97)
        salvar(fig, base)


def plot_cdf_passo(base):
    """Distribuição acumulada do erro relativo do passo x_{n+1} - x_n: esfera pequena (x_S < pi) e grande (x_S >= pi)."""
    with plt.rc_context(RC_FIG | {"font.size": 15}):
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
        for ax, grupo in zip(axes, ("pequena", "grande")):
            for chave, est in ESTILO.items():
                a = AVAL[chave]
                mk = a["x_S"] < np.pi if grupo == "pequena" else a["x_S"] >= np.pi
                e = np.sort(a["erro_passo"][mk]) * 100
                e = e[np.isfinite(e) & (e > 0)]
                if len(e):
                    ax.plot(e, np.arange(1, len(e) + 1) / len(e), linewidth=2.0, label=rotulo_modelo(chave), **est)
            ax.set_xscale("log")
            ax.set_ylim(0.0, 1.02)
            ax.set_xlabel(T("Relative step error", "Erro relativo do passo") + f" ({PCT})", fontsize=16)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: fdec(val, 2)))
            moldura(ax, 13)
            ax.set_title(T("Small sphere", "Esfera pequena") + rf" ($x_{IDX()}<\pi$)" if grupo == "pequena"
                         else T("Large sphere", "Esfera grande") + rf" ($x_{IDX()}\geq\pi$)", fontsize=15, pad=8)
        axes[0].set_ylabel(T("Fraction of test transitions", "Fração das transições de teste"), fontsize=16)
        fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))
        h, lab = axes[0].get_legend_handles_labels()
        legenda(fig, handles=h, labels=lab, loc="lower center", ncol=3, fontsize=12, bbox_to_anchor=(0.5, 0.0))
        salvar(fig, base)


def plot_barras(base):
    """Métricas de teste lado a lado, esfera pequena x grande, uma barra por forma."""
    metricas = [("R2", r"$R^2$ " + T("of", "de") + r" $x_{n+1}$", False),
                ("MAPE", T("MAPE of", "MAPE de") + rf" $x_{{n+1}}$ ({PCT})", True),
                ("passo_med", T("Median step error", "Erro mediano do passo") + f" ({PCT})", True),
                ("seq_med", T("Median sequence deviation", "Desvio mediano da sequência") + f" ({PCT})", True),
                ("N_erro", T("Median error in $N$", "Erro mediano em $N$") + f" ({PCT})", True)]
    cores = {k: v["color"] for k, v in ESTILO.items()}
    chaves = list(MODELOS)
    with plt.rc_context(RC_FIG | {"font.size": 14}):
        fig, axes = plt.subplots(2, 3, figsize=(15.0, 9.2))
        axes = axes.ravel()
        largura = 0.8 / len(chaves)
        for ax, (nome, rot, log) in zip(axes[:len(metricas)], metricas):
            for k, chave in enumerate(chaves):
                vals = np.array([AVAL[chave][f"{nome}_{g}"] for g in ("pequena", "grande")], float)
                pos = np.arange(2) + (k - (len(chaves) - 1) / 2) * largura
                ax.bar(pos, np.nan_to_num(vals, nan=0.0), width=largura, color=cores[chave], edgecolor="black",
                       linewidth=0.8, label=rotulo_modelo(chave), zorder=3, hatch="//" if chave[0] == "fam" else None)
                for p, val in zip(pos, vals):           # valor nulo ou sem dado (nenhuma transição no grupo) fica escrito
                    if not np.isfinite(val) or (log and val == 0):
                        ax.text(p, 0.0, "--" if not np.isfinite(val) else "0", ha="center", va="bottom", fontsize=11, zorder=4)
            ax.set_xticks([0, 1])
            ax.set_xticklabels([T("Small", "Pequena") + "\n" + rf"$x_{IDX()}<\pi$", T("Large", "Grande") + "\n" + rf"$x_{IDX()}\geq\pi$"], fontsize=13)
            if log:
                ax.set_yscale("symlog", linthresh=1e-2)
                ax.set_ylim(bottom=0.0)
            else:
                ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: fdec(val, 2)))
            ax.set_title(rot, fontsize=14, pad=8)
            moldura(ax, 12)
        h, lab = axes[0].get_legend_handles_labels()
        axes[-1].axis("off")
        legenda(axes[-1], handles=h, labels=lab, loc="center", fontsize=14)
        fig.tight_layout()
        salvar(fig, base)


def plot_contra_m(base):
    """Onde cada forma falha: erro mediano do passo e erro médio em N contra m = c_M/c_S (uma curva por forma)."""
    with plt.rc_context(RC_FIG | {"font.size": 15}):
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
        for chave, est in ESTILO.items():
            a = AVAL[chave]
            df = pd.DataFrame({"m": a["m_trans"], "e": 100 * a["erro_passo"]}).dropna()
            med = df.groupby("m")["e"].median()
            axes[0].plot(med.index, med.to_numpy(), marker="o", markersize=4, linewidth=1.8, label=rotulo_modelo(chave), **est)
            c = a["contagem"].groupby("m")["erro_N"].mean()
            axes[1].plot(c.index, c.to_numpy(), marker="o", markersize=4, linewidth=1.8, **est)
        axes[0].set_yscale("log")
        axes[1].set_yscale("symlog", linthresh=1e-1)
        axes[1].set_ylim(bottom=0.0)
        for ax in axes:
            ax.set_xscale("log")
            ax.axvline(np.pi, color="0.35", linestyle=":", linewidth=1.15)
            ax.set_xlabel(rotulo_m(), fontsize=16)
            moldura(ax, 13)
        axes[0].set_ylabel(T("Median step error", "Erro mediano do passo") + f" ({PCT})", fontsize=16)
        axes[1].set_ylabel(T("Mean error in $N$", "Erro médio em $N$") + f" ({PCT})", fontsize=16)
        fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))
        h, lab = axes[0].get_legend_handles_labels()
        legenda(fig, handles=h, labels=lab, loc="lower center", ncol=3, fontsize=12, bbox_to_anchor=(0.5, 0.0))
        salvar(fig, base)


for IDIOMA["atual"] in IDIOMAS:
    sufixo = IDIOMA["atual"]
    for chave in MODELOS:
        plot_dispersao(chave, os.path.join(DIRS["recorrencia"], f"grafico_recorrencia_{chave[0]}_{chave[1]}_{sufixo}"))
    plot_cdf_passo(os.path.join(DIRS["recorrencia"], f"grafico_erro_do_passo_{sufixo}"))
    plot_barras(os.path.join(DIRS["recorrencia"], f"grafico_metricas_pequena_grande_{sufixo}"))
    plot_contra_m(os.path.join(DIRS["recorrencia"], f"grafico_erro_contra_m_{sufixo}"))
diz(f"recorrência salva em {DIRS['recorrencia']}")


# ############################################################################
# PARTE 4: CUSTO COMPUTACIONAL
#   cálculo completo: runtime_s que o gerador salvou em cada curva (máquina onde o dataset rodou);
#   relações, medidas aqui, por configuração: P_res e a sequência inteira até x_M = 1 a partir do 1o pico
#   (laço escalar, uma configuração por vez, mediana de N_REPETICOES).
# ############################################################################
diz("\n" + "=" * 96 + "\nPARTE 4. CUSTO\n" + "=" * 96)
N_REPETICOES = 5
t_exist = []
for _ in range(N_REPETICOES):
    t0 = time.perf_counter()
    p_res(resumo["u"].to_numpy(), resumo["v"].to_numpy())
    t_exist.append((time.perf_counter() - t0) / len(resumo))
t_exist = float(np.median(t_exist))


def sequencia_escalar(mod, m, rho, x0):
    """Uma configuração por vez: x_{n+1} = anc A + B x_n até passar de x_M = 1 (A e B calculados uma vez só)."""
    Tuv = termos_uv(np.log(m), np.log(rho))[0]
    A, B = float(Tuv @ mod.coef[:6]), float(Tuv @ mod.coef[6:])
    anc = float(mod.ancora(np.array([m]), np.array([rho]))[0])
    seq, x = [x0], x0
    while True:
        x1 = anc * A + B * x
        if not (x1 > x and x1 <= X_MAX) or len(seq) > 50000:
            return seq
        seq.append(x1)
        x = x1


primeiros = TESTE_RES.groupby("global")["xM_peak"].first()
amostra = CURVAS.loc[primeiros.index].copy()
amostra["x0"] = primeiros
custo = []
for g, c in amostra.iterrows():
    linha = {"global": g, "m": c["m"], "rho_ratio": c["rho_ratio"], "N_obs": c["N"], "runtime_completo_s": c["runtime_s"]}
    for tipo in ("ref", "esp"):
        tempos = []
        for _ in range(N_REPETICOES):
            t0 = time.perf_counter()
            sequencia_escalar(MODELOS[("wt", tipo)], c["m"], c["rho_ratio"], c["x0"])
            tempos.append(time.perf_counter() - t0)
        linha[f"recorrencia_{tipo}_s"] = float(np.median(tempos)) + t_exist
    custo.append(linha)
custo = pd.DataFrame(custo)
custo.to_csv(os.path.join(DIRS["custo"], "custo_por_configuracao.csv"), index=False)
diz(f"cálculo completo (gerador): mediana {resumo['runtime_s'].median():.3g} s, média {resumo['runtime_s'].mean():.3g} s por configuração")
diz(f"existência: {t_exist:.3g} s por configuração")
for tipo in ("ref", "esp"):
    r = custo[f"recorrencia_{tipo}_s"]
    diz(f"existência + recorrência (âncora {tipo}): mediana {r.median():.3g} s, máximo {r.max():.3g} s por configuração | "
        f"ganho mediano {np.median(custo['runtime_completo_s'] / r):.3g}x")


def plot_custo(base):
    with plt.rc_context(RC_FIG | {"font.size": 15}):
        fig, ax = plt.subplots(figsize=(6.5, 5.0))
        ax.scatter(resumo["m"], resumo["runtime_s"], s=22, facecolors="#D9D9D9", edgecolors="#8A8A8A", linewidths=0.45,
                   alpha=0.75, zorder=2, label=T("Full calculation (generator)", "Cálculo completo (gerador)"))
        for tipo, cor, mk in (("ref", "#0f223d", "o"), ("esp", "#810000", "s")):
            ax.scatter(custo["m"], custo[f"recorrencia_{tipo}_s"], s=22, marker=mk, facecolors=cor, edgecolors="black",
                       linewidths=0.45, alpha=0.85, zorder=3,
                       label=T("Relations, anchor ", "Relações, âncora ") + rotulo_modelo(("wt", tipo), curto=True))
        ax.axhline(t_exist, color="black", linestyle=":", linewidth=1.15, label=T("Existence only", "Só a existência"))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(rotulo_m(), fontsize=18, labelpad=7)
        ax.set_ylabel(T("Time per configuration (s)", "Tempo por configuração (s)"), fontsize=18, labelpad=7)
        moldura(ax)
        legenda(ax, loc="center left")
        fig.subplots_adjust(left=0.18, right=0.97, bottom=0.17, top=0.97)
        salvar(fig, base)


for IDIOMA["atual"] in IDIOMAS:
    plot_custo(os.path.join(DIRS["custo"], f"grafico_custo_{IDIOMA['atual']}"))

with open(os.path.join(SAIDA, "resumo_resultados.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(LOG))
diz("\n" + "=" * 96 + f"\nTUDO SALVO EM: {SAIDA}\n" + "=" * 96)
