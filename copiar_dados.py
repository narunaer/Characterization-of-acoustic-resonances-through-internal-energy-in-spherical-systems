#!/usr/bin/env python3
"""Copia os bancos de dados da tese (grandes demais para o git) para a pasta dados/ deste repositório.

Uso (na raiz do repositório):
    python copiar_dados.py --origem "C:\\Users\\Narun\\Downloads" --dry-run
    python copiar_dados.py --origem "C:\\Users\\Narun\\Downloads"
    python copiar_dados.py --origem "C:\\Users\\Narun\\Downloads" --grupos cap3 cap5 --sha256

Só usa a biblioteca padrão. Arquivos que já existem no destino com o mesmo tamanho são pulados.
Nada é apagado nem movido na origem.
"""
import argparse
import fnmatch
import hashlib
import os
import shutil
import sys
from pathlib import Path

# grupo: lista de (origem relativa, destino relativo a dados/, padrões a excluir)
GRUPOS = {
    "cap3": [("sphere_final", "cap3_esfera_fluida", [])],
    "cap4": [
        ("coreshell_final/resultados_mix", "cap4_coreshell_fluido/resultados_mix", []),
        ("checkpoint_ressonancias_coreshell_lmax10_density_final_reparado_z4.npz",
         "cap4_coreshell_fluido/checkpoints/checkpoint_ressonancias_coreshell_lmax10_density_final_reparado_z4.npz", []),
        ("checkpoint_ressonancias_coreshell_lmax10_mixed_final_reparado_z4.npz",
         "cap4_coreshell_fluido/checkpoints/checkpoint_ressonancias_coreshell_lmax10_mixed_final_reparado_z4.npz", []),
    ],
    "cap4_reparo": [("reparo_velocity", "cap4_coreshell_fluido/reparo_velocity", ["__pycache__", "*.py", "*.pyc"])],
    "cap4_relacoes": [("relacoes_existencia_recorrencia_final",
                       "cap4_coreshell_fluido/relacoes_existencia_recorrencia_final", ["*.svg"])],
    "cap5": [
        ("visco_final/viscoelastico_lmax10_elastico", "cap5_viscoelastico/viscoelastico_lmax10_elastico", []),
        ("visco_final/viscoelastico_lmax10_maxwell", "cap5_viscoelastico/viscoelastico_lmax10_maxwell", []),
        ("visco_final/viscoelastico_lmax10_voigt", "cap5_viscoelastico/viscoelastico_lmax10_voigt", []),
    ],
    "extras": [(f, "extras/" + f, []) for f in (
        "dataset8.csv", "dataset8.npz", "dataset_maxwell.csv", "dataset_maxwell.npz",
        "dataset_voith.csv", "dataset_voith.npz", "mix_leve.npz",
        "predicoes_recorrencia_esfera_Q400_xM1_grid40.csv")],
}


def excluido(nome, padroes):
    return any(fnmatch.fnmatch(nome, p) for p in padroes)


def arquivos(origem, destino, padroes):
    """Gera pares (arquivo de origem, arquivo de destino)."""
    if origem.is_file():
        yield origem, destino
        return
    for raiz, dirs, nomes in os.walk(origem):
        dirs[:] = [d for d in dirs if not excluido(d, padroes)]
        for nome in nomes:
            if excluido(nome, padroes):
                continue
            src = Path(raiz) / nome
            yield src, destino / src.relative_to(origem)


def sha256(caminho, bloco=1 << 22):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        while True:
            b = f.read(bloco)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def gb(n):
    return f"{n / 1e9:.2f} GB"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--origem", required=True, help="pasta onde estão os dados (a Downloads de origem)")
    ap.add_argument("--destino", default=str(Path(__file__).resolve().parent / "dados"), help="pasta dados/ do repositório")
    ap.add_argument("--grupos", nargs="+", choices=sorted(GRUPOS), default=sorted(GRUPOS))
    ap.add_argument("--dry-run", action="store_true", help="só lista o que seria copiado")
    ap.add_argument("--sha256", action="store_true", help="grava dados/SHA256SUMS.txt com o hash de cada arquivo copiado")
    a = ap.parse_args()

    origem_raiz, destino_raiz = Path(a.origem), Path(a.destino)
    if not origem_raiz.is_dir():
        sys.exit(f"origem não encontrada: {origem_raiz}")

    total, copiados, pulados, faltando = 0, 0, 0, []
    soma = []
    for g in a.grupos:
        print(f"== {g}")
        for rel_o, rel_d, pad in GRUPOS[g]:
            src = origem_raiz / rel_o
            if not src.exists():
                faltando.append(rel_o)
                print(f"   (não encontrado) {rel_o}")
                continue
            for s, d in arquivos(src, destino_raiz / rel_d, pad):
                tam = s.stat().st_size
                total += tam
                if d.exists() and d.stat().st_size == tam:
                    pulados += 1
                    soma.append(d)
                    continue
                print(f"   {'copiaria' if a.dry_run else 'copiando'}: {s.relative_to(origem_raiz)}  ({gb(tam)})")
                if not a.dry_run:
                    d.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(s, d)
                    copiados += 1
                    soma.append(d)
    print(f"\ntotal na origem: {gb(total)} | copiados: {copiados} | já presentes: {pulados} | grupos/itens ausentes: {len(faltando)}")

    if a.sha256 and not a.dry_run:
        saida = destino_raiz / "SHA256SUMS.txt"
        with open(saida, "w", encoding="utf8") as f:
            for d in sorted(set(soma)):
                f.write(f"{sha256(d)}  {d.relative_to(destino_raiz).as_posix()}\n")
        print("hashes gravados em", saida)


if __name__ == "__main__":
    main()
