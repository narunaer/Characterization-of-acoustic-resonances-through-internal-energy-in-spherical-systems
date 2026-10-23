# Manifesto dos dados

Tamanhos aproximados, medidos em 19/10/2026 na pasta `Downloads` de origem.

| Grupo | Origem (relativa a `Downloads`) | Destino (em `dados/`) | Tamanho |
|---|---|---|---|
| `cap3` | `sphere_final/` | `cap3_esfera_fluida/` | 3,1 GB |
| `cap4` | `coreshell_final/resultados_mix/coreshell_lmax10_mixed_mix{0,00 a 1,00}/` (13 fatores de mistura f) | `cap4_coreshell_fluido/resultados_mix/` | 6,1 GB |
| `cap4_reparo` | `reparo_velocity/` (sem os `.py`) | `cap4_coreshell_fluido/reparo_velocity/` | 1,9 GB |
| `cap4_relacoes` | `relacoes_existencia_recorrencia_final/` (sem `.svg`) | `cap4_coreshell_fluido/relacoes_existencia_recorrencia_final/` | centenas de MB |
| `cap4` (checkpoints z4) | `checkpoint_ressonancias_coreshell_lmax10_{density,mixed}_final_reparado_z4.npz` | `cap4_coreshell_fluido/checkpoints/` | 23 MB |
| `cap5` | `visco_final/viscoelastico_lmax10_{elastico,maxwell,voigt}/` | `cap5_viscoelastico/` | 7,5 GB |
| `extras` | `dataset8.csv/.npz`, `dataset_maxwell.csv/.npz`, `dataset_voith.csv/.npz`, `mix_leve.npz`, `predicoes_recorrencia_esfera_Q400_xM1_grid40.csv` | `extras/` | 0,3 GB |

## Conteúdo de cada rodada do gerador (resumo)

Cada pasta de rodada tem, além do `LEIAME.txt` e do `parametros_*.json` (versionados):

* `checkpoint_*.npz`: tudo o que foi calculado, curva a curva;
* `resumo_*.csv`: uma linha por curva (contagem final `N_res_energy` ou `N_res`);
* `ressonancias_*.csv`: uma linha por ressonância da contagem final;
* `ressonancias_malha_*.csv`: uma linha por pico encontrado na malha de energia;
* `candidatos_*.csv`: uma linha por polo (zero de D_l) da busca;
* `hipercubo_*.csv/.npz` (só no caso sólido): formato antigo, uma linha por pico.

Geradores: `cap3_esfera_fluida/gerador_esfera.py` (versão 3), `cap4_casca_nucleo_fluido/gerador_coreshell.py` (versão 8) e
`cap5_esfera_viscoelastica/gerador_viscoelastico.py` (versão 2). Parâmetros e semente de cada rodada: `parametros_*.json`.
