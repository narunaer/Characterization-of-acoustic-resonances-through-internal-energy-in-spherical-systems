# Bancos de dados

Os bancos de dados da tese somam cerca de 19 GB (CSV e NPZ com até 1,6 GB cada). O GitHub recusa arquivos acima de
100 MB e desaconselha repositórios acima de poucos GB, então **os dados não são versionados**. Nesta pasta ficam só a
documentação (`LEIAME.txt` e `parametros_*.json` de cada rodada, que descrevem colunas, constantes e semente).

## Como trazer os dados para cá

No computador onde estão os dados (pasta `Downloads`), na raiz do repositório:

```
python copiar_dados.py --origem "C:\Users\Narun\Downloads" --dry-run      # só lista o que seria copiado
python copiar_dados.py --origem "C:\Users\Narun\Downloads"                # copia (pula o que já está igual)
python copiar_dados.py --origem "C:\Users\Narun\Downloads" --sha256       # copia e grava dados/SHA256SUMS.txt
python copiar_dados.py --grupos cap3 cap5 ...                               # só alguns grupos
```

Grupos: `cap3` (esfera fluida), `cap4` (core-shell fluido, 13 fatores de mistura), `cap4_reparo` (reparo do contraste de
velocidade), `cap4_relacoes` (tabelas e gráficos das relações de existência e de recorrência), `cap5` (esfera elástica,
Voigt e Maxwell) e `extras` (bancos antigos do dataset8 e `mix_leve.npz`).

## Para publicar os dados

O caminho mais simples é depositar os arquivos em um repositório de dados com DOI (por exemplo Zenodo, que em geral
aceita dezenas de GB por depósito) e colocar o DOI no `README.md` da raiz. Depois de rodar `copiar_dados.py --sha256`,
o arquivo `dados/SHA256SUMS.txt` permite conferir a integridade dos arquivos baixados por quem for reproduzir.

## Variáveis de ambiente lidas pelos notebooks

| Notebook | Variável | Valor sugerido |
|---|---|---|
| `cap3_esfera_fluida/esfera_fluida_tese.ipynb` | `ESFERA_DADOS` | `dados/cap3_esfera_fluida` |
| `cap4_casca_nucleo_fluido/coreshell_fluido_tese.ipynb` | `CORESHELL_DADOS` | `dados/cap4_coreshell_fluido/resultados_mix` |
| `cap5_esfera_viscoelastica/esfera_viscoelastica_tese.ipynb` | `VISCO_DADOS` | `dados/cap5_viscoelastico` |

Cada notebook também aceita `ESFERA_SAIDA`, `CORESHELL_SAIDA` e `VISCO_SAIDA` para escolher onde gravar figuras e tabelas
(por padrão, uma pasta `resultados/` ao lado de onde o notebook roda).
