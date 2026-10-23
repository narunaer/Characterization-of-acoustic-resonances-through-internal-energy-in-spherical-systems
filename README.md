# Caracterização de ressonâncias acústicas por meio da energia interna em sistemas esféricos

Código, notebooks, figuras e documentação dos bancos de dados da tese de doutorado de **Naruna Esselin Rodrigues**
(Programa de Pós-Graduação em Física, Instituto de Física de São Carlos, Universidade de São Paulo; orientador Odemir Bruno,
coorientador Alexandre Souto).

A tese caracteriza ressonâncias acústicas de espalhadores esféricos pela energia interna W_T = W_P + W_K (potencial mais
cinética). As ressonâncias são os picos de W_T no eixo real (proeminência mínima de 0,1 em log10), e o resultado é acompanhado
dos polos complexos do coeficiente de espalhamento. Para cada família de espalhadores, os bancos de dados alimentam uma
**relação de existência** (regressão logística: quando há ressonância) e uma **relação de recorrência** (onde elas aparecem).

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `cap3_esfera_fluida/` | Cap. 3, esfera fluida: notebook da tese, gerador do banco de dados (v3), relações de existência e de recorrência, mapa de estados ressonantes |
| `cap4_casca_nucleo_fluido/` | Cap. 4, centro espalhador casca-núcleo fluido (core-shell): notebook da tese, gerador (v8), códigos de cluster, funções de apoio |
| `cap5_esfera_viscoelastica/` | Cap. 5, esfera elástica e viscoelástica (Kelvin-Voigt e Maxwell): notebook da tese, gerador (v2), código físico autocontido |
| `apendice_H_casca_viscoelastica/` | Apêndice H, casca viscoelástica: limites, conservação de energia, capsídeos virais (HSV-1, HBV T4) |
| `validacao/` | Verificações numéricas independentes: integral de Clapeyron e integral radial por elementos finitos |
| `artigo_coreshell/` | Repositório do artigo sobre o core-shell fluido (mantido como estava, com README próprio) |
| `dados/` | Documentação dos bancos de dados; os arquivos grandes **não** estão no git (ver `dados/README.md`) |
| `tese/` | Lugar do PDF final depositado |
| `copiar_dados.py` | Copia os bancos de dados da pasta de origem para `dados/` e gera hashes |

## Do código à tese

| Parte da tese | Notebook | Gerador do banco de dados | Pasta de dados |
|---|---|---|---|
| Cap. 3 | `cap3_esfera_fluida/esfera_fluida_tese.ipynb` | `gerador_esfera.py` | `dados/cap3_esfera_fluida/` |
| Cap. 4 | `cap4_casca_nucleo_fluido/coreshell_fluido_tese.ipynb` | `gerador_coreshell.py` | `dados/cap4_coreshell_fluido/resultados_mix/` (13 fatores de mistura f de 0,00 a 1,00) |
| Cap. 5 | `cap5_esfera_viscoelastica/esfera_viscoelastica_tese.ipynb` | `gerador_viscoelastico.py` | `dados/cap5_viscoelastico/` (elástico, Maxwell, Voigt) |
| Apêndice H | `apendice_H_casca_viscoelastica/coreshell_viscoelastico_tese.ipynb` | calcula na hora (não usa banco) | |

Cada notebook apresenta primeiro as equações, na notação do texto, e depois o código e as figuras correspondentes. As
primeiras seções de cada notebook só precisam do próprio arquivo; as seções de banco de dados, mapas, relações de existência e de
recorrência e custo computacional leem a saída do gerador.

Figuras e tabelas do Apêndice H (Figs. 35 a 39) já estão em `apendice_H_casca_viscoelastica/figuras/`, com o eixo horizontal
rotulado y_M = k_M b (raio externo b) e, nas figuras dos capsídeos, sem a caixa de parâmetros dentro do gráfico.

## Como reproduzir

1. Ambiente (Python 3):

   ```
   python -m venv .venv
   .venv\Scripts\activate          # Windows;  no Linux/macOS: source .venv/bin/activate
   pip install -r requirements.txt
   ```

   Os rótulos das figuras usam LaTeX (`text.usetex`). É preciso ter uma distribuição LaTeX com `dvipng` e `cm-super`
   (MiKTeX ou TeX Live). Sem LaTeX os notebooks caem para o mathtext do matplotlib e as figuras saem com fonte um pouco diferente.

2. Dados: na raiz do repositório, `python copiar_dados.py --origem "<pasta onde estão os dados>"` (detalhes em `dados/README.md`).

3. Variáveis de ambiente que apontam cada notebook para seus dados (exemplo no PowerShell; no Linux/macOS use `export`):

   ```
   $env:ESFERA_DADOS    = "dados/cap3_esfera_fluida"
   $env:CORESHELL_DADOS = "dados/cap4_coreshell_fluido/resultados_mix"
   $env:VISCO_DADOS     = "dados/cap5_viscoelastico"
   jupyter lab
   ```

   Os notebooks gravam figuras e tabelas em `resultados/`, ao lado de onde são executados. No Colab, montam o Drive e usam
   as pastas `sphere_final`, `coreshell_final` e `visco_final`.

4. Geradores (cálculo pesado, pensado para cluster): `gerador_esfera.py`, `gerador_coreshell.py` e
   `gerador_viscoelastico.py` aceitam `--help`. Cada rodada grava `checkpoint_*.npz`, `resumo_*.csv`,
   `ressonancias_*.csv`, `ressonancias_malha_*.csv`, `candidatos_*.csv` e um `parametros_*.json` com todas as constantes
   e a semente, para a rodada poder ser repetida.

## Como as ressonâncias são contadas

* Curvas normais: máximos de W_T em uma malha base de 5000 pontos em x_M, com proeminência de pelo menos 0,1 em log10.
* Curvas de risco (contrastes que concentram picos estreitos): busca dos zeros complexos do denominador D_ℓ (ℓ = 0 a 10,
  Newton mais princípio do argumento), malha refinada com passo calculado pela largura dos picos, e contagem pelos polos
  com Q ≥ 2 e fração de área mínima da energia na janela de espaçamento médio ΔW.
* ℓ_max = 10 em todos os bancos. O detalhe de cada versão está no `LEIAME.txt` de cada rodada.

## Notas de versão

* Os notebooks foram salvos **sem saídas**, para o diff do git ficar legível. As figuras são regeneradas ao executar.
* Para cada capítulo foi mantida a versão mais recente do notebook; cópias antigas (`(1)`, `(2)`, `_corrigido`) não foram
  incluídas, exceto `apendice_H_casca_viscoelastica/auxiliares/coreshell_viscoelastico_tese_corrigido_malha.ipynb`, que é uma
  variante com ajustes de malha e **não** é a usada nas figuras da tese.
* No notebook do Apêndice H, o código e as equações mantêm o nome `xM`; apenas os rótulos das figuras 35 a 39 mostram y_M.
* `artigo_coreshell/` guarda o repositório do artigo (core-shell fluido) como estava, inclusive seus resultados já calculados.

## Como citar

Ver `CITATION.cff`. Licença ainda a definir; sem licença explícita, o código fica com todos os direitos reservados à autora.
