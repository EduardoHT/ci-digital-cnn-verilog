# CI Digital — CNN em Python e SystemVerilog

Projeto acadêmico completo para classificar cinco figuras geométricas (`círculo`, `quadrado`, `triângulo`, `cruz` e `estrela`) com uma CNN treinada em Python e uma implementação de inferência em SystemVerilog.

O repositório inclui o dataset sintético, notebook, código Python, módulos e testbenches SystemVerilog, pesos treinados e de demonstração, checkpoint, métricas, resultados, vetores de referência e documentação.

## Comece por aqui

- [Guia passo a passo de Python](docs/01_Guia_Python.pdf)
- [Guia passo a passo de SystemVerilog](docs/02_Guia_SystemVerilog.pdf)
- [Notebook de treinamento e exportação](notebooks/01_treino_exportacao.ipynb)
- [Prévia do dataset](dataset/preview.png)
- [Estado e limites da validação](reports/validation_status.json)
- [Auditoria técnica da versão 2](AUDITORIA_V2.md)
- [README original do pacote](README_ORIGINAL.md)

## Conteúdo principal

```text
dataset/               2.000 imagens PNG + manifesto + prévia
docs/                  dois guias em PDF
notebooks/             notebook editável
python/                geração, treino, quantização e exportação
rtl/                   módulos SystemVerilog
tb/                    testbenches
exports/trained/       pesos treinados, metadados e vetores
exports/demo/          pesos reproduzíveis de demonstração
eda_playground/        versões autocontidas para simulação online
runs/baseline/         checkpoint, histórico, métricas e predições
reports/               logs e estado real da validação
tests/                 testes Python e verificação escalar
```

O dataset tem 2.000 imagens de 28 × 28 pixels em tons de cinza, distribuídas em 1.400 imagens de treino, 300 de validação e 300 de teste.

## Resultados do treinamento de referência

| Avaliação sobre 300 imagens de teste | Resultado |
| --- | ---: |
| CNN em ponto flutuante | 288/300 — 96,00% |
| Referência inteira quantizada | 289/300 — 96,33% |
| Concordância de classe entre as versões | 299/300 — 99,67% |

Esses números descrevem apenas a execução incluída e o dataset sintético deste projeto; não constituem validação com imagens reais de drones.

## Estado da validação

A parte Python foi executada e testada. A compilação e a simulação SystemVerilog ainda estão pendentes, pois o ambiente que gerou o pacote não dispunha de simulador HDL. Os testbenches e os comandos necessários estão incluídos, mas não há um resultado real `PASS_ALL` de HDL neste repositório.

Consulte o [README original](README_ORIGINAL.md) para os comandos completos de instalação, treinamento, exportação e simulação, além do contrato numérico usado entre Python e RTL.

## Proveniência

Os arquivos do pacote original foram preservados. A identificação do ZIP-fonte está registrada em [SOURCE_ARCHIVE_SHA256.txt](SOURCE_ARCHIVE_SHA256.txt).
