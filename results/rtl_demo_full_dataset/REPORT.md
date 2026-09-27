# Campanha RTL completa — pesos sintéticos — 2.000 imagens

Execução real do `tiny_cnn` em Icarus Verilog. Python foi usado apenas para preparar a referência inteira e agregar as linhas emitidas pelo simulador.

- Execução: [36336686174](https://github.com/EduardoHT/ci-digital-cnn-verilog/actions/runs/36336686174)
- Imagens processadas: **2000**
- Comparações RTL × referência aprovadas: **2000/2000**
- Duração até a agregação: **1136 s**

## total

| verdadeira \ prevista | circulo | quadrado | triangulo | cruz | estrela | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| circulo | 400 | 0 | 0 | 0 | 0 | 400 |
| quadrado | 400 | 0 | 0 | 0 | 0 | 400 |
| triangulo | 400 | 0 | 0 | 0 | 0 | 400 |
| cruz | 400 | 0 | 0 | 0 | 0 | 400 |
| estrela | 400 | 0 | 0 | 0 | 0 | 400 |

Acurácia funcional de distribuição: **400/2000 = 0.200000**. 
Macro F1: **0.066667**. Weighted F1: **0.066667**.

## train

| verdadeira \ prevista | circulo | quadrado | triangulo | cruz | estrela | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| circulo | 280 | 0 | 0 | 0 | 0 | 280 |
| quadrado | 280 | 0 | 0 | 0 | 0 | 280 |
| triangulo | 280 | 0 | 0 | 0 | 0 | 280 |
| cruz | 280 | 0 | 0 | 0 | 0 | 280 |
| estrela | 280 | 0 | 0 | 0 | 0 | 280 |

Acurácia funcional de distribuição: **280/1400 = 0.200000**. 
Macro F1: **0.066667**. Weighted F1: **0.066667**.

## val

| verdadeira \ prevista | circulo | quadrado | triangulo | cruz | estrela | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| circulo | 60 | 0 | 0 | 0 | 0 | 60 |
| quadrado | 60 | 0 | 0 | 0 | 0 | 60 |
| triangulo | 60 | 0 | 0 | 0 | 0 | 60 |
| cruz | 60 | 0 | 0 | 0 | 0 | 60 |
| estrela | 60 | 0 | 0 | 0 | 0 | 60 |

Acurácia funcional de distribuição: **60/300 = 0.200000**. 
Macro F1: **0.066667**. Weighted F1: **0.066667**.

## test

| verdadeira \ prevista | circulo | quadrado | triangulo | cruz | estrela | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| circulo | 60 | 0 | 0 | 0 | 0 | 60 |
| quadrado | 60 | 0 | 0 | 0 | 0 | 60 |
| triangulo | 60 | 0 | 0 | 0 | 0 | 60 |
| cruz | 60 | 0 | 0 | 0 | 0 | 60 |
| estrela | 60 | 0 | 0 | 0 | 0 | 60 |

Acurácia funcional de distribuição: **60/300 = 0.200000**. 
Macro F1: **0.066667**. Weighted F1: **0.066667**.

## Interpretação

Estas matrizes descrevem a distribuição de saídas dos pesos sintéticos e validam o pipeline em escala. Não são métricas de generalização. Apenas a matriz `test` usa o split reservado, e mesmo ela não mede um modelo treinado.

Divisão por zero: precision, recall ou F1 recebem `0.0` quando o denominador correspondente é zero.

Cobertura: todas as 2.000 imagens tiveram logits, classe, protocolo e ciclos comparados; os mapas internos completos foram comparados nos dez vetores demo pelo testbench detalhado.
