# Contrato RTL e simulação online da CNN

Este documento registra o contrato que permite trocar os pesos sintéticos pelos pesos treinados **sem reescrever o RTL**, desde que arquitetura, dimensões, ordem dos dados e regras numéricas permaneçam iguais.

## Arquitetura implementada

```text
Entrada 28 x 28 x 1, uint8
Conv1 valid 3 x 3, 1 -> 8 canais       = 26 x 26 x 8
ReLU + requantização + MaxPool 2 x 2   = 13 x 13 x 8
Conv2 valid 3 x 3, 8 -> 16 canais      = 11 x 11 x 16
ReLU + requantização + MaxPool 2 x 2   = 5 x 5 x 16
Flatten C,H,W                           = 400 valores
Dense                                  = 5 logits int32
Argmax                                 = classe 0..4
```

Classes: `0=circulo`, `1=quadrado`, `2=triangulo`, `3=cruz`, `4=estrela`.

## Entrada e handshake

- Carregar exatamente 784 pixels enquanto `busy=0`.
- `input_addr = linha * 28 + coluna`, com endereços de 0 a 783.
- `input_data` é um pixel `uint8` em tons de cinza; não normalizar novamente no RTL.
- Após a carga, aplicar `start=1` por um ciclo.
- `busy` permanece ativo durante a inferência.
- `done` é um pulso de um ciclo e coincide com a queda de `busy`.
- A implementação atual deve terminar em 206.520 ciclos de processamento por imagem.

A imagem real futura deve ser convertida para 28 × 28, um canal, 8 bits, na mesma ordem linha-a-linha. Trocar somente os 784 pixels não exige alterar o RTL.

## Pesos, biases e ordem

| Arquivo | Tipo | Forma lógica | Quantidade | Ordem linear |
| --- | --- | --- | ---: | --- |
| `conv1_w.mem` | int8 | `[8,1,3,3]` | 72 | `O,I,Ky,Kx` |
| `conv1_b.mem` | int32 | `[8]` | 8 | canal de saída |
| `conv2_w.mem` | int8 | `[16,8,3,3]` | 1.152 | `O,I,Ky,Kx` |
| `conv2_b.mem` | int32 | `[16]` | 16 | canal de saída |
| `fc_w.mem` | int8 | `[5,400]` | 2.000 | classe, índice flatten |
| `fc_b.mem` | int32 | `[5]` | 5 | classe |
| `shifts.mem` | uint8 | `[2]` | 2 | Conv1, Conv2 |

Os mapas internos usam ordem `C,H,W`, row-major. As convoluções são correlações válidas; o kernel não é invertido. O flatten percorre os 16 canais e, dentro de cada canal, linha e coluna.

## Contrato numérico

- Ativações e pixels: `uint8`.
- Pesos: `int8`, intervalo usado na exportação `[-127,127]`.
- Biases, MACs e logits: `int32`.
- Conv1 e Conv2: MAC → ReLU → arredondamento meio para cima → deslocamento à direita → saturação em `[0,255]`.
- Dense: produz cinco logits `int32` na mesma escala.
- Argmax usa comparação estrita; em empate vence o menor índice.
- Entrada Python equivalente: `pixel / 256`, não `pixel / 255`.

Nos pesos sintéticos de `exports/demo/`, a geração é reproduzível com a seed `20260925`. Eles exercitam todas as camadas, mas não foram treinados e não demonstram acurácia.

## Vetor do quadrado

O vetor 1 de `exports/demo/vectors/` vem de `dataset/test/quadrado/quadrado_0000.png` e tem rótulo real 1. Com os pesos sintéticos, a classe esperada da referência inteira é 0; isso é intencional. O teste exige igualdade exata entre SystemVerilog e a referência inteira em:

- 5.408 saídas da Conv1;
- 1.352 saídas do primeiro pooling;
- 1.936 saídas da Conv2;
- 400 saídas do segundo pooling;
- cinco logits e o argmax.

São 9.102 valores conferidos por vetor. O testbench também verifica reset, carga, timeout, `busy`, pulso de `done`, contagem de ciclos, arquivos incompletos, arredondamento, saturação, sinal do MAC e extremos de overflow.

## Executar online

1. Abra a aba **Actions** do repositório privado.
2. Selecione **Simulação HDL da CNN completa**.
3. Clique em **Run workflow** e confirme a branch `main`.
4. Abra a execução para ver os passos e baixar o artefato `resultados-simulacao-hdl`.

A execução instala o Icarus Verilog no runner privado, compila o RTL original e roda primeiro `exports/demo`, depois `exports/trained`, sem mudar uma linha do hardware entre os dois conjuntos de pesos.

## O que a simulação prova — e o que não prova

Um `PASS_ALL` prova concordância bit a bit com a referência inteira para os vetores incluídos e valida o protocolo simulado. Não prova acurácia dos pesos sintéticos, síntese, frequência máxima, utilização de recursos, timing, consumo, integração com câmera ou funcionamento em FPGA física.
