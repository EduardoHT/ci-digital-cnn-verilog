# Verificação HDL online — 27/09/2026

## Resultado

**PASS** na execução privada do GitHub Actions com Icarus Verilog.

- [Execução completa e logs](https://github.com/EduardoHT/ci-digital-cnn-verilog/actions/runs/36334747450)
- Commit testado: `8ca93aa6904e09d0ba3fdb3a7e54d6a35645aa10`
- Duração total: 1 min 8 s
- Artefato de logs: `resultados-simulacao-hdl`
- Digest do artefato: `sha256:289345986011e9452449ea109eb600d9b156eaa8a581bef8ec93906626ef29d7`

## Pesos sintéticos determinísticos

Os dez testes aritméticos básicos passaram (`PASS_ARITHMETIC tests=10`). A CNN completa passou nos dez vetores de `exports/demo/`:

```text
PASS_ALL vectors=10 exact_values_per_vector=9102 labeled_correct=1/5
```

O vetor 1 é a imagem `test/quadrado/quadrado_0000.png`. Resultado exato da simulação:

```text
PASS vector=1 class=0 cycles=206520 logits=[60969 -54749 2570 -13455 -45341]
```

A classe 0 não é tratada como erro: os pesos são sintéticos e não foram treinados. O teste importante é a igualdade bit a bit dos 9.102 valores por vetor, incluindo Conv1, Pool1, Conv2, Pool2, logits e argmax.

## Troca para pesos treinados

Sem qualquer alteração no RTL ou testbench, a execução repetiu a simulação usando `exports/trained/`:

```text
PASS_ALL vectors=20 exact_values_per_vector=9102 labeled_correct=13/15
```

Isso valida o mecanismo de substituição de pesos para conjuntos que respeitem o mesmo contrato de topologia, ordem, dimensões, tipos e escalas. O resultado 13/15 é apenas o subconjunto de vetores do testbench, não a métrica do conjunto de teste de 300 imagens.

## Controles efetivamente verificados

- compilação SystemVerilog do RTL completo e dos dois testbenches;
- MAC `uint8 × int8` com acumulador `int32`;
- ReLU, arredondamento meio para cima, saturação e pooling unsigned;
- valores exatos de todos os mapas intermediários;
- cinco logits e argmax;
- carga de 784 pixels e arquivos de parâmetros completos;
- `busy`, pulso de `done`, timeout e reset inicial;
- contagem de 206.520 ciclos por inferência;
- execução com pesos sintéticos e treinados na mesma RTL.

## Limites

Esta evidência é de simulação funcional. Ela não substitui síntese, análise de timing, place-and-route, medição de recursos/energia ou teste em FPGA física. Também não mede acurácia dos pesos sintéticos.
