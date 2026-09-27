# Auditoria da primeira versão e alterações na v2

## Problemas identificados no pacote anterior

O `tiny_cnn_top.sv` anterior apenas avançava contadores. Ele não recebia pixels, não calculava
convoluções, não aplicava os pesos à imagem e fixava `class_id=0`. Os módulos MAC/ReLU existiam,
mas não estavam integrados a um caminho de dados de inferência. Portanto, não era correto chamar
aquele RTL de um classificador funcional.

A exportação Python escrevia alguns pesos int8, mas deixava biases e escalas de ativação sem fechar.
Não havia uma referência inteira bit a bit. O testbench verificava apenas que a máquina chegava
a `done`, não se os números estavam certos. As imagens escolhidas por índices absolutos do dataset
não eram garantidamente exclusivas da partição de teste feita por `random_split`.

O gerador antigo não implementava toda a diversidade discutida, e a entrega não incluía um conjunto
de PNGs já organizado em pastas. Também faltavam validação separada, controle de sobrescrita,
tratamento explícito de sinais, saturação e teste da borda ímpar do pooling.

## Mudanças implementadas

Foi adotada, de forma consistente nos dois lados, a arquitetura 1→8→16 canais com Dense400→5.
O aumento de camadas não era uma correção obrigatória: foi uma escolha de capacidade para o dataset
com mais variação. Há 3.253 parâmetros, contra 3.425 da primeira arquitetura 1→4 com Dense676→5;
a segunda convolução aumenta o cálculo, mas o vetor denso menor reduz a contagem total de parâmetros.

A v2 implementa uma entrada de pixels, quatro memórias de feature maps, um MAC compartilhado,
requantização/ReLU, max pooling, Dense e argmax. Os estados controlam contas reais. O resultado de
cada produto é acumulado antes de gravar a saída em um estado separado, evitando perder o último MAC
por uso incorreto de atribuições não bloqueantes.

Há um único contrato numérico. A normalização é pixel/256; pesos são int8; ativações uint8;
biases e acumuladores int32. A calibração usa treino. O exportador calcula escalas, verifica limites
de soma, escreve todos os `.mem`, relê os arquivos e exige igualdade com os tensores quantizados.

Os dados incluem 2.000 imagens únicas por hash, com divisão explícita 1.400/300/300 e classe balanceada.
A ordem de classes é fixa. O checkpoint é escolhido pela validação. A avaliação final usa o teste reservado.

O novo testbench compara todas as ativações quantizadas, pools, logits e classes, não só a previsão final.
Inclui watchdog, contagem esperada de ciclos e verificação de que `done` dura um ciclo. Os testes aritméticos
cobrem sinal, pixels acima de 127, pesos negativos, rounding, saturação e INT32_MIN.

## Estado real da execução

O Python foi executado. Treino de 40 épocas, melhor checkpoint na época 36. Teste float: 288/300;
teste inteiro: 289/300. Concordância de classe entre os dois: 299/300.
Nove testes unitários Python passaram. O notebook inteiro passou por um smoke test de uma época.
O endereçamento equivalente ao RTL foi reimplementado em Python escalar e comparado com a referência
NumPy em 30 vetores, somando 273.060 valores iguais.

**Não houve compilação nem simulação de HDL nesta sessão.** Icarus e Verilator não estavam disponíveis;
as tentativas de instalação/download falharam. A implementação escalar não verifica sintaxe HDL,
semântica de eventos, suporte do simulador nem temporização de atribuições SystemVerilog.
O arquivo `reports/rtl_environment_check.log` registra a ausência do simulador, não uma falha de compilação.

O pacote online foi gerado e conferido como arquivo autocontido, mas não executado no site.
A validação física em FPGA foi retirada do escopo por sua decisão.

## Critério de aceitação técnica da próxima execução

Aprovar os testes aritméticos e todas as comparações do testbench, sem X/Z, divergência ou timeout.
Salvar o log real com `PASS_ALL` e identificar o simulador e a exportação usados. Não basta observar
uma classe visualmente plausível. Só após essa etapa haverá evidência de equivalência Python inteiro↔HDL.

O projeto classifica figuras sintéticas; não foi avaliado como sistema de pouso autônomo ou detector real.
A diferença de um acerto após quantização não sustenta uma conclusão estatística de melhoria.
