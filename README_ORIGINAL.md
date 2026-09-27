# CI Digital — classificador de figuras em Python e SystemVerilog

**Versão 2.0 · 25/09/2026 · Escopo fechado em simulação, sem DE10-Lite.**

## Comece aqui

Extraia este ZIP em uma pasta nova. **Não misture os arquivos com a versão anterior.**
A arquitetura, a quantização, as memórias e o testbench foram substituídos em conjunto.

- **Para estudar/treinar:** abra `docs/01_Guia_Python.pdf` e depois `notebooks/01_treino_exportacao.ipynb`.
- **Para começar pelo Verilog sem esperar treino:** abra `docs/02_Guia_SystemVerilog.pdf` e use os dois arquivos de `eda_playground/trained/` no navegador.
- **Para conhecer os limites da validação:** leia `reports/validation_status.json` e `AUDITORIA_V2.md`.

## O que foi realmente executado

Foram gerados e conferidos 2.000 PNGs, executado um treino completo de 40 épocas em CPU,
feito um smoke test do notebook inteiro com uma época e aprovados nove testes Python.
Uma segunda implementação escalar do endereçamento comparou 273.060 valores em 30 vetores.
**Essa segunda implementação é Python, não um simulador HDL.**

**A compilação e a simulação SystemVerilog NÃO foram executadas nesta sessão.** O ambiente não tinha
Icarus/Verilator e as tentativas de obter o simulador não funcionaram. Não existe um log de `PASS_ALL`
real de HDL neste pacote. O testbench e o comando de simulação estão prontos para executar essa etapa.
A versão do navegador também foi preparada, mas não executada na conta do EDA Playground.

## Resultado do treinamento de referência

| Avaliação sobre o teste reservado | Resultado |
| --- | ---: |
| Modelo em ponto flutuante | 288/300 = 96,00% |
| Referência inteira quantizada | 289/300 = 96,33% |
| Concordância de classe float/inteiro | 299/300 = 99,67% |

O melhor checkpoint foi escolhido na época 36, usando validação, não teste.
Um acerto a mais após quantização não prova superioridade geral do modelo inteiro.
Esses números descrevem apenas este dataset sintético e esta execução.

## Dataset e arquitetura

Cinco classes: `0=circulo`, `1=quadrado`, `2=triangulo`, `3=cruz`, `4=estrela`.
São 400 imagens por classe, das quais 280 são treino, 60 validação e 60 teste.
Total: **1.400/300/300**. Arquivos PNG grayscale de **28×28 pixels e 8 bits por pixel**.
Rotação, translação, escala, contraste, ruído e borrão leves já estão presentes nas imagens.

```
28x28x1
Conv3x3 (1 -> 8), valid       -> 26x26x8
ReLU + MaxPool2x2            -> 13x13x8
Conv3x3 (8 -> 16), valid      -> 11x11x16
ReLU + MaxPool2x2            ->  5x 5x16
Flatten                     -> 400
Dense                       -> 5 logits
Argmax                      -> classe 0..4
```

**3.253 parâmetros; 190.064 multiplicações-acumulações por imagem.**
O último pooling usa floor, descartando a última linha/coluna do mapa 11×11.
O circuito faz inferência; o treinamento ocorre exclusivamente em Python.

## Pastas principais

```
ci_digital_v2/
  docs/                  dois PDFs passo a passo
  notebooks/             notebook editável
  python/                gerador, modelo, treino e quantização
  dataset/               2.000 PNGs + manifesto + preview
  rtl/                   quatro módulos SystemVerilog
  tb/                    testbench integral + testes aritméticos
  exports/trained/       pesos aprendidos + metadados + vetores
  exports/demo/          pesos aleatórios reproduzíveis + vetores
  eda_playground/trained/ design.sv e testbench.sv, autocontidos
  eda_playground/demo/    a mesma CNN com pesos não treinados
  runs/baseline/         checkpoint, histórico e métricas reais
  scripts/               preparação online e execução do simulador
  tests/                 testes Python e modelo escalar independente
  reports/               logs reais e estado da validação
```

Os PNGs estão em `dataset/train|val|test/<classe>/`. O mosaico `dataset/preview.png`
fica fora das pastas de treinamento. `dataset/manifest.csv` registra seeds, variações e hashes.

## Windows: ambiente Python

Abra um terminal na pasta que contém este README. Não é preciso ativar a venv no PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m jupyter lab
```

Abra `notebooks/01_treino_exportacao.ipynb` no Jupyter. A célula de configuração usa
`TREINAR_AGORA=True`; altere para `False` para explorar o baseline já fornecido.
Um ambiente que já tenha PyTorch funcionando também serve. CUDA é opcional.

## Linha de comando, alternativa ao notebook

Com o interpretador correto, sempre **na raiz do projeto**:

```text
python -m python.train --out runs/meu_treino --epochs 40
python -m python.export --checkpoint runs/meu_treino/best.pt --out exports/meu_treino
python scripts/make_eda.py --export exports/meu_treino --out eda_playground/meu_treino
python -m unittest discover -s tests -v
```

No Windows sem ativar a venv, troque o `python` inicial por `.venv\Scripts\python.exe`.
O `python` após `-m` é o nome da pasta de módulos, não outro interpretador.
As pastas de treino/exportação existentes são protegidas contra sobrescrita acidental.
Use outro nome ou `--overwrite` explicitamente.

## Contrato numérico obrigatório

Entrada float = pixel/256, **não /255**. Pixels e ativações usam uint8, pesos usam int8,
biases e acumuladores usam int32. As escalas são potências de dois por camada, calibradas em
500 imagens do treino. Pesos e biases usam arredondamento offline `np.rint`; ativações usam
ReLU, metade para cima, shift e saturação. Os logits finais compartilham a mesma escala.
A classe em caso de empate é o menor índice.

Não exporte só os pesos: copie também biases, `shifts.mem`, metadados e novos valores esperados.
Não aplique o pré-processamento de outro dataset ou normalize a imagem novamente no RTL.

## Rodar no navegador

Consulte as instruções ilustradas por tabelas e exemplos no PDF Verilog. Entre no EDA Playground,
selecione Verilog/SystemVerilog e um simulador compatível (Icarus, quando disponível), cole o
conteúdo de `eda_playground/trained/design.sv` no painel Design e o de `testbench.sv` no painel
Testbench. Não é necessário enviar os `.mem` nesta variante. Ambos precisam vir da mesma pasta.
São cinco vetores, um de cada classe, com comparação de todos os mapas intermediários.

**O sucesso deverá produzir `PASS_ALL` no console. Isso é um resultado esperado, não um log
já obtido nesta sessão.** Login e disponibilidade de simuladores dependem da plataforma.
Para ondas, use a opção de execução `+WAVES` e Open EPWave after run.

## Rodar localmente, se houver Icarus

```text
iverilog -V
vvp -V
python scripts/run_sim.py --export exports/trained
python scripts/run_sim.py --export exports/demo
```

A execução treinada usa 20 vetores: 15 figuras de teste + 5 estresses aritméticos sem rótulo.
O script compila `tb_arithmetic.sv` e `tb_cnn.sv`, exige mensagens de conclusão e guarda logs.
Sem `iverilog`/`vvp`, ele para com erro claro. Não instala programas automaticamente.

## Limites e conclusão do projeto

Não há detecção, localização do marcador, classe fundo, fotos reais, câmera nem comando de pouso.
Uma imagem vazia ainda recebe uma das cinco classes; os estresses não contam na acurácia.
Não foram executados síntese, place-and-route, timing ou teste em placa. A contagem prevista de
206.520 ciclos vem da FSM proposta; não é um benchmark de FPGA.

Para fechar a validação, execute o HDL, resolva eventuais mensagens do simulador e guarde o log
com comparação integral. Confirme com a coordenação do curso o formato e os documentos de submissão.
Os PDFs têm referências oficiais para PyTorch, EDA Playground e Icarus Verilog.
