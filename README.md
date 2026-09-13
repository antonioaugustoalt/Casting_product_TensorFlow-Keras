# Machine Learning e Visão Computacional — Inspeção de Peças

## 1. Objetivo

Este projeto tem como objetivo desenvolver um pipeline de inspeção visual de peças metálicas combinando técnicas clássicas de processamento de imagens com aprendizado profundo.

Na primeira etapa, foram utilizadas técnicas de Visão Computacional com OpenCV para realizar uma análise exploratória das imagens, aplicando conversão para escala de cinza, suavização, limiarização, detecção de bordas e operações morfológicas.

Na segunda etapa, foi desenvolvido um classificador utilizando uma Rede Neural Convolucional (CNN) com TensorFlow/Keras, capaz de classificar as peças em duas categorias:

* **OK**
* **Defeituosa**

O projeto também utiliza Data Augmentation para aumentar a variedade das imagens utilizadas no treinamento e reduzir a dependência do modelo em condições específicas de posição, zoom e iluminação.

---

## 2. Dataset

Foi utilizado o dataset **Casting Product Image Data for Quality Inspection**, composto por imagens de peças metálicas destinadas à inspeção de qualidade.

Neste projeto, as imagens foram organizadas em duas classes:

```text
def_front/
    → Peças defeituosas

ok_front/
    → Peças OK
```

O conjunto utilizado possui:

* **1.300 imagens**
* **2 classes**
* Aproximadamente **781 imagens defeituosas**
* Aproximadamente **519 imagens OK**

As imagens foram carregadas automaticamente utilizando `image_dataset_from_directory()`.

---

## 3. Estrutura do Projeto

```text
projeto_tensor_keras/
│
├── def_front/
│   └── imagens de peças defeituosas
│
├── ok_front/
│   └── imagens de peças OK
│
├── main.ipynb
├── funcoes.py
├── requirements.txt
└── README.md
```

O arquivo `main.ipynb` contém o desenvolvimento principal do projeto, incluindo a análise exploratória, preparação dos dados, Data Augmentation, construção da CNN e treinamento.

---

# 4. Sprint 2 — Análise Exploratória Clássica

## 4.1 Grayscale

As imagens foram convertidas para escala de cinza utilizando o OpenCV:

```python
imagem_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

As imagens utilizadas no dataset já apresentavam aparência monocromática. Dessa forma, a conversão para Grayscale não produziu uma alteração visual significativa, porém a operação foi realizada para garantir que as imagens utilizadas no processamento clássico fossem representadas explicitamente por uma única matriz de intensidade.

A escala de intensidade utilizada possui valores entre:

```text
0   → preto
255 → branco
```

A conversão foi importante para preparar as imagens para as etapas posteriores de análise de características.

---

## 4.2 Gaussian Blur

Após a conversão para escala de cinza, foi aplicado o filtro Gaussian Blur:

```python
imagem_blur = cv2.GaussianBlur(
    imagem_gray,
    (5, 5),
    0
)
```

O objetivo do filtro foi suavizar pequenas variações e ruídos presentes na imagem antes das etapas de detecção de características.

Foi utilizado um kernel de `5 × 5` e sigma automático.

O efeito visual foi relativamente discreto com esse tamanho de kernel, mas suficiente para realizar uma suavização sem remover excessivamente as características estruturais da peça.

---

# 5. Sprint 3 — Destaque de Características

Nesta etapa foram utilizadas técnicas para destacar visualmente possíveis regiões defeituosas.

O pipeline desenvolvido foi:

```text
Imagem Original
       ↓
Grayscale
       ↓
Gaussian Blur
       ↓
Threshold
       ↓
Canny
       ↓
Operações Morfológicas
```

---

## 5.1 Threshold

Foi aplicada uma limiarização utilizando:

```python
_, imagem_thresh = cv2.threshold(
    imagem_blur,
    80,
    255,
    cv2.THRESH_BINARY
)
```

O threshold utilizado foi definido empiricamente durante a análise exploratória.

O valor `80` apresentou um resultado satisfatório para as imagens analisadas, embora não tenha sido o valor perfeito para todas as imagens.

A operação converte os valores de intensidade em uma representação binária:

```text
intensidade < 80  → 0
intensidade ≥ 80  → 255
```

Essa transformação permitiu separar visualmente diferentes regiões da superfície da peça.

---

## 5.2 Detecção de Bordas com Canny

Foi utilizado o algoritmo Canny:

```python
imagem_canny = cv2.Canny(
    imagem_blur,
    50,
    150
)
```

O Canny foi utilizado para identificar mudanças bruscas de intensidade na imagem.

Nas imagens com acabamento mais uniforme, o algoritmo conseguiu destacar principalmente o defeito localizado na região central da peça.

Em imagens com acabamento superficial mais irregular, também foram identificadas estruturas adicionais nas bordas da peça. Isso ocorre porque o Canny detecta transições de intensidade e não possui conhecimento sobre o que representa necessariamente um defeito.

Dessa forma, irregularidades de acabamento também podem ser interpretadas como bordas.

---

## 5.3 Erosão

A erosão foi testada experimentalmente utilizando diferentes tamanhos de kernel.

Exemplo:

```python
kernel = np.ones((3, 3), np.uint8)

imagem_erodida = cv2.erode(
    imagem_canny,
    kernel,
    iterations=1
)
```

Também foi avaliado um kernel menor.

Nos testes realizados, a erosão reduziu excessivamente as estruturas produzidas pelo Canny, chegando a praticamente eliminar as linhas detectadas.

Por esse motivo, a erosão foi mantida no notebook para fins de demonstração e documentação da análise exploratória, mas não foi utilizada como parte do resultado final escolhido.

---

## 5.4 Dilatação

A dilatação apresentou um resultado mais adequado:

```python
kernel = np.ones((3, 3), np.uint8)

imagem_dilatada = cv2.dilate(
    imagem_canny,
    kernel,
    iterations=1
)
```

A operação tornou as estruturas detectadas pelo Canny mais espessas e contínuas.

Nas imagens analisadas, o defeito central tornou-se mais evidente e as bordas apresentaram estruturas mais contínuas.

Dessa forma, a dilatação foi escolhida como a operação morfológica de maior interesse para a análise final.

---

# 6. Sprint 4 — Ingestão de Dados e Data Augmentation

## 6.1 Ingestão do Dataset

O dataset completo foi carregado automaticamente utilizando:

```python
tf.keras.utils.image_dataset_from_directory()
```

Foi utilizada uma divisão de:

```text
80% → Treinamento
20% → Validação
```

Resultado:

```text
1.300 imagens
│
├── 1.040 → Treinamento
└──   260 → Validação
```

Foram utilizadas as seguintes configurações:

```text
image_size = (224, 224)
batch_size = 32
seed = 42
```

As imagens foram portanto padronizadas para:

```text
224 × 224 × 3
```

onde os três canais representam RGB.

---

## 6.2 Data Augmentation

Foi utilizado Data Augmentation dinâmico com:

```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomBrightness(0.1)
])
```

As transformações utilizadas foram:

* **RandomRotation** — pequenas variações de rotação;
* **RandomZoom** — pequenas variações de aproximação/afastamento;
* **RandomBrightness** — variações de iluminação.

Essas transformações são aplicadas dinamicamente durante o treinamento.

As imagens originais não são modificadas permanentemente no disco.

O objetivo é fazer com que a CNN observe diferentes versões da mesma peça e aprenda características mais gerais, reduzindo a dependência em condições específicas de captura.

O Data Augmentation foi aplicado ao conjunto de treinamento e não ao conjunto de validação.

---

## 6.3 Otimização do Pipeline

Para melhorar o fluxo de dados durante o treinamento foram utilizados:

```python
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(
    buffer_size=AUTOTUNE
)

val_ds = val_ds.cache().prefetch(
    buffer_size=AUTOTUNE
)
```

O `cache()` reduz operações repetidas de leitura e preparação dos dados.

O `prefetch()` permite preparar novos lotes enquanto o modelo ainda processa o lote anterior.

---

# 7. Sprint 5 — Arquitetura CNN e Treinamento

## 7.1 Arquitetura

Foi construída uma CNN utilizando a API `Sequential` do Keras.

Arquitetura utilizada:

```text
Data Augmentation
        ↓
Conv2D — 32 filtros — kernel 3×3 — ReLU
        ↓
MaxPooling2D — 2×2
        ↓
Conv2D — 64 filtros — kernel 3×3 — ReLU
        ↓
MaxPooling2D — 2×2
        ↓
Flatten
        ↓
Dense — 64 neurônios — ReLU
        ↓
Dense — 1 neurônio — Sigmoid
```

Código da arquitetura:

```python
model = tf.keras.Sequential([
    data_augmentation,

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])
```

---

## 7.2 Funcionamento das camadas

### Conv2D

As camadas convolucionais são responsáveis pela extração de características espaciais das imagens.

A primeira camada utiliza 32 filtros e a segunda utiliza 64 filtros.

Esses filtros aprendem automaticamente diferentes padrões presentes nas imagens durante o treinamento.

### MaxPooling2D

As camadas de pooling reduzem a dimensão espacial das características extraídas, preservando as respostas mais relevantes.

### Flatten

A camada `Flatten` transforma os mapas de características tridimensionais em um vetor unidimensional, preparando os dados para as camadas densas.

### Dense

A camada intermediária possui 64 neurônios e combina as características extraídas pela parte convolucional.

### Sigmoid

A camada final possui um único neurônio com função de ativação `sigmoid`, adequada para a classificação binária.

As classes utilizadas são:

```text
0 → def_front
1 → ok_front
```

---

## 7.3 Parâmetros do Modelo

O modelo apresentou:

```text
Total de parâmetros: 11.963.457
Parâmetros treináveis: 11.963.457
```

Grande parte dos parâmetros está concentrada na camada `Dense` após o `Flatten`.

---

# 8. Compilação

O modelo foi compilado utilizando:

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

### Optimizer

Foi utilizado o **Adam**, conforme solicitado pela tarefa.

### Loss

Foi utilizada a **Binary Crossentropy**, adequada ao problema de classificação binária.

### Métrica

Foi utilizada a **Accuracy** para acompanhar a proporção de classificações corretas durante o treinamento.

---

# 9. Treinamento

O treinamento foi realizado durante 20 épocas:

```python
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20
)
```

Foram acompanhadas as seguintes métricas:

```text
accuracy
loss
val_accuracy
val_loss
```

---

# 10. Resultados

## 10.1 Melhor resultado de validação

O melhor resultado observado foi na **Epoch 18**:

```text
Val Accuracy: 84,23%
Val Loss:     0,3607
```

Os valores finais na Epoch 20 foram:

```text
Val Accuracy: 83,08%
Val Loss:     0,3660
```

Resultados:

| Métrica      | Melhor resultado | Última Epoch |
| ------------ | ---------------: | -----------: |
| Val Accuracy |       **84,23%** |       83,08% |
| Val Loss     |       **0,3607** |       0,3660 |

---

# 11. Auditoria de Overfitting

Foram analisadas as curvas de:

* Loss de treinamento;
* Loss de validação;
* Accuracy de treinamento;
* Accuracy de validação.

A Loss de treinamento apresentou redução geral de aproximadamente:

```text
0,96 → 0,41
```

A Loss de validação apresentou redução geral de aproximadamente:

```text
0,63 → 0,37
```

As curvas apresentaram algumas oscilações ao longo do treinamento, incluindo uma elevação temporária da `val_loss` na Epoch 17.

Entretanto, essa elevação foi seguida por recuperação na Epoch 18:

```text
Epoch 17 → val_loss ≈ 0,78
Epoch 18 → val_loss ≈ 0,36
```

Não foi observada uma divergência persistente na qual a Loss de treinamento continuasse diminuindo enquanto a Loss de validação aumentasse continuamente.

Dessa forma, considerando as 20 épocas analisadas, **não foi identificado um quadro claro e persistente de overfitting**.

O comportamento geral foi considerado predominantemente saudável, embora o modelo ainda apresente espaço para melhoria.

---

# 12. Conclusão

O projeto demonstrou a integração entre técnicas clássicas de Visão Computacional e aprendizado profundo para inspeção de peças metálicas.

Na análise exploratória, as técnicas de Grayscale, Gaussian Blur, Threshold e Canny permitiram visualizar características e defeitos presentes nas peças. As operações morfológicas também foram avaliadas, com a dilatação apresentando melhor resultado que a erosão para as imagens analisadas.

Na etapa de aprendizado profundo, o dataset foi carregado automaticamente, dividido em treinamento e validação e submetido a Data Augmentation com variações geométricas e de iluminação.

Por fim, foi desenvolvida uma CNN utilizando `Conv2D`, `MaxPooling2D`, `Flatten` e `Dense`, treinada com Adam e Binary Crossentropy.

O melhor resultado obtido na validação foi de aproximadamente **84,23% de acurácia**, com `val_loss` de **0,3607** na Epoch 18.

A análise das curvas de treinamento não indicou overfitting persistente nas 20 épocas executadas.

O projeto demonstra, portanto, um pipeline completo que parte da análise clássica das imagens e chega à classificação automatizada das peças utilizando aprendizado profundo.

---

# 13. Tecnologias utilizadas

* Python
* OpenCV
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Jupyter Notebook
* Git
* GitHub

---

# 14. Organização por Sprints

### Sprint 1 — Configuração e Versionamento

* Configuração do ambiente;
* Criação do repositório Git;
* Download e organização do dataset;
* Versionamento do projeto.

### Sprint 2 — Análise Exploratória

* Grayscale;
* Gaussian Blur;
* Análise visual das imagens.

### Sprint 3 — Destaque de Características

* Threshold;
* Canny;
* Erosão;
* Dilatação.

### Sprint 4 — Ingestão e Augmentation

* `image_dataset_from_directory`;
* Divisão treino/validação;
* Redimensionamento;
* Data Augmentation;
* Cache e Prefetch.

### Sprint 5 — CNN e Treinamento

* Conv2D;
* MaxPooling2D;
* Flatten;
* Dense;
* Sigmoid;
* Adam;
* Binary Crossentropy;
* Treinamento por 20 épocas.

### Sprint 6 — Auditoria e Entrega

* Gráfico de Loss;
* Gráfico de Accuracy;
* Análise de Overfitting;
* Documentação;
* Preparação do vídeo técnico.

---

# 15. Arquivos principais

```text
main.ipynb
```

Notebook contendo todo o desenvolvimento do projeto.

```text
funcoes.py
```

Arquivo destinado às funções auxiliares utilizadas no projeto.

```text
requirements.txt
```

Lista de dependências necessárias para execução do projeto.
