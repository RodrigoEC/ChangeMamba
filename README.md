<h1 align="center">Reprodução experimento - ChangeMamba</h1>

<h3>Reprodução do ChangeMamba: Avaliação dos Modelos
MambaBCD e MambaSCD para Detecção de Mudança</h3>



## 🔭Overview

Esse repositório é um *fork* do repositório do [**ChangeMamba**](https://ieeexplore.ieee.org/document/10565926). A partir do kit de reprodução ja estabelecido pelos autores, foram criados outros scripts para a automatização da execução do experimento, focando na execução do código e formatação das imagens, pacotes de imagens e execução de várias inferencias sequencialmente.

**📚 Documentação:**
- **Este arquivo (README.md)** - Quick start e instruções básicas
- **[TROUBLESHOOTING.md](./TROUBLESHOOTING.md)** - Guia completo de reproducibilidade, problemas comuns e soluções


## 2. Setup do Ambiente

### 2.1 Instalação

Devido ao uso de algumas dependências do código, é recomendado que ele seja executado em um ambiente **Linux**. A execução dele em algum ambiente como Windows ou MacOS pode acarretar em erros de execução.


```bash
# 1. Clone e ambiente
git clone https://github.com/RodrigoEC/ChangeMamba.git
cd ChangeMamba

conda create -n changemamba python=3.8
conda activate changemamba

# 2. Dependências (versões FIXADAS)
pip install -r requirements.txt

# 3. Kernel customizado
cd changedetection/kernels/selective_scan
pip install .
cd ../../..

# 4. Validação
python -c "import torch; print(torch.__version__)"

```
<!-- 
***Dependencies for "Detection" and "Segmentation" (optional in VMamba)***

```bash
pip install mmengine==0.10.1 mmcv==2.1.0 opencv-python-headless ftfy regex
pip install mmdet==3.3.0 mmsegmentation==1.2.2 mmpretrain==1.2.0
``` -->

### `B. Download dos pesos pré-treinados utilizados na reprodução`

Faça o Download de todos os pesos pré-treinados utilizados nos experimentos. Todos eles estão no [zenodo](https://zenodo.org/records/14037769). Uma vez instalados, coloque eles no seguinte caminho:

```bash
project_path/ChangeMamba/pretrained_weight/
```

Os pesos utilizados na reprodução feita são os seguintes

**Detecção de Mudança Binária**
- MambaBCD_Base_LEVIRCD+_F1_0.8823.pth
- MambaBCD_Small_LEVIRCD+_F1_0.8825.pth
- MambaBCD_Tiny_LEVIRCD+_F1_0.8803.pth
---
- MambaBCD_Base_SYSU_F1_0.8331.pth
- MambaBCD_Small_SYSU_F1_0.8336.pth
- MambaBCD_Tiny_SYSU_F1_0.8316.pth
---
- MambaBCD_Base_WHU_F1_0.9419.pth
- MambaBCD_Small_WHU_F1_0.9404.pth
- MambaBCD_Tiny_WHU_F1_0.9409.pth

**Detecção de Mudança Semântica**
- MambaSCD_Tiny_SECOND_SeK_0.2208.pth
- MambaSCD_Base_SECOND_SeK_0.2292.pth

### `C. Preparação dos dados`
***Binary change detection***

Os três datasets utilizados na Detecção de Mudança Binária (BCD) devem seguir a mesma estrutura a seguir:

```
${DATASET_ROOT}   # Dataset root directory, for example: /home/username/data/SYSU
├── test
│   ├── T1
│   │   ├──00001.png
│   │   ├──00002.png
│   │   ├──00003.png
│   │   ...
│   │
│   ├── T2
│   │   ├──00001.png
│   │   ... 
│   │
│   └── GT
│       ├──00001.png 
│       ...   
│   
└── test_set.txt    # Data name list, recording all the names of testing data
```

Segue os a fonte e pré-processamento feito em cada um dos três *datasets*:

- SYSU: https://github.com/liumency/SYSU-CD

- LEVIR-CD+: https://www.kaggle.com/datasets/mdrifaturrahman33/levir-cd-change-detection?resource=download

- WHU-CD: http://gpcv.whu.edu.cn/data/building_dataset.html (base de testes, seção 2.2). 

#### Pré-processamento do WHU-CD

**Por que pré-processar?**
- Dataset WHU vem como imagens `.tif` de alta resolução (~4000×4000 px)
- Modelo espera imagens de 256×256 px
- Solução: dividir em tiles (pedaços) de 256×256 px e converter para `.png`

**Como fazer:**

O repositório fornece `FileManager.split_image_into_tiles()` para automatizar isso.

**Passo 1: Organize os arquivos TIF**

Após baixar WHU-CD, você terá:
```
whu-data/
├── 2012/whole_image/test/image/2012_test.tif      (imagem T1)
├── 2016/whole_image/test/image/2016_test.tif      (imagem T2)
└── change_label/test/change_label.tif             (GT binário)
```

**Passo 2: Execute o pré-processamento**

Edite `scripts/file_manager.py` e na seção `if __name__ == "__main__":`:

```python
if __name__ == "__main__":
    # Substitua os caminhos pelos seus paths locais
    whu_data_path = "/seu/caminho/para/whu-data"
    dataset_output = "/seu/caminho/para/dataset/WHU"
    
    # Processa as 3 imagens
    FileManager.split_image_into_tiles(
        f"{whu_data_path}/2012/whole_image/test/image/2012_test.tif",
        f"{dataset_output}/T1"
    )
    
    FileManager.split_image_into_tiles(
        f"{whu_data_path}/2016/whole_image/test/image/2016_test.tif",
        f"{dataset_output}/T2"
    )
    
    FileManager.split_image_into_tiles(
        f"{whu_data_path}/change_label/test/change_label.tif",
        f"{dataset_output}/GT"
    )
```

Depois execute:
```bash
cd scripts
python file_manager.py
```

**Resultado:** 
- Você terá ~12,096 imagens de 256×256 px em cada pasta (T1, T2, GT)
- Todos em formato `.png` (pronto para o modelo)

***Semantic change detection***

Baixe o dataset SECOND pré-processado no [Zenodo](https://zenodo.org/records/14037769) e siga a seguinte estrutura:

```
${DATASET_ROOT}   # Dataset root directory, for example: /home/username/data/SECOND
├── test
│   ├── T1
│   │   ├──00001.png
│   │   ├──00002.png
│   │   ├──00003.png
│   │   ...
│   │
│   ├── T2
│   │   ├──00001.png
│   │   ... 
│   │
│   ├── GT_CD   # Binary change map
│   │   ├──00001.png 
│   │   ... 
│   │
│   ├── GT_T1   # Land-cover map of T1
│   │   ├──00001.png 
│   │   ...  
│   │
│   └── GT_T2   # Land-cover map of T2
│       ├──00001.png 
│       ...  
│   
├── train
│   ├── ...
│   ...
│ 
├── train_set.txt
└── test_set.txt
```


### `D. Treinamento do modelo`

Apesar de ser possível o treinamento do modelo usando o código desse repositório, vamos pular essa seção, uma vez que vamos utilizar os pesos pré-treinados.

### `E. Inferencia`

#### Passo 1 - Configuração da execução

Para definir qual modelo, dataset e numero de iteracoes, criei uma entidade `Config`, presente em `/scripts/config.py`. Ela será passada para a execução uma vez que ela seja definida. A entidade config pode ser definida da seguinte forma:

```python
from config import Config

config = Config(
  data_path = "/seu/caminho/dataset/test"),
  data_list_path = "/seu/caminho/dataset/test_set.txt",
  dataset_name = "nome-dataset",
  model_name = "nome-modelo",
  model_path = "/seu/caminho/model/modelo.pth",
  config_path =  "changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml", # Para cada variação do modelo (tiny, small e base existe uma config correta)
  iterations = 5,
  script_name = "infer_MambaBCD" # ou infer_MambaSCD
)
```

#### Passo 2 - Executar main script

Com uma configuração em mãos, basta instanciar e executar o nosso `MainClient`.

```python
client = MainClient(config=config)
client.run(use_batch=False) # parametro para usarmos o dataset inteiro ou batches aleatorias de até 500 imagens
```

**Pronto!** Resultados em `scripts/summary.json`


#### Passo extra caso queira xecutar varias configurações de uma vez

No arquivo `/scripts/main.py`, dentro de `if __name__ == "__main__":`, existe um for loop para a execução direta de varias configurações de execução. Ele é ótimo para colocar para rodar e não precisar ficar mudando as configurações e executando novamente o código para rodar outras inferencias para outros modelos e datasets. Só uma dica mesmo hehe.

---

## 7. Resultados

### Formato JSON (summary.json)

```json
[
  {
    "run": 1,
    "timestamp": "2026-06-18T10:30:00.123456",
    "dataset": "LEVIR-CD+",
    "model": "MambaBCD_Tiny",
    "metrics": {
      "F1": 0.8803,
      "Recall": 0.8705,
      "Precision": 0.8905
    },
    "batch": 0,
    "success": true
  },
  {
    "run": 2,
    ...
  }
]
```

### Diretórios Gerados

```
scripts/
├── batch/           # Dados temporários
├── results/         # Mapas preditos
└── summary.json     # Resultados finais
```
