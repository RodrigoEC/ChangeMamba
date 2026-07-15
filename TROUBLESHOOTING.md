# Troubleshooting - Inferências ChangeMamba

Guia rápido para **resolver problemas comuns** ao executar experimentos com `scripts/main.py`.

---

## 1. Problemas de Instalação

### Issue: "ModuleNotFoundError: No module named 'torch'"

**Solução:**
```bash
pip install -r requirements.txt
```

### Issue: "NameError: name 'selective_scan_cuda_oflex' is not defined"

**Solução:** O kernel customizado não foi compilado.

```bash
cd changedetection/kernels/selective_scan
pip install .
cd ../../..
```

### Issue: "CUDA out of memory"

**Solução:** Reduza batch size ou crop size em `scripts/config.py`:

```python
@dataclass
class Config:
    batch_size: int = 8      # reduzir de 16
    crop_size: int = 128     # reduzir de 256
```

### Issue: "No module named 'selective_scan'"

**Solução:** Kernels não instalado. Execute:

```bash
cd changedetection/kernels/selective_scan
pip install .
```

---

## 2. Problemas de Configuração

### Issue: "FileNotFoundError: No such file or directory"

**Verificar:**
- [ ] Paths em `scripts/config.py` são absolutos (não relativos)?
- [ ] Arquivos existem? `ls /seu/caminho/dataset/test/T1`
- [ ] Estrutura de pastas está correta? (T1, T2, GT)

**Solução:**
```python
# ✓ Correto (absoluto)
data_path: str = "/Users/rodrigoec/datasets/LEVIR-CD+/test"

# ✗ Errado (relativo)
data_path: str = "../datasets/LEVIR-CD+/test"
```

### Issue: "Dataset structure is invalid"

**Verificar estrutura:**
```bash
ls /seu/dataset/test/
# Deve conter: T1  T2  GT  test_set.txt

ls /seu/dataset/test/T1/ | head -5
# Deve conter imagens: 00001.png  00002.png  ...
```

### Issue: "Config path not found"

**Verificar:**
```bash
ls changedetection/configs/vssm1/
# Modelos disponíveis:
# - vssm_tiny_224_0229flex.yaml
# - vssm_small_224.yaml
# - vssm_base_224.yaml
```

**Solução:** Use path relativo correto:
```python
config_path: str = "changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml"
```

---

## 3. Problemas de Modelo

### Issue: "F1 Score muito baixo (0.72 ao invés de esperado 0.88)"

**Causa mais comum:** Usando encoder ao invés de modelo completo.

**Verificar em `scripts/main.py` linha 71:**

```python
# ✓ CORRETO (modelo completo)
'--model_checkpoint_path', self.config.model_path

# ✗ ERRADO (só encoder)
'--encoder_pretrained_path', self.config.model_path
```

**Se arquivo é encoder:** Use `--encoder_pretrained_path` mas terá F1 ruim (esperado)

### Issue: "Model expects different input size"

**Verificar alinhamento:**

| Modelo | Config YAML | Crop Size |
|--------|-----------|-----------|
| MambaBCD-Tiny | vssm_tiny_224_0229flex.yaml | 256 |
| MambaBCD-Small | vssm_small_224.yaml | 256 |
| MambaBCD-Base | vssm_base_224.yaml | 256 |
| MambaSCD-Tiny | vssm_tiny_224_0229flex.yaml | 512 |
| MambaSCD-Base | vssm_base_224.yaml | 512 |

**Solução:** Certifique que config_path corresponde ao modelo.

### Issue: "Model file corrupted"

**Solução:** Re-download do Zenodo:
```bash
wget https://zenodo.org/records/14037769
# Extrai arquivo .pth
mv modelo.pth pretrained_weight/
```

---

## 4. Problemas de Reproducibilidade

### Issue: "Resultados variam muito entre runs (>1%)"

**Causa:** Seed não está sendo aplicado.

**Solução:** Adicione seed em `changedetection/script/infer_MambaBCD.py`:

```python
import torch
import numpy as np
import random

# No início do script
seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### Issue: "Diferentes resultados em diferentes máquinas"

**Causas normais:**
- GPU diferente (NVIDIA vs AMD)
- CUDA version diferente
- PyTorch version diferente

**Para minimizar variação:**
```bash
# Use as versões fixadas
pip install -r requirements.txt

# Verifique PyTorch
python -c "import torch; print(torch.__version__)"

# Verifique CUDA
nvidia-smi
```

### Issue: "Variação esperada > 1%"

**Checklist:**
- [ ] Seed = 1234 em `config.py`?
- [ ] `num_workers = 0` em DataLoader?
- [ ] CUDA deterministic mode ativado?
- [ ] Batch size é fixo?

---

## 5. Problemas de Dados

### Issue: "test_set.txt vazio ou inválido"

**Validação:**
```bash
# Arquivo deve ter lista de nomes, um por linha
cat /seu/dataset/test_set.txt | head -5
# Saída esperada:
# 00001
# 00002
# 00003

# Deve corresponder aos arquivos
ls /seu/dataset/test/T1/ | wc -l
wc -l /seu/dataset/test_set.txt
# Números devem bater
```

### Issue: "Imagens T1 e T2 têm tamanhos diferentes"

**Verificação:**
```bash
# T1 e T2 devem ter mesmo número de arquivos
ls /seu/dataset/test/T1/ | wc -l
ls /seu/dataset/test/T2/ | wc -l
# Devem ser iguais

# GT deve ter mesmo número
ls /seu/dataset/test/GT/ | wc -l
```

**Solução:** Recria dataset seguindo estrutura correta.

### Issue: "Dataset SECOND com estrutura inválida"

**Para SECOND, a estrutura é diferente:**
```
test/
├── T1/
├── T2/
├── GT_CD/        # ← Mudança binária (NÃO GT)
├── GT_T1/        # ← Ground truth semântico T1
├── GT_T2/        # ← Ground truth semântico T2
├── GT_T1_COLORED/
├── GT_T2_COLORED/
└── test_set.txt
```

**Solução:** Use dataset pré-processado do Zenodo:
https://zenodo.org/records/14037769

---

## 6. Problemas de Execução

### Issue: "Script trava/não termina"

**Verificação:**
```bash
# Quantas imagens tem no dataset?
ls /seu/dataset/test/T1/ | wc -l

# Se > 10,000 imagens, pode demorar muito
# Use batches em config.py:
num_batches: int = 5      # divide em 5 lotes
images_per_batch: int = 500
```

### Issue: "Timeout após 1 hora"

**Causa:** Dataset muito grande ou modelo lento.

**Solução:**
```python
# scripts/config.py
iterations: int = 1        # Reduza de 5 para 1
batch_size: int = 8        # Reduza de 16
```

### Issue: "Resultados não salvam em summary.json"

**Verificação:**
```bash
# Arquivo deve existir e ter conteúdo
ls -lh scripts/summary.json

# Conteúdo deve ser JSON válido
python -m json.tool scripts/summary.json
```

**Se vazio:** Rerun script com debug:
```bash
cd scripts
python main.py 2>&1 | tee debug.log
```

---

## 7. Problemas de WHU-CD Preprocessing

### Issue: "split_image_into_tiles não funciona"

**Verificação:**
```bash
# Arquivo TIF existe?
ls -lh /seu/caminho/2012_test.tif

# Pode ler como imagem?
python -c "from PIL import Image; Image.open('/seu/caminho/2012_test.tif')"
```

### Issue: "Sem espaço em disco durante preprocessing"

**Problema:** Imagens de alta resolução (~4000×4000 px) geram ~12,000 tiles.

**Estimativa de espaço:**
- T1: ~300 MB (12,096 imagens de 256×256 px)
- T2: ~300 MB
- GT: ~300 MB
- **Total: ~1 GB por dataset**

**Solução:** Libere espaço ou use SSD para preprocessamento.

### Issue: "PNG converter perdendo informação"

**Se problemas com qualidade:**
```python
# scripts/file_manager.py - linha 39
image.convert("RGB").save(png_path, "PNG")

# Para preservar melhor (lossless):
image.save(png_path, "PNG")  # Remove .convert() se image já é RGB
```

---

## 8. Problemas de Output

### Issue: "Pasta results/ vazia"

**Verificação:**
```bash
ls scripts/results/
# Deve conter mapas preditos

# Se vazia: verificar logs
cat scripts/summary.json | grep success
# Se "success": false, há erro nas métricas
```

### Issue: "summary.json com formato estranho"

**Validação:**
```bash
python -m json.tool scripts/summary.json

# Deve conter:
# - "run": número
# - "metrics": F1, Recall, Precision
# - "success": true/false
```

**Se erro:** Arquivo corrupto, delete e rerun:
```bash
rm scripts/summary.json
python main.py
```

---

## 9. Perguntas Frequentes

**P: Posso rodar em Windows/MacOS?**  
R: Não recomendado. Kernel `selective_scan` é compilado para Linux. Possível em WSL2 ou Docker.

**P: Preciso da GPU?**  
R: Sim. CPU é muito lento. GPU com 24GB+ VRAM recomendada.

**P: Quanto tempo leva?**  
R: ~5-10 min por dataset (5 runs), depende do tamanho do dataset e GPU.

**P: Posso usar modelo diferente?**  
R: Sim, desde que baixe de Zenodo e coloque em `pretrained_weight/`.

**P: O que é "batch" em config.py?**  
R: Divisão aleatória do dataset em subconjuntos menores. Use `use_batches=False` para usar tudo.

---

## 10. Reportar Problema

Se encontrar erro não listado, inclua:

```bash
# 1. Versões
python --version
python -c "import torch; print(torch.__version__)"
nvidia-smi

# 2. Arquivo de log
cd scripts && python main.py 2>&1 | tee debug.log
cat debug.log  # Copie saída completa

# 3. Config.py
cat scripts/config.py

# 4. Estrutura do dataset
ls -la /seu/dataset/test/
```

---

## Changelog

| Data | Alteração |
|------|-----------|
| 2026-07-13 | Criação - Guia de troubleshooting |
