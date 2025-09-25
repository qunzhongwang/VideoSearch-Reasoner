###  Installation
```bash
pip install git+ssh://git@github.com/huggingface/transformers.git@89d27fa6fff206c0153e9670ae09e2766eb75cdf
cd cs_rft_train/install
pip install -r requirements.txt
pip install wandb==0.18.3
pip install tensorboardx
pip install flash-attn --no-build-isolation
pip install git+ssh://git@github.com/cjakfskvnad/Qwen-Agent.git
pip install qwen-vl-utils
pip install torch torchvision
```