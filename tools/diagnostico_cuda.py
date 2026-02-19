import torch
import sys

print("--- Diagnóstico PyTorch/CUDA ---")
print(f"Versão do Python: {sys.version}")
print(f"Versão do PyTorch: {torch.__version__}")

cuda_available = torch.cuda.is_available()
print(f"CUDA disponível: {cuda_available}")

if cuda_available:
    print(f"Número de GPUs detectadas: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"GPU atual: {torch.cuda.current_device()}")
    print(f"Memória total da GPU 0: {torch.cuda.get_device_properties(0).total_memory // (1024**2)} MB")
    # Teste simples de tensor na GPU
    try:
        x = torch.rand(3, 3).to('cuda')
        print("Tensor criado na GPU com sucesso:", x)
    except Exception as e:
        print("Erro ao criar tensor na GPU:", e)
else:
    print("Nenhuma GPU CUDA detectada ou PyTorch sem suporte a CUDA.")
    print("Sugestão: verifique drivers, instalação do CUDA Toolkit e se o PyTorch foi instalado com suporte a CUDA.")
