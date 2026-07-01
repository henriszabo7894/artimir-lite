import torch

def test_cuda_environment():
    print("=== Validation Environnement CUDA (Artimir 2.0) ===")
    
    is_available = torch.cuda.is_available()
    print(f"[AI Engine] CUDA Disponible: {is_available}")
    
    if is_available:
        device_count = torch.cuda.device_count()
        print(f"[AI Engine] Nombre de GPU(s) détectés: {device_count}")
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            print(f"  -> GPU {i} : {device_name}")
            
        current_device = torch.cuda.current_device()
        print(f"[AI Engine] GPU sélectionné par défaut: {current_device} ({torch.cuda.get_device_name(current_device)})")
        
        # Simple test tensor creation on GPU
        try:
            tensor = torch.ones(2, 2).cuda()
            print(f"[+] Test Tensor créé avec succès sur le GPU: {tensor.device}")
        except Exception as e:
            print(f"[-] Erreur lors de la création d'un test tensor: {e}")
    else:
        print("[-] CUDA n'est pas disponible. Vérifiez vos drivers NVIDIA et l'installation de PyTorch.")

if __name__ == "__main__":
    test_cuda_environment()
