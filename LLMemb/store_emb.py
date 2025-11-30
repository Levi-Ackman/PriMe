import torch
import sys
import os
import time
import h5py
import argparse
from torch.utils.data import DataLoader
from data_provider.data_loader_emb import APAVALoader, ADFTDLoader, PTBLoader,PTBXLLoader, TDBRAINLoader,MIMICIVLoader
from gen_prompt_emb import GenPromptEmb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--data", type=str, default="APAVA")
    parser.add_argument("--root_path", type=str, default="./dataset/APAVA/")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--divide", type=str, default="train")
    parser.add_argument("--num_workers", type=int, default=min(32, os.cpu_count()))
    return parser.parse_args()

def get_dataset(data,root_path, flag):
    datasets = {
        'APAVA': APAVALoader,
        'ADFTD': ADFTDLoader,
        'PTB': PTBLoader,
        'PTB-XL': PTBXLLoader,
        'TDBRAIN': TDBRAINLoader,
        'MIMIC': MIMICIVLoader,
    }
    dataset_class = datasets[data]
    return dataset_class(root_path=root_path,flag=flag)

def save_embeddings(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_set = get_dataset(args.data,args.root_path, 'train')
    test_set = get_dataset(args.data,args.root_path, 'test')
    val_set = get_dataset(args.data,args.root_path, 'val')

    data_loader = {
        'train': DataLoader(train_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers),
        'test': DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers),
        'val': DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    }[args.divide]

    gen_prompt_emb = GenPromptEmb(
        device=device, # type: ignore
        data=args.data,
        model_name=args.model_name,
        d_model=args.d_model,
    ).to(device)

    save_path = f"./Embeddings/{args.data}/{args.divide}/"
    os.makedirs(save_path, exist_ok=True)

    for i, (x, y) in enumerate(data_loader):
        embeddings = gen_prompt_emb.generate_embeddings(x.to(device))

        file_path = f"{save_path}{i}.h5"
        with h5py.File(file_path, 'w') as hf:
            hf.create_dataset('embeddings', data = embeddings.cpu().numpy())
    
if __name__ == "__main__":
    args = parse_args()
    t1 = time.time()
    save_embeddings(args)
    t2 = time.time()
    print(f"Total time spent: {(t2 - t1)/60:.4f} minutes")