<div align="center">
  <h2><b> Code for Paper:</b></h2>
  <h2><b> Signals Meet Sentences: Medical Time Series Classification with Integrated LLM Priors </b></h2>
</div>

## Get Started

1. Install requirements. ```pip install -r requirements.txt```
2. Download data. You can download *APAVA, ADFTD, TDBrain, PTB, and PTB-XL* from [**Medformer**](https://github.com/DL4mHealth/Medformer). **All the datasets are well pre-processed** *(except for the TDBrain dataset, which requires permission first)* and can be used easily thanks to their efforts. For the *MIMIC* dataset, we refer to [**Cardioformer**](https://github.com/KMobin555/Cardioformer). Then, place all datasets under a folder ```./dataset```. Here, we provide an example, i.e., **APAVA** dataset.
3. Get the LLM embedding. You can use comments below './scripts/get_llm_emb' to get the LLM embeddings of all datasets. Such as ```bash scripts/get_llm_emb/APAVA.sh```
4. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`. Such as ```bash ./scripts/APAVA.sh ``` to get the result of  **APAVA**. You can find the training history and results under the './logs' folder.

## Acknowledgement

This project is constructed based on the code in repo [**Medformer**](https://github.com/DL4mHealth/Medformer) and [**Cardioformer**](https://github.com/KMobin555/Cardioformer).
Thanks a lot for their amazing work!

***Please also star their project and cite their paper if you find this repo useful.***
```
@article{wang2024medformer,
  title={Medformer: A multi-granularity patching transformer for medical time-series classification},
  author={Wang, Yihe and Huang, Nan and Li, Taida and Yan, Yujun and Zhang, Xiang},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={36314--36341},
  year={2024}
}
```
```
@misc{mobin2025cardioformeradvancingaiecg,
      title={Cardioformer: Advancing AI in ECG Analysis with Multi-Granularity Patching and ResNet}, 
      author={Md Kamrujjaman Mobin and Md Saiful Islam and Sadik Al Barid and Md Masum},
      year={2025},
      eprint={2505.05538},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.05538}, 
}
```


