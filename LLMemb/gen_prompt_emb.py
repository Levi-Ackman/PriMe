import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model
import torch.nn.functional as F

local_dir='/data/gqyu/alib/weight/gpt2_local'

class GenPromptEmb(nn.Module):
    def __init__(
        self,
        data = 'APAVA',
        model_name = "gpt2",
        device = 'cuda:0',
        d_model = 768,
    ):  
        super(GenPromptEmb, self).__init__()
        self.data = data
        self.device = device
        self.model_name = model_name
        self.d_model = d_model
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(local_dir)
        self.model = GPT2Model.from_pretrained(local_dir).to(self.device)

    def _prepare_prompt(self, input_template, in_data, i, j):
        # Time series value
        values = in_data[i, :, j].flatten().tolist()
        values_str = ", ".join([str(int(value)) for value in values])

        # Prompt
        in_prompt           = input_template.replace("value1, ..., valuen", values_str)

        tokenized_prompt = self.tokenizer.encode(in_prompt, return_tensors="pt").to(self.device)
        return tokenized_prompt

    def forward(self, tokenized_prompt):
        with torch.no_grad():
            prompt_embeddings = self.model(tokenized_prompt).last_hidden_state
        return prompt_embeddings

    def generate_embeddings(self, in_data):
            input_templates = {
                'APAVA': "The values: value1, ..., valuen are EEG signals with a frequency of 256 Hz.",
                'ADFTD': "The values: value1, ..., valuen are EEG signals with a frequency of 256 Hz.",
                'TDBRAIN': "The values: value1, ..., valuen are EEG signals with a frequency of 256 Hz.",
                'PTB': "The values: value1, ..., valuen are ECG signals with a frequency of 250 Hz.",
                'PTB-XL': "The values: value1, ..., valuen are ECG signals with a frequency of 250 Hz.",
                'MIMIC': "The values: value1, ..., valuen are ECG signals with a frequency of 250 Hz.",
            }

            input_template = input_templates[self.data]
            
            tokenized_prompts = []
            max_token_count = 0
            for i in range(len(in_data)):
                for j in range(in_data.shape[2]):
                    tokenized_prompt = self._prepare_prompt(input_template, in_data, i, j).to(self.device)
                    max_token_count = max(max_token_count, tokenized_prompt.shape[1])
                    tokenized_prompts.append((i, tokenized_prompt.to(self.device), j))

            in_prompt_emb = torch.zeros((len(in_data), max_token_count, self.d_model, in_data.shape[2]), dtype=torch.float32, device=self.device)

            for i, tokenized_prompt, j in tokenized_prompts:
                prompt_embeddings = self.forward(tokenized_prompt)
                padding_length = max_token_count - tokenized_prompt.shape[1]
                if padding_length > 0:
                    last_token_embedding = prompt_embeddings[:, -1, :].unsqueeze(1)
                    padding = last_token_embedding.repeat(1, padding_length, 1)
                    prompt_embeddings_padded = torch.cat([prompt_embeddings, padding], dim=1)
                else:
                    prompt_embeddings_padded = prompt_embeddings
                        
                in_prompt_emb[i, :max_token_count, :, j] = prompt_embeddings_padded
                last_token_emb = in_prompt_emb[:, max_token_count-1:max_token_count, :, :]
                last_token_emb = last_token_emb.squeeze()

            return last_token_emb