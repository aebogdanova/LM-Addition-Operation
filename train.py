import json 
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import get_linear_schedule_with_warmup


class AdditionOperatorDataset(torch.utils.data.Dataset):
    
    def __init__(self, tokenized, max_len, tokenizer):
        self.tokenized = tokenized
        self.max_len = max_len
        self.eos_idx = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        
    def __len__(self):
        return len(self.tokenized)
    
    def __getitem__(self, index):
        sequence = self.tokenized[index]
        sequence += [self.eos_idx] * (self.max_len - len(sequence))
        return torch.tensor(sequence).long()

def load_dataset(file_path):
	with open(file_path, "r", encoding="utf-8") as datas_file:
	    datas = datasfile.readlines()
	return datas

def tokenize(datas, tokenizer):
	return [tokenizer.encode(data) for data in datas]

def get_max_len(tokenized_datas):
	lengths = [len(data) for data in tokenized_datas]
	return max(lengths)

def pack_tensor(new_tensor, packed_tensor, max_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None

def train(train_loader, model, epochs, batch_size, device):

    losses = []
    perplexities = []
    best_loss = 1e+6

    model.train()

    accumulating_batch_count = 0
    input_tensor = None

    for epoch in range(epochs):

        print(f"Epoch: {epoch + 1}")
        epoch_losses = []

        for idx, sequence in tqdm(enumerate(train_loader)):
            input_tensor, carry_on, remainder = pack_tensor(sequence, input_tensor, 768)
            if carry_on and idx != len(train_loader) - 1:
                continue
            input_tensor = input_tensor.to(device)
            outputs = model(input_tensor, labels=input_tensor)
            loss = outputs[0]
            loss.backward()
            epoch_losses.append(loss.item())

            if (accumulating_batch_count % batch_size) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            accumulating_batch_count += 1
            input_tensor = None

        mean_loss = np.mean(epoch_losses)
        losses.append(mean_loss)
        perplexities.append(np.exp(mean_loss))

        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model.state_dict(), "./model/state_dict_model.pth")

        with open("./model/info.json", "w") as info_file:
                info = {
                    'losses': losses,
                    'perplexities': perplexities
                }
                info_file.write(json.dumps(info, indent=2))


if __name__ == "__main__":
	
	DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	MODEL_NAME = "gpt2"
	LEARNING_RATE = 2e-5
	EPOCHS = 150
	BATCH_SIZE = 20

	SEED = 42
	torch.manual_seed(SEED)
	torch.cuda.manual_seed(SEED)

	tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
	model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

	train_data = load_datas("./data/train.txt")
	tokenized = tokenize(train_data, tokenizer)
	max_len = get_max_len(tokenized)

	train_dataset = AdditionOperatorDataset(
		tokenized=tokenized, max_len=max_len, tokenizer=tokenizer
	)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
	
	num_training_steps = int(len(train_loader) * EPOCHS)
	warmup_steps = int(num_training_steps * 0.1)
	optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
	scheduler = get_linear_schedule_with_warmup(
	    optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
	)

	train(
		train_loader=train_loader,
		model=model,
		epochs=EPOCHS,
		batch_size=BATCH_SIZE,
		device=DEVICE
	)
