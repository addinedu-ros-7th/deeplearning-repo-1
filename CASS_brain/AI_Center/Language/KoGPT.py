import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from dataset import ChatDataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

U_TKN = '<usr>'
S_TKN = '<sys>'
MASK = '<unused0>'
SENT = '<unused1>'
tokenizer = PreTrainedTokenizerFast.from_pretrained("EasthShin/Youth_Chatbot_Kogpt2-base",
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token=MASK)

train_dataset = ChatDataset('data', tokenizer, 128)
train_Loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

model = GPT2LMHeadModel.from_pretrained('EasthShin/Youth_Chatbot_Kogpt2-base').cuda()
model.load_state_dict(torch.load('src/DL/LM/checkpoints/model.pt'))
# LoRA만 업데이트
for name, param in model.named_parameters():
    if 'LoRA' in name or 'attn' in name:
        # print(name)
        param.requires_grad = True
    else:
        param.requires_grad = False    
model.lm_head.weight.requires_grad = True


optimizer = optim.AdamW(model.parameters(), lr=1e-6)
criterion = nn.CrossEntropyLoss()
best = 100
for epoch in range(1000):
    model.train()
    total_loss = 0
    dlen = 0
    pbar = tqdm(train_Loader)
    for x_input_ids, y_input_ids in pbar:
        input_ids, token_type_ids, attention_mask = x_input_ids.to('cuda').values()
        labels = y_input_ids.to('cuda')

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, 
                token_type_ids=token_type_ids, 
                attention_mask=attention_mask, 
                labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        dlen += len(labels)
        tloss = total_loss / dlen
        pbar.set_description(f'Epoch: {epoch + 1}, Loss: {tloss}')

    if tloss < best:
        best = tloss
        print('save model_'+str(epoch))
        torch.save(model.state_dict(), 'src/DL/LM/checkpoints/model.pt')