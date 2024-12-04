from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
U_TKN = '<usr>'
S_TKN = '<sys>'
MASK = '<unused0>'
SENT = '<unused1>'
tokenizer = PreTrainedTokenizerFast.from_pretrained("EasthShin/Youth_Chatbot_Kogpt2-base",
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token=MASK)

model = GPT2LMHeadModel.from_pretrained('EasthShin/Youth_Chatbot_Kogpt2-base')

model = model.cuda()

while True:
    text = input('You : ')
    if text=='q': break
    input_ids = tokenizer.encode(text, return_tensors='pt').cuda()
    gen_ids = model.generate(
        input_ids,
        max_length=128,
        repetition_penalty=2.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        use_cache=True
    )
    generated = tokenizer.decode(gen_ids[0])
    print('koGPT :', generated)