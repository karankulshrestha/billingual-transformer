import warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from dataset import BillingualDataset, casual_mask
from model import build_transformer
from torch.utils.tensorboard import SummaryWriter
from config import get_weights_file_path, get_config

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from tqdm import tqdm


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")
    
    # precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        
        # Build mask for the target decoder input
        decoder_mask = casual_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        
        # calculate the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        
        # get the next token
        prob = model.project(out[:, -1])
        # select the token with the maximum probability
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)
        
        if next_word == eos_idx:
            break
    
    return decoder_input.squeeze(0)
        


def run_Validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples=2):
    model.eval()
    count = 0
    
    source_texts = []
    expected_text = []
    predicted = []
    
    # size of the control window (just use the default value)
    control_window = 80
    
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
            
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            
            source_texts.append(source_text)
            expected_text.append(target_text)
            predicted.append(model_out_text)
            
            # print to the console
            print_msg("-"*control_window)
            print_msg(f"SOURCE: {source_text}")
            print_msg(f"TARGET: {target_text}")
            print_msg(f"PREDICTED: {model_out_text}")
            
            if count == num_examples:
                break
            



def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[EOS]", "[SOS]", "[PAD]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer



def get_ds(config):
    ds_raw = load_dataset(config['datasource'], f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')
    
    # build the tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
    
    # keep 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    
    train_ds = BillingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BillingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    
    max_len_src = 0
    max_len_tgt = 0
    
    for item in ds_raw:
        max_len_src = max(max_len_src, len(tokenizer_src.encode(item['translation'][config['lang_src']]).ids))
        max_len_tgt = max(max_len_tgt, len(tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids))
    
    print(f"Maximum langth of the source token: {max_len_src}")
    print(f"Maximum langth of the target token: {max_len_tgt}")
    
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt




def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model
    


def train_model(config):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')
    
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    #tensorboard
    writer = SummaryWriter(config['experiment_name'])
    
    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    
    initial_epoch = 0
    global_step = 0
    
    if config['pre_load']:
        model_filename = get_weights_file_path(config, config['pre_load'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1).to(device)
    
    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        
        for batch in batch_iterator:
            
            encoder_input = batch['encoder_input'].to(device) # (Batch, Sequence)
            decoder_input = batch['decoder_input'].to(device) # (Batch, Sequence)
            encoder_mask = batch['encoder_mask'].to(device) # (Batch, 1, 1, sequence)
            decoder_mask = batch['decoder_mask'].to(device) # (Batch, 1, sequence, sequence)
            
            # Run the tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask) # (Batch, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (Batch, seq_len, d_model)
            project_output = model.project(decoder_output) # (Batch, seq_len, d_model) -> (Batch, seq_len, vocab_tgt_len)
            
            label = batch['label'].to(device) # (Batch, seq_len)
            
            # (B, seq_len, tgt_vocab_size) ---> (B * seq_len, tgt_vocab_size)
            loss = loss_fn(project_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})
            
            # log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()
            
            # backpropogate
            loss.backward()
            
            # update the weights
            optimizer.step()
            optimizer.zero_grad()
            
            
            global_step += 1
        
        run_Validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
        
        # save the model
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
            
            
    
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)