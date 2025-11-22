import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import GenerationConfig, T5TokenizerFast
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

import sys

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0
PAD_MIN = -100

def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=0,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=0,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=0,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)

    args = parser.parse_args()
    return args

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    epochs_since_improvement = 0
    
    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir
    experiment_name = 'ft_experiment'
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(args, model, dev_loader,
                                                                         gt_sql_path, model_sql_path,
                                                                         gt_record_path, model_record_path)
        print(f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
        print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        if args.use_wandb:
            result_dict = {
                'train/loss' : tr_loss,
                'dev/loss' : eval_loss,
                'dev/record_f1' : record_f1,
                'dev/record_em' : record_em,
                'dev/sql_em' : sql_em,
                'dev/error_rate' : error_rate,
            }
            wandb.log(result_dict, step=epoch)

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False)
        if epochs_since_improvement == 0:
            save_model(checkpoint_dir, model, best=True)

        if epochs_since_improvement >= args.patience_epochs:
            break

def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    epoch_accum_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
        )['logits']

        non_pad = decoder_targets != PAD_MIN
        loss = criterion(logits[non_pad], decoder_targets[non_pad])
        loss.backward()
        optimizer.step()
        if scheduler is not None: 
            scheduler.step()

        with torch.no_grad():
            num_tokens = torch.sum(non_pad).item()
            epoch_accum_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return epoch_accum_loss / total_tokens
        
def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    You must implement the evaluation loop to be using during training. We recommend keeping track
    of the model step_loss on the SQL queries, the metrics compute_metrics returns (save_queries_and_records should be helpful)
    and the model's syntax error rate. 

    To compute non-loss metrics, you will need to perform generation with the model. Greedy decoding or beam search
    should both provide good results. If you find that this component of evaluation takes too long with your compute,
    we found the cross-entropy loss (in the evaluation set) to be well (albeit imperfectly) correlated with F1 performance.
    '''
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    tokenizer = T5TokenizerFast.from_pretrained("t5-small")

    token_loss_sum = 0.0
    token_count = 0
    generated_queries = []

    # Decoding configuration
    decode_cfg = GenerationConfig(
        max_length=512,
        num_beams=1,
    )

    with torch.no_grad():
        for enc_ids, enc_mask, dec_inp, dec_lbls, dec_init in tqdm(dev_loader):

            # Move batch to device
            enc_ids   = enc_ids.to(DEVICE)
            enc_mask  = enc_mask.to(DEVICE)
            dec_inp   = dec_inp.to(DEVICE)
            dec_lbls  = dec_lbls.to(DEVICE)
            dec_init  = dec_init.to(DEVICE)

            # -------------------------------
            #  Compute decoder-side CE loss
            # -------------------------------
            outputs = model(
                input_ids=enc_ids,
                attention_mask=enc_mask,
                decoder_input_ids=dec_inp
            )["logits"]

            not_pad_mask = (dec_lbls != -100)

            if not_pad_mask.any():
                batch_loss = loss_fn(outputs[not_pad_mask], dec_lbls[not_pad_mask])
                n_tokens = not_pad_mask.sum().item()
                token_loss_sum += batch_loss.item() * n_tokens
                token_count += n_tokens

            # -------------------------------
            #  Autoregressive generation
            # -------------------------------
            # T5 requires a single decoder start token per batch
            start_token = int(dec_init[0].item())

            seq_ids = model.generate(
                input_ids=enc_ids,
                attention_mask=enc_mask,
                decoder_start_token_id=start_token,
                generation_config=decode_cfg
            )

            # Convert token ids → text
            decoded = tokenizer.batch_decode(seq_ids, skip_special_tokens=True)
            cleaned = [txt.strip().replace("\n", " ") for txt in decoded]
            generated_queries.extend(cleaned)

    # -----------------------------------------
    # Aggregate loss across all tokens
    # -----------------------------------------
    avg_loss = token_loss_sum / token_count if token_count > 0 else 0.0

    # Save generated SQL and record outputs
    save_queries_and_records(generated_queries, model_sql_path, model_record_path)

    # Compute SQL + record metrics
    sql_em, rec_em, rec_f1, err_list = compute_metrics(
        gt_sql_pth,
        model_sql_path,
        gt_query_records=gt_record_path,
        model_query_records=model_record_path
    )

    # Error rate = fraction of invalid SQL generations
    if err_list:
        num_invalid = sum(1 for x in err_list if x is not None)
        error_rate = num_invalid / len(err_list)
    else:
        error_rate = 0.0

    return avg_loss, rec_f1, rec_em, sql_em, error_rate
        
def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    You must implement inference to compute your model's generated SQL queries and its associated 
    database records. Implementation should be very similar to eval_epoch.
    '''
    model.eval()
    tokenizer = T5TokenizerFast.from_pretrained("t5-small")

    decoded_sql = []

    # Configuration for autoregressive generation
    decode_settings = GenerationConfig(
        max_length=512,
        num_beams=1,
    )

    with torch.no_grad():
        for enc_ids, enc_mask, start_tokens in tqdm(test_loader):

            # Move batch tensors to device
            enc_ids = enc_ids.to(DEVICE)
            enc_mask = enc_mask.to(DEVICE)
            start_tokens = start_tokens.to(DEVICE)

            # Use the first start token (T5 requires a single decoder start per batch)
            start_token_id = int(start_tokens[0].item())

            # Run greedy decoding (beam search disabled by num_beams=1)
            seq_output = model.generate(
                input_ids=enc_ids,
                attention_mask=enc_mask,
                decoder_start_token_id=start_token_id,
                generation_config=decode_settings
            )

            # Convert token IDs to text
            text_batch = tokenizer.batch_decode(
                seq_output,
                skip_special_tokens=True
            )

            cleaned_output = [s.strip().replace("\n", " ") for s in text_batch]
            decoded_sql.extend(cleaned_output)

    # Save SQL + record predictions
    save_queries_and_records(decoded_sql, model_sql_path, model_record_path)

def main():
    # Get key arguments
    args = get_args()
    if args.use_wandb:
        # Recommended: Using wandb (or tensorboard) for result logging can make experimentation easier
        setup_wandb(args)
    
    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Train 
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Evaluate
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    # Dev set
    experiment_name = 'ft_experiment'
    model_type = 'ft' if args.finetune else 'scr'
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    dev_loss, dev_record_em, dev_record_f1, dev_sql_em, dev_error_rate = eval_epoch(args, model, dev_loader,
                                                                                    gt_sql_path, model_sql_path,
                                                                                    gt_record_path, model_record_path)
    print("Dev set results: Loss: {dev_loss}, Record F1: {dev_record_f1}, Record EM: {dev_record_em}, SQL EM: {dev_sql_em}")
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

    # Test set
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_test.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_test.pkl')
    test_inference(args, model, test_loader, model_sql_path, model_record_path)

if __name__ == "__main__":
    main()
