import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        assert split in {"train", "dev", "test"}
        self.data_folder = data_folder
        self.split = split
        
        self.tokenizer = T5TokenizerFast.from_pretrained("t5-small")
        
        self.extra_id = self.tokenizer.convert_tokens_to_ids("<extra_id_0>")

        self.max_source_length = 128   # NL
        self.max_target_length = 768   # SQL
        self.examples = []

        self.stats_before = None
        self.stats_after = None

        self.process_data(self.data_folder, self.split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f"{split}.nl")
        if not os.path.exists(nl_path):
            raise FileNotFoundError(f"Could not find file: {nl_path}")

        raw_nls = []
        raw_sqls = []

        with open(nl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_nls.append(line)

        num_examples = len(raw_nls)

        if split in {"train", "dev"}:
            sql_path = os.path.join(data_folder, f"{split}.sql")
            if not os.path.exists(sql_path):
                raise FileNotFoundError(f"Could not find file: {sql_path}")
                
            with open(sql_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        raw_sqls.append(line)

            if len(raw_sqls) != len(raw_nls):
                raise ValueError(f"Mismatch between {split}.nl ({len(raw_nls)}) and {split}.sql ({len(raw_sqls)})")
        else:
            raw_sqls = [None] * num_examples

        if split in {"train", "dev"} and num_examples > 0:
            nl_lengths = [len(nl.split()) for nl in raw_nls]
            sql_lengths = [len(sql.split()) for sql in raw_sqls if sql]

            nl_vocab = Counter()
            sql_vocab = Counter()

            for nl in raw_nls:
                nl_vocab.update(nl.split())
            for sql in raw_sqls:
                if sql:
                    sql_vocab.update(sql.split())

            self.stats_before = {
                "Number of examples": num_examples,
                "Mean sentence length": sum(nl_lengths) / len(nl_lengths),
                "Mean SQLquery length": (
                    sum(sql_lengths) / len(sql_lengths) if sql_lengths else 0.0
                ),
                "Vocabulary size (natural language)": len(nl_vocab),
                "Vocabulary size (SQL)": len(sql_vocab),
            }

        all_encoder_token_ids = []
        all_decoder_token_ids = []

        for nl, sql in tqdm(zip(raw_nls, raw_sqls), total=num_examples, desc=f"Tokenizing {split}"):
            enc = tokenizer(
                nl,
                truncation=True,
                max_length=self.max_source_length,
            )
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]

            decoder_input_ids = None
            labels = None

            if sql is not None:
                tgt = tokenizer(
                    sql,
                    truncation=True,
                    max_length=self.max_target_length,
                )
                target_ids = tgt["input_ids"]

                decoder_input_ids = [self.extra_id] + target_ids[:-1]

                pad_id = tokenizer.pad_token_id
                labels = [(-100 if tid == pad_id else tid) for tid in target_ids]

                all_decoder_token_ids.extend(
                    [tid for tid in target_ids if tid != pad_id]
                )

            pad_id = tokenizer.pad_token_id
            all_encoder_token_ids.extend(
                [tid for tid, m in zip(input_ids, attention_mask) if m == 1 and tid != pad_id]
            )

            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "decoder_input_ids": decoder_input_ids,
                "labels": labels,
                "nl": nl,
                "sql": sql,
            })

        if split in {"train", "dev"} and len(self.examples) > 0:
            enc_lengths = []
            dec_lengths = []

            pad_id = tokenizer.pad_token_id
            for ex in self.examples:
                enc_len = sum(1 for tid in ex["input_ids"] if tid != pad_id)
                enc_lengths.append(enc_len)

                if ex["decoder_input_ids"] is not None:
                    dec_len = sum(1 for tid in ex["decoder_input_ids"] if tid != pad_id)
                    dec_lengths.append(dec_len)

            self.stats_after = {
                "Number of examples": len(self.examples),
                "Mean sentence length": (sum(enc_lengths) / len(enc_lengths)),
                "Mean SQLquery length": (sum(dec_lengths) / len(dec_lengths) if dec_lengths else 0.0),
                "Vocabulary size (natural language)": len(set(all_encoder_token_ids)),
                "Vocabulary size (SQL)": len(set(all_decoder_token_ids)),
            }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        input_ids = torch.tensor(ex["input_ids"])
        attention_mask = torch.tensor(ex["attention_mask"])

        if self.split == "test" or ex["decoder_input_ids"] is None:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "nl": ex["nl"],
                "extra_id": torch.tensor(self.extra_id)
            }

        decoder_input_ids = torch.tensor(ex["decoder_input_ids"])
        labels = torch.tensor(ex["labels"])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        }

def print_stats(ds, split):
    print(f"\n===== {split.upper()} STATISTICS =====")

    before = ds.stats_before
    after = ds.stats_after

    # BEFORE
    print("\n--- BEFORE PREPROCESSING ---")
    if before is None:
        print("No BEFORE stats available.")
    else:
        for k, v in before.items():
            print(f"{k:<35} : {v}")

    # AFTER
    print("\n--- AFTER PREPROCESSING ---")
    if after is None:
        print("No AFTER stats available.")
    else:
        for k, v in after.items():
            print(f"{k:<35} : {v}")

    print("\n============================\n")

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = []
    encoder_mask = []
    decoder_inputs = []
    decoder_targets = []
    initial_decoder_inputs = []

    for item in batch:
        encoder_ids.append(torch.as_tensor(item["input_ids"], dtype=torch.long))
        encoder_mask.append(torch.as_tensor(item["attention_mask"], dtype=torch.long))

        di = torch.as_tensor(item["decoder_input_ids"], dtype=torch.long)
        dt = torch.as_tensor(item["labels"], dtype=torch.long)

        decoder_inputs.append(di)
        decoder_targets.append(dt)

        initial_decoder_inputs.append(di[0])

    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_mask, batch_first=True, padding_value=PAD_IDX)

    decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(decoder_targets, batch_first=True, padding_value=-100)

    initial_decoder_inputs = torch.stack(initial_decoder_inputs)

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = []
    encoder_mask = []
    initial_decoder_inputs = []

    for item in batch:
        encoder_ids.append(torch.as_tensor(item["input_ids"], dtype=torch.long))
        encoder_mask.append(torch.as_tensor(item["attention_mask"], dtype=torch.long))
        initial_decoder_inputs.append(torch.as_tensor(item["extra_id"], dtype=torch.long))

    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_mask, batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = torch.stack(initial_decoder_inputs)

    return encoder_ids, encoder_mask, initial_decoder_inputs
    
def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    
    if split in {"train", "dev"}:
        print_stats(dset, split)
        
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,             # GPU transfer optimization
        persistent_workers=True      # workers stay alive across epochs
    )
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines