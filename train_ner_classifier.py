from main import *
from model import *

# import argparse
# import pprint
# import json

# import torch
# import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from tqdm import tqdm

# from data import QADataset, Tokenizer, Vocabulary

# from model import BaselineReader
# from utils import cuda, search_span_endpoints, unpack





_TQDM_BAR_SIZE = 75
_TQDM_LEAVE = False
_TQDM_UNIT = ' batches'
_TQDM_OPTIONS = {
    'ncols': _TQDM_BAR_SIZE, 'leave': _TQDM_LEAVE, 'unit': _TQDM_UNIT
}

def _early_stop(args, eval_history):
    """
    Determines early stopping conditions. If the evaluation loss has
    not improved after `args.early_stop` epoch(s), then training
    is ended prematurely. 

    Args:
        args: `argparse` object.
        eval_history: List of booleans that indicate whether an epoch resulted
            in a model checkpoint, or in other words, if the evaluation loss
            was lower than previous losses.

    Returns:
        Boolean indicating whether training should stop.
    """
    return (
        len(eval_history) > args.early_stop
        and not any(eval_history[-args.early_stop:])
    )

def calc_ner_loss(pred_ner_type_logits, true_ner_types, weights):
    # criterion = nn.CrossEntropyLoss(ignore_index=ignored_index)

    criterion = nn.CrossEntropyLoss(weights.float())
    # weights = 0.1 if ntype == 18 else 1 for ntype in true_ner_types]
    loss = criterion(pred_ner_type_logits, true_ner_types)
    # end_loss = criterion(end_logits, end_positions)
    # loss = -torch.sum(torch.log(torch.Tensor([pred_ner_type_logits[idx,true_ner_types[idx]] for idx in range(len(true_ner_types))])))
    return loss

def make_ner_samples(batch):
    return [batch['passages'][idx,batch['start_positions'][idx]:batch['end_positions'][idx]] for idx in range(len(batch))]

def train_ner_classifier(args, epoch, model, dataset):
    # Set the model in "train" mode.
    model.train()

    # Cumulative loss and steps.
    train_loss = 0.0
    train_steps = 0

    # Set up optimizer.
    optimizer = optim.Adam(
        model.parameters(),
        lr = 0.5,
        weight_decay = 0.0
        # lr=args.learning_rate,
        # weight_decay=args.weight_decay,
    )

    # Set up training dataloader. Creates `args.batch_size`-sized
    # batches from available samples.
    train_dataloader = tqdm(
        dataset.get_batch(shuffle_examples=args.shuffle_examples),
        **_TQDM_OPTIONS,
    )

    weights = torch.zeros(19).double()

    for batch in train_dataloader:
        true_ner_types = batch['ner_types']
        weights += np.bincount(true_ner_types)

    weights = weights.double()
    weights = 1.0 / weights
    weights.clamp_(0.0001,1)
    weights = weights / weights.sum()
    print('training weights: ', weights)

    train_dataloader = tqdm(
        dataset.get_batch(shuffle_examples=args.shuffle_examples),
        **_TQDM_OPTIONS,
    )

    for batch in train_dataloader:
        # Zero gradients.
        optimizer.zero_grad()

        # Forward inputs, calculate loss, optimize model.
        # batch['']
        # print(batch)
        # print(batch['detected_answers'])
        # ner_parse_values = [nlp(sample['text']) for sample in batch['detected_answers']]
        # true_ner_types = [map_ner_label_to_idx(ner_parse.ents[0].label_) for ner_parse in ner_parse_values]
        # for ent in ner_parse.ents:
        # print(ent.text, ent.start_char, ent.end_char, ent.label_)
        true_ner_types = batch['ner_types']
        pred_ner_type_logits = model(batch)

        loss = calc_ner_loss(pred_ner_type_logits, true_ner_types, weights)
        print('loss: ', loss.item())
        loss.backward()

        if args.grad_clip > 0.:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # Update tqdm bar.
        train_loss += loss.item()
        train_steps += 1
        train_dataloader.set_description(
            f'[training NER classifier] epoch = {epoch}, loss = {train_loss / train_steps:.6f}'
        )

    return train_loss / train_steps


def evaluate_ner_classifier(args, epoch, model, dataset):
    """
    Evaluates the model for a single epoch using the development dataset.

    Args:
        args: `argparse` object.
        epoch: Epoch number (used in the `tqdm` bar).
        model: Instance of the PyTorch model.
        dataset: Development dataset.

    Returns:
        Evaluation cross-entropy loss normalized across all samples.
    """
    # Set the model in "evaluation" mode.
    model.eval()

    # Cumulative loss and steps.
    eval_loss = 0.
    eval_steps = 0

    # Set up evaluation dataloader. Creates `args.batch_size`-sized
    # batches from available samples. Does not shuffle.
    eval_dataloader = tqdm(
        dataset.get_batch(shuffle_examples=False),
        **_TQDM_OPTIONS,
    )

    weights = torch.zeros(19).double()

    for batch in eval_dataloader:
        true_ner_types = batch['ner_types']
        weights += np.bincount(true_ner_types)

    weights = weights.double()
    weights = 1.0 / weights
    weights.clamp_(0.01,1)
    weights = weights / weights.sum()
    print('test weights: ', weights)

    eval_dataloader = tqdm(
        dataset.get_batch(shuffle_examples=False),
        **_TQDM_OPTIONS,
    )

    with torch.no_grad():
        for batch in eval_dataloader:
            # Forward inputs, calculate loss.
            # start_logits, end_logits = model(batch)
            # ner_parse_values = [nlp(sample['text']) for sample in batch['detected_answers']]
            # true_ner_types = [map_ner_label_to_idx(ner_parse.ents[0].label_) for ner_parse in ner_parse_values]
            true_ner_types = batch['ner_types']

            pred_ner_type_logits = model(batch)
            
            loss = calc_ner_loss(pred_ner_type_logits, true_ner_types, weights)

            # loss = _calculate_loss(
            #     start_logits,
            #     end_logits,
            #     batch['start_positions'],
            #     batch['end_positions'],
            # )


            # Update tqdm bar.
            eval_loss += loss.item()
            eval_steps += 1
            eval_dataloader.set_description(
                f'[eval] epoch = {epoch}, loss = {eval_loss / eval_steps:.6f}'
            )

    return eval_loss / eval_steps


def main_ner(args):
    # Print arguments.
    print('\nusing arguments:')
    # _print_arguments(args)
    print()

    # Check if GPU is available.
    if not args.use_gpu and torch.cuda.is_available():
        print('warning: GPU is available but args.use_gpu = False')
        print()

    # Set up datasets.
    train_dataset = QADataset(args, args.train_path)
    dev_dataset = QADataset(args, args.dev_path)

    # Create vocabulary and tokenizer.
    vocabulary = Vocabulary(train_dataset.samples, args.vocab_size)
    tokenizer = Tokenizer(vocabulary)
    for dataset in (train_dataset, dev_dataset):
        dataset.register_tokenizer(tokenizer)
    args.vocab_size = len(vocabulary)
    args.pad_token_id = tokenizer.pad_token_id
    print(f'vocab words = {len(vocabulary)}')

    # Print number of samples.
    print(f'train samples = {len(train_dataset)}')
    print(f'dev samples = {len(dev_dataset)}')
    print()

    # Select model.
    model = NERClassifier(19, args)
    num_pretrained = model.load_pretrained_embeddings(
        vocabulary, args.embedding_path
    )
    pct_pretrained = round(num_pretrained / len(vocabulary) * 100., 2)
    print(f'using pre-trained embeddings from \'{args.embedding_path}\'')
    print(
        f'initialized {num_pretrained}/{len(vocabulary)} '
        f'embeddings ({pct_pretrained}%)'
    )
    print()

    if args.use_gpu:
        model = cuda(args, model)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'using model \'{args.model}\' ({params} params)')
    print(model)
    print()

    if args.do_train:
        # Track training statistics for checkpointing.
        eval_history = []
        best_eval_loss = float('inf')

        # Begin training.
        for epoch in range(1, args.epochs + 1):
            # Perform training and evaluation steps.
            train_loss = train_ner_classifier(args, epoch, model, train_dataset)
            
            # 'accuracy on dev'
            # dev_acc = model.predict()
            eval_loss = evaluate_ner_classifier(args, epoch, model, dev_dataset)

            # If the model's evaluation loss yields a global improvement,
            # checkpoint the model.
            eval_history.append(eval_loss < best_eval_loss)
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save(model.state_dict(), args.model_path)
            
            print(
                f'epoch = {epoch} | '
                f'train loss = {train_loss:.6f} | '
                f'eval loss = {eval_loss:.6f} | '
                f"{'saving model!' if eval_history[-1] else ''}"
            )

            # If early stopping conditions are met, stop training.
            if _early_stop(args, eval_history):
                suffix = 's' if args.early_stop > 1 else ''
                print(
                    f'no improvement after {args.early_stop} epoch{suffix}. '
                    'early stopping...'
                )
                print()
                break
    if args.do_test:
        # Write predictions to the output file. Use the printed command
        # below to obtain official EM/F1 metrics.
        # np.shuffle(dev_dataset)
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
        model.eval()

        eval_dataloader = tqdm(
            dataset.get_batch(shuffle_examples=False),
            **_TQDM_OPTIONS,
        )

        num_right = 0
        num_total = 0
        with torch.no_grad():
            for batch in eval_dataloader:
                true_ner_types = batch['ner_types']
                pred_ner_type_logits = model(batch)
                pred_ner_types = np.argmax(pred_ner_type_logits,axis=1)
                print(pred_ner_types)
                print(true_ner_types)
                print(num_right, num_total)
                print(pred_ner_types.numpy().flatten() == true_ner_types.numpy().flatten())
                num_right += np.sum(pred_ner_types.numpy().flatten() == true_ner_types.numpy().flatten())
                num_total += len(batch['ner_types'])

        print('acc: ', num_right/num_total)
        # write_predictions(args, model, dev_dataset)
        # eval_cmd = (
        #     'python3 evaluate.py '
        #     f'--dataset_path {args.dev_path} '
        #     f'--output_path {args.output_path}'
        # )
        # print()
        # print(f'predictions written to \'{args.output_path}\'')
        # print(f'compute EM/F1 with: \'{eval_cmd}\'')
        # print()

if __name__ == '__main__':
    main_ner(parser.parse_args())
