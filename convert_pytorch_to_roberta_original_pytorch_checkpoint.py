# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert huggingface RoBERTa checkpoint to fairseq roberta."""


import argparse
import logging
import pathlib

import fairseq
import torch
from fairseq.models.roberta import RobertaModel as FairseqRobertaModel
from fairseq.modules import TransformerSentenceEncoderLayer
from packaging import version
import numpy as np
import os

from transformers.modeling_bert import BertIntermediate, BertLayer, BertOutput, BertSelfAttention, BertSelfOutput
from transformers.modeling_roberta import RobertaConfig, RobertaForMaskedLM, RobertaForSequenceClassification
from transformers import RobertaTokenizer

if version.parse(fairseq.__version__) < version.parse("0.8.0"):
    raise Exception("requires fairseq >= 0.8.0")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_TEXT = "Hello world! cécé herlolip"


def convert_pytorch_to_roberta_checkpoint(
    pytorch_checkpoint_path: str, roberta_dump_folder_path: str
):
    """
    Copy/paste/tweak roberta's weights to our BERT structure.
    """
    import pickle
    model = RobertaForMaskedLM.from_pretrained(pytorch_checkpoint_path)
    config = RobertaConfig.from_pretrained(pytorch_checkpoint_path)
    from argparse import Namespace

    huggingface_train_args = Namespace(**vars(torch.load(f"{pytorch_checkpoint_path}/training_args.bin")))
    model.eval()  # disable dropout

    # tokenizer = RobertaTokenizer.from_pretrained(roberta_checkpoint_path)
    if config.num_hidden_layers == 12:
        roberta = FairseqRobertaModel.from_pretrained("roberta.base")
    elif config.num_hidden_layers == 24:
        roberta = FairseqRobertaModel.from_pretrained("roberta.large")
    else:
        raise Exception("Only roberta LM is supported!")
    roberta.eval()
    # roberta_sent_encoder = roberta.model.decoder.sentence_encoder

    # update config from huggingface and reuse lots of settings from fairseq pretrained
    roberta.args.warmup_updates = huggingface_train_args.warmup_steps
    roberta.args.weight_decay = huggingface_train_args.weight_decay
    roberta.args.adam_eps = huggingface_train_args.adam_epsilon
    roberta.args.clip_norm = huggingface_train_args.max_grad_norm
    roberta.args.max_update = huggingface_train_args.max_steps
    roberta.args.total_num_update = huggingface_train_args.max_steps
    roberta.args.save_interval_updates = huggingface_train_args.save_steps

    roberta.args.attention_dropout = config.attention_probs_dropout_prob
    roberta.args.encoder_embed_dim = config.hidden_size
    roberta.args.encoder_ffn_embed_dim = config.intermediate_size
    roberta.args.activation_fn = config.hidden_act
    roberta.args.activation_dropout = config.hidden_dropout_prob
    roberta.args.encoder_layers = config.num_hidden_layers
    roberta.args.encoder_attention_heads = config.num_attention_heads
    roberta.args.__dict__.update(huggingface_train_args.__dict__)

    roberta.model.decoder.sentence_encoder.embed_tokens.weight = model.roberta.embeddings.word_embeddings.weight
    roberta.model.decoder.sentence_encoder.embed_positions.weight = model.roberta.embeddings.position_embeddings.weight
    model.roberta.embeddings.token_type_embeddings.weight.data = torch.zeros_like(
        model.roberta.embeddings.token_type_embeddings.weight
    )  # just zero them out b/c RoBERTa doesn't use them.
    roberta.model.decoder.sentence_encoder.emb_layer_norm.weight = model.roberta.embeddings.LayerNorm.weight
    roberta.model.decoder.sentence_encoder.emb_layer_norm.bias = model.roberta.embeddings.LayerNorm.bias

    for i in range(config.num_hidden_layers):
        # Encoder: start of layer
        layer: BertLayer = model.roberta.encoder.layer[i]
        # roberta.model.decoder.sentence_encoder.layers[i]: TransformerSentenceEncoderLayer = roberta.model.decoder.sentence_encoder.layers[i]

        # self attention
        self_attn: BertSelfAttention = layer.attention.self
        assert (
            roberta.model.decoder.sentence_encoder.layers[i].self_attn.k_proj.weight.data.shape
            == roberta.model.decoder.sentence_encoder.layers[i].self_attn.q_proj.weight.data.shape
            == roberta.model.decoder.sentence_encoder.layers[i].self_attn.v_proj.weight.data.shape
            == torch.Size((config.hidden_size, config.hidden_size))
        )

        roberta.model.decoder.sentence_encoder.layers[i].self_attn.q_proj.weight = self_attn.query.weight
        roberta.model.decoder.sentence_encoder.layers[i].self_attn.q_proj.bias = self_attn.query.bias
        roberta.model.decoder.sentence_encoder.layers[i].self_attn.k_proj.weight = self_attn.key.weight
        roberta.model.decoder.sentence_encoder.layers[i].self_attn.k_proj.bias = self_attn.key.bias
        roberta.model.decoder.sentence_encoder.layers[i].self_attn.v_proj.weight = self_attn.value.weight
        roberta.model.decoder.sentence_encoder.layers[i].self_attn.v_proj.bias = self_attn.value.bias

        # self-attention output
        self_output: BertSelfOutput = layer.attention.output
        assert self_output.dense.weight.shape == roberta.model.decoder.sentence_encoder.layers[i].self_attn.out_proj.weight.shape
        roberta.model.decoder.sentence_encoder.layers[i].self_attn.out_proj.weight = self_output.dense.weight
        roberta.model.decoder.sentence_encoder.layers[i].self_attn.out_proj.bias = self_output.dense.bias
        roberta.model.decoder.sentence_encoder.layers[i].self_attn_layer_norm.weight = self_output.LayerNorm.weight
        roberta.model.decoder.sentence_encoder.layers[i].self_attn_layer_norm.bias = self_output.LayerNorm.bias

        # intermediate
        intermediate: BertIntermediate = layer.intermediate
        assert intermediate.dense.weight.shape == roberta.model.decoder.sentence_encoder.layers[i].fc1.weight.shape
        roberta.model.decoder.sentence_encoder.layers[i].fc1.weight = intermediate.dense.weight
        roberta.model.decoder.sentence_encoder.layers[i].fc1.bias = intermediate.dense.bias

        # output
        bert_output: BertOutput = layer.output
        assert bert_output.dense.weight.shape == roberta.model.decoder.sentence_encoder.layers[i].fc2.weight.shape
        roberta.model.decoder.sentence_encoder.layers[i].fc2.weight = bert_output.dense.weight
        roberta.model.decoder.sentence_encoder.layers[i].fc2.bias = bert_output.dense.bias
        roberta.model.decoder.sentence_encoder.layers[i].final_layer_norm.weight = bert_output.LayerNorm.weight
        roberta.model.decoder.sentence_encoder.layers[i].final_layer_norm.bias = bert_output.LayerNorm.bias

    # LM Head
    roberta.model.decoder.lm_head.dense.weight = model.lm_head.dense.weight
    roberta.model.decoder.lm_head.dense.bias = model.lm_head.dense.bias
    roberta.model.decoder.lm_head.layer_norm.weight = model.lm_head.layer_norm.weight
    roberta.model.decoder.lm_head.layer_norm.bias = model.lm_head.layer_norm.bias
    roberta.model.decoder.lm_head.weight = model.lm_head.decoder.weight
    roberta.model.decoder.lm_head.bias = model.lm_head.decoder.bias

    input_ids: torch.Tensor = roberta.encode(SAMPLE_TEXT).unsqueeze(0)  # batch of size 1
    their_output = model(input_ids)[0]
    our_output = roberta.model(input_ids)[0]

    print(our_output.shape, their_output.shape)
    max_absolute_diff = torch.max(torch.abs(our_output - their_output)).item()
    print(f"max_absolute_diff = {max_absolute_diff}")  # ~ 1e-7
    copy_success = torch.allclose(our_output, their_output, atol=1e-3)
    print("Do both models output the same tensors?", "🔥" if copy_success else "💩")
    if not copy_success:
        raise Exception("Something went wRoNg")

    pathlib.Path(roberta_dump_folder_path).mkdir(parents=True, exist_ok=True)
    print(f"Saving model to {roberta_dump_folder_path}")
    from fairseq import checkpoint_utils
    state_dict = {
        "args": roberta.args,
        "model": roberta.model.state_dict(),
        # these last two were copied from fairseq pretrained just to make .from_pretrain() function works
        "extra_state": {'train_iterator': {'epoch': 0}, 'val_loss': 1.4955725940408326},
        "optimizer_history": [{'criterion_name': 'MaskedLmLoss',
                               'optimizer_name': 'MemoryEfficientFP16Optimizer',
                               'lr_scheduler_state': {'best': 1.495530066777925},
                               'num_updates': 500000}]
    }
    from fairseq import checkpoint_utils
    # checkpoint_utils.save_state(f"{roberta_dump_folder_path}/model.pt", roberta.args, roberta.state_dict(), )
    # del model
    checkpoint_utils.torch_persistent_save(state_dict, f"{roberta_dump_folder_path}/model.pt")
    loaded_model = FairseqRobertaModel.from_pretrained(roberta_dump_folder_path)
    loaded_model.eval()

    # roberta.model(input_ids)
    # loaded_model.model(input_ids)

    del state_dict
    copied_dict = roberta.state_dict()
    loaded_dict = loaded_model.state_dict()
    assert loaded_model.state_dict().keys() == roberta.state_dict().keys()
    for k in roberta.state_dict().keys():
        loaded_val = loaded_dict[k]
        copied_val = copied_dict[k]
        if not torch.allclose(loaded_val, copied_val, atol=1e-3):
            print(k)
    loaded_output = loaded_model.model(input_ids)[0]
    save_success = torch.allclose(our_output, loaded_output, atol=1e-3)
    print("Do both models output the same tensors?", "🔥" if save_success else "💩")
    if not save_success:
        raise Exception("Something went wRoNg")
    # except:
    #     print("Fail to save")
    # torch.save(roberta, f"{roberta_dump_folder_path}/model.pt")
    print("Done")


def parse_range(string: str):
    """Return a tuple that represent the range, inclusive on both sides"""
    import itertools
    def h(elem):
        if "-" in elem:
            pair = elem.split("-")
            assert len(pair) == 2
            lower, upper = int(pair[0]), int(pair[1])
            return list(range(lower, upper + 1))
        else:
            assert elem.isdigit()
            return [int(elem)]

    string = string.strip()
    lst = []
    if string == "None":
        return lst
    assert '[' == string[0] and ']' == string[-1]
    string = string[1:-1]
    if "," not in string:
        lst = h(string)
    else:
        elems = string.split(",")
        lsts = [h(elem.strip()) for elem in elems if elem != ""]
        lst = list(set(itertools.chain(*lsts)))
    return sorted(lst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--pytorch_checkpoint_root", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--roberta_dump_folder_root", default=None, type=str, required=True, help="Path the official PyTorch dump."
    )
    parser.add_argument(
        "--save_every", default=None, type=int, required=False, help="Path the official PyTorch dump."
    )
    parser.add_argument(
        "--chkpts", default=None, type=str, required=False, help="Path the official PyTorch dump."
    )
    parser.add_argument(
        "--dict_folder", default=None, type=str, required=True, help="Path the official PyTorch dump."
    )
    # parser.add_argument(
    #     "--classification_head", action="store_true", help="Whether to convert a final classification head."
    # )
    args = parser.parse_args()
    if args.chkpts is not None and args.chkpts.lower() != "none":
        assert "[" in args.chkpts and "]" in args.chkpts
    import numpy as np
    import os

    chkpts = np.array(parse_range(args.chkpts)).astype(int)
    if args.save_every is not None:
        assert type(args.save_every) is int
        chkpts *= args.save_every

    if not os.path.exists(args.roberta_dump_folder_root):
        os.makedirs(args.roberta_dump_folder_root)

    from shutil import copy
    if len(chkpts) == 0:
        convert_pytorch_to_roberta_checkpoint(
            args.pytorch_checkpoint_root,
            args.roberta_dump_folder_root
        )
    else:
        assert os.path.exists(args.dict_folder)
        assert os.path.exists(f"{args.dict_folder}/dict.txt")
        for chkpt in chkpts:
            assert os.path.exists(f"{args.pytorch_checkpoint_root}/checkpoint-{chkpt}")

        for chkpt in chkpts:
            print(f"Checkpoint: {chkpt}")
            if not os.path.exists(f"{args.roberta_dump_folder_root}/checkpoint-{chkpt}"):
                os.makedirs(f"{args.roberta_dump_folder_root}/checkpoint-{chkpt}")
            copy(f"{args.dict_folder}/dict.txt", f"{args.roberta_dump_folder_root}/checkpoint-{chkpt}")

            convert_pytorch_to_roberta_checkpoint(
                f"{args.pytorch_checkpoint_root}/checkpoint-{chkpt}",
                f"{args.roberta_dump_folder_root}/checkpoint-{chkpt}"
            )
