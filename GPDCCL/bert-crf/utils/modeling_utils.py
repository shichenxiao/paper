import torch
import torch.nn as nn
from transformers import PreTrainedModel
from dataclasses import dataclass
from typing import Any, Optional, List, Tuple, Union, Dict

@dataclass
class BaseModelOutput:
    logits: Any = None
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    loss: Any = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


def Bertpooling(bert_outputs, pool_type='pooler'):
    encoded_layers = bert_outputs.hidden_states
    sequence_output = bert_outputs.last_hidden_state

    if pool_type == 'pooler':
        pooled_output = bert_outputs.pooler_output

    elif pool_type == 'first-last-avg':
        first = encoded_layers[1]
        last = encoded_layers[-1]
        seq_length = first.size(1)

        first_avg = torch.avg_pool1d(first.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # batch, hid_size
        last_avg = torch.avg_pool1d(last.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # batch, hid_size
        pooled_output = torch.avg_pool1d(
            torch.cat([first_avg.unsqueeze(1), last_avg.unsqueeze(1)], dim=1).transpose(1, 2),
            kernel_size=2).squeeze(-1)

    elif pool_type == 'last-avg':
        pooled_output = torch.mean(sequence_output, 1)

    elif pool_type == 'cls':
        pooled_output = sequence_output[:, 0]

    else:
        raise TypeError('Please the right pool_type from cls, pooler, first-last-avg and last-avg !!!')

    return pooled_output




class PreTrainedModelWrapper(nn.Module):
    r"""
    A wrapper class around a (`transformers.PreTrainedModel`) to be compatible with the
    (`~transformers.PreTrained`) class in order to keep some attributes and methods of the
    (`~transformers.PreTrainedModel`) class.

    Attributes
    ----------
    pretrained_model: (`transformers.PreTrainedModel`)
        The model to be wrapped.
    parent_class: (`transformers.PreTrainedModel`)
        The parent class of the model to be wrapped.
    supported_args: (`list`)
        The list of arguments that are supported by the wrapper class.
    """
    transformers_parent_class = None
    supported_args = None

    def __init__(self, pretrained_model=None, **kwargs):
        super().__init__()
        self.pretrained_model = pretrained_model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Instantiates a new model from a pretrained model.

        Parameters
        ----------
        pretrained_model_name_or_path: (`str` or `transformers.PreTrainedModel`)
            The path to the pretrained model or its name.
        *model_args:
            Additional positional arguments passed along to the underlying model's
            `from_pretrained` method.
        **kwargs:
            Additional keyword arguments passed along to the underlying model's
            `from_pretrained` method. We also pre-process the kwargs to extract
            the arguments that are specific to the `transformers.PreTrainedModel`
            class and the arguments that are specific to trl models.
        """


        # First, load the pre-trained model using the parent-class
        # either `AutoModelForCausalLM` or `AutoModelForSeq2SeqLM`
        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model = cls.transformers_parent_class.from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )
        elif isinstance(pretrained_model_name_or_path, PreTrainedModel):
            pretrained_model = pretrained_model_name_or_path
        else:
            raise ValueError(
                "pretrained_model_name_or_path should be a string or a PreTrainedModel, "
                f"but is {type(pretrained_model_name_or_path)}"
            )

        # Then, create the full model by instantiating the wrapper class
        model = cls(pretrained_model)

        return model

    @classmethod
    def _split_kwargs(cls, kwargs):
        """
        Separate the kwargs from the arguments that we support inside
        `supported_args` and the ones that we don't.
        """
        supported_kwargs = {}
        unsupported_kwargs = {}

        for key, value in kwargs.items():
            if key in cls.supported_args:
                supported_kwargs[key] = value
            else:
                unsupported_kwargs[key] = value

        return supported_kwargs, unsupported_kwargs

    def push_to_hub(self, *args, **kwargs):
        r"""
        Push the pretrained model to the hub.
        """
        return self.pretrained_model.push_to_hub(*args, **kwargs)

    def save_pretrained(self, *args, **kwargs):
        r"""
        Save the pretrained model to a directory.
        """
        return self.pretrained_model.save_pretrained(*args, **kwargs)



from transformers.activations import ACT2FN

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        self.config = config

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = ACT2FN[self.config.hidden_act](x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        # 例如，self.emb = nn.Embedding(5000, 100)
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)  # 默认为2范数
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}
        self.emb_name = 'word_embeddings'    # embeddings
        self.r_at = None

    def attack(self, epsilon=1., alpha=0.3, is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    if self.r_at is None:
                        self.r_at = alpha * param.grad / norm
                        param.data.add_(self.r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]



