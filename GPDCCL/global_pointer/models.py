import torch
import torch.nn as nn
from utils.modeling_utils import BaseModelOutput
from transformers import AutoModel, DebertaModel
from utils import PreTrainedModelWrapper, BaseModelOutput, LossFct, ClassificationHead
from config import args


class GlobalPointerModel(PreTrainedModelWrapper):
    transformers_parent_class = AutoModel

    def __init__(self, pretrained_model):
        super().__init__(pretrained_model)
        self.config = self.pretrained_model.config

        self.inner_dim = int(self.config.hidden_size / self.config.num_attention_heads)
        self.dropout = nn.Dropout(0.3)
        self.fc = torch.nn.Linear(self.config.hidden_size, args.num_labels * self.inner_dim * 2)
        self.fc2 = nn.Linear(self.config.hidden_size, args.num_domain_labels)

        # pgd
        self.eps = 0.02
        self.alpha = 0.05
        self.steps = 3

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)
        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)],
                                 dim=-1)
        embeddings = embeddings.repeat(
            (batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to('cuda')
        return embeddings

    def pgd_attack(self, input_ids, attention_mask, domain_labels):
        input_embedding = self.get_input_embeddings(input_ids)
        adv_inputs = input_embedding.clone().detach()
        for _ in range(self.steps):
            adv_inputs.requires_grad = True
            pooled_output_adv = self.get_input_representations(inputs_embeds=adv_inputs,attention_mask=attention_mask).pooler_output

            domain_logits_adv = self.fc2(pooled_output_adv)
            domain_loss = self.loss_fct(domain_logits_adv.view(-1, args.num_domain_labels), domain_labels.view(-1))
            cost = - domain_loss


            grad = torch.autograd.grad(cost, adv_inputs,
                                       retain_graph=False, create_graph=False)[0]

            adv_inputs = adv_inputs.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_inputs - input_embedding, min=-self.eps, max=self.eps)
            adv_inputs = torch.clamp(input_embedding + delta, min=0, max=1).detach()

        return delta

    def get_input_embeddings(self, input_ids=None):
        word_embeddings = self.pretrained_model.embeddings(input_ids=input_ids)
        return word_embeddings

    def get_input_representations(self, input_ids=None, inputs_embeds=None, attention_mask=None):
        if input_ids is not None:
            text_output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask,
                                                output_hidden_states=True)
        else:
            text_output = self.pretrained_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                                output_hidden_states=True)

        return text_output

    def get_table(self, input_ids=None, attention_mask=None, seq_output=None, text=None):
        if seq_output is None:
            seq_output = self.get_input_representations(input_ids=input_ids,
                                                        attention_mask=attention_mask).last_hidden_state
        batch_size, seq_len = seq_output.size(0), seq_output.size(1)
        seq_output = self.dropout(seq_output)

        outputs = self.fc(seq_output)
        outputs = torch.split(outputs, self.inner_dim * 2,
                              dim=-1)
        outputs = torch.stack(outputs, dim=-2)
        qw, kw = outputs[..., :self.inner_dim], outputs[...,
                                                self.inner_dim:]

        pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len,
                                                     self.inner_dim)

        cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2,
                                                             dim=-1)
        sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
        qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
        qw2 = qw2.reshape(qw.shape)
        qw = qw * cos_pos + qw2 * sin_pos
        kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
        kw2 = kw2.reshape(kw.shape)
        kw = kw * cos_pos + kw2 * sin_pos


        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, args.num_labels, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12

        return logits / self.inner_dim ** 0.5

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                domain_labels=None, domain_attention_mask=None, domain_input_ids=None, step=None):
        logits = self.get_table(input_ids, attention_mask)
        gp_loss = self.compute_loss(logits, labels)

        if step < 700 or not args.domain_adapation:
            loss = gp_loss
        if step >= 700 and args.domain_adapation:
            delta = self.pgd_attack(domain_input_ids, domain_attention_mask, domain_labels)
            word_embeddings = self.get_input_embeddings(input_ids)
            adv_inputs = torch.clamp(word_embeddings + delta, min=0, max=1).detach()
            output_adv = self.get_input_representations(inputs_embeds=adv_inputs, attention_mask=attention_mask)
            seq_output_adv = output_adv.last_hidden_state
            logits_adv = self.get_table(seq_output=seq_output_adv, attention_mask=attention_mask)

            logits_filtered = logits
            logits_filtered_adv = logits_adv

            kl_loss = self.compute_kl_loss(logits_filtered, logits_filtered_adv, labels)
            contrastive_loss = self.compute_contrastive_loss(logits_filtered, logits_filtered_adv, labels)


            loss = gp_loss + kl_loss * 6 + contrastive_loss
        return BaseModelOutput(loss=loss, logits=logits)

    def compute_loss(self, pred, labels):
        batch_size, num_labels = labels.shape[:2]
        y_true = labels.reshape(batch_size * num_labels, -1)
        y_pred = pred.reshape(batch_size * num_labels, -1)
        loss = self.multilabel_categorical_crossentropy(y_pred, y_true)
        return loss

    def compute_kl_loss(self, logits_filtered, logits_filtered_adv, labels):
        kl_loss = kl_div_table(logits_filtered, logits_filtered_adv) + kl_div_table(logits_filtered_adv,
                                                                                    logits_filtered)
        return kl_loss

    def compute_contrastive_loss(self, logits_filtered, logits_filtered_adv, labels):
        label_indices = torch.nonzero(labels)
        contrastive_loss = torch.tensor(0.).to(logits_filtered.device)
        span_logits_with_postive = {}
        span_logits_adv_batch = {}
        for indice in label_indices:
            label_idx = int(indice[1].item())
            span_logits = logits_filtered[indice[0], :, indice[2], indice[3]]
            span_logits_adv = logits_filtered_adv[indice[0], :, indice[2], indice[3]]
            if not span_logits_with_postive.get(label_idx):
                span_logits_with_postive[label_idx] = []
            # if not span_logits_adv_batch.get(label):
            #     span_logits_adv_batch[label] = []
            span_logits_with_postive[label_idx].append((span_logits, span_logits_adv))
            # span_logits_adv_batch[label].append(span_logits_adv)
        all_labels = list(span_logits_with_postive) # 1 2 3 4
        for label_idx, span_logits_all in span_logits_with_postive.items():
            for span_logits, span_logits_adv in span_logits_all:
                pos_cos = torch.cosine_similarity(span_logits.unsqueeze(0), span_logits_adv.unsqueeze(0))
                label_neg_idxs = [idx for idx in all_labels if idx != label_idx]
                negative_all = []
                for label_neg_idx in label_neg_idxs:
                    negative_all.extend([item[0] for item in span_logits_with_postive[label_neg_idx]])
                    negative_all.extend([item[1] for item in span_logits_with_postive[label_neg_idx]])

                if len(negative_all) == 0:
                    continue
                negative_all = torch.stack(negative_all, 0)
                neg_cos = torch.nn.functional.cosine_similarity(span_logits.unsqueeze(0), negative_all)
                similarity = torch.cat((pos_cos, neg_cos), 0)
                contrastive_ = -torch.nn.functional.log_softmax(similarity, dim=-1)[0]
                contrastive_loss += contrastive_

        contrastive_loss = contrastive_loss / (logits_filtered.size(0) * logits_filtered.size(2))

        return contrastive_loss

    def multilabel_categorical_crossentropy(self, y_pred, y_true):
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return (neg_loss + pos_loss).mean()

    def evaluate(self, input_ids=None, attention_mask=None, labels=None):
        table_logits = self.get_table(input_ids, attention_mask)
        if labels is None:
            loss = None
        else:
            loss = self.compute_loss(table_logits, labels)
        return BaseModelOutput(loss=loss, logits=table_logits)


def kl_div_table(table_scores_stu, table_scores_teach):


    table_scores_stu = torch.clamp(table_scores_stu, min=1e-7, max=1 - 1e-7)
    table_scores_stu_1 = 1 - table_scores_stu
    table_score_dist_stu = torch.cat(
        [table_scores_stu.unsqueeze(-1), table_scores_stu_1.unsqueeze(-1)], dim=-1
    )

    table_scores_teach = torch.clamp(table_scores_teach, min=1e-7, max=1 - 1e-7)
    table_scores_teach_1 = 1 - table_scores_teach
    table_score_dist_teach = torch.cat(
        [table_scores_teach.unsqueeze(-1), table_scores_teach_1.unsqueeze(-1)], dim=-1
    )

    """Kullback-Leibler divergence D(P || Q) for discrete distributions"""
    p = torch.log(table_score_dist_stu.view(-1, 2))
    q = torch.log(table_score_dist_teach.view(-1, 2))


    scores = torch.sum(p.exp() * (p - q), axis=-1)
    return scores.mean()