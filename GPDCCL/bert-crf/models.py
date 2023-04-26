import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoConfig, BertForTokenClassification
from utils import PreTrainedModelWrapper, BaseModelOutput, LossFct
from config import args
conf = AutoConfig.from_pretrained(args.model_path)

class BERTModel(PreTrainedModelWrapper):
    transformers_parent_class = AutoModel

    def __init__(self, pretrained_model):
        super().__init__(pretrained_model)
        self.config = self.pretrained_model.config
        self.fc = nn.Linear(self.config.hidden_size, args.num_labels)
        self.fc2 = nn.Linear(self.config.hidden_size, args.num_domain_labels)
        self.loss_fct = LossFct(args=args, ignore_index=-1, num_labels=args.num_labels)
        self.dropout = nn.Dropout(0.3)

        self.eps = 0.02
        self.alpha = 0.05
        self.steps = 3

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
        word_embeddings = self.bert.embeddings(input_ids=input_ids)
        return word_embeddings

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                domain_labels=None, domain_attention_mask=None, domain_input_ids=None, step=None):
        output = self.get_input_representations(input_ids=input_ids, attention_mask=attention_mask)
        seq_output = output.last_hidden_state

        logits = self.fc(seq_output)
        ce_loss = self.loss_fct(logits.view(-1, args.num_labels), labels.view(-1))
        if step < 600 or not args.domain_adapation:
            loss = ce_loss
        if step>=600 and args.domain_adapation:

            delta = self.pgd_attack(domain_input_ids, domain_attention_mask, domain_labels)
            word_embeddings = self.get_input_embeddings(input_ids)
            adv_inputs = torch.clamp(word_embeddings + delta, min=0, max=1).detach()
            output_adv = self.get_input_representations(inputs_embeds=adv_inputs, attention_mask=attention_mask)
            seq_output_adv = output_adv.last_hidden_state
            logits_adv = self.fc(seq_output_adv)

            logits_filtered = logits
            logits_filtered_adv = logits_adv

            kl_loss, contrastive_loss = self.compute_loss(logits_filtered, logits_filtered_adv, labels)

            loss = ce_loss+kl_loss#+contrastive_loss*0.1

        return BaseModelOutput(logits=logits, loss=loss)

    def compute_loss(self, logits_filtered, logits_filtered_adv, labels):
        kl_loss = torch.tensor(0.).to(logits_filtered.device)
        contrastive_loss = torch.tensor(0.).to(logits_filtered.device)
        for i in range(logits_filtered.size(0)):
            for j in range(logits_filtered.size(1)):
                if labels[i][j] == -1:
                    continue
                kl_loss_fct = nn.KLDivLoss(reduction='mean')
                kl_ = kl_loss_fct(torch.softmax(logits_filtered_adv[i][j], -1),
                                  torch.softmax(logits_filtered[i][j], -1)) + kl_loss_fct(
                    torch.softmax(logits_filtered[i][j], -1),
                    torch.softmax(logits_filtered_adv[i][j], -1))
                kl_loss += kl_

                negative = torch.randn((120, len(logits_filtered[i][j])), device=logits_filtered.device, requires_grad=True)
                virtual_loss = -torch.nn.functional.log_softmax(negative, dim=-1)[:, labels[i][j].item()]
                top_indices = torch.topk(virtual_loss, 40)[1]
                selected_negative = negative[top_indices]
                total_samples = torch.cat((logits_filtered_adv[i][j].unsqueeze(0), selected_negative), 0)
                similarity = F.cosine_similarity(logits_filtered[i][j].unsqueeze(0), total_samples)
                contrastive_ = -torch.nn.functional.log_softmax(similarity, dim=-1)[0]
                contrastive_loss += contrastive_

        kl_loss = - kl_loss / (logits_filtered.size(0) * logits_filtered.size(1))
        contrastive_loss = contrastive_loss / (logits_filtered.size(0) * logits_filtered.size(1))
        return kl_loss, contrastive_loss


    def get_input_embeddings(self, input_ids=None):
        word_embeddings = self.pretrained_model.embeddings(input_ids=input_ids)
        return word_embeddings

    def get_input_representations(self, input_ids=None, inputs_embeds=None, attention_mask=None):
        if input_ids is not None:
            text_output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        else:
            text_output = self.pretrained_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_hidden_states=True)

        return text_output

    def evaluate(self, input_ids=None, attention_mask=None, labels=None,
                 domain_labels=None, domain_attention_mask=None, domain_input_ids=None, step=None):
        output = self.get_input_representations(input_ids=input_ids, attention_mask=attention_mask)
        seq_output = output.last_hidden_state
        seq_output = self.dropout(seq_output)
        logits = self.fc(seq_output)
        loss = self.loss_fct(logits.view(-1, args.num_labels), labels.view(-1))
        return BaseModelOutput(logits=logits, loss=loss)


