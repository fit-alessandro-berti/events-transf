# File: components/moe_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .meta_learner import MetaLearner


class MoEModel(nn.Module):
    """
    A Mixture of Experts (MoE) wrapper.
    It holds multiple `MetaLearner` instances (experts).

    - During Training: Routes a batch to a single, randomly chosen expert.
    - During Evaluation: Aggregates outputs from all experts.
    """

    def __init__(self, num_experts, strategy, **kwargs):
        super().__init__()
        self.num_experts = num_experts
        self.strategy = strategy  # For compatibility with testing scripts

        # Create N independent experts
        self.experts = nn.ModuleList([
            MetaLearner(strategy=strategy, **kwargs)
            for _ in range(num_experts)
        ])
        print(f"✅ Initialized MoEModel with {num_experts} expert(s).")

    def set_char_vocab(self, char_to_id: dict):
        """Passes the character vocabulary to all experts."""
        if self.strategy == 'learned':
            for expert in self.experts:
                expert.set_char_vocab(char_to_id)

    def _process_batch(self, batch_of_sequences):
        """
        Processes a batch by averaging the embeddings from all experts.
        This is used by the retrieval-augmented and sklearn-baseline
        evaluation scripts.
        """
        if not self.experts:
            return None

        # Get embeddings from all experts (in eval mode)
        all_expert_embeddings = [
            expert._process_batch(batch_of_sequences)
            for expert in self.experts
        ]

        # Stack and average
        stacked_embeddings = torch.stack(all_expert_embeddings)
        avg_embeddings = torch.mean(stacked_embeddings, dim=0)
        return avg_embeddings

    def _aggregate_outputs(self, expert_outputs, task_type, true_labels):
        """Aggregates outputs from all experts for inference."""

        if task_type == 'regression':
            # expert_outputs is list of (prediction, _, confidence)
            # prediction and confidence shape: [N_q]
            all_preds = torch.stack([out[0] for out in expert_outputs])  # [E, N_q]
            all_confs = torch.stack([out[2] for out in expert_outputs])  # [E, N_q]

            # Weighted average: sum(pred * conf) / sum(conf)
            weighted_preds = all_preds * all_confs
            sum_weighted_preds = weighted_preds.sum(dim=0)
            sum_confs = all_confs.sum(dim=0).clamp_min(1e-8)  # Avoid div by zero
            final_preds = sum_weighted_preds / sum_confs

            # For final confidence, we can average the expert confidences
            final_confidence = all_confs.mean(dim=0)
            return final_preds, true_labels, final_confidence

        elif task_type == 'classification':
            # expert_outputs is list of (logits, _, confidence)
            # logits and confidence shape: [N_q, C]
            # We use the confidence (softmax) as requested

            all_confs_stacked = torch.stack([out[2] for out in expert_outputs])  # [E, N_q, C]

            # Sum confidences as requested: "confidence sums for classification"
            summed_confs = all_confs_stacked.sum(dim=0)  # [N_q, C]

            # The 'predictions' are the summed confidences.
            # The testing script will take argmax of this.
            final_predictions = summed_confs

            # The final confidence is the max of the normalized summed confs
            # (i.e., the strength of the winning class)
            norm_confs = F.normalize(summed_confs, p=1, dim=-1)  # Normalize sums to 1
            final_confidence, _ = torch.max(norm_confs, dim=-1)

            return final_predictions, true_labels, final_confidence

    def forward(self, support_set, query_set, task_type, expert_id=None):
        """
        Forward pass.
        - In train mode: Routes to a single expert specified by `expert_id`.
        - In eval mode: Aggregates outputs from all experts.
        """
        if self.training:
            # --- TRAINING PATH ---
            if expert_id is None:
                raise ValueError("MoEModel.forward() requires an 'expert_id' during training.")
            if expert_id >= self.num_experts:
                raise IndexError(f"Invalid expert_id {expert_id}. Max is {self.num_experts - 1}.")

            # Route to a single, specific expert
            return self.experts[expert_id](support_set, query_set, task_type)

        else:
            # --- EVALUATION PATH ---
            expert_outputs = []
            all_true_labels = None

            for expert in self.experts:
                # Call expert (which is in eval mode)
                preds, labels, confs = expert(support_set, query_set, task_type)

                if preds is None: continue

                expert_outputs.append((preds, labels, confs))
                if all_true_labels is None:
                    all_true_labels = labels  # All experts get same support, so labels are same

            if not expert_outputs:
                return None, None, None

            # Combine all expert outputs
            return self.aggregate_outputs(expert_outputs, task_type, all_true_labels)