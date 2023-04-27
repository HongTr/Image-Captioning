import torch
from torch.nn.functional import log_softmax
from nltk.translate.bleu_score import corpus_bleu

class BeamSearch:
    def __init__(self, model, beam_size, max_len, sos_token, eos_token, device):
        self.model = model
        self.beam_size = beam_size
        self.max_len = max_len
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.device = device

    def search(self, src, trg_vocab):
        # Initialize the beam
        beam = [(torch.tensor([self.sos_token], device=self.device), [self.sos_token], 0)]
        # Initialize the completed sequences
        completed = []

        # Loop over the maximum allowed sequence length
        for _ in range(self.max_len):
            # Create a list to hold the next beam candidates
            candidates = []
            # Loop over the current beam
            for sequence, tokens, score in beam:
                # Get the last predicted token
                last_token = torch.tensor([tokens[-1]], device=self.device)
                # Predict the next token using the model
                output = self.model(src, last_token)
                log_probs = log_softmax(output, dim=1)
                top_log_probs, top_indices = log_probs.topk(self.beam_size)
                # Add the top beam_size candidates to the list
                for i in range(self.beam_size):
                    next_token = top_indices[:, i].unsqueeze(1)
                    log_prob = top_log_probs[:, i].item()
                    candidate_sequence = torch.cat([sequence, next_token], dim=0)
                    candidate_tokens = tokens + [next_token.item()]
                    candidate_score = score + log_prob
                    # Check if the sequence ends with the end-of-sequence token
                    if candidate_tokens[-1] == self.eos_token:
                        completed.append((candidate_sequence, candidate_tokens, candidate_score))
                    else:
                        candidates.append((candidate_sequence, candidate_tokens, candidate_score))
            # Sort the candidates by score and keep the top beam_size candidates
            candidates = sorted(candidates, key=lambda x: x[2], reverse=True)[:self.beam_size]
            # Check if there are no more candidates
            if not candidates:
                break
            # Update the beam with the new candidates
            beam = candidates

        # Get the completed sequences
        sequences = completed + beam
        # Sort the sequences by score
        sequences = sorted(sequences, key=lambda x: x[2], reverse=True)
        # Get the top-scoring sequence
        top_sequence = sequences[0][1]

        # Convert the predicted sequence to tokens and remove the start-of-sequence token
        predicted_tokens = [trg_vocab.itos[idx] for idx in top_sequence][1:]
        # Compute the BLEU-4 score between the predicted sequence and the target sequence
        bleu_score = corpus_bleu([[predicted_tokens]], [[trg_vocab.itos[idx] for idx in trg_vocab][1:]])

        return predicted_tokens, bleu_score
