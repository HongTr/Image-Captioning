from hyperparameters import *
import random
from torch import Tensor

def teacher_forcing(decoder_output: Tensor, target: Tensor, teacher_forcing_ratio=TEACHER_FORCING_RATE) -> Tensor:
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        decoder_input = target.unsqueeze(0) # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        decoder_input = decoder_output.argmax(2)
    
    return decoder_input