import editdistance
import torch

from asr.data.alphabet import Alphabet


class Decoder:
    """
    Basic decoder class from which all other decoders inherit. Implements several
    helper functions. Subclasses should implement the decode() method.

    Arguments:
        alphabet (Alphabet): the alphabet
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet

    @staticmethod
    def wer(ref, hyp):
        """
        Computes the Word Error Rate

        Arguments:
            ref (string): space-separated sentence
            hyp (string): space-separated sentence
        """
        return editdistance.eval(Alphabet.word_tokenize(ref), Alphabet.word_tokenize(hyp))

    @staticmethod
    def cer(ref, hyp, remove_space=False):
        """
        Computes the Character Error Rate

        Arguments:
            ref (string): space-separated sentence
            hyp (string): space-separated sentence
        """
        return editdistance.eval(Alphabet.char_tokenize(ref, remove_space=remove_space),
                                 Alphabet.char_tokenize(hyp, remove_space=remove_space))

    def decode(self, log_probs, sizes=None):
        """
        Given a matrix of character probabilities, returns the decoder's
        best guess of the transcription

        Arguments:
            log_probs (tensor): Tensor of log probabilities with shape (B, T, L), 
                where `log_probs[b, t, l]` is the log probability of character `c` at time `t` 
                in batch `b`
            sizes (optional): Size of each sequence in the batch
        Returns:
            scores (tensor): tensor of size B the negative log probability 
            decoded (list of string): sequence of the model's best guess for the transcription
            offsets (tensor): time-step per character predicted
        """
        raise NotImplementedError

    def tensor2str(self, tensor, sizes=None):
        """ Convert tensor into string given the alphabet

        Args:
            tensor (Tensor): tensor of size (B, T), where `B` is the batch size and `T` is the 
                maximum sequence length in the batch or size (B*T,)
            sizes (Tensor): tensor of size (B,) with the size of each sequence in the batch
        Returns:
            list of string containing the transcriptions
        """
        if tensor.dim() not in (1, 2):
            raise ValueError('`tensor.dim()` != 1 or 2.')

        if tensor.dtype not in (torch.int, torch.int16, torch.int32, torch.int64):
            raise ValueError('`tensor` dtype must be int.')

        transcripts = []
        for int_transcripts in self.tensor2list(tensor, sizes):
            transcript = self.alphabet.idx2str(int_transcripts)
            transcripts.append(transcript)

        return transcripts

    def list2str(self, input, sizes=None):
        transcripts = []
        for i, int_transcripts in enumerate(input):
            if sizes is not None:
                int_transcripts = int_transcripts[:i]
            transcript = self.alphabet.idx2str(int_transcripts)
            transcripts.append(transcript)

        return transcripts

    @staticmethod
    def tensor2list(tensor, sizes=None, dim=-1):
        """ Convert tensor in list of tensors given the size

        Args:
            tensor (Tensor): tensor of size (B, *)
            sizes (Tensor): tensor of sixze (B,) with the size of each sequence in the batch
            dim (int): which dimension to crop using `sizes`
        
        Returns:
            list of B tensors limited by the sizes
        """
        if sizes is None:
            if tensor.dim() == 1:
                raise ValueError('Arg. `sizes` must be set if tensor.ndim() == 1')
            sizes = [t.shape[dim] for t in tensor]

        sizes = torch.as_tensor(sizes)

        if sizes.dtype not in (torch.int, torch.int16, torch.int32, torch.int64):
            raise ValueError('`sizes.dtype` must be int.')

        if sizes.dim() != 1:
            raise ValueError('`sizes.dim()` must be 1')

        if tensor.dim() == 2:
            if sizes.numel() != tensor.shape[0]:
                raise ValueError('`sizes.numel()` != tensor.shape[0]. '
                                 f'Expected {tensor.shape[0]}, found {sizes.numel()}')

            return [t.narrow(dim, 0, s) for t, s in zip(tensor, sizes)]

        if sizes.sum().item() != tensor.numel():
            raise ValueError('sum(sizes) != tensor(numel).')

        return list(torch.split(tensor, sizes.tolist()))


class GreedyCTCDecoder(Decoder):
    def decode(self, log_probs, sizes=None):
        """
        Given a matrix of character probabilities, returns the decoder's
        best guess of the transcription

        Arguments:
            log_probs (tensor): Tensor of log probabilities with shape (B, T, L), 
                where `log_probs[b, t, l]` is the log probability of character `c` at time `t` 
                in batch `b`
            sizes (optional): Size of each sequence in the batch
        Returns:
            decoded (list of string): sequence of the model's best guess for the transcription
            scores (tensor): tensor of size B the negative log probability 
            offsets (tensor): time-step per character predicted
        """
        assert isinstance(log_probs, torch.Tensor) and log_probs.dim(
        ) == 3, "The param `log_prob` must be a `torch.Tensor` with shape (B, T, L)."

        scores, out, offsets = [], [], []
        for b, log_prob in enumerate(log_probs):
            if sizes is not None:
                log_prob = log_prob[:sizes[b]]

            curr_scores, curr_decoded_sequence, curr_offsets = self._decode(log_prob)

            scores.append(curr_scores)
            out.append(curr_decoded_sequence)
            offsets.append(curr_offsets)

        scores = torch.tensor(scores)
        strings = self.list2str(out)

        return strings, scores, offsets

    def _decode(self, log_probs_seq):
        """CTC greedy (best path) decoder.

        Path consisting of the most probable tokens are further post-processed to
        remove consecutive repetitions and all blanks.

        Args:
            log_probs_seq: 3-D tensor containing the log probability of a character given each
                timestep per batch
        Returns:
            tuple containing (score, decoded sequence, timesteps)
        """
        # argmax to get the best index for each time step
        max_probs, max_indexes = torch.max(log_probs_seq, 1)
        # remove consecutive duplicate indexes
        mask = torch.cat([
            torch.tensor([1], dtype=torch.bool, device=log_probs_seq.device),
            ((max_indexes[:-1] - max_indexes[1:]).abs() > 0)
        ])
        # remove blank indexes
        mask = mask * (max_indexes != self.alphabet.blank_index)

        return -max_probs.sum(), max_indexes[mask], mask.nonzero().squeeze()


class BeamCTCDecoder(Decoder):
    def __init__(self,
                 alphabet,
                 lm_path=None,
                 alpha=0,
                 beta=0,
                 cutoff_top_n=40,
                 cutoff_prob=1.0,
                 beam_width=100,
                 num_processes=4):
        super().__init__(alphabet)

        try:
            from ctcdecode import CTCBeamDecoder
        except ImportError:
            raise ImportError("BeamCTCDecoder requires ctcdecode package.")

        self._decoder = CTCBeamDecoder(alphabet.tokens,
                                       lm_path,
                                       alpha,
                                       beta,
                                       cutoff_top_n,
                                       cutoff_prob,
                                       beam_width,
                                       num_processes,
                                       alphabet.blank_index,
                                       log_probs_input=True)

    def decode(self, log_probs, sizes=None):
        """
        Given a matrix of character probabilities, returns the decoder's
        best guess of the transcription

        Arguments:
            log_probs (tensor): Tensor of log probabilities with shape (B, T, L), 
                where `log_probs[b, t, l]` is the log probability of character `c` at time `t` 
                in batch `b`
            sizes (optional): Size of each sequence in the batch
        Returns:
            decoded (list of string): sequence of the model's best guess for the transcription
            scores (tensor): tensor of size B the negative log probability 
            offsets (tensor): time-step per character predicted
        """
        log_probs = log_probs.cpu()

        out, scores, offsets, seq_lens = self._decoder.decode(log_probs, sizes)

        strings = self.tensor2str(out[:, 0, :], seq_lens[:, 0])

        scores = scores[:, 0]
        offsets = offsets[:, 0]

        return strings, scores, offsets

    def reset_params(self, alpha, beta):
        self._decoder.reset_params(alpha, beta)
