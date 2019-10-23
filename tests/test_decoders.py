import pytest
from asr.decoders import Decoder, BeamCTCDecoder, GreedyCTCDecoder
from asr.data import Alphabet
import torch
import os

base_dir = os.path.split(__file__)[0]


def test_base_decoder():
    with pytest.raises(TypeError) as excinfo:
        decoder = Decoder()

    assert "missing 1 required positional argument: 'alphabet'" in str(
        excinfo.value)

    alphabet = Alphabet('-abc ', blank_index=0)
    decoder = Decoder(alphabet)

    assert hasattr(decoder, 'alphabet')

    assert decoder.wer('a bcd c', 'a dcc c') == 1
    assert decoder.cer('a bc c', 'a dcc c') == 2
    assert decoder.cer('a bcc', 'a dcc c', remove_space=True) == 2

    with pytest.raises(NotImplementedError):
        decoder.decode(None, None)


def test_tensor2list():

    tensor = torch.tensor([
        [1, 1, 1, 1, 0],
        [2, 2, 2, 0, 0],
        [3, 3, 0, 0, 0],
        [4, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ],
                          dtype=torch.int)
    expected = [
        tensor[0][:-1],
        tensor[1][:-2],
        tensor[2][:-3],
        tensor[3][:-4],
        torch.tensor([], dtype=torch.int),
    ]
    sizes = [4, 3, 2, 1, 0]

    output = Decoder.tensor2list(tensor, sizes)
    for o, e in zip(output, expected):
        assert torch.all(o == e)

    with pytest.raises(ValueError) as excinfo:
        Decoder.tensor2list(tensor, [3.2, 3, 2, 1, 0])

    assert 'must be in' in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        Decoder.tensor2list(tensor, [3, 3, 2, 1])

    assert 'sizes.numel()` != tensor.shape[0].' in str(excinfo.value)

    output = Decoder.tensor2list(tensor)
    for o, e in zip(output, torch.chunk(tensor, tensor.shape[0])):
        assert torch.all(o == e)


def test_1dtensor2list():

    tensor = torch.tensor([
        [1, 1, 1, 1, 0],
        [2, 2, 2, 0, 0],
        [3, 3, 0, 0, 0],
        [4, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ],
                          dtype=torch.int)
    expected = [
        tensor[0][:-1],
        tensor[1][:-2],
        tensor[2][:-3],
        tensor[3][:-4],
        torch.tensor([], dtype=torch.int),
    ]
    tensor = torch.cat(expected)
    sizes = [4, 3, 2, 1, 0]

    output = Decoder.tensor2list(tensor, sizes)
    for o, e in zip(output, expected):
        assert torch.all(o == e)

    with pytest.raises(ValueError) as excinfo:
        Decoder.tensor2list(tensor, [3.2, 3, 2, 1, 0])

    assert 'must be in' in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        Decoder.tensor2list(tensor, [3, 3, 2, 1])

    assert 'sum(sizes) != tensor(numel)' in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        Decoder.tensor2list(tensor)

    assert 'Arg. `sizes` must be set if tensor.ndim() == 1' in str(
        excinfo.value)


def test_tensor2str():
    alphabet = Alphabet('-abc ', blank_index=0)
    decoder = Decoder(alphabet)

    expected = ['ab c', 'aa-', 'c ', 'a', '']
    tensor = torch.tensor([
        [1, 2, 4, 3, 0],
        [1, 1, 0, 0, 0],
        [3, 4, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ],
                          dtype=torch.int)

    sizes = [4, 3, 2, 1, 0]

    with pytest.raises(ValueError) as excinfo:
        decoder.tensor2str(torch.tensor([[[1, 2, 3]]]))

    assert '`tensor.dim()` != 1 or 2' in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        decoder.tensor2str(tensor.float())

    assert 'must be int' in str(excinfo.value)

    output = decoder.tensor2str(tensor, sizes)
    for o, e in zip(output, expected):
        assert o == e

    output = decoder.tensor2str(tensor)
    for o, t in zip(output, tensor):
        assert o == alphabet.idx2str(t)


def test_beam_search_decoder():

    alphabet = ['\'', ' ', 'a', 'b', 'c', 'd', '-']
    beam_width = 20
    probs_seq1 = [[
        0.06390443, 0.21124858, 0.27323887, 0.06870235, 0.0361254, 0.18184413,
        0.16493624
    ],
                  [
                      0.03309247, 0.22866108, 0.24390638, 0.09699597,
                      0.31895462, 0.0094893, 0.06890021
                  ],
                  [
                      0.218104, 0.19992557, 0.18245131, 0.08503348, 0.14903535,
                      0.08424043, 0.08120984
                  ],
                  [
                      0.12094152, 0.19162472, 0.01473646, 0.28045061,
                      0.24246305, 0.05206269, 0.09772094
                  ],
                  [
                      0.1333387, 0.00550838, 0.00301669, 0.21745861,
                      0.20803985, 0.41317442, 0.01946335
                  ],
                  [
                      0.16468227, 0.1980699, 0.1906545, 0.18963251, 0.19860937,
                      0.04377724, 0.01457421
                  ]]
    probs_seq2 = [[
        0.08034842, 0.22671944, 0.05799633, 0.36814645, 0.11307441, 0.04468023,
        0.10903471
    ],
                  [
                      0.09742457, 0.12959763, 0.09435383, 0.21889204,
                      0.15113123, 0.10219457, 0.20640612
                  ],
                  [
                      0.45033529, 0.09091417, 0.15333208, 0.07939558,
                      0.08649316, 0.12298585, 0.01654384
                  ],
                  [
                      0.02512238, 0.22079203, 0.19664364, 0.11906379,
                      0.07816055, 0.22538587, 0.13483174
                  ],
                  [
                      0.17928453, 0.06065261, 0.41153005, 0.1172041,
                      0.11880313, 0.07113197, 0.04139363
                  ],
                  [
                      0.15882358, 0.1235788, 0.23376776, 0.20510435,
                      0.00279306, 0.05294827, 0.22298418
                  ]]
    log_probs_seq1 = torch.log(torch.as_tensor(probs_seq1))
    log_probs_seq2 = torch.log(torch.as_tensor(probs_seq2))

    greedy_result = ["ac'bdc", "b'da"]
    beam_search_result = ['acdc', "b'a"]

    alphabet = Alphabet(alphabet, blank_index=alphabet.index('-'))
    decoder = BeamCTCDecoder(alphabet, beam_width=beam_width)

    log_probs_seq = log_probs_seq1[None, ...]
    beam_result, beam_scores, timesteps = decoder.decode(log_probs_seq)

    assert beam_result[0] == beam_search_result[0]

    log_probs_seq = log_probs_seq2[None, ...]
    beam_result, beam_scores, timesteps = decoder.decode(log_probs_seq)

    assert beam_result[0] == beam_search_result[1]

    # Test batch

    log_probs_seq = torch.stack([log_probs_seq1, log_probs_seq2])

    beam_results, beam_scores, timesteps = decoder.decode(log_probs_seq)

    assert beam_results[0] == beam_search_result[0]
    assert beam_results[1] == beam_search_result[1]


def test_real_ctc_beam_decoder():
    labels = ' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_'

    alphabet = Alphabet(labels, blank_index=labels.index('_'))

    log_input = torch.load(
        os.path.join(base_dir, "data/rnn_output_log_softmax.pth"))
    sizes = torch.tensor([log_input.shape[1]])
    # greedy using beam
    decoder = BeamCTCDecoder(alphabet, beam_width=1)

    decode_result, scores, timesteps = decoder.decode(log_input, sizes)

    assert "the fak friend of the fomly hae tC" == decode_result[0]

    # default beam decoding
    decoder = BeamCTCDecoder(alphabet, beam_width=25)
    decode_result, scores, timesteps = decoder.decode(log_input, sizes)

    assert "the fak friend of the fomcly hae tC" == decode_result[0]

    # lm-based decoding
    decoder = BeamCTCDecoder(alphabet,
                             lm_path=os.path.join(base_dir, 'data',
                                                  'bigram.arpa'),
                             beam_width=25,
                             alpha=2,
                             beta=0)
    decode_result, scores, timesteps = decoder.decode(log_input, sizes)
    assert "the fake friend of the family, like the" == decode_result[0]


def test_greedy_decoder():
    """ Code adapted from tensorflow
    """
    max_time_steps = 6

    seq_len_0 = 4
    input_prob_matrix_0 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],  # t=0
            [0.0, 0.0, 0.4, 0.6],  # t=1
            [0.0, 0.0, 0.4, 0.6],  # t=2
            [0.0, 0.9, 0.1, 0.0],  # t=3
            [0.0, 0.0, 0.0, 0.0],  # t=4 (ignored)
            [0.0, 0.0, 0.0, 0.0]
        ],  # t=5 (ignored)
        dtype=torch.float32)
    input_log_prob_matrix_0 = input_prob_matrix_0.log()

    seq_len_1 = 5
    # dimensions are time x depth
    input_prob_matrix_1 = torch.tensor(
        [
            [0.1, 0.9, 0.0, 0.0],  # t=0
            [0.0, 0.9, 0.1, 0.0],  # t=1
            [0.0, 0.0, 0.1, 0.9],  # t=2
            [0.0, 0.9, 0.1, 0.1],  # t=3
            [0.9, 0.1, 0.0, 0.0],  # t=4
            [0.0, 0.0, 0.0, 0.0]
        ],  # t=5 (ignored)
        dtype=torch.float32)
    input_log_prob_matrix_1 = input_prob_matrix_1.log()

    # len max_time_steps array of batch_size x depth matrices
    inputs = torch.stack([input_log_prob_matrix_0, input_log_prob_matrix_1])
    # batch_size length vector of sequence_lengths
    seq_lens = torch.tensor([seq_len_0, seq_len_1], dtype=torch.int32)

    # batch_size length vector of negative log probabilities
    log_prob_truth = torch.tensor([
        -(torch.tensor([1.0, 0.6, 0.6, 0.9]).log()).sum().item(),
        -(torch.tensor([0.9, 0.9, 0.9, 0.9, 0.9]).log()).sum().item()
    ])

    decode_truth = ['ab', 'bba']
    offsets_truth = [
        torch.tensor([0, 3]),
        torch.tensor([0, 3, 4]),
    ]

    alphabet = Alphabet('abc-', blank_index=3)

    decoder = GreedyCTCDecoder(alphabet)
    out, scores, offsets = decoder.decode(inputs, seq_lens)

    assert out[0] == decode_truth[0]
    assert out[1] == decode_truth[1]

    assert torch.allclose(scores, log_prob_truth)

    assert torch.all(offsets[0] == offsets_truth[0])
    assert torch.all(offsets[1] == offsets_truth[1])