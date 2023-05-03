import torch


def single_speaker_collate(batch):
    # Right zero-pad all one-hot text sequences to max input length
    _, ids_sorted_decreasing = torch.sort(
        input=torch.LongTensor([x[1].size(1) for x in batch]),
        dim=0,
        descending=True
    )

    max_text_len = max([len(x[0]) for x in batch])
    max_spec_len = max([x[1].size(1) for x in batch])
    max_wav_len = max([x[2].size(1) for x in batch])

    text_lengths = torch.LongTensor(len(batch))
    spec_lengths = torch.LongTensor(len(batch))
    wav_lengths = torch.LongTensor(len(batch))

    text_padded = torch.zeros((len(batch), max_text_len), dtype=torch.long)
    spec_padded = torch.zeros((len(batch), batch[0][1].size(0), max_spec_len), dtype=torch.float)
    wav_padded = torch.zeros((len(batch), 1, max_wav_len), dtype=torch.float)

    for i, batch_id in enumerate(ids_sorted_decreasing):
        text, spec, wav = batch[batch_id]

        text_padded[i, :text.size(0)] = text
        text_lengths[i] = text.size(0)

        spec_padded[i, :, :spec.size(1)] = spec
        spec_lengths[i] = spec.size(1)

        wav_padded[i, :, :wav.size(1)] = wav
        wav_lengths[i] = wav.size(1)

    return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths
