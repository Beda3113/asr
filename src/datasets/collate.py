import torch


def collate_fn(dataset_items: list[dict], time_reduction: int = 2):
    """
    Collate function for dataloader.
    
    Args:
        dataset_items: list of dicts from dataset __getitem__
        time_reduction: reduction factor from CNN layers
    
    Returns:
        dict with batched tensors
    """
    # Filter items where spectrogram is valid
    filtered_items = []
    for item in dataset_items:
        if item is None:
            continue
        if "spectrogram" not in item or item["spectrogram"] is None:
            continue
        if "text_encoded" not in item or item["text_encoded"] is None:
            continue
            
        spec_len = item["spectrogram"].shape[-1]
        text_len = item["text_encoded"].shape[-1]
        output_len = spec_len // time_reduction
        
        if output_len >= text_len:
            filtered_items.append(item)
    
    # If no valid items, return None (will be skipped in trainer)
    if len(filtered_items) == 0:
        return None
    
    # Extract data
    text = [item["text"] for item in filtered_items]
    audio_path = [item["audio_path"] for item in filtered_items]
    text_encoded = [item["text_encoded"].squeeze(0) for item in filtered_items]
    audio = [item["audio"].squeeze(0) for item in filtered_items]
    spectrograms = [item["spectrogram"].squeeze(0) for item in filtered_items]
    
    # Get lengths
    audio_lengths = torch.LongTensor([x.shape[-1] for x in audio])
    spectrogram_length = torch.LongTensor([x.shape[-1] for x in spectrograms])
    text_encoded_length = torch.LongTensor([len(x) for x in text_encoded])
    
    # Max lengths for padding
    text_encoded_max_length = int(text_encoded_length.max())
    mels_max_length = int(spectrogram_length.max())
    audio_max_length = int(audio_lengths.max())
    
    # Pad tensors
    text_encoded_padded = torch.zeros(
        len(text_encoded), text_encoded_max_length, dtype=torch.long
    )
    audio_padded = torch.zeros(len(audio), audio_max_length, dtype=torch.float32)
    
    n_feats = spectrograms[0].shape[0]
    spectrogram_padded = torch.zeros(
        len(spectrograms), n_feats, mels_max_length, dtype=torch.float32
    )
    
    for i in range(len(filtered_items)):
        text_encoded_padded[i, : text_encoded[i].shape[0]] = text_encoded[i]
        audio_padded[i, : audio[i].shape[-1]] = audio[i]
        spectrogram_padded[i, :, : spectrograms[i].shape[-1]] = spectrograms[i]
    
    return {
        "text": text,
        "text_encoded": text_encoded_padded,
        "text_encoded_length": text_encoded_length,
        "audio_path": audio_path,
        "audio": audio_padded,
        "spectrogram": spectrogram_padded,
        "spectrogram_length": spectrogram_length,
    }