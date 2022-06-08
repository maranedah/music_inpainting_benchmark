
def select_midi_process(process):
    if process == "noteseq":
        from .encoded_midi_noteseq import EncodedMidiNoteSeq
        midi_process = EncodedMidiNoteSeq
    if process == "mxl_noteseq":
        from .encoded_mxl_noteseq import EncodedMxlNoteSeq
        midi_process = EncodedMxlNoteSeq
    if process == "remi":
        from .encoded_midi_remi import EncodedMidiRemi
        midi_process = EncodedMidiRemi
    return midi_process

class EncodedMidi:

    def from_path(file, process, dataset_name, fraction, min_length):
        midi_process = select_midi_process(process)
        return midi_process(file, fraction, dataset_name, min_length)
        