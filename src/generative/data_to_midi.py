import pretty_midi as pyd


def noteseq_to_midi(data, output="test.mid"):
    gen_midi = pyd.PrettyMIDI()
    melodies = pyd.Instrument(program = pyd.instrument_name_to_program('Acoustic Grand Piano'))
    
    minStep = (60/120) * (4 / 6) / 4 
    rest_pitch = 129
    hold_pitch = 128
    local_duration = 0
    time_shift = 0.0
    local_duration = 0
    prev = rest_pitch
    for note in data:
        note = int(note)
        if note < 0 or note > 129:
            continue
        if note == hold_pitch:
            local_duration += 1
        elif note == rest_pitch:
            time_shift += minStep
        else:
            if prev == rest_pitch:
                prev = note
                local_duration = 1
            else:
                i_note = pyd.Note(velocity = 90, pitch = prev, 
                    start = time_shift, end = time_shift + local_duration * minStep)
                melodies.notes.append(i_note)
                prev = note
                time_shift += local_duration * minStep
                local_duration = 1
    if prev != rest_pitch:
        i_note = pyd.Note(velocity = 90, pitch = prev, 
                    start = time_shift, end = time_shift + local_duration * minStep)
        melodies.notes.append(i_note)
    gen_midi.instruments.append(melodies)
    #gen_midi.write(output)
    #print("finish render midi on " + output)