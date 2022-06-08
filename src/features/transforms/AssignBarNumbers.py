import copy
class AssignBarNumbers:
    def __init__(self, ctxt_size):
        self.ctxt_size = ctxt_size

    def __call__(self, ctxt):
        # assign bar number to each note (ranging from 0 ~ n_bars_per_x - 1)
        for i in range(self.ctxt_size):
            for note_tuple in ctxt[0][i]:
                note_tuple[1] = i

        # flatten list from [n_bars, n_notes, 5] to [n_bars * n_notes, 5]
        ctxt = [copy.deepcopy(note_tuple) for bar in ctxt[0] for note_tuple in bar]
        
        return ctxt
