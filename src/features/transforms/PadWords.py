import torch
class PadWords:
	def __init__(self, fraction):
		self.fraction = fraction
		self.e2w, self.w2e = construct_dict(fraction)



	def __call__(self, data):
		bos_word = []
		eos_word = []
		pad_word = []
		for etype in self.e2w:
			bos_word.append(self.e2w[etype]['%s <BOS>' % etype])
			eos_word.append(self.e2w[etype]['%s <EOS>' % etype])
			pad_word.append(self.e2w[etype]['%s <PAD>' % etype])

		max_len = 512
		if len(data) <= max_len:
			while len(data) < max_len:
				data.append(pad_word)
		else:
			data = data[0:max_len]
			
		
		return torch.tensor(data)



def construct_dict(fraction):
	event2word = {}
	word2event = {}
	tempo_quantize_step = 4
	ctxt_size = 16
	velocity_bins = 32

	for etype in ['Tempo', 'Bar', 'Position', 'Pitch', 'Duration', 'Velocity']:
		count = 0
		e2w = {}

		# Tempo 30 ~ 210
		if etype == 'Tempo':
			for i in range(28, 211, tempo_quantize_step):
				e2w['Tempo %d' % i] = count
				count += 1

		# Bar 0 ~ 15
		elif etype == 'Bar':
			for i in range(ctxt_size):
				e2w['Bar %d' % i] = count
				count += 1

		# Position: 0/16 ~ 15/16
		elif etype == 'Position':
			for i in range(0, fraction):
				e2w['Position %d/16' % i] = count
				count += 1

		# Pitch: 22 ~ 107
		elif etype == 'Pitch':
			for i in range(22, 108):
				e2w['Pitch %d' % i] = count
				count += 1

		# Duration: 0 ~ 63
		elif etype == 'Duration':
			for i in range(64):
				e2w['Duration %d' % i] = count
				count += 1

		# Velocity: 0 ~ 21
		elif etype == 'Velocity':
			for i in range(velocity_bins):
				e2w['Velocity %d' % i] = count
				count += 1

		else:
			raise Exception('etype error')


		e2w['%s <BOS>' % etype] = count
		count += 1
		e2w['%s <EOS>' % etype] = count
		count += 1
		e2w['%s <PAD>' % etype] = count
		count += 1

		event2word[etype] = e2w
		word2event[etype] = {e2w[key]: key for key in e2w}

	return event2word, word2event

