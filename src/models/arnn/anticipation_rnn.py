import torch
import random


from torch import nn
from torch.autograd import Variable
from torch.nn import ModuleList, Embedding
from tqdm import tqdm

import torch.nn.functional as F

import numpy as np
import os

def cuda_variable(tensor, volatile=False):
    if torch.cuda.is_available():
        return Variable(tensor.cuda(), volatile=volatile)
    else:
        return Variable(tensor, volatile=volatile)


def to_numpy(variable: Variable):
    if torch.cuda.is_available():
        return variable.data.cpu().numpy()
    else:
        return variable.data.numpy()

class AnticipationRNN(nn.Module):
    def __init__(self, 
                 note_embedding_dim=20,
                 metadata_embedding_dim=2, #30
                 num_lstm_constraints_units=256,
                 num_lstm_generation_units=256,
                 linear_hidden_size=256, #128
                 num_layers=2, #1
                 dropout_input_prob=0.2,
                 dropout_prob=0.2, #0.5
                 unary_constraint=True,
                 no_metadata=False
                 ):
        super(AnticipationRNN, self).__init__()
        #self.chorale_dataset = chorale_dataset

        # === parameters ===
        # --- common parameters
        self.num_layers = num_layers
        self.num_units_linear = linear_hidden_size
        self.unary_constraint = unary_constraint
        unary_constraint_size = 1 if self.unary_constraint else 0
        self.no_metadata = no_metadata

        # --- notes
        self.note_embedding_dim = note_embedding_dim
        self.num_lstm_generation_units = num_lstm_generation_units
        self.num_notes_per_voice = [131]#, 130, 130, 130]
        # use also note_embeddings to embed unary constraints
        self.note_embeddings = ModuleList(
            [
                Embedding(num_embeddings + unary_constraint_size, self.note_embedding_dim)
                for num_embeddings in self.num_notes_per_voice
            ]
        )
        # --- metadatas
        self.metadata_embedding_dim = metadata_embedding_dim
        self.num_elements_per_metadata = [4] # metadatas = [TickMetadata(subdivision=4)] [metadata.num_values for metadata in self.chorale_dataset.metadatas]
        # must add the number of voices
        self.num_elements_per_metadata.append(1) #voices_ids = [0] #self.chorale_dataset.num_voices)
        #print(self.num_elements_per_metadata)
        # embeddings for all metadata except unary constraints
        if not no_metadata:
            self.metadata_embeddings = ModuleList(
                [
                    Embedding(num_embeddings, self.metadata_embedding_dim)
                    for num_embeddings in self.num_elements_per_metadata
                ]
            )
            self.num_metadata = len(self.num_elements_per_metadata)
        else:
            self.num_metadata = 0
        # nn hyper parameters
        #print(self.metadata_embeddings)
        self.num_lstm_constraints_units = num_lstm_constraints_units
        self.dropout_input_prob = dropout_input_prob
        self.dropout_prob = dropout_prob

        # trainable parameters
        self.lstm_constraint = nn.LSTM(
            input_size=self.metadata_embedding_dim * self.num_metadata
                       + self.note_embedding_dim * unary_constraint_size,
            hidden_size=self.num_lstm_constraints_units,
            num_layers=self.num_layers,
            dropout=dropout_prob,
            batch_first=True)
        #print("meta embed dim", self.metadata_embedding_dim)
        #print("num metadata", self.num_metadata)
        #print("note embed dime", self.note_embedding_dim)
        #print("unary const", unary_constraint_size)
        #print("lstm", self.lstm_constraint)

        self.lstm_generation = nn.LSTM(
            input_size=self.note_embedding_dim + self.num_lstm_constraints_units,
            hidden_size=self.num_lstm_generation_units,
            num_layers=self.num_layers,
            dropout=dropout_prob,
            batch_first=True)

        self.linear_1 = nn.Linear(self.num_lstm_generation_units, linear_hidden_size)
        self.linear_ouput_notes = ModuleList(
            [
                nn.Linear(self.num_units_linear, num_notes)
                for num_notes in self.num_notes_per_voice
            ]
        )
        self.dropout_layer = nn.Dropout2d(p=dropout_input_prob)
        self.hidden_state_h_generation = nn.Parameter(
            data=torch.zeros(num_layers, num_lstm_generation_units)
        )
        self.hidden_state_c_generation = nn.Parameter(
            data=torch.zeros(num_layers, num_lstm_generation_units)
        )

        self.hidden_state_h_constraint = nn.Parameter(
            data=torch.zeros(num_layers, num_lstm_generation_units)
        )
        self.hidden_state_c_constraint = nn.Parameter(
            data=torch.zeros(num_layers, num_lstm_generation_units)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.filepath = os.path.join('models/',
                                     self.__repr__())

    def __repr__(self):
        return f'AnticipationRNN(' \
               f'{self.note_embedding_dim},' \
               f'{self.metadata_embedding_dim},' \
               f'{self.num_lstm_constraints_units},' \
               f'{self.num_lstm_generation_units},' \
               f'{self.num_units_linear},' \
               f'{self.num_layers},' \
               f'{self.dropout_input_prob},' \
               f'{self.dropout_prob},' \
               f'{self.unary_constraint},' \
               f'{self.no_metadata}' \
               f')'

    def generation(self,
                   tensor_chorale,
                   tensor_metadata,
                   temperature,
                   time_index_range):
        # process arguments
        if tensor_chorale is None:
            chorales_gen = self.chorale_dataset.chorale_iterator_gen()
            # original_chorale = next(chorales_gen)
            original_chorale = next(chorales_gen)
        else:
            original_chorale = self.chorale_dataset.tensor_chorale_to_score(tensor_chorale)
        (tensor_chorale,
         tensor_metadata) = self.chorale_dataset.transposed_chorale_and_metadata_tensors(
            original_chorale,
            semi_tone=0)

        constraints_location = torch.zeros_like(tensor_chorale)
        a, b = time_index_range
        if a > 0:
            constraints_location[:, :a] = 1
        if b < constraints_location.size(1) - 1:
            constraints_location[:, b:] = 1
        score, gen_chorale, tensor_metadata = self.generate(
            original_tensor_chorale=tensor_chorale,
            tensor_metadata=tensor_metadata,
            constraints_location=constraints_location,
            temperature=temperature)
        return score, gen_chorale, tensor_metadata

    def init_hidden(self, batch_size, type,
                    volatile=False):
        if type == 'constraint':
            hidden_state_h = self.hidden_state_h_constraint
            hidden_state_c = self.hidden_state_c_constraint
        elif type == 'generation':
            hidden_state_h = self.hidden_state_h_generation
            hidden_state_c = self.hidden_state_c_generation
        else:
            raise NotImplementedError

        hidden = (
            hidden_state_h[:, None, :].expand(self.num_layers,
                                              batch_size,
                                              self.num_lstm_generation_units).contiguous(),

            hidden_state_c[:, None, :].expand(self.num_layers,
                                              batch_size,
                                              self.num_lstm_generation_units).contiguous()
        )
        return hidden

    @staticmethod
    def flatten_chorale(chorale: Variable):
        """
        :param chorale:(batch, num_voices, length, embedding_dim)
        :return: (batch, num_voices * length) with num_voices varying faster
        """
        batch_size, num_voices, length, embedding_dim = chorale.size()
        chorale = chorale.transpose(1, 2).contiguous()
        chorale = chorale.view(batch_size, num_voices * length, embedding_dim)
        return chorale

    @staticmethod
    def flatten_metadata(metadata: Variable):
        batch_size, num_voices, length, num_metadatas = metadata.size()
        metadata = metadata.transpose(1, 2).contiguous()
        metadata = metadata.view(batch_size, num_voices * length, num_metadatas)
        return metadata

    def forward(self, chorale: Variable, metadata: Variable, constraints_loc):
        """
        :param chorale: (batch, num_voices, length in ticks)
        :param metadata: (batch, num_voices, length in ticks, num_metadatas)
        :return: list of probabilities per voice (batch, chorale_length, num_notes)
        """
        batch_size, num_voices, chorale_length = chorale.size()
        sequence_length = num_voices * chorale_length

        # === embed as wrapped sequence ===
        # --- chorale
        x = self.embed_chorale(chorale)

        # --- metadata
        m = self.embed_metadata(metadata, chorale, constraints_location=constraints_loc)

        # === LSTM on constraints ===
        output_constraints = self.output_lstm_constraints(m)

        # === LSTM on notes ===
        offset_seq = torch.cat(
            [cuda_variable(torch.zeros(batch_size, 1, self.note_embedding_dim)),
             x[:, :sequence_length - 1, :]
             ], 1)
        #print(offset_seq.shape)

        if self.dropout_input_prob > 0:
            offset_seq = self.drop_input(offset_seq)

        input = torch.cat([offset_seq, output_constraints], 2)

        hidden = self.init_hidden(batch_size=batch_size, type='generation')

        output_gen, hidden = self.lstm_generation(input, hidden)
        #print(output_gen.shape)

        # distributed NN on output
        weights = [F.relu(self.linear_1(time_slice))
                   for time_slice
                   in output_gen.split(split_size=1,
                                       dim=1)]
        #print([len(weights), len(weights[0]), len(weights[0][0])])
        weights = torch.cat(weights, 1)
        #print(weights.shape)
        weights = weights.view(batch_size, chorale_length, num_voices, self.num_units_linear)
        #print(weights.shape)
        
        # CrossEntropy includes a LogSoftMax layer
        weights = [
            linear_layer(voice[:, :, 0, :])
            for voice, linear_layer
            in zip(weights.split(split_size=1, dim=2), self.linear_ouput_notes)
        ]
        #print([len(weights), len(weights[0]), len(weights[0][0]), len(weights[0][0][0])])
        weights = torch.stack(weights)
        #print(weights.shape)
        return weights

    def drop_input(self, x):
        """
        :param x: (batch_size, seq_length, num_features)
        :return:
        """
        return self.dropout_layer(x[:, :, :, None])[:, :, :, 0]

    def embed_chorale(self, chorale):
        separate_voices = chorale.split(split_size=1, dim=1)
        #print("separate voice", separate_voices[0].shape)
        separate_voices = [
            embedding(voice[:, 0, :])[:, None, :, :]
            for voice, embedding
            in zip(separate_voices, self.note_embeddings)
        ]
        #print(separate_voices[0].shape)
        x = torch.cat(separate_voices, 1)
        x = self.flatten_chorale(chorale=x)
        return x

    def output_lstm_constraints(self, flat_embedded_metadata):
        """

        :param flat_embedded_metadata: (batch_size, length, total_embedding_dim)
        :return:
        """
        batch_size = flat_embedded_metadata.size(0)
        hidden = self.init_hidden(
            batch_size=batch_size, type='constraint'
        )
        # reverse seq
        idx = [i for i in range(flat_embedded_metadata.size(1) - 1, -1, -1)]
        idx = torch.Tensor(idx).long().to(torch.cuda.current_device())#cuda_variable(torch.LongTensor(idx))
        flat_embedded_metadata = flat_embedded_metadata.index_select(1, idx)
        #print(flat_embedded_metadata.shape, hidden[0].shape)
        output_constraints, hidden = self.lstm_constraint(flat_embedded_metadata, hidden)
        output_constraints = output_constraints.index_select(1, idx)
        return output_constraints

    def embed_metadata(self, metadata, chorale=None, constraints_location=None):
        """

        :param metadata: (batch_size, num_voices, chorale_length, num_metadatas)
        :return: (batch_size, num_voices * chorale_length, embedding_dim * num_metadatas
        + note_embedding_dim * (1 if chorale else 0))
        """
        if not self.no_metadata:
            batch_size, num_voices, chorale_length, num_metadatas = metadata.size()
            m = self.flatten_metadata(metadata=metadata)
            separate_metadatas = m.split(split_size=1,
                                         dim=2)
            #print("separate_metadata", separate_metadatas[0].shape)
            #print("separate_metadata", separate_metadatas[0])
            #print("num_metadata", num_metadatas)
            #print(separate_metadatas)
            separate_metadatas = [
                embedding(separate_metadata[:, :, 0])[:, :, None, :]
                for separate_metadata, embedding
                in zip(separate_metadatas, self.metadata_embeddings)
            ]
            m = torch.cat(separate_metadatas, 2)
            # concat all
            m = m.view((batch_size, num_voices * chorale_length, -1))

        # append unary constraints
        if chorale is not None:
            masked_chorale = self.mask_chorale(chorale,
                                               constraints_location=constraints_location)
            #print(masked_chorale)
            masked_chorale_embed = self.embed_chorale(masked_chorale)
            if not self.no_metadata:
                m = torch.cat([m, masked_chorale_embed], 2)
            else:
                m = masked_chorale_embed
        return m

    def mask_chorale(self, chorale, constraints_location=None):
        """
        (batch_size, num_voices, chorale_length)
        :param chorale:
        :return:
        """
        p = random.random() * 0.5
        if constraints_location is None:
            constraints_location = cuda_variable((torch.rand(*chorale.size()) < p).long())
        else:
            assert constraints_location.size() == chorale.size()
            constraints_location = cuda_variable(constraints_location)

        batch_size, num_voices, chorale_length = chorale.size()
        no_constraint = torch.from_numpy(
            np.array([130])
        )
        no_constraint = no_constraint[None, :, None]
        no_constraint = no_constraint.long().clone().repeat(batch_size, 1, chorale_length)
        no_constraint = cuda_variable(no_constraint)
        #print("no_constraint", no_constraint.shape)
        #print(no_constraint)
        return chorale * constraints_location + no_constraint * (1 - constraints_location)

    @staticmethod
    def mean_crossentropy_loss(weights, targets):
        """

        :param weights: list (batch, num_notes) one for each voice
        since num_notes are different
        :param targets:(voice, batch)
        :return:
        """
        cross_entropy = nn.CrossEntropyLoss(size_average=True)
        sum = 0
        for weight, target in zip(weights, targets):
            ce = cross_entropy(weight, target)
            sum += ce
        return sum / len(weights)

    @staticmethod
    def mean_accuracy(weights, targets):
        sum = 0
        for weight, target in zip(weights, targets):
            max_values, max_indices = weight.max(1)
            correct = max_indices == target
            sum += correct.float().mean()

        return sum / len(weights)

    def generate(self,
                 original_tensor_chorale,
                 tensor_metadata,
                 constraints_location,
                 temperature=1.):
        self.eval()
        original_tensor_chorale = cuda_variable(original_tensor_chorale, volatile=True)

        num_voices, chorale_length, num_metadatas = tensor_metadata.size()

        # generated chorale
        gen_chorale = torch.ones([num_voices, chorale_length]).long()
        #gen_chorale = self.chorale_dataset.empty_chorale(chorale_length)

        m = cuda_variable(tensor_metadata[None, :, :, :], volatile=True)
        m = self.embed_metadata(m, original_tensor_chorale[None, :, :],
                                constraints_location=constraints_location[None, :, :])

        output_constraints = self.output_lstm_constraints(m)

        hidden = self.init_hidden(batch_size=1, type='generation')
        subdivision = 4
        for tick_index in range(num_voices * 4 * subdivision - 1):#* self.chorale_dataset.subdivision - 1):
            voice_index = tick_index % num_voices
            # notes
            time_slice = gen_chorale[voice_index, 0]
            time_slice = torch.from_numpy(np.array([time_slice]))[None, :]
            note = self.note_embeddings[voice_index](
                cuda_variable(time_slice, volatile=True)
            )
            time_slice = note
            time_slice_cat = torch.cat(
                (time_slice, output_constraints[:,
                             tick_index + 1: tick_index + 2, :]
                 ), 2)

            output_gen, hidden = self.lstm_generation(time_slice_cat, hidden)

        # generation:
        for tick_index in range(-1, chorale_length * num_voices - 1):
            voice_index = tick_index % num_voices
            time_index = (tick_index - voice_index) // num_voices
            next_voice_index = (tick_index + 1) % num_voices
            next_time_index = (tick_index + 1 - next_voice_index) // num_voices

            if tick_index == -1:
                last_start_symbol = gen_chorale[-1, 0]
                last_start_symbol = torch.from_numpy(np.array([last_start_symbol]))[None, :]
                time_slice = self.note_embeddings[-1](
                    cuda_variable((last_start_symbol)
                                  , volatile=True)
                )
            else:
                time_slice = gen_chorale[voice_index, time_index]
                time_slice = torch.from_numpy(np.array([time_slice]))[None, :]
                note = self.note_embeddings[voice_index](
                    cuda_variable(time_slice, volatile=True)
                )
                time_slice = note

            time_slice_cat = torch.cat(
                (time_slice, output_constraints[:,
                             tick_index + 1: tick_index + 2, :]
                 ), 2
            )

            output_gen, hidden = self.lstm_generation(time_slice_cat, hidden)

            weights = F.relu(self.linear_1(output_gen[:, 0, :]))
            weights = self.linear_ouput_notes[next_voice_index](weights)

            # compute predictions
            # temperature
            weights = weights * temperature
            preds = F.softmax(weights)

            # first batch element
            preds = to_numpy(preds[0])
            new_pitch_index = np.random.choice(np.arange(
                self.num_notes_per_voice[next_voice_index]
            ), p=preds)

            gen_chorale[next_voice_index, next_time_index] = int(new_pitch_index)

        #score = self.chorale_dataset.tensor_chorale_to_score(tensor_chorale=gen_chorale)
        return gen_chorale, tensor_metadata

    def fill(self, ascii_input):
        self.eval()
        # constants
        num_voices = self.chorale_dataset.num_voices
        padding_size = self.chorale_dataset.num_voices * 8 * self.chorale_dataset.subdivision
        temperature = 1.
        chorale_length = len(ascii_input[0])
        # preprocessing
        constraint_metadata = [[
            d[c] if c != 'NC' else len(d) for c in ascii_voice
        ] for d, ascii_voice
            in zip(self.chorale_dataset.note2index_dicts, ascii_input)]

        constraint_metadata = torch.from_numpy(np.array(constraint_metadata)).long()

        constraint_metadata = self.chorale_dataset.extract_metadata_with_padding(
            constraint_metadata[:, :, None], -padding_size, end_tick=chorale_length + padding_size
        )[:, :, 0]
        constraint_metadata = cuda_variable(constraint_metadata, volatile=True)
        constraint_metadata = self.embed_chorale(constraint_metadata[None, :, :])

        other_metadata = cuda_variable(torch.from_numpy(np.array([
            metadata.generate(chorale_length + 2 * padding_size)
            for metadata in self.chorale_dataset.metadatas])), volatile=True)
        # add voice index?!
        other_metadata = torch.cat([other_metadata,
                                    torch.zeros_like(other_metadata)],
                                   0)
        other_metadata = other_metadata.transpose(0, 1)
        other_metadata = other_metadata[None, None, :, :]
        other_metadata = self.embed_metadata(other_metadata)

        tensor_metadata = torch.cat([
            other_metadata,
            constraint_metadata,
        ], 2)

        # generated chorale
        gen_chorale = self.chorale_dataset.empty_chorale(chorale_length)

        output_constraints = self.output_lstm_constraints(tensor_metadata)

        hidden = self.init_hidden(
            batch_size=1,
            type='generation'
        )

        # 1 bar of start symbols
        for tick_index in range(padding_size):
            voice_index = tick_index % self.chorale_dataset.num_voices
            # notes
            time_slice = gen_chorale[voice_index, 0]
            time_slice = torch.from_numpy(np.array([time_slice]))[None, :]
            note = self.note_embeddings[voice_index](
                cuda_variable(time_slice, volatile=True)
            )
            time_slice = note
            # concat with first metadata
            time_slice_cat = torch.cat(
                (time_slice, output_constraints[:,
                             tick_index: tick_index + 1, :]
                 ), 2)

            output_gen, hidden = self.lstm_generation(time_slice_cat, hidden)

        output_constraints = output_constraints[:, padding_size:-padding_size, :]
        # generation:
        for tick_index in range(-1, chorale_length * num_voices - 1):
            voice_index = tick_index % num_voices
            time_index = (tick_index - voice_index) // num_voices
            next_voice_index = (tick_index + 1) % num_voices
            next_time_index = (tick_index + 1 - next_voice_index) // num_voices

            if tick_index == -1:
                last_start_symbol = gen_chorale[-1, 0]
                last_start_symbol = torch.from_numpy(np.array([last_start_symbol]))[None, :]
                time_slice = self.note_embeddings[-1](
                    cuda_variable((last_start_symbol)
                                  , volatile=True)
                )
            else:
                time_slice = gen_chorale[voice_index, time_index]
                time_slice = torch.from_numpy(np.array([time_slice]))[None, :]
                note = self.note_embeddings[voice_index](
                    cuda_variable(time_slice, volatile=True)
                )
                time_slice = note

            time_slice_cat = torch.cat(
                (time_slice, output_constraints[:,
                             tick_index + 1: tick_index + 2, :]
                 ), 2
            )

            output_gen, hidden = self.lstm_generation(time_slice_cat, hidden)

            weights = F.relu(self.linear_1(output_gen[:, 0, :]))
            weights = self.linear_ouput_notes[next_voice_index](weights)

            # compute predictions
            # temperature
            weights = weights * temperature
            preds = F.softmax(weights)

            # first batch element
            preds = to_numpy(preds[0])
            new_pitch_index = np.random.choice(np.arange(
                self.num_notes_per_voice[next_voice_index]
            ), p=preds)

            gen_chorale[next_voice_index, next_time_index] = int(new_pitch_index)

        score = self.chorale_dataset.tensor_chorale_to_score(tensor_chorale=gen_chorale)
        return score, gen_chorale, tensor_metadata
