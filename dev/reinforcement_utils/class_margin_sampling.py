from PIL import Image
import random
import torch
from torch.autograd import Variable
import numpy as np
from utils import transforms
import copy


class ClassMarginSampler:

    def __init__(self, args, transform, sentence_length=12):
        # Class Margin Setup
        self.margin_size = args.margin_size
        self.margin_time = args.margin_time

        # Training Setup
        self.nof_classes = args.class_vector_size
        self.episode_size = args.episode_size

        # Image Transforms
        self.transform = transform

        # Text Embedding Setup
        self.sentence_length = sentence_length

        # Statistics
        self.all_margins = []
        self.low_margins = []
        self.all_choices = []

    def sample_images(self, image_batch, label_batch, q_network, batch_size, statistics):

        # Size should be: (episode_size, batch_size, PIXEL_X, PIXEL_Y)
        state = []
        for b in range(batch_size):
            state.append([0 for j in range(self.nof_classes)])

        # Initial parameters:
        hidden = q_network.reset_hidden(batch_size)
        rand_label = random.randint(0, self.nof_classes)
        current_margin_time = 0
        next_class = False
        current_class = 0
        margins = torch.zeros(self.margin_size * self.nof_classes, batch_size)
        choices = np.zeros(self.nof_classes + 1)

        rotations = [np.random.choice(4, batch_size, replace=True) for i in range(int(self.margin_size))]

        # We want to iterate over all images m_c --> [0, 119]:
        for m_c in range(len(image_batch)):

            image_class_batch = image_batch[m_c]

            # So we don't have to transform ALL images (AKA also the ones were not gonna use):
            if next_class and current_class == label_batch[m_c][0]:
                continue
            else:
                next_class = False
                current_class = label_batch[m_c][0]
            
            # Transforming images
            margin_image_batch = self.transform_images(image_class_batch, batch_size, rotations, current_class)
            
            # Get class prediction value:
            # Tensoring the state
            state = torch.FloatTensor(state)
            
            # Need to add image to the state vector:
            state = torch.cat((state, margin_image_batch.view(batch_size, -1)), 1).view(batch_size, -1)
            with torch.no_grad():
                margin, hidden = q_network(Variable(state), hidden)

            # Create new state:
            state = []
            for b in range(batch_size):
                state.append([1 if j == rand_label else 0 for j in range(self.nof_classes)])
            
            # First time new class:
            if current_margin_time < self.margin_time:
                for m_ind in margin.data.max(1)[1]:
                    choices[m_ind] += 1
                margins[current_class] = torch.abs(margin.data.max(1)[0]) + margins[current_class]
                current_margin_time += 1
            
            # n >= margin-time:
            else:
                # Initialize new zero-state
                state = []
                for b in range(batch_size):
                    state.append([0 for j in range(self.nof_classes)])
                # Reinitialize hidden state
                hidden = q_network.reset_hidden(batch_size)
                rand_label = random.randint(0, self.nof_classes)
                current_margin_time = 0
                next_class = True

        # Get the c lowest margin class indexes:
        margin_class_batch = self.compare_margins(margins)

        # Storing the max margin:
        statistics.push_variables({'all_margins': torch.mean(margins.t().max(1)[0]).item(),
                                   'low_margins': torch.mean(margins.t().min(1)[0]).item(),
                                   'all_choices': np.array([float(c/batch_size) for c in choices])
                                   })

        episode_batch_final = torch.FloatTensor(int(self.nof_classes*10), batch_size, 20, 20)
        label_batch_final = torch.LongTensor(int(self.nof_classes*10), batch_size)

        # Iterate over classes to select (meaning batch):
        b = 0
        for m_c in margin_class_batch.t():
            images = []
            for c in m_c:
                for i in range(c*20, c*20 + 20):
                    images.append((image_batch[i][b], c))

            class_indexes = np.random.choice(len(images), int(self.nof_classes*10), replace=False)

            labels = {}
            t = 0
            inds = np.random.choice(self.nof_classes, self.nof_classes, replace=False)
            ind = 0

            for c_i in class_indexes:
                img, label = images[c_i]

                img = self.transform_image(img, rotations[label][b])

                if label not in labels:
                    labels[label] = int(inds[ind])
                    ind += 1

                pseudo_label = labels[label]

                episode_batch_final[t][b] = img
                label_batch_final[t][b] = pseudo_label

                t += 1
            b += 1

        return episode_batch_final, label_batch_final

    def sample_text(self, text_batch, label_batch, q_network, batch_size, statistics):
        # Size should be: (episode_size, Batch_size, SEN_LEN, WORDS)
        state = []
        for b in range(batch_size):
            state.append([0 for j in range(self.nof_classes)])

        # Initial parameters:
        hidden = q_network.reset_hidden(batch_size)
        rand_label = random.randint(0, self.nof_classes)
        current_margin_time = 0
        next_class = False
        current_class = 0
        margins = torch.zeros(self.margin_size * self.nof_classes, batch_size)
        choices = np.zeros(self.nof_classes + 1)

        # We want to iterate over all images m_c --> [0, 119]:
        for m_c in range(len(text_batch[0])):

            text_class_batch = text_batch[:, m_c].squeeze()
            label_class_batch = label_batch[:, m_c]

            # So we dont have to transform ALL images (AKA also the ones were not gonna use):
            if next_class == True and current_class == label_class_batch[0]:
                continue
            else:
                next_class = False
                current_class = label_class_batch[0]

            # Get class prediction value:
            # Tensoring the state:
            state = Variable(torch.FloatTensor(state))

            # Need to add text to the state vector:
            with torch.no_grad():
                margin, hidden = q_network(Variable(text_class_batch), hidden, class_vector=state, seq=text_class_batch.size()[1])

            # Create new state:
            state = []
            for b in range(batch_size):
                state.append([1 if j == rand_label else 0 for j in range(self.nof_classes)])

            # First time new class:
            if current_margin_time < self.margin_time:
                for m_ind in margin.data.max(1)[1]:
                    choices[m_ind] += 1
                margins[current_class] = torch.abs(margin.data.max(1)[0]) + margins[current_class]
                current_margin_time += 1

            # n >= margin-time:
            else:
                state = []
                for b in range(batch_size):
                    state.append([0 for j in range(self.nof_classes)])
                hidden = q_network.reset_hidden(batch_size)
                rand_label = random.randint(0, self.nof_classes)
                current_margin_time = 0
                next_class = True

        # Get the c lowest margin class indexes:
        margin_class_batch = self.compare_margins(margins)

        # Storing the max margin:
        statistics.push_variables({'all_margins': torch.mean(margins.t().max(1)[0]).item(),
                                   'low_margins': torch.mean(margins.t().min(1)[0]).item(),
                                   'all_choices': np.array([float(c / batch_size) for c in choices])
                                   })

        episode_batch_final = torch.zeros(batch_size, int(self.nof_classes * 10), 1,
                                          self.sentence_length).type(torch.LongTensor)
        # episode_batch_final = torch.LongTensor(batch_size, int(self.nof_classes*10), 1)
        label_batch_final = torch.LongTensor(batch_size, int(self.nof_classes * 10))

        # Iterate over classes to select (meaning batch):
        b = 0
        for m_c in margin_class_batch.t():
            texts = []
            for c in m_c:
                for i in range(c*10, c*10 + 10):
                    texts.append(([text_batch[b][i]], label_batch[b][i]))

            class_indexes = np.random.choice(len(texts), int(self.nof_classes*10), replace=False)

            labels = {}
            t = 0
            inds = np.random.choice(self.nof_classes, self.nof_classes, replace=False)
            ind = 0

            for c_i in class_indexes:
                text, label = texts[c_i]
                label = label.item()

                if label not in labels:
                    labels[label] = int(inds[ind])
                    ind += 1

                pseudo_label = labels[label]
                episode_batch_final[b][t] = text[0]

                label_batch_final[b][t] = pseudo_label
                t += 1
            b += 1
        return episode_batch_final, label_batch_final

    def transform_images(self, image_class_batch, batch_size, rotations, current_class):
        margin_image_batch = torch.Tensor(batch_size, 20, 20)

        # Transforming image batch (EITHER HERE OR IN DATALOADER):
        for b in range(batch_size):
            img = Image.fromarray(image_class_batch[b].numpy())
            if self.transform is not None:
                # Applying class specific rotations:
                if rotations[current_class][b] == 1:
                    img = transforms.vflip(img)
                elif rotations[current_class][b] == 2:
                    img = transforms.hflip(img)
                elif rotations[current_class][b] == 3:
                    img = transforms.hflip(transforms.vflip(img))

            img = self.transform(img)

            # Normalizing (pixels are binary):
            threshold = torch.Tensor([0.0])
            img = (img == threshold).float() * 1

            # Adding to new tensor:
            margin_image_batch[b] = img

        return margin_image_batch

    def transform_image(self, image, rotation):
        img = Image.fromarray(image.numpy())
        if self.transform is not None:
            # Applying class specific rotations:
            if rotation == 1:
                img = transforms.vflip(img)
            elif rotation == 2:
                img = transforms.hflip(img)
            elif rotation == 3:
                img = transforms.hflip(transforms.vflip(img))

        img = self.transform(img)

        # Normalizing (pixels are binary):
        threshold = torch.Tensor([0.0])
        img = (img == threshold).float() * 1

        return img

    def compare_margins(self, margins):
        # Get the classes with the lowest margin:
        margin_classes = margins.sort(0, descending=False)[1][0:self.nof_classes, :]

        return margin_classes
