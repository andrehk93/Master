import torch
from torch.autograd import Variable
import numpy as np

def train(model, epoch, optimizer, train_loader, args, episode, criterion, batch_size=32, multi_class=False):

    # Initialize training:
    model.train()

    # Collect all episode images w/labels:
    image_batch, label_batch = train_loader.__iter__().__next__()

    image_batch_sequence = torch.cat([images for images in image_batch])
    label_batch_sequence = torch.cat([labels for labels in label_batch]).type(torch.LongTensor)

    if (args.cuda):
        image_batch_sequence = image_batch_sequence.squeeze().view(args.episode_size, batch_size, -1).cuda()
        label_batch_sequence = Variable(label_batch_sequence.view(args.episode_size, batch_size)).cuda()
    else:
        image_batch_sequence = image_batch_sequence.squeeze().view(args.episode_size, batch_size, -1)
        label_batch_sequence = Variable(label_batch_sequence.view(-1, batch_size))

    # Episode Statistics:
    episode_loss = 0.0
    episode_optimized = 0.0
    episode_correct = 0.0
    episode_predict = 0.0
    episode_optimized = 0.0
    episode_iter = 0.0
    total_loss = 0.0

    # Create initial state:
    initial_state_batch = []
    label_dict = []
    for i in range(batch_size):
        label_dict.append({})
        if (multi_class):
            initial_state_batch.append([[0 for i in range(args.class_vector_size)] for c in range(args.class_vector_size)])
        else:
            initial_state_batch.append([0 for i in range(args.class_vector_size)])

    state_batch = []
    label_to_string = None
    for j in range(args.episode_size - 1):
        if (multi_class):
            string_labels, label_to_string = get_multiclass_representations(batch_size, args.class_vector_size, label_batch_sequence[j], label_to_string=label_to_string)
            state_batch.append(string_labels)
        else:
            state_batch.append(get_singleclass_representations(batch_size, args.class_vector_size, label_batch_sequence[j]))
    
    if (args.cuda):
        initial_state_batch = torch.Tensor(initial_state_batch).cuda()
        state_batch = torch.Tensor(state_batch).cuda()
    else:
        initial_state_batch = torch.Tensor(initial_state_batch).view(1, batch_size, args.class_vector_size, args.class_vector_size)
        state_batch = torch.Tensor(state_batch) 


    # Creating states:
    episode_pre_states = torch.cat((initial_state_batch, state_batch)).view(args.episode_size, batch_size, -1)
    episode_states = torch.cat((episode_pre_states, image_batch_sequence), 2)

    
    # Initialize model between each episode:
    hidden = model.reset_hidden(batch_size)

    predictions, hidden = model(Variable(episode_states), hidden, seq=args.episode_size)

    if (args.cuda):
        loss = Variable(torch.zeros(1).type(torch.Tensor)).cuda()
    else:
        loss = Variable(torch.zeros(1).type(torch.Tensor))

    for e in range(args.episode_size):
        pred_lab = Variable(torch.LongTensor(batch_size, args.class_vector_size))
        for b in range(batch_size):
            preds = Variable(torch.LongTensor(label_to_string[b][label_batch_sequence[e][b].data[0]]))
            loss += criterion(predictions[e][b].view(args.class_vector_size, args.class_vector_size), preds)

        

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    accuracy_dict = {1: [], 2: [], 5: [], 10: []}

    for t in range(args.episode_size):
        for b in range(batch_size):

            true_label = label_batch_sequence[t][b].data[0]

            # Logging label occurences:
            if (true_label not in label_dict[b]):
                label_dict[b][true_label] = 1
            else:
                label_dict[b][true_label] += 1

        for b in range(batch_size):

            true_label = label_batch_sequence[t][b].data[0]
            string_label = label_to_string[b][label_batch_sequence[t][b].data[0]]
            predicted_label = []
            for c in range(args.class_vector_size):
                predicted_label.append(predictions[t][b].data[c*args.class_vector_size : (c+1)*args.class_vector_size].view(1, -1).max(1)[1])


            # Logging accuracy:
            if (np.array(predicted_label)==np.array(string_label)).all():
                episode_correct += 1.0
                if (label_dict[b][true_label] in accuracy_dict):
                    accuracy_dict[label_dict[b][true_label]].append(1)
            else:
                if (label_dict[b][true_label] in accuracy_dict):
                    accuracy_dict[label_dict[b][true_label]].append(0)
            episode_predict += 1.0

    # More status update:
    total_loss = loss.data[0]

    for key in accuracy_dict.keys():
        accuracy_dict[key] = float(sum(accuracy_dict[key])/len(accuracy_dict[key]))


    print("\n--- Epoch " + str(epoch) + ", Episode " + str(episode + i + 1) + " Statistics ---")
    print("Instance\tAccuracy")       
    for key in accuracy_dict.keys():
        accuracy = accuracy_dict[key]
        
        print("Instance " + str(key) + ":\t" + str(100.0*accuracy)[0:4] + " %")
    

    # Even more status update:
    print("\n+------------------STATISTICS----------------------+")
    total_accuracy = float((100.0 * episode_correct) / episode_predict)
    print("Batch Average Accuracy = " + str(total_accuracy)[:5] +  " %")
    total_loss = float(total_loss)
    print("Batch Average Loss = " + str(total_loss)[:5])
    print("+--------------------------------------------------+\n")


    return [total_accuracy, total_loss], accuracy_dict


def get_multiclass_representations(batch_size, classes, timestep_label_batch, label_to_string=None):
    label_list = ['a', 'b', 'c', 'd', 'e']
    if (label_to_string == None):
        label_to_string = []
        bits = np.array([[np.array(np.array(np.random.choice(len(label_list), len(label_list), replace=True))) for c in range(classes)] for b in range(batch_size)])
        for b in bits:
            label_to_string.append(b)
    one_hot_vectors = np.array([np.array(np.zeros((len(label_list), len(label_list)))) for b in range(batch_size)])
    for b in range(batch_size):
        true_label = label_to_string[b][timestep_label_batch[b].data[0]]
        for c in range(classes):
            one_hot_vectors[b][c] = [0 if true_label[c] != j else 1 for j in range(classes)]
    return one_hot_vectors, label_to_string

def get_singleclass_representations(batch_size, classes, timestep_label_batch):
    one_hot_labels = []
    for b in range(batch_size):
        true_label = timestep_label_batch[b].data[0]
        one_hot_labels.append([1 if j == true_label else 0 for j in range(classes)])

    return one_hot_labels