import torch
from torch.autograd import Variable
from transition import Transition
import torch.nn.functional as F

def validate(model, epoch, optimizer, test_loader, args, writer, accuracy_dict, episode, criterion):

    # Initialize training:
    model.eval()

    # Collect all episode images w/labels:
    image_batch, label_batch = test_loader.__iter__().__next__()

    # Episode Statistics:
    episode_loss = 0.0
    episode_optimized = 0.0
    episode_correct = 0.0
    episode_predict = 0.0
    episode_optimized = 0.0
    episode_iter = 0.0

    # Create initial state:
    state = []
    label_dict = []
    for i in range(args.batch_size):
        label_dict.append({})
        state.append([0 for i in range(args.class_vector_size)])

    # Initialize model between each episode:
    hidden = model.reset_hidden(args.batch_size)

    # Accuracy statistics:
    for v in accuracy_dict.values():
        v.append([])

    # Initiate empty loss Variable:
    if (args.cuda):
        loss = Variable(torch.zeros(args.batch_size).type(torch.Tensor)).cuda()
    else:
        loss = Variable(torch.zeros(args.batch_size).type(torch.Tensor))

    predictions = []
    test_labels = []

    for i_e in range(len(label_batch)):

        # Collect timestep images/labels:
        episode_images = image_batch[i_e]
        episode_labels = label_batch[i_e]

        # Tensoring the state:
        state = torch.FloatTensor(state)
        
        # Need to add image to the state vector:
        flat_images = episode_images.squeeze().view(args.batch_size, -1)

        # Concatenating possible labels/zero vector with image, thus creating the state:
        state = torch.cat((state, flat_images), 1)


        # Generating actions to choose from the model:
        if (args.cuda):
            actions, hidden = model(Variable(state).type(torch.FloatTensor).cuda(), hidden)
        else:
            actions, hidden = model(Variable(state).type(torch.FloatTensor), hidden)

        predictions.append([pred for pred in F.softmax(actions[0]).data])
        test_labels.append(episode_labels[0])

        if (args.cuda):
            current_loss = criterion(actions, Variable(episode_labels).cuda())
        else:
            current_loss = criterion(actions, Variable(episode_labels))
        
        loss = loss.add(current_loss)

        actions = actions.data.max(1)[1].squeeze()

        one_hot_labels = []
        for i in range(args.batch_size):
            true_label = episode_labels[i]

            # Creating one hot labels:
            one_hot_labels.append([1 if j == true_label else 0 for j in range(args.class_vector_size)])

            # Logging label occurences:
            if (true_label not in label_dict[i]):
                label_dict[i][true_label] = 1
            else:
                label_dict[i][true_label] += 1

            # Logging accuracy:
            if (actions[i] == true_label):
                episode_correct += 1.0
                episode_predict += 1.0
                if (label_dict[i][true_label] in accuracy_dict):
                    accuracy_dict[label_dict[i][true_label]][-1].append(1)
            else:
                episode_predict += 1.0
                if (label_dict[i][true_label] in accuracy_dict):
                    accuracy_dict[label_dict[i][true_label]][-1].append(0)

        # Update next state:
        state = one_hot_labels
            
        ### END EPISODE LOOP ###

    # Averaging the loss over the batch (SGD):
    avg_loss = torch.div(loss.sum(), args.batch_size)

    # More status update:
    total_loss = avg_loss.data[0]

    print("\n--- Epoch " + str(epoch) + ", Episode " + str(episode + i + 1) + " Statistics ---")
    print("Instance\tAccuracy")       
    for key in accuracy_dict.keys():
        prob_list = accuracy_dict[key]
        
        latest = prob_list[len(prob_list)-1:]
        probs = 0.0
        prob = 0.0
        for l in latest:
            prob += sum(l)
            probs += len(l)
        prob /= probs
        print("Instance " + str(key) + ":\t" + str(100.0*prob)[0:4] + " %")
    

    # Even more status update:
    print("\n+------------------STATISTICS----------------------+")
    total_accuracy = float((100.0 * episode_correct) / episode_predict)
    print("Batch Average Accuracy = " + str(total_accuracy)[:5] +  " %")
    total_loss = float(total_loss)
    print("Batch Average Loss = " + str(total_loss)[:5])
    print("+--------------------------------------------------+\n")

    ### LOGGING TO TENSORBOARD ###
    data = {
        'test_total_accuracy': total_accuracy,
        'test_total_loss': total_loss,
    }

    for tag, value in data.items():
        writer.scalar_summary(tag, value, epoch)
    ### DONE LOGGING ###

    return total_accuracy, total_loss, accuracy_dict, predictions, test_labels