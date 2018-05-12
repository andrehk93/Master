import torch
from torch.autograd import Variable

def train(model, epoch, optimizer, train_loader, args, episode, criterion, batch_size=50):

    # Initialize training:
    model.train()

    # Collect all episode images w/labels:
    image_batch, label_batch = train_loader.__iter__().__next__()

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
    for i in range(batch_size):
        label_dict.append({})
        state.append([0 for i in range(args.class_vector_size)])

    # Initialize model between each episode:
    hidden = model.reset_hidden(batch_size)

    # Initiate empty loss Variable:
    if (args.cuda):
        loss = Variable(torch.zeros(1).type(torch.Tensor)).cuda()
    else:
        loss = Variable(torch.zeros(1).type(torch.Tensor))

    predictions = []
    test_labels = []

    accuracy_dict = {1: [], 2: [], 5: [], 10: []}

    for i_e in range(len(label_batch)):

        # Collect timestep images/labels:
        episode_images = image_batch[i_e]
        episode_labels = label_batch[i_e]

        # Tensoring the state:
        state = torch.FloatTensor(state)
        
        # Need to add image to the state vector:
        flat_images = episode_images.squeeze().view(batch_size, -1)

        # Concatenating possible labels/zero vector with image, thus creating the state:
        state = torch.cat((state, flat_images), 1)


        # Generating actions to choose from the model:
        if (args.cuda):
            actions, hidden = model(Variable(state).type(torch.FloatTensor).cuda(), hidden)
        else:
            actions, hidden = model(Variable(state).type(torch.FloatTensor), hidden)

        if (args.cuda):
            current_loss = criterion(actions, Variable(episode_labels).cuda())
        else:
            current_loss = criterion(actions, Variable(episode_labels))
        
        loss += current_loss


        # Logging stats:
        actions = actions.data.max(1)[1].squeeze()

        one_hot_labels = []
        for i_e in range(args.batch_size):

            true_label = episode_labels[i]

            # Creating one hot labels:
            one_hot_labels.append([1 if j == true_label else 0 for j in range(args.class_vector_size)])

            # Logging label occurences:
            if (true_label not in label_dict[i_e]):
                label_dict[i_e][true_label] = 1
            else:
                label_dict[i_e][true_label] += 1

        for i_e in range(args.batch_size):

            true_label = episode_labels[i]

            # Logging accuracy:
            if (actions[i_e] == true_label):
                episode_correct += 1.0
                episode_predict += 1.0
                if (label_dict[i_e][true_label] in accuracy_dict):
                    accuracy_dict[label_dict[i_e][true_label]].append(1)
            else:
                episode_predict += 1.0
                if (label_dict[i_e][true_label] in accuracy_dict):
                    accuracy_dict[label_dict[i_e][true_label]].append(0)

        # Update next state:
        state = one_hot_labels
            
        ### END EPISODE LOOP ###

    # More status update:
    total_loss = loss.data[0]

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    for key in accuracy_dict.keys():
        accuracy_dict[key] = sum(accuracy_dict[key])/len(accuracy_dict[key])


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