import torch
from torch.autograd import Variable
import torch.nn.functional as F
from transition import Transition

def validate(model, epoch, optimizer, test_loader, args, writer, accuracy_dict, episode, print_stats, criterion):

    # Initialize training:
    model.eval()

    # Epoch Statistics:
    total_correct = 0.0
    total_loss = 0.0
    total_predict = 0.0
    total_episodes = 0.0
    total_episode_optimized = 0.0
    total_iterations = 0.0
    update_every = 10


    # Training batch loop:
    for i, testdata in enumerate(test_loader):

        # Collect all episode images w/labels:
        episode_images, episode_labels = testdata

        # Episode Statistics:
        episode_loss = 0.0
        episode_optimized = 0.0
        episode_correct = 0.0
        episode_predict = 0.0
        episode_optimized = 0.0
        episode_iter = 0.0

        # Create initial state:
        state = [0 for i in range(args.class_vector_size)]

        # Initialize model between each episode:
        hidden = model.reset_hidden(1)

        # Statistics again:
        label_dict = {}
        for v in accuracy_dict.values():
            v.append([])
        
        if (args.cuda):
            loss = Variable(torch.zeros(1).type(torch.Tensor), volatile=True).cuda()
        else:
            loss = Variable(torch.zeros(1).type(torch.Tensor), volatile=True)
        for i_e in range(len(episode_labels)):
            true_label = episode_labels[i_e][0]
            episode_image = episode_images[i_e]

            episode_iter += 1

            # Tensoring the state:
            state = torch.FloatTensor(state)
            
            # Need to add image to the state vector:
            flat_image = episode_image.squeeze().view(-1)
            
            one_hot_label = [1 if j == true_label else 0 for j in range(args.class_vector_size)]

            # Concatenating possible labels/zero vector with image:
            state = torch.cat((state, flat_image), 0)

            # Logging statistics:
            if (true_label not in label_dict):
                label_dict[true_label] = 1
            else:
                label_dict[true_label] += 1

            # Selecting an action to perform (Epsilon Greedy),
            # Could maybe also be implemented using memory techniques:
            if (args.cuda):
                action, hidden = model(Variable(state, volatile=True).type(torch.FloatTensor).cuda(), hidden)
            else:
                action, hidden = model(Variable(state, volatile=True).type(torch.FloatTensor), hidden)
            if (args.cuda):
                current_loss = criterion(action, Variable(episode_labels[i_e], volatile=True).cuda())
            else:
                current_loss = criterion(action, Variable(episode_labels[i_e], volatile=True))
            loss = loss.add(current_loss)

            action = action.data.max(1)[1].squeeze()

            # Just some statistics logging:
            if (action[0] == true_label):
                episode_correct += 1.0
                episode_predict += 1.0
                if (label_dict[true_label] in accuracy_dict):
                    accuracy_dict[label_dict[true_label]][-1].append(1)
            else:
                episode_predict += 1.0
                if (label_dict[true_label] in accuracy_dict):
                    accuracy_dict[label_dict[true_label]][-1].append(0)

            # Update next state:
            state = one_hot_label
                
            ### END TRAIN LOOP ###

        loss = torch.div(loss.sum(), args.episode_size)

        # More status update:
        total_correct += episode_correct
        total_predict += episode_predict
        total_episodes += 1
        total_iterations += episode_iter
        total_loss += loss.data[0]

        if (i == args.batch_size - 1):
            break

    print("\n--- Epoch " + str(epoch) + ", Episode " + str(episode + i + 1) + " Statistics ---")
    print("Instance\tAccuracy")       
    for key in accuracy_dict.keys():
        prob_list = accuracy_dict[key]
        
        latest = prob_list[len(prob_list)-int(total_episodes):]
        probs = 0.0
        prob = 0.0
        for l in latest:
            prob += sum(l)
            probs += len(l)
        prob /= probs
        print("Instance " + str(key) + ":\t" + str(100.0*prob)[0:4] + " %")
    

    # Even more status update:
    print("\n+------------------STATISTICS----------------------+")
    total_accuracy = float((100.0 * total_correct) / total_predict)
    print("Batch Average Accuracy = " + str(total_accuracy)[:5] +  " %")
    total_loss = float((total_loss / total_episodes))
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

    return total_accuracy, total_loss, accuracy_dict