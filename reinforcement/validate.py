import torch
from torch.autograd import Variable
import torch.nn.functional as F
from transition import Transition

def validate(model, epoch, optimizer, test_loader, args, writer, reinforcement_learner, request_dict, accuracy_dict, episode, print_stats):

    # Initialize training:
    model.eval()

    # Epoch Statistics:
    total_correct = 0.0
    total_requests = 0.0
    total_predict = 0.0
    total_reward = 0.0
    total_episodes = 0.0
    total_iterations = 0.0


    # Training batch loop:
    for i, testdata in enumerate(test_loader):

        # Collect all episode images w/labels:
        episode_images, episode_labels = testdata

        # Episode Statistics:
        episode_correct = 0.0
        episode_predict = 0.0
        episode_request = 0.0
        episode_reward = 0.0
        episode_iter = 0.0

        # Create initial state:
        state = [0 for i in range(args.class_vector_size)]

        # Re-Initialize hidden layer between each episode:
        hidden = model.init_hidden(1)

        # Testing + Training Statistics:
        label_dict = {}
        for v in request_dict.values():
            v.append([])
        for v in accuracy_dict.values():
            v.append([])
        
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
            probs = F.softmax(action)
            action = action.data.max(1)[1].view(1)[0]
            action = torch.LongTensor([action]).view(1, 1)
           
            # Collect Reward:
            reward = reinforcement_learner.collect_reward(action, one_hot_label)

            # Printing softmax scores:
            if (print_stats and i == 0):
                description = "Image Nr\tPredict 0\tPredict 1\tPredict 2\tRequest\tTrue Label"
                string = str(i_e) + ":\t\t"
                for p in probs.data.squeeze():
                    string += str(100.0*p)[0:5] + " %\t\t"
                string += str(true_label)
                if (i_e == 0):
                    print(description)
                print(string)

            # accumulating the reward:
            episode_reward += reward

            # Just some statistics logging:
            if (reward == reinforcement_learner.request_reward):
                episode_request += 1
                episode_predict += 1
                if (label_dict[true_label] in request_dict):
                    request_dict[label_dict[true_label]][-1].append(1)
                if (label_dict[true_label] in accuracy_dict):
                    accuracy_dict[label_dict[true_label]][-1].append(0)
            elif (reward == reinforcement_learner.prediction_reward):
                episode_correct += 1.0
                episode_predict += 1.0
                if (label_dict[true_label] in request_dict):
                    request_dict[label_dict[true_label]][-1].append(0)
                if (label_dict[true_label] in accuracy_dict):
                    accuracy_dict[label_dict[true_label]][-1].append(1)
            else:
                episode_predict += 1.0
                if (label_dict[true_label] in request_dict):
                    request_dict[label_dict[true_label]][-1].append(0)
                if (label_dict[true_label] in accuracy_dict):
                    accuracy_dict[label_dict[true_label]][-1].append(0)


            # Observe next state:
            next_state_start = reinforcement_learner.next_state(action, one_hot_label)

            # Non-final state:
            if (i_e < args.episode_size - 1):
                next_state = torch.cat((torch.FloatTensor(next_state_start), episode_images[i_e + 1][0].squeeze().view(-1)), 0)

            # Final state:
            else:
                next_state = None

            # Update current state:
            state = next_state_start

            ### END TRAIN LOOP ###

        # More status update:
        total_correct += episode_correct
        total_requests += episode_request
        total_predict += episode_predict
        total_reward += float(episode_reward)
        total_episodes += 1
        total_iterations += episode_iter

        if (i == args.batch_size - 1):
            break
    
    print("\n--- Epoch " + str(epoch) + ", Episode " + str(episode + i + 1) + " Statistics ---")
    print("Instance\tAccuracy\tRequests")       
    for key in accuracy_dict.keys():
        prob_list = accuracy_dict[key]
        req_list = request_dict[key]
        
        latest = prob_list[len(prob_list)-i:]
        latest_req = req_list[len(req_list)-i:]
        probs = 0.0
        reqs = 0.0
        req = 0.0
        prob = 0.0
        for l, r in zip(latest, latest_req):
            prob += sum(l)
            probs += len(l)
            reqs += len(r)
            req += sum(r)
        prob /= probs
        req /= reqs
        print("Instance " + str(key) + ":\t" + str(100.0*prob)[0:4] + " %" + "\t\t" + str(100.0*req)[0:4] + " %")
    

    # Even more status update:
    print("\n+------------------STATISTICS----------------------+")
    total_accuracy = float((100.0 * total_correct) / total_predict)
    print("Batch Average Accuracy = " + str(total_accuracy)[:5] +  " %")
    total_requests = float((100.0 * total_requests) / total_iterations)
    print("Batch Average Requests = " + str(total_requests)[:5] + " %")
    total_reward = float(total_reward / total_episodes)
    print("Batch Average Reward = " + str(total_reward)[:5])
    print("+--------------------------------------------------+\n")

    ### LOGGING TO TENSORBOARD ###
    data = {
        'testing_total_requests': total_requests,
        'testing_total_accuracy': total_accuracy,
        'testing_average_reward': total_reward
    }

    for tag, value in data.items():
        writer.scalar_summary(tag, value, epoch)

    return total_requests, total_accuracy, total_reward, request_dict, accuracy_dict


