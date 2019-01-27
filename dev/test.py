import torch
from torch.autograd import Variable

# Discount:
GAMMA = 0.5


def validate(q_network, epoch, test_loader, args, reinforcement_learner,
             statistics, text_dataset):
    # Initialize training:
    q_network.eval()

    # Collect a random batch:
    sample_batch, label_batch = test_loader.__iter__().__next__()

    # Episode Statistics:
    episode_correct = 0.0
    episode_predict = 0.0
    episode_request = 0.0
    episode_reward = 0.0

    # Create initial state:
    state = []
    label_dict = []
    for i in range(args.test_batch_size):
        state.append([0 for i in range(args.class_vector_size)])
        label_dict.append({})

    # Initialize q_network between each episode:
    hidden = q_network.reset_hidden(args.test_batch_size)

    # Statistics again:
    request_dict = {1: [], 2: [], 5: [], 10: []}
    accuracy_dict = {1: [], 2: [], 5: [], 10: []}
    prediction_accuracy_dict = {1: [], 2: [], 5: [], 10: []}

    # EPISODE LOOP:
    for i_e in range(args.episode_size):

        # Collecting timestep image/label batch:
        if text_dataset:
            episode_labels, episode_samples = label_batch[:, i_e], sample_batch[:, i_e]
        else:
            episode_labels, episode_samples = label_batch[i_e], sample_batch[i_e]

        # Tensoring the state:
        state = Variable(torch.FloatTensor(state))

        # Create possible next states and update stats:
        one_hot_labels = []
        for i in range(args.test_batch_size):
            true_label = episode_labels.squeeze()[i].item()

            # Creating one hot labels:
            one_hot_labels.append([1 if j == true_label else 0 for j in range(args.class_vector_size)])

            # Logging statistics:
            if true_label not in label_dict[i]:
                label_dict[i][true_label] = 1
            else:
                label_dict[i][true_label] += 1

        # Selecting an action to perform (Epsilon Greedy):
        if text_dataset:
            if args.cuda:
                q_values, hidden = q_network(Variable(episode_samples).type(torch.LongTensor).cuda(), hidden,
                                             class_vector=state, seq=episode_samples.size()[1])
            else:
                q_values, hidden = q_network(Variable(episode_samples).type(torch.LongTensor), hidden,
                                             class_vector=state, seq=episode_samples.size()[1])
        else:
            # Need to add image to the state vector:
            flat_images = episode_samples.squeeze().view(args.test_batch_size, -1)

            # Concatenating possible labels/zero vector with image, to create the environment state:
            state = torch.cat((state, flat_images), 1)

            if args.cuda:
                q_values, hidden = q_network(Variable(state, volatile=True).type(torch.FloatTensor).cuda(),
                                             hidden)
            else:
                q_values, hidden = q_network(Variable(state, volatile=True).type(torch.FloatTensor),
                                             hidden)

        # Choosing the largest Q-values:
        q_network_actions = q_values.data.max(1)[1].view(args.test_batch_size)

        # NOT Performing Epsilon Greedy Exploration:
        agent_actions = q_network_actions

        # Collect rewards:
        rewards = reinforcement_learner.collect_reward_batch(agent_actions,
                                                             one_hot_labels, args.test_batch_size)

        # Collecting average reward at time t over the batch:
        episode_reward += float(sum(rewards) / args.test_batch_size)

        # Log statistics:
        stats = update_dicts(args, episode_labels, rewards, reinforcement_learner, label_dict,
                             request_dict, accuracy_dict, prediction_accuracy_dict)
        episode_predict += stats[0]
        episode_correct += stats[1]
        episode_request += stats[2]

        # Observe next state and images:
        next_state_start = reinforcement_learner.next_state_batch(agent_actions, one_hot_labels, args.test_batch_size)

        # Update current state:
        state = next_state_start

        # End test loop

    for key in request_dict.keys():
        request_dict[key] = sum(request_dict[key]) / len(request_dict[key])
        accuracy_dict[key] = sum(accuracy_dict[key]) / len(accuracy_dict[key])
        prediction_accuracy_dict[key] = float(
            sum(prediction_accuracy_dict[key]) / max(1, len(prediction_accuracy_dict[key])))

    # Validation done
    print("\n---Validation Statistics---\n")

    print("\n--- Epoch " + str(epoch) + " Statistics ---")
    print("Instance\tAccuracy\tRequests")
    for key in accuracy_dict.keys():
        accuracy = accuracy_dict[key]
        request_percentage = request_dict[key]

        print("Instance " + str(key) + ":\t" + str(100.0 * accuracy)[0:4] + " %" + "\t\t" + str(
            100.0 * request_percentage)[0:4] + " %")

    # Even more status update
    print("\n+------------------STATISTICS----------------------+")
    total_prediction_accuracy = float((100.0 * episode_correct) / max(1, episode_predict - episode_request))
    print("Batch Average Prediction Accuracy = " + str(total_prediction_accuracy)[:5] + " %")
    total_accuracy = float((100.0 * episode_correct) / episode_predict)
    print("Batch Average Accuracy = " + str(total_accuracy)[:5] + " %")
    total_requests = float((100.0 * episode_request) / (args.test_batch_size * args.episode_size))
    print("Batch Average Requests = " + str(total_requests)[:5] + " %")
    total_reward = float(episode_reward)
    print("Batch Average Reward = " + str(total_reward)[:5])
    print("+--------------------------------------------------+\n")

    # Update statistics dictionary
    statistics.update(
        {
            'total_test_prediction_accuracy': total_prediction_accuracy,
            'total_test_requests': total_requests,
            'total_test_accuracy': total_accuracy,
            'total_test_reward': total_reward
        },
        {
            'test_req_dict': request_dict,
            'test_acc_dict': accuracy_dict,
            'test_pred_dict': prediction_accuracy_dict,
        }
    )


def update_dicts(args, episode_labels, rewards, reinforcement_learner, label_dict, request_dict, accuracy_dict,
                 prediction_accuracy_dict):
    predict = 0.0
    request = 0.0
    correct = 0.0
    for i in range(args.test_batch_size):
        true_label = episode_labels[i].item()

        # Statistics:
        reward = rewards[i]
        if reward == reinforcement_learner.request_reward:
            request += 1.0
            predict += 1.0
            if label_dict[i][true_label] in request_dict:
                request_dict[label_dict[i][true_label]].append(1)
            if label_dict[i][true_label] in accuracy_dict:
                accuracy_dict[label_dict[i][true_label]].append(0)
        elif reward == reinforcement_learner.prediction_reward:
            correct += 1.0
            predict += 1.0
            if label_dict[i][true_label] in request_dict:
                request_dict[label_dict[i][true_label]].append(0)
            if label_dict[i][true_label] in accuracy_dict:
                accuracy_dict[label_dict[i][true_label]].append(1)
                prediction_accuracy_dict[label_dict[i][true_label]].append(1)
        else:
            predict += 1.0
            if label_dict[i][true_label] in request_dict:
                request_dict[label_dict[i][true_label]].append(0)
            if label_dict[i][true_label] in accuracy_dict:
                accuracy_dict[label_dict[i][true_label]].append(0)
                prediction_accuracy_dict[label_dict[i][true_label]].append(0)

    return predict, correct, request
