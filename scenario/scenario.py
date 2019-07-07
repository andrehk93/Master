import torch
from torch.autograd import Variable
import torch.nn.functional as F


def run(q_network, scenario_loader, args, reinforcement_learner, text_dataset):
    # Statistics again:
    requests = []
    accuracies = []
    request_percentage = []
    prediction_accuracy_percentage = []
    accuracy_percentage = []

    # Initialize training:
    q_network.eval()

    # Collect a random batch:
    sample_batch, label_batch = scenario_loader.__iter__().__next__()

    # Create initial state:
    state = []
    label_dict = []
    for i in range(args.batch_size):
        state.append([0 for i in range(args.class_vector_size)])
        label_dict.append({})

    # Initialize q_network between each episode:
    hidden = q_network.reset_hidden(args.batch_size)

    # EPISODE LOOP:
    if text_dataset:
        scenario_length = len(label_batch[0])
    else:
        scenario_length = len(label_batch)
    for i_e in range(scenario_length):

        # Collecting timestep image/label batch:
        if text_dataset:
            episode_labels, episode_samples = label_batch[:, i_e], sample_batch[:, i_e]
        else:
            episode_labels, episode_samples = label_batch[i_e], sample_batch[i_e]

        # Tensoring the state:
        state = Variable(torch.FloatTensor(state))

        # Create possible next states and update stats:
        one_hot_labels = []
        for i in range(args.batch_size):
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
            flat_images = episode_samples.squeeze().view(args.batch_size, -1)

            # Concatenating possible labels/zero vector with image, to create the environment state:
            state = torch.cat((state, flat_images), 1)

            if args.cuda:
                q_values, hidden = q_network(Variable(state, volatile=True).type(torch.FloatTensor).cuda(), hidden)
            else:
                q_values, hidden = q_network(Variable(state, volatile=True).type(torch.FloatTensor), hidden)

        q_values = F.softmax(q_values, dim=1)

        # Mean the last output nodes (requests)
        requests.append(torch.mean(q_values.data[:, -1]).item())

        # Mean over all classes softmax values except the last
        batch_mean_accuracies = torch.mean(q_values.data[:, : args.class_vector_size], 0)
        accuracies.append(batch_mean_accuracies.tolist())

        # Choosing the largest Q-values:
        model_actions = q_values.data.max(1)[1].view(args.batch_size)

        # Logging action:
        reqs = 0
        total = 0
        accs = 0
        for a in model_actions:
            if a == args.class_vector_size:
                reqs += 1
            elif a == episode_labels[total]:
                accs += 1
            total += 1

        request_percentage.append(float(reqs / total))
        prediction_accuracy_percentage.append(float(accs / max(1, (total - reqs))))
        accuracy_percentage.append(float(accs / total))

        # NOT Performing Epsilon Greedy Exploration:
        agent_actions = model_actions

        # Observe next state and sample:
        next_state_start = reinforcement_learner.next_state_batch(agent_actions, one_hot_labels, args.batch_size)

        # Update current state:
        state = next_state_start

    return requests, accuracies, request_percentage, prediction_accuracy_percentage, accuracy_percentage


def get_singleclass_representations(batch_size, classes, episode_labels):
    one_hot_labels = []
    for b in range(batch_size):
        true_label = episode_labels.squeeze()[b]
        one_hot_labels.append([1 if j == true_label else 0 for j in range(classes)])

    return one_hot_labels
