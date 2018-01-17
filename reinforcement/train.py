import torch
from torch.autograd import Variable
from transition import Transition

def train(model, epoch, optimizer, train_loader, args, writer, reinforcement_learner, memory, request_dict, accuracy_dict, episode):

    # Initialize training:
    model.train()

    # Epoch Statistics:
    total_correct = 0.0
    total_loss = 0.0
    total_requests = 0.0
    total_predict = 0.0
    total_reward = 0.0
    total_episodes = 0.0
    total_episode_optimized = 0.0
    total_iterations = 0.0

    # HYPER PARAMETER:
    update_every = 10


    # Training batch loop:
    for i, traindata in enumerate(train_loader):

        # Collect all episode images w/labels:
        episode_images, episode_labels = traindata

        # Episode Statistics:
        episode_loss = 0.0
        episode_optimized = 0.0
        episode_correct = 0.0
        episode_predict = 0.0
        episode_request = 0.0
        episode_reward = 0.0
        episode_optimized = 0.0
        episode_iter = 0.0

        # Create initial state:
        state = [0 for i in range(args.class_vector_size)]

        # Initialize model between each episode:
        hidden = model.reset_hidden(1)
        
        # Zeroing the episode-buffer:
        memory.flush()

        # Statistics again:
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
            action, hidden, _ = reinforcement_learner.select_action(model, state, one_hot_label, hidden, args.cuda, i + episode)
            
            # Collect Reward:
            reward = reinforcement_learner.collect_reward(action, one_hot_label)
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

            # Tensoring the reward:
            reward = torch.Tensor([reward])

            # Observe next state:
            next_state_start = reinforcement_learner.next_state(action, one_hot_label)

            # Non-final state:
            if (i_e < args.episode_size - 1):
                next_state = torch.cat((torch.FloatTensor(next_state_start), episode_images[i_e + 1][0].squeeze().view(-1)), 0)

            # Final state:
            else:
                next_state = None

            # Add transition to memory:
            memory.push_transition(state, action, next_state, reward)
            
            # Update current state:
            state = next_state_start

            if (i_e % update_every == 0):
                loss = optimize_model(model, optimizer, memory, args.cuda, 32)
                if loss != None:
                    total_episode_optimized += 1.0
                    total_loss += loss
                
            ### END TRAIN LOOP ###

        # Add the entire episode to memory:
        memory.push(memory.episode)

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
        
        latest = prob_list[len(prob_list)-int(total_episodes):]
        latest_req = req_list[len(req_list)-int(total_episodes):]
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
    total_prediction_accuracy = float((100.0 * total_correct) / max(1, total_predict-total_requests))
    print("Batch Average Prediction Accuracy = " + str(total_prediction_accuracy)[:5] +  " %")
    total_accuracy = float((100.0 * total_correct) / total_predict)
    print("Batch Average Accuracy = " + str(total_accuracy)[:5] +  " %")
    total_loss = float((total_loss / max(1, total_episode_optimized)))
    print("Batch Average Loss = " + str(total_loss)[:5])
    total_requests = float((100.0 * total_requests) / total_iterations)
    print("Batch Average Requests = " + str(total_requests)[:5] + " %")
    total_reward = float(total_reward / total_episodes)
    print("Batch Average Reward = " + str(total_reward)[:5])
    print("+--------------------------------------------------+\n")

    ### LOGGING TO TENSORBOARD ###
    data = {
        'training_total_requests': total_requests,
        'training_total_accuracy': total_accuracy,
        'training_total_loss': total_loss,
        'training_average_reward': total_reward
    }

    for tag, value in data.items():
        writer.scalar_summary(tag, value, epoch)
    ### DONE LOGGING ###

    return total_prediction_accuracy, total_requests, total_accuracy, total_loss, total_reward, request_dict, accuracy_dict




# Discount factor for future rewards:
GAMMA = 0.5


# Optimizes the model based on a randomly sampled set of size batch size transitions:
def optimize_model(model, optimizer, memory, gpu, batch_size):
    if (len(memory) < batch_size):
        return None

    current_hidden = model.reset_hidden(batch_size)

    ### OPTIMIZING BASED ON PREVIOUSLY STORED TRANSITIONS IN MEMORY ###

    # Collecting a batch of previously seen state-transitions:
    episodes = memory.sample(batch_size)

    size = len(episodes[0])
    if (gpu):
        avg_loss = Variable(torch.zeros(1).type(torch.Tensor)).cuda()
        loss = Variable(torch.zeros(batch_size).type(torch.Tensor)).cuda()
    else:
        avg_loss = Variable(torch.zeros(1).type(torch.Tensor))
        loss = Variable(torch.zeros(batch_size).type(torch.Tensor))

    """
    BPTT:
        Iterating over all episodes in a batch, from start to finish.
    """
    for b in range(size):
        if (gpu):
            action = Variable(torch.cat([episodes[a][b][1] for a in range(batch_size)])).cuda()
            state = Variable(torch.cat([episodes[a][b][0] for a in range(batch_size)])).cuda()
            reward = Variable(torch.cat([episodes[a][b][3] for a in range(batch_size)])).cuda()
        else:
            action = Variable(torch.cat([episodes[a][b][1] for a in range(batch_size)]))
            state = Variable(torch.cat([episodes[a][b][0] for a in range(batch_size)]))
            reward = Variable(torch.cat([episodes[a][b][3] for a in range(batch_size)]))

        predict = True

        if (b < len(episodes[0]) - 1):
            all_next_states = [episodes[a][b][2] if episodes[a][b][2] is not None else None for a in range(batch_size)]
            if (gpu):
                non_final_next_states = Variable(torch.cat([s for s in all_next_states
                                                            if s is not None]), volatile=True).cuda()
                non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
                                              all_next_states))).cuda()
            else:
                non_final_next_states = Variable(torch.cat([s for s in all_next_states
                                                            if s is not None]), volatile=True)
                non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
                                              all_next_states)))
        else:
            predict = False
            discounted_target_value = reward

        # Get Q value, KEEP hidden:
        current_q_values, current_hidden = model(state, current_hidden)
        current_q_value = current_q_values.gather(1, action)

        # Get next Q value:
        if (predict):
            if (gpu):
                target_value = Variable(torch.zeros(batch_size).type(torch.Tensor)).cuda()
            else:
                target_value = Variable(torch.zeros(batch_size).type(torch.Tensor))
            target_value[non_final_mask] = model(non_final_next_states, current_hidden)[0].detach().max(1)[0]
            target_value.volatile = False
            # Calulating the discounted target Q values:
            discounted_target_value = (target_value*GAMMA) + reward

        # Calculating Bellman Loss:
        difference = current_q_value.squeeze().sub(discounted_target_value)
        loss_squared = difference.pow(2).squeeze()
        loss = loss.add(loss_squared * -1.0)

    # Average loss:
    avg_loss = torch.div(loss.sum(), batch_size)

    # Optimize the model:
    optimizer.zero_grad()

    # Backpropagating the loss:
    avg_loss.backward()

    # Do one optimizer step:
    optimizer.step()

    # Record loss:
    return avg_loss.data[0]
