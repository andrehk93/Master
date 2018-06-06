import os
from models import reinforcement_models
import torch
import numpy as np

# Saves checkpoint to disk
def save_checkpoint(state, name, filename='checkpoint.pth.tar'):
    directory = "pretrained/%s/" % (name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    print("Checkpoint successfully saved!")

name = 'reinforced_lstm_cm3_Q_AVG'
checkpoint = 'pretrained/reinforced_lstm_r1_cm3_last/checkpoint.pth.tar'

# Set batch_size:
batch_size = 32
nof_classes = 3
IMAGE_SCALE = 20
IMAGE_SIZE = IMAGE_SCALE*IMAGE_SCALE
output_classes = nof_classes

# Choose network:
LSTM = True
NTM = False
LRUA = False

if LSTM:
    q_network = reinforcement_models.ReinforcedRNN(batch_size, False, nof_classes, IMAGE_SIZE, output_classes=output_classes)
elif NTM:
    q_network = reinforcement_models.ReinforcedNTM(batch_size, False, nof_classes, IMAGE_SIZE, output_classes=output_classes)
elif LRUA:
    q_network = reinforcement_models.ReinforcedLRUA(batch_size, False, nof_classes, IMAGE_SIZE, output_classes=output_classes)

### LOADING PREVIOUS NETWORK ###
if os.path.isfile(checkpoint):
    print("=> loading checkpoint '{}'".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    episode = checkpoint['episode']
    req_dict = checkpoint['requests']
    acc_dict = checkpoint['accuracy']
    total_requests = checkpoint['tot_requests']
    total_accuracy = checkpoint['tot_accuracy']
    total_prediction_accuracy = checkpoint['tot_pred_acc']
    total_loss = checkpoint['tot_loss']
    total_reward = checkpoint['tot_reward']
    best = checkpoint['best']
    q_network.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epoch']
    all_margins = checkpoint['all_margins']
    low_margins = checkpoint['low_margins']
    all_choices = checkpoint['all_choices']

new_choices = []
epochs = 100000
margin_classes = 3
classes = nof_classes
margin_time = 4
print("LENGTH OF MARGINS: ", len(all_choices), " ==", epochs)
input("OK?")
assert(len(all_choices) == epochs)


total = margin_time*margin_classes*classes

for i in range(len(all_choices)):
    new_choices.append([])
for i in range(0, len(all_choices)):
    for m in range(len(all_choices[i])):
       new_choices[i].append(float(all_choices[i][m]/total))

print(len(new_choices), " ==", epochs)
print("EPOCHS: ", epoch)
print(new_choices[-10:-1])
input("OK FINAL?")

input("SAVE?")

save_checkpoint({
    'epoch': epoch,
    'episode': episode,
    'state_dict': q_network.state_dict(),
    'requests': req_dict,
    'accuracy': acc_dict,
    'tot_accuracy': total_accuracy,
    'tot_requests': total_requests,
    'tot_pred_acc': total_prediction_accuracy,
    'tot_loss': total_loss,
    'tot_reward': total_reward,
    'all_margins': all_margins,
    'low_margins': low_margins,
    'all_choices': new_choices,
    'best': best
}, name)

