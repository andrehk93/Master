import torch
import test


def validate_model(args, statistics, q_network, test_loader, train_loader,
                   text_dataset, reinforcement_module):
    checkpoint_name = "pretrained/" + args.name + "/best.pth.tar"
    print("=> loading checkpoint '{}'".format(checkpoint_name))
    checkpoint = torch.load(checkpoint_name)
    q_network.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(checkpoint_name, checkpoint['epoch']))

    parsed = False
    test_epochs = 0
    while not parsed:
        test_epochs = input("\nHow many epochs to test? \n[0, N]: ")
        try:
            test_epochs = int(test_epochs)
            parsed = True
        except:
            parsed = False

    print("\n--- Testing for", int(test_epochs), "epochs ---\n")
    for epoch in range(args.epochs + 1, args.epochs + 1 + test_epochs):

        # Validating on test-set
        test.validate(q_network, epoch, test_loader, args,
                      reinforcement_module, statistics, text_dataset)

        # Validating on train-set
        test.validate(q_network, epoch, train_loader, args,
                      reinforcement_module, statistics, text_dataset)
