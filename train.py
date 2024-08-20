from load_data import *
from model import *
import pandas as pd
import pennylane as qml


###############   Loading Dataset   ###############
data = pd.read_csv("financial_fraud.csv")
dataset = FinancialFraudDataset(data)
train_loader, test_loader = dataset.create_train_test_loaders(n_components=64)

###############   Defining Model   ###############
discriminator_base = Discriminator()
discriminator_unsup = DiscriminatorUnsupervised(discriminator_base).to(device)
discriminator_sup = DiscriminatorSupervised(discriminator_base).to(device)
generator = PatchQuantumGenerator(n_generators, qml.device("lightning.qubit", wires=n_qubits)).to(device)


###########   Defining Loss Functions   ###########
# Binary cross entropy
criterion = nn.BCELoss()
# Categorical cross entropy
cat_criterion = nn.CrossEntropyLoss()


#############   Defining Optimisers   #############
lrG = 0.3  # Learning rate for the generator
lrD = 0.01  # Learning rate for the discriminator
optD1 = optim.SGD(discriminator_unsup.parameters(), lr=lrD)
optD2 = optim.SGD(discriminator_sup.parameters(), lr=lrD)
optG = optim.SGD(generator.parameters(), lr=lrG)


#############                         #############
real_labels = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
fake_labels = torch.full((batch_size,), 0.0, dtype=torch.float, device=device)

# Fixed noise allows us to visually track the generated images throughout training
fixed_noise = torch.rand(8, n_qubits, device=device) * math.pi / 2

counter = 0  # Iteration counter
num_iter = 500  # Number of training iterations

##########   Lists to store Loss Values   ##########
sup_disc_loss_values = []
unsup_disc_loss_values = []
generator_loss_values = []
train_accuracies = []

results = []

while True:
    for i, (data, label, cat_label) in enumerate(train_loader):
        cat_label = cat_label.to(device)

        # Data for training the discriminator
        #data = data.reshape(-1, image_size * image_size)
        real_data = data.to(device)

        # Noise follwing a uniform distribution in range [0,pi/2)
        noise = torch.rand(batch_size, n_qubits, device=device) * math.pi / 2
        fake_data = generator(noise).reshape(batch_size, 1, 8, 8)

        # Training the supervised discriminator
        discriminator_sup.zero_grad()
        outD_sup = discriminator_sup(real_data).view(-1)
        # accuracy
        acc = (outD_sup.argmax(dim=0).item() == label).sum()
        train_accuracies.append(acc)
        errD_sup = cat_criterion(outD_sup, cat_label.view(2))
        # Propagate gradients
        errD_sup.backward()

        optD2.step()

        # Training the unsupervised discriminator
        discriminator_unsup.zero_grad()
        outD_real = discriminator_unsup(real_data).view(-1)
        outD_fake = discriminator_unsup(fake_data.detach()).view(-1)

        errD_real = criterion(outD_real, real_labels)
        errD_fake = criterion(outD_fake, fake_labels)
        # Propagate gradients
        errD_real.backward()
        errD_fake.backward()

        errD = errD_real + errD_fake
        optD1.step()

        # Training the generator
        generator.zero_grad()
        outD_fake = discriminator_unsup(fake_data).view(-1)
        errG = criterion(outD_fake, fake_labels)
        errG.backward()
        optG.step()

        counter += 1

        sup_disc_loss_values.append(errD_sup.cpu())
        unsup_disc_loss_values.append(errD.cpu())
        generator_loss_values.append(errG.cpu())

        # Show loss values
        if counter % 10 == 0:
            print(f'Iteration: {counter}, Supervised Discriminator Loss: {errD_sup:0.3f}, Accuracy: {int(acc)}, Unsupervised Discriminator Loss: {errD:0.3f}, Generator Loss: {errG:0.3f}')

            test_embeddings = generator(fixed_noise).view(1,image_size,image_size).cpu().detach()
            if counter % 50 == 0:
                results.append(test_embeddings)

        if counter == num_iter:
            break
    if counter == num_iter:
        break
