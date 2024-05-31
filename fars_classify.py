import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pygraphviz as pgv
from sklearn.metrics import confusion_matrix
from torchview import draw_graph

from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu')  # Mogucnost treniranja pomocu CPU-a ukoliko CUDA jezgra nisu dostupna


def plot_cm(cm): # Konfuziona matrica
    class_names = ['No_Injury', 'Possible_Injury', 'Nonincapaciting_Evident_Injury', 'Incapaciting_Injury',
                   'Fatal_Injury', 'Injured_Severity_Unknown', 'Died_Prior_to_Accident', 'Unknown']
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=(20, 20))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    sns.set(font_scale=10.0)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('confusion_matrix.png')
    plt.show()


def analyze_correlation(model, x_data): #Korelacija parametara
    weights = model.state_dict()
    weight_list = []
    for key in weights:
        if key.startswith("fc"):
            weight_list.extend(weights[key].cpu().numpy().flatten())

    df = pd.DataFrame(x_data)
    df = df.loc[:, (df != 0).any(axis=0)]
    corr_matrix = df.corr()
    plt.figure(figsize=(32, 32))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
    plt.title('Parameter Correlation')
    plt.xlabel('Parameter')
    plt.ylabel('Parameter')
    plt.savefig('correlation_matrix.png')
    plt.show()


def count(y_data):
    fig = sns.countplot(x='Class', hue='Class', data=pd.DataFrame(y_data, columns=['Class']), palette='magma')
    plt.title('Klase')
    fig.figure.savefig("countplot.png")
    plt.close()
    plt.cla()
    plt.clf()


def main():
    # Ucitavanje FARS dataseta koristeci ARFF biblioteku
    dataarff = arff.loadarff('fars.dat')
    data = pd.DataFrame(dataarff[0])

    
    # Dodeljivanje numerickih vrednosti atributima 
    label_encoder = LabelEncoder()
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = label_encoder.fit_transform(data[column])

    # Podela dataseta
    X = data.drop('INJURY_SEVERITY'.strip(), axis=1).values
    y = data['INJURY_SEVERITY'].values

    # SPodela podataka na skupove za treniranje i testiranje
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)

    # Convert numpy arrays to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    # Definisanje neuronske mrze
   
    class Net(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, num_classes)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x

    # Hiper paraemtri
    input_size = X_train.shape[1]
    hidden_size = 128
    num_classes = len(np.unique(y_train))

    model = Net(input_size, hidden_size, num_classes).to(device)

    # Funkcija gubitka i optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    # Treniranje
    num_epochs = 4000
    batch_size = 50000
    X_train.to(device)
    y_train.to(device)
    for epoch in range(num_epochs):
        model.train()
        for i in range(0, X_train.shape[0], batch_size):
            batch_X = X_train[i:i + batch_size].to(device)
            batch_y = y_train[i:i + batch_size].to(device)

            # Forward pass
            outputs = model(batch_X).to(device)
            loss = criterion(outputs, batch_y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 2 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluacija
    model.eval().to(device)  
    with torch.no_grad():
        outputs = model(X_test).to(device)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        print(f'Test accuracy: {accuracy:.4f}')

    cm = confusion_matrix(y_test.cpu().numpy(), predicted.cpu().numpy())
    model_graph = draw_graph(model, input_size=(len(X_train[0]),), expand_nested=True)
    graph = pgv.AGraph(string=str(model_graph.visual_graph))
    graph.layout(prog="dot")
    graph.draw("model.png")
    count(y)
    analyze_correlation(model, X)
    plot_cm(cm)


if __name__ == '__main__':
    main()
