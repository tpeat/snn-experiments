import pandas as pd
import ast
import matplotlib.pyplot as plt

def parse_csv(file_path):
    df = pd.read_csv(file_path)
    
    # Convert stringified lists to actual lists using ast.literal_eval
    df['train_losses'] = df['train_losses'].apply(ast.literal_eval)
    df['test_losses'] = df['test_losses'].apply(ast.literal_eval)
    df['test_accuracies'] = df['test_accuracies'].apply(ast.literal_eval)
    
    return df

def plot_train_losses(df, var_name, save_to_disk=False):
    plt.figure(figsize=(10, 6))
    hyperparams = df['hyperparam']
    
    for i, hyperparam in enumerate(hyperparams):
        plt.plot(df['train_losses'][i], label=f"{var_name}: {hyperparam}")
    
    plt.title('Train Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    if save_to_disk:
        plt.savefig(f"train_losses_{var_name}.png")
    
    plt.show()

def plot_test_losses(df, var_name, save_to_disk=False):
    plt.figure(figsize=(10, 6))
    hyperparams = df['hyperparam']
    
    for i, hyperparam in enumerate(hyperparams):
        plt.plot(df['test_losses'][i], label=f"{var_name}: {hyperparam}")
    
    plt.title('Test Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    if save_to_disk:
        plt.savefig(f"test_losses_{var_name}.png")
    
    plt.show()

def plot_test_accuracies(df, var_name, save_to_disk=False):
    plt.figure(figsize=(10, 6))
    hyperparams = df['hyperparam']
    
    for i, hyperparam in enumerate(hyperparams):
        plt.plot(df['test_accuracies'][i], label=f"{var_name}: {hyperparam}")
    
    plt.title('Test Accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    if save_to_disk:
        plt.savefig(f"test_accuracies_{var_name}.png")
    
    plt.show()

def plot_train_test_losses(df, var_name, save_to_disk=False):
    plt.figure(figsize=(10, 6))
    hyperparams = df['hyperparam']
    
    for i, hyperparam in enumerate(hyperparams):
        plt.plot(df['train_losses'][i], label=f"Train Loss (Hyperparam: {hyperparam})", linestyle='-')
        plt.plot(df['test_losses'][i], label=f"Test Loss (Hyperparam: {hyperparam})", linestyle='--')


    plt.title('Train and Test Losses for Different Hyperparameters')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save_to_disk:
        plt.savefig(f"train_test_losses_{var_name}.png")
    
    plt.show()


def parse_and_plot_csv(file_path, var_name, save_to_disk=False):
    df = parse_csv(file_path)
    
    plot_train_losses(df, var_name, save_to_disk)
    plot_test_losses(df, var_name, save_to_disk)
    plot_test_accuracies(df, var_name, save_to_disk)
    plot_train_test_losses(df, var_name, save_to_disk)