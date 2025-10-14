# main.py
from data_generator import ProcessSimulator, get_task_data
from components.meta_learner import MetaLearner
from training import train
from testing import test

# --- Configuration ---
CONFIG = {
    # Model Hyperparameters
    'd_model': 64,
    'n_heads': 4,
    'n_layers': 2,
    'dropout': 0.1,
    'num_numerical_features': 3,  # cost, time_from_start, time_from_previous

    # Meta-Learning Parameters
    'num_shots_range': (3, 10),  # Vary K between 3 and 10 during training
    'num_queries': 5,
    'num_shots_test': [1, 5, 10],  # K values to evaluate

    # Training Parameters
    'lr': 1e-4,
    'epochs': 5,
    'episodes_per_epoch': 200,
    'num_test_episodes': 100,
}


def main():
    print("1. Generating data...")
    simulator = ProcessSimulator(num_cases=500)

    # Vocabularies for the embedder
    cat_vocabs = {
        'activity': len(simulator.vocab['activity']),
        'resource': len(simulator.vocab['resource']),
    }

    # Generate logs from different process models for training
    log_a = simulator.generate_data_for_model('A')
    log_b = simulator.generate_data_for_model('B')
    log_c = simulator.generate_data_for_model('C')

    # Create task pools for meta-training
    training_tasks = {
        'classification': [
            get_task_data(log_a, 'classification'),
            get_task_data(log_b, 'classification'),
            get_task_data(log_c, 'classification'),
        ],
        'regression': [
            get_task_data(log_a, 'regression'),
            get_task_data(log_b, 'regression'),
            get_task_data(log_c, 'regression'),
        ]
    }

    print("2. Initializing model...")
    model = MetaLearner(
        cat_vocabs=cat_vocabs,
        num_feat_dim=CONFIG['num_numerical_features'],
        d_model=CONFIG['d_model'],
        n_heads=CONFIG['n_heads'],
        n_layers=CONFIG['n_layers'],
        dropout=CONFIG['dropout']
    )

    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    print("\n3. Starting training...")
    train(model, training_tasks, CONFIG)

    print("\n4. Preparing test data from an unseen process...")
    # Generate data from a new process model the learner has never seen
    unseen_log = simulator.generate_data_for_model('D_unseen')
    test_tasks = {
        'classification': get_task_data(unseen_log, 'classification'),
        'regression': get_task_data(unseen_log, 'regression')
    }

    print("\n5. Starting testing...")
    test(model, test_tasks, CONFIG['num_shots_test'], CONFIG['num_test_episodes'])


if __name__ == '__main__':
    main()
