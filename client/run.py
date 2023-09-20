from federated_learning.core.client import Client
from federated_learning.core.utils import ClientInitConfig
from federated_learning.examples.anti_fraud_fl.client.client import AntiFraudClient, create_client


def get_init_config() -> ClientInitConfig:
    init_config = ClientInitConfig(
        id="client_1",
        config={'hidden_dim': 32, 'n_features': 30},
        init_weights=None
    )
    return init_config


def get_client(dataset_path, global_weights_save_folder_path, local_weights_save_folder_path) -> AntiFraudClient:
    init_config = get_init_config()
    client = create_client(init_config, dataset_path, global_weights_save_folder_path, local_weights_save_folder_path)
    return client


def main(dataset_path, global_weights_save_folder_path, local_weights_save_folder_path):
    client = get_client(dataset_path, global_weights_save_folder_path, local_weights_save_folder_path)
    return client


if __name__ == '__main__':
    main()
