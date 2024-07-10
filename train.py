import argparse
import trainer


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cuda", default=False, action="store_true")
    parser.add_argument("-m", "--model_non_linear", default=True, action="store_true", help='')  #
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=800)
    parser.add_argument("--past_len", type=int, default=20)
    parser.add_argument("--future_len", type=int, default=40)
    parser.add_argument("--learning_rate", type=int, default=0.00001)
    parser.add_argument("--dataset_file", default="kitti_dataset.json", help="dataset file")
    # parser.add_argument("--info", type=str, default='')
    return parser.parse_args()


def main(config):
    t = trainer.Trainer(config)
    print('start training')
    t.fit()


if __name__ == "__main__":
    config = parse_config()
    main(config)
